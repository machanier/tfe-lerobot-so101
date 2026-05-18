"""
src.pipeline - Orchestration perception -> planning -> control pour pick-and-place.

C'est le module qui MET TOUT ENSEMBLE :
  1. Capture caméras + état robot.
  2. Détection 2D (HSV ou HF).
  3. Reconstruction 3D dans le repère base.
  4. Trouve l'objet cible dans la scène.
  5. Grasp planning (TopDownGrasp).
  6. IK pour les 3 poses (approach/grasp/retract).
  7. Génération de trajectoire articulaire complète :
       config_courante -> approach (pince ouverte)
                       -> grasp (descendre)
                       -> grasp (fermer pince)
                       -> retract (remonter)
                       -> drop_above (au-dessus de la boite)
                       -> drop_release (ouvrir pince)
                       -> rest (config zero)
  8. Execution sur le robot via MotorController.

Usage via le CLI :
  python scripts/pick_and_place.py --target orange_cube --detector hf
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ============================================================
# Constantes display
# ============================================================
# Ordre des cameras dans le display horizontal. cam_1 a GAUCHE pour respecter
# la perspective de l'utilisateur derriere le robot : cam_1 est physiquement
# a gauche du robot vu de face (et donc a droite vu de derriere/par cam_0).
# cam_2 (eye-in-hand) toujours a droite.
CAM_DISPLAY_ORDER = ("cam_1", "cam_0", "cam_2")

# Taille de chaque tile dans la mosaic (640x360 = taille originale, 960x540
# = x1.5). Mosaic finale = 3 * tile_width x tile_height.
DISPLAY_TILE_SIZE = (960, 540)  # 1.5x la taille originale

from src.calibration.forward_kinematics import ARM_JOINTS
from src.control.ik import IKResult, IKSolver
from src.control.motor_controller import MotorController
from src.control.trajectory import (
    JointTrajectory,
    chain_trajectories,
    estimate_duration_safe,
    quintic_trajectory,
)
from src.perception.camera_io import MultiCamera
from src.perception.detector import (
    HFDetector,
    HSVDetector,
    default_hf_labels,
    default_hsv_specs,
    load_hf_specs,
    load_hsv_specs,
)
from src.perception.pose_estimator import PoseEstimator
from src.perception.robot_state import RobotStateProvider
from src.perception.scene import Scene
from src.planning.grasp import TopDownGrasp


# ============================================================
# Configuration du pipeline
# ============================================================


@dataclass
class PipelineConfig:
    """Hyperparametres du pipeline pick-and-place."""
    target_label: str = "orange_cube"
    detector_kind: str = "hsv"          # "hsv" ou "hf"
    hsv_specs_path: Optional[Path] = None
    hf_specs_path: Optional[Path] = None
    scene_config_path: Optional[Path] = None    # configs/scene.json
    motor_port: str = ""                # a fournir
    max_velocity_rad_s: float = 0.5     # vitesse articulaire max (prudent)
    grip_close_pct: float = 5.0         # fermer la pince a 5% pour grasper
    grip_open_pct: float = 100.0
    pause_grasp_s: float = 1.0          # attente apres fermeture pince
    pause_release_s: float = 0.5
    dry_run: bool = False               # True : pas d'envoi moteur, juste log
    closed_loop: bool = True            # Sprint 4 : raffinement cam_2 avant grasp
    closed_loop_max_correction_mm: float = 80.0  # rejette correction > 8 cm (sanity)
    display: bool = False               # Affiche les frames camera (cv2.imshow) aux moments cles

    # Pose intermediaire "safe" entre drop_release et home (utile pour
    # eviter que le bras traverse la zone de l'objet pendant le retour).
    safe_intermediate_angles_rad: Optional[dict] = None

    # STRATEGIE DE RETOUR FINAL :
    # Si True (defaut) : le robot revient EXACTEMENT a la pose dans laquelle
    # il etait au lancement du script (capturee dans run()). Pratique :
    # Maxence place le robot en position stable + camera bien orientee, lance
    # la commande, et le robot y revient apres la pose.
    # Si False : utilise la pose fixe `home_angles_rad` ci-dessous.
    home_from_session_start: bool = True

    # Pose "home" fixe (utilisee uniquement si home_from_session_start=False).
    # Defaut : bras replie au-dessus de lui-meme, pince vers le bas.
    home_angles_rad: Optional[dict] = None

    # Vitesse RALENTIE pour la transition finale (home), pour eviter
    # l'impression de chute. Defaut : 0.3 rad/s (vs 0.5 par defaut).
    home_max_velocity_rad_s: float = 0.3

    # Compensation SYSTEMATIQUE des biais de calibration mesures
    # empiriquement (cf D11 : biais Y ~+28mm sur ce poste). Sera soustraite
    # a toutes les positions detectees par la stereo.
    # = np.array([dx, dy, dz]) en metres. None = pas de compensation.
    systematic_bias_correction_m: Optional[object] = None  # ndarray (3,)


# ============================================================
# Orchestrateur principal
# ============================================================


class PickAndPlacePipeline:
    """Pipeline complet pour saisir un objet et le poser dans la boite.

    Usage typique :
        config = PipelineConfig(target_label="orange_cube", motor_port="/dev/...")
        pipeline = PickAndPlacePipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._load_scene()
        self._build_perception()
        self._grasp_strategy = TopDownGrasp()
        self._ik = IKSolver()
        # IK SPECIFIQUE A LA DEPOSE : on reduit drastiquement le poids de la
        # rotation (0.01 vs 0.1 par defaut). Justification : pour la depose,
        # on ouvre la pince et on lache le cube -- l'orientation exacte de la
        # pince importe peu (elle peut etre inclinee de 10-15deg sans gener).
        # En revanche la POSITION doit etre precise pour viser la boite.
        # Avec rot_w=0.1, l'IK choisissait de preserver l'orientation au prix
        # de ~5-13 cm d'erreur position (cf logs : drop_above approx
        # trans=47mm rot=0.8deg). Avec rot_w=0.01 : trans<1mm rot~10deg.
        self._ik_drop = IKSolver(rotation_weight=0.01)
        self._provider = RobotStateProvider()

    def _load_scene(self):
        """Charge la position de la boite de depose."""
        scene_path = self.config.scene_config_path or (REPO / "configs" / "scene.json")
        if not scene_path.exists():
            raise FileNotFoundError(f"scene.json manquant : {scene_path}")
        data = json.load(open(scene_path))
        box = data["drop_box"]
        self.drop_position = np.array(box["center_base_m"], dtype=np.float64)
        # Ajout d'une marge en Z pour relacher au-dessus de la boite
        box_height = float(box["dimensions_m"][2])
        self.drop_above = self.drop_position + np.array([0.0, 0.0, 0.05 + box_height / 2])
        self.drop_release = self.drop_position + np.array([0.0, 0.0, 0.02 + box_height / 2])

    def _build_perception(self):
        """Construit detecteur + estimator selon la config."""
        if self.config.detector_kind == "hsv":
            specs_path = self.config.hsv_specs_path or (REPO / "configs/perception/hsv_specs.json")
            specs = load_hsv_specs(specs_path) if specs_path.exists() else default_hsv_specs()
            self._detector = HSVDetector(specs)
            self._specs_meta = {s.label: s.meta for s in specs}
            self._label_mapping = {}  # HSV : labels = label internes directement
        elif self.config.detector_kind == "hf":
            hf_path = self.config.hf_specs_path or (REPO / "configs/perception/hf_specs.json")
            if hf_path.exists():
                cfg = load_hf_specs(hf_path)
                labels = cfg["labels"]
                model_name = cfg.get("model_name", "google/owlv2-base-patch16-ensemble")
                threshold = float(cfg.get("score_threshold", 0.15))
                self._label_mapping = cfg.get("_label_mapping") or {}
            else:
                labels = default_hf_labels()
                model_name = "google/owlv2-base-patch16-ensemble"
                threshold = 0.15
                self._label_mapping = {}
            self._detector = HFDetector(prompt_labels=labels,
                                        model_name=model_name,
                                        score_threshold=threshold)
            self._specs_meta = {}
        else:
            raise ValueError(f"detector_kind inconnu: {self.config.detector_kind}")
        self._estimator = PoseEstimator(specs_by_label=self._specs_meta)

    # ----- main entry point -----------------------------------------------

    def run(self):
        """Execute un cycle complet de pick-and-place."""
        print("=" * 70)
        print(f" PICK-AND-PLACE — cible: '{self.config.target_label}', "
              f"detector: {self.config.detector_kind}")
        print("=" * 70)
        print()

        # ============================================================
        # 1. Connexion
        # ============================================================
        # IMPORTANT : initialiser controller AVANT le try, sinon le finally
        # referencera un nom non defini si MotorController() leve.
        controller = None
        try:
            controller = MotorController()
            if not self.config.dry_run:
                self._provider.connect_live(self.config.motor_port)
                controller.connect(self.config.motor_port)
                controller.enable_torque()
        except Exception as e:
            print(f"!! Connexion robot/moteur impossible : {e}")
            if not self.config.dry_run:
                print("   Bascule en mode --dry-run.")
                self.config.dry_run = True

        # IMPORTANT : la MultiCamera DOIT rester ouverte pendant TOUT le
        # pipeline car Sprint 4 (closed_loop) a besoin de cam_2 APRES la
        # phase 1 d'execution. Sinon le 'with MultiCamera() as mc' se ferme
        # apres la perception initiale et le refinement crash avec
        # "MultiCamera n'est pas ouvert".
        try:
            mc = MultiCamera()
            mc.open()

            # ============================================================
            # 2. Perception (UNE seule capture initiale)
            # ============================================================
            print(">> Perception en cours...")
            rs = (self._provider.read_live() if not self.config.dry_run
                  else self._provider.from_angles({j: 0.0 for j in ARM_JOINTS}))
            # Warmup pour stabiliser autoexposure
            for _ in range(3):
                mc.grab(robot_state=rs)
                time.sleep(0.1)
            frames = mc.grab(robot_state=rs)
            dets_by_cam = self._detector.detect_multi(frames)
            # Si HF avec mapping : renomme les labels
            if self._label_mapping:
                for cam_key, dets in dets_by_cam.items():
                    for d in dets:
                        if d.label in self._label_mapping:
                            d.label = self._label_mapping[d.label]
            # === DEBUG : detections brutes par camera (TOUTES) ===
            print("   Detections brutes par camera :")
            for cam_key in ("cam_0", "cam_1", "cam_2"):
                if cam_key in dets_by_cam:
                    dets = dets_by_cam[cam_key]
                    if dets:
                        dets_sorted = sorted(dets, key=lambda d: -d.score)
                        print(f"     {cam_key} : {len(dets)} detections")
                        for d in dets_sorted:
                            bbox_size = ""
                            if d.bbox:
                                w = d.bbox[2] - d.bbox[0]
                                h = d.bbox[3] - d.bbox[1]
                                bbox_size = f"  bbox={int(w)}x{int(h)}px"
                            print(f"        {d.label:<22} s={d.score:.2f}"
                                  f"  center=({int(d.center_px[0])},{int(d.center_px[1])}){bbox_size}")
                    else:
                        print(f"     {cam_key} : 0 detection")
            scene = self._estimator.build_scene(dets_by_cam, frames)
            # === DEBUG : si scene vide alors qu'on avait des dets ===
            if not scene.objects and any(dets_by_cam.get(k) for k in dets_by_cam):
                print()
                print("   [DIAGNOSTIC] Detections 2D presentes mais aucun objet 3D :")
                rejections = getattr(self._estimator, "_last_rejections", {})
                if rejections:
                    for label, reason in rejections.items():
                        print(f"     {label:<22} : {reason}")
                from src.perception.pose_estimator import (
                    PoseEstimator, PoseEstimatorConfig)
                cfg_loose = PoseEstimatorConfig(max_reproj_error_px=200.0)
                est_no_filter = PoseEstimator(
                    config=cfg_loose,
                    specs_by_label=self._specs_meta,
                    load_scene_config=False,
                )
                scene_raw = est_no_filter.build_scene(dets_by_cam, frames)
                if scene_raw.objects:
                    print("   Avec seuils relaches (reproj 200 px, sans scene.json) :")
                    for o in scene_raw.objects:
                        p = o.position_base_m * 1000
                        err = o.meta.get("reproj_error_px", -1)
                        print(f"     {o.label:<22} pos=({p[0]:+6.1f},{p[1]:+6.1f},{p[2]:+6.1f}) mm  "
                              f"reproj_err={err:.1f} px")
            print(scene.pretty())
            print()

            # === DISPLAY OPTIONNEL : montre les 3 frames avec detections ===
            if self.config.display:
                self._show_perception_snapshot(frames, dets_by_cam, scene,
                                                title="Perception initiale (Sprint 2)")

            # ============================================================
            # 3. Trouve l'objet cible
            # ============================================================
            target = next((o for o in scene.objects if o.label == self.config.target_label), None)
            if target is None:
                print(f"!! Objet '{self.config.target_label}' NON DETECTE dans la scene. Annule.")
                return
            print(f">> Cible '{target.label}' a position "
                  f"({target.position_base_m[0]*1000:+.1f}, "
                  f"{target.position_base_m[1]*1000:+.1f}, "
                  f"{target.position_base_m[2]*1000:+.1f}) mm")
            print()

            # ============================================================
            # 4. Grasp planning
            # ============================================================
            grasp_pose = self._grasp_strategy.plan(target)
            if grasp_pose is None:
                print(f"!! Grasp planning a echoue (objet trop haut ?). Annule.")
                return
            print(f">> Grasp planifie ({grasp_pose.meta.get('strategy')}, "
                  f"yaw={np.degrees(grasp_pose.meta.get('yaw_rad', 0)):+.0f}deg)")

            # ============================================================
            # 5. IK pour les 3 poses + drop
            # ============================================================
            print(">> Cinematique inverse...")
            current_state = (self._provider.read_live() if not self.config.dry_run
                             else self._provider.from_angles({j: 0.0 for j in ARM_JOINTS}))
            q_current = current_state.joint_angles_rad
            # MEMORISE la pose de depart : c'est notre future "home" si
            # home_from_session_start=True. Le robot reviendra exactement la.
            q_session_start = dict(q_current)

            r_app, r_grp, r_ret = self._ik.solve_grasp_pose(grasp_pose, q_init=q_current)
            # IK pour la pose drop_above et drop_release : on utilise l'IK
            # specialise self._ik_drop qui a un poids rotation reduit (0.01).
            # Cela permet de privilegier la POSITION (precision au mm pour
            # viser la boite) au prix d'une orientation pince approximative
            # (acceptable : la pince ouvre puis le cube tombe -- l'incli-
            # naison de ~10deg n'empeche pas le drop). Avec l'IK standard,
            # la pose drop ratait la cible de 5-13 cm (sous-actuation 5/6 DDL).
            from src.planning.grasp import _rotation_top_down, _se3
            T_drop_above = _se3(_rotation_top_down(0.0), self.drop_above)
            T_drop_release = _se3(_rotation_top_down(0.0), self.drop_release)
            r_drop_above = self._ik_drop.solve(T_drop_above, q_init=r_ret.joint_angles_rad)
            r_drop_release = self._ik_drop.solve(T_drop_release, q_init=r_drop_above.joint_angles_rad)

            for label, r in [("approach", r_app), ("grasp", r_grp),
                             ("retract", r_ret), ("drop_above", r_drop_above),
                             ("drop_release", r_drop_release)]:
                tag = "OK" if r.converged else "approx"
                print(f"   {label:<14} {tag:<6} "
                      f"trans={r.translation_err_mm:5.1f}mm rot={r.rotation_err_deg:5.1f}deg")
            print()

            # ============================================================
            # 6. Generation trajectoire + execution
            # Si closed_loop : on EXECUTE jusqu'a approach, puis on raffine
            # avec cam_2, puis on EXECUTE la suite avec correction.
            # Sinon : trajectoire complete d'un coup.
            # ============================================================
            # Helper : callback display live qui rafraichit cv2.imshow
            # pendant l'execution moteur (toutes les ~30 frames de traj).
            #
            # IMPORTANT : on N'APPELLE PAS le detecteur ici. Avec HF (OWL-ViTv2)
            # une detection prend ~3-5 sec sur M4 -> le callback bloquerait
            # chaque rafraichissement et ralentirait la trajectoire d'un
            # facteur 10x (0.2 fps observe par Maxence). Le display sert juste
            # a "voir" le bras pendant le mouvement, pas a re-detecter.
            def make_live_callback():
                if not self.config.display:
                    return None
                import cv2
                def on_step(i, trajectory):
                    # Lit la pose courante du robot pour mettre a jour T_base_cam2
                    try:
                        rs = self._provider.read_live()
                    except Exception:
                        return
                    frames = mc.grab(robot_state=rs)
                    # Mosaic des 3 cams : on affiche les frames BRUTES (pas de
                    # detection) -> rafraichissement quasi-instantane meme en HF.
                    import numpy as np
                    tiles = [self._annotate_frame(frames.get(k), [], None)
                             for k in CAM_DISPLAY_ORDER]
                    mosaic = np.hstack(tiles)
                    cv2.putText(mosaic, f"LIVE exec frame {i}/{len(trajectory)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Pipeline perception", mosaic)
                    cv2.waitKey(1)
                return on_step

            # Determine la pose home finale : soit la pose de depart de la
            # session (defaut, le robot revient ou Maxence l'avait place), soit
            # la pose fixe configuree dans home_angles_rad.
            if self.config.home_from_session_start:
                q_home_final = q_session_start
                home_origin = "pose de depart session"
            else:
                q_home_final = self.config.home_angles_rad
                home_origin = "pose fixe (home_angles_rad)"

            if self.config.closed_loop and not self.config.dry_run:
                # --- Phase 1 : trajectoire courant -> approach ---
                print(">> Phase 1 : courant -> approach (boucle fermee Sprint 4)")
                traj_phase1 = self._build_phase1_trajectory(q_current, r_app)
                print(f"   {len(traj_phase1)} points, duree {traj_phase1.duration_s:.1f}s")
                controller.execute_trajectory(traj_phase1, verbose=True,
                                              on_step=make_live_callback())
                print()

                # --- Raffinement cam_2 ---
                print(">> Raffinement cam_2 (eye-in-hand)...")
                from src.control.closed_loop import (
                    apply_correction_to_grasp_pose,
                    refine_grasp_with_cam2,
                )
                # Re-lit l'etat robot (on est arrive a approach, qui peut differer
                # un peu de la cible IK a cause de la precision des moteurs)
                rs_at_approach = self._provider.read_live()
                # IMPORTANT : la MultiCamera est encore ouverte (cf 'with mc' plus haut)
                refinement = refine_grasp_with_cam2(
                    target_label=self.config.target_label,
                    detector=self._detector,
                    multi_camera=mc,
                    robot_state=rs_at_approach,
                    # IMPORTANT : on passe la hauteur ATTENDUE de l'objet
                    # (= Z de la pose grasp triangulée) pour que la formule
                    # de conversion pixel->m utilise la VRAIE distance
                    # cam_2 -> objet, pas une valeur en dur.
                    target_z_base_m=float(grasp_pose.T_base_gripper_grasp[2, 3]),
                    label_mapping=self._label_mapping,
                )
                print(f"   {refinement.message}")
                if refinement.confidence < 0.1:
                    print("   [WARN] confiance faible, correction NON appliquee "
                          "(grasp utilise la pose stereo seule)")
                elif refinement.delta_norm_mm > self.config.closed_loop_max_correction_mm:
                    print(f"   [WARN] correction {refinement.delta_norm_mm:.1f} mm > seuil "
                          f"{self.config.closed_loop_max_correction_mm} mm. "
                          "Suspect, on NE corrige PAS (peut etre fausse detection).")
                else:
                    apply_correction_to_grasp_pose(grasp_pose, refinement.delta_base_m)
                    print(f"   Correction appliquee (norme {refinement.delta_norm_mm:.1f} mm)")
                    # Refaire l'IK pour les poses corrigees
                    r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                        grasp_pose, q_init=rs_at_approach.joint_angles_rad)
                    print(f"   IK re-resolue avec poses corrigees.")
                print()

                # --- Phase 2 : trajectoire approach -> grasp -> retract -> drop ---
                print(f">> Phase 2 : descente + saisie + depot + retour ({home_origin})")
                q_at_approach = rs_at_approach.joint_angles_rad
                traj_phase2 = self._build_phase2_trajectory(
                    q_at_approach,
                    [r_app, r_grp, r_ret, r_drop_above, r_drop_release],
                    q_home=q_home_final,
                )
                print(f"   {len(traj_phase2)} points, duree {traj_phase2.duration_s:.1f}s")
                controller.execute_trajectory(traj_phase2, verbose=True,
                                              on_step=make_live_callback())
                print(f">> Termine. Robot revenu a la {home_origin}.")
                if self.config.display:
                    import cv2
                    cv2.destroyAllWindows()

            else:
                # Mode sans boucle fermee OU dry-run : trajectoire complete
                print(">> Generation trajectoire complete (sans boucle fermee)...")
                traj = self._build_full_trajectory(
                    q_current,
                    [r_app, r_grp, r_ret, r_drop_above, r_drop_release],
                    q_home=q_home_final,
                )
                print(f"   {len(traj)} points, duree {traj.duration_s:.1f}s")
                print()
                if self.config.dry_run:
                    print(">> DRY RUN : pas d'execution sur le robot.")
                else:
                    print(f">> Execution sur le robot (jusqu'au retour {home_origin})...")
                    controller.execute_trajectory(traj, verbose=True,
                                                  on_step=make_live_callback())
                    print(f">> Termine. Robot revenu a la {home_origin}.")
                    if self.config.display:
                        import cv2
                        cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\nInterrompu par utilisateur.")
            self._safe_stop_with_torque(controller)
        except Exception as e:
            print(f"\n!! EXCEPTION pendant le pipeline : {type(e).__name__}: {e}")
            self._safe_stop_with_torque(controller)
            raise
        finally:
            try:
                mc.close()
            except Exception:
                pass
            try:
                self._provider.disconnect_live()
            except Exception:
                pass
            if controller is not None:
                try:
                    controller.disconnect()
                except Exception:
                    pass

    def _annotate_frame(self, frame, dets, scene):
        """Helper : annote une frame avec detections + reprojections 3D.

        Taille des tuiles fixee par DISPLAY_TILE_SIZE (constante au top du module).
        """
        import cv2
        import numpy as np
        from src.perception.pose_estimator import _projection_matrix
        tw, th = DISPLAY_TILE_SIZE
        if frame is None:
            return np.zeros((th, tw, 3), dtype=np.uint8)
        img = frame.image.copy()
        # Bandeau noir avec le nom de la cam (utile vu que l'ordre n'est plus
        # alphabetique : cam_1 a gauche maintenant).
        cv2.rectangle(img, (0, 0), (260, 50), (0, 0, 0), -1)
        cv2.putText(img, frame.cam_key, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        for d in dets:
            if d.bbox:
                x0, y0, x1, y1 = (int(v) for v in d.bbox)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cx, cy = int(d.center_px[0]), int(d.center_px[1])
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{d.label} ({d.score:.2f})", (cx + 8, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Reprojection des objets 3D estimes (rouge)
        if scene is not None:
            P = _projection_matrix(frame.K, frame.T_base_cam)
            for obj in scene.objects:
                X = np.hstack([obj.position_base_m, 1.0])
                uvw = P @ X
                if uvw[2] > 0:
                    u, v = uvw[0]/uvw[2], uvw[1]/uvw[2]
                    cv2.circle(img, (int(u), int(v)), 8, (0, 0, 255), 2)
        return cv2.resize(img, (tw, th))

    def _show_perception_snapshot(self, frames, dets_by_cam, scene, title=""):
        """Affiche les 3 frames camera avec detections, sauvegarde aussi
        dans outputs/perception/. Non-bloquant : utilise cv2.waitKey(1500) au
        lieu de input().
        """
        import cv2
        import numpy as np
        from datetime import datetime

        tiles = []
        for k in CAM_DISPLAY_ORDER:
            tiles.append(self._annotate_frame(frames.get(k), dets_by_cam.get(k, []), scene))
        mosaic = np.hstack(tiles)
        cv2.putText(mosaic, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Sauvegarde
        out_dir = REPO / "outputs" / "perception"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"pipeline_snapshot_{stamp}.png"
        cv2.imwrite(str(out_path), mosaic)
        print(f"   [display] snapshot sauve : {out_path}")

        # Affichage NON BLOQUANT : 1.5 sec puis on continue
        cv2.imshow("Pipeline perception", mosaic)
        cv2.waitKey(1500)   # 1.5s = visualisation suffisante
        # La fenetre reste ouverte ; sera mise a jour ou fermee plus tard

    def _live_display_during_execution(self, mc, robot_state_provider,
                                        every_n_frames: int = 30):
        """Generateur de callback : a appeler periodiquement pendant
        execute_trajectory pour rafraichir l'affichage des cameras.

        Utilise comme : passer ce callable a controller.execute_trajectory
        si on veut un display live. Sinon ne fait rien.
        """
        # Pour l'instant on n'integre pas dans execute_trajectory (trop
        # invasif). On laisse l'utilisateur lancer pick_and_place puis
        # voir le snapshot initial + le robot bouge a vue d'oeil.
        # TODO : ajouter un thread separe qui rafraichit la fenetre.
        pass

    def _safe_stop_with_torque(self, controller):
        """En cas d'exception ou Ctrl+C : MAINTIENT le torque pour eviter
        que le bras tombe sous son poids. Demande a l'utilisateur de
        placer le bras en securite manuellement.

        Sans cette procedure, le `finally` appellerait disconnect() qui
        coupe le torque -> le bras tombe.
        """
        if controller is None or controller._bus is None:
            return
        if not controller._torque_enabled:
            return
        print()
        print("=" * 70)
        print(" SECURITE : torque MAINTENU pour eviter que le bras tombe.")
        print(" 1. Soutiens manuellement le bras avec ta main libre.")
        print(" 2. Appuie sur ENTREE quand tu maintiens le bras et qu'il")
        print("    est en securite (pas suspendu en l'air).")
        print(" 3. Le torque sera ALORS coupe.")
        print("=" * 70)
        try:
            input("Pret a couper le torque ? Appuie ENTREE : ")
        except (EOFError, KeyboardInterrupt):
            pass
        try:
            controller.disable_torque()
            print("Torque coupe. Tu peux relacher le bras.")
        except Exception as e:
            print(f"[WARN] Echec disable_torque : {e}. Coupe l'alim manuellement.")

    # ----- helpers --------------------------------------------------------

    def _build_phase1_trajectory(self,
                                  q_current: dict[str, float],
                                  ik_approach: IKResult
                                  ) -> JointTrajectory:
        """Sprint 4 boucle fermee : trajectoire courant -> approach uniquement.

        Pince ouverte. Apres l'execution, on raffine avec cam_2 puis on
        genere la phase 2 (approach corrige -> grasp -> ...).
        """
        c = self.config
        q_app = ik_approach.joint_angles_rad
        return quintic_trajectory(
            q_current, q_app,
            duration_s=estimate_duration_safe(q_current, q_app,
                                               max_velocity_rad_s=c.max_velocity_rad_s),
            gripper_start=c.grip_open_pct, gripper_end=c.grip_open_pct,
        )

    def _build_phase2_trajectory(self,
                                  q_at_approach: dict[str, float],
                                  ik_results: list[IKResult],
                                  q_home: Optional[dict] = None,
                                  ) -> JointTrajectory:
        """Sprint 4 phase 2 : approach -> grasp -> retract -> drop_above
        -> drop_release -> safe -> home.

        Identique a _build_full_trajectory mais part de q_at_approach
        (pas q_current) et n'inclut pas le premier segment.

        Args:
            q_home : pose finale a rejoindre apres la depose. Si None,
                fallback sur self.config.home_angles_rad ou la pose hardcodee.
                Typiquement, le pipeline appelle avec q_home=q_session_start
                pour que le robot retourne ou l'utilisateur l'avait place.
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad
        c = self.config
        gp_o = c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

        # Pose intermediaire "safe" : bras releve haut, centre.
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
        }
        # Pose "home" finale STABLE : priorite au parametre fourni (= pose de
        # depart session typiquement), fallback sur config, fallback sur
        # pose hardcodee "bras replie".
        q_home = q_home or c.home_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.0,
            "wrist_flex": -0.7,    # poignet replie pour pince vers le bas
            "wrist_roll": 0.0,
        }
        # Vitesse RALENTIE pour la transition vers home (anti-chute visuel)
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )

        segs = [
            # 1. q_at_approach -> approach corrige (petit deplacement si correction)
            quintic_trajectory(q_at_approach, q_app, duration_s=dur(q_at_approach, q_app),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 2. approach -> grasp
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 3. STATIQUE : ferme la pince
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
            # 4. grasp -> retract (pince fermee, bras remonte avec l'objet)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 5. retract -> drop_above (deplace au-dessus de la boite)
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 6. drop_above -> drop_release (descend dans la boite)
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 7. STATIQUE : relache (ouvre la pince)
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # 8. drop_release -> SAFE intermediaire (releve le bras AVANT
            #    le retour a home, pour eviter de traverser la zone du cube).
            quintic_trajectory(q_rel, q_safe, duration_s=dur(q_rel, q_safe),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 9. SAFE -> HOME : transition RALENTIE pour atterrissage doux.
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=gp_o, gripper_end=gp_o),
        ]
        return chain_trajectories(segs)

    def _build_full_trajectory(self,
                                q_current: dict[str, float],
                                ik_results: list[IKResult],
                                q_home: Optional[dict] = None,
                                ) -> JointTrajectory:
        """Concatene les sous-trajectoires : current -> approach -> grasp ->
        (fermer pince) -> retract -> drop_above -> drop_release ->
        (ouvrir pince) -> safe -> home.

        On utilise quintic pour chaque segment (lisse) + pauses pour la
        fermeture/ouverture de pince.

        Args:
            q_home : pose finale a rejoindre. Si None, fallback sur
                config.home_angles_rad ou la pose hardcodee.
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad

        c = self.config
        gp_o = c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

        # Pose safe intermediaire (idem _build_phase2_trajectory)
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": 1.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0,
        }
        # Pose finale : parametre fourni > config > defaut hardcode.
        # FIX (etait q_rest=config_zero precedemment, ce qui en plus pouvait
        # crasher sur q_home/dur_home non definis -- NameError).
        q_home = q_home or c.home_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.0,
            "wrist_flex": -0.7,
            "wrist_roll": 0.0,
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )

        segs = [
            quintic_trajectory(q_current, q_app, duration_s=dur(q_current, q_app),
                                gripper_start=gp_o, gripper_end=gp_o),
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # Safe intermediate AVANT home (evite que la pince traverse la zone
            # du cube pendant le retour).
            quintic_trajectory(q_rel, q_safe, duration_s=dur(q_rel, q_safe),
                                gripper_start=gp_o, gripper_end=gp_o),
            # Transition finale vers home : RALENTIE pour atterrissage doux
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=gp_o, gripper_end=gp_o),
        ]
        return chain_trajectories(segs)
