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

    # Ouverture pince a la fin (apres retour home). 0 = totalement fermee,
    # 100 = grand ouvert. Defaut 5 = presque fermee, pour eviter qu'une
    # pince ouverte accroche un cable / objet de la scene apres execution.
    home_gripper_pct: float = 5.0

    # Rafraichissement des detections HF en --display live. La detection HF
    # prend ~3-5s sur M4. Si on detecte a chaque rafraichissement du callback
    # (toutes les 30 frames de traj), le bras bloque pendant la detection.
    # Solution : detecter tous les N rafraichissements + cacher les dernieres
    # detections pour les afficher entre-temps. N=5 -> detection toutes les
    # ~5s, fluide pendant les intervalles. Mettre N=1 pour HSV (rapide).
    display_detect_every_n: int = 5

    # Compensation SYSTEMATIQUE des biais de calibration mesures
    # empiriquement (cf D11 : biais Y ~+28mm sur ce poste). Sera soustraite
    # a toutes les positions detectees par la stereo.
    # = np.array([dx, dy, dz]) en metres. None = pas de compensation.
    systematic_bias_correction_m: Optional[object] = None  # ndarray (3,)

    # ---------- B0 : restreindre HF a un seul label (la cible) ----------
    # Par defaut hf_specs.json contient 10 labels ("orange_cube", "tissue_box",
    # "pen", "robot_arm", ...). Sur le poste de Maxence, ces labels parasites
    # generent des fausses detections (e.g. tissue_box detecte a (+125,-243,+42)
    # alors qu'il n'y a rien la). Quand True (defaut), on ne charge que le
    # prompt qui mappe vers target_label, ce qui :
    #   - elimine les bbox parasites dans le display
    #   - reduit legerement le temps de detection HF (moins de comparaisons texte)
    #   - rend les logs lisibles (1 seul objet a discuter)
    # Mettre a False pour debugger ou comparer detections multi-objets.
    hf_restrict_to_target: bool = True

    # ---------- P1 : feedback de saisie + retry ----------
    # Apres la fermeture pince (consigne = grip_close_pct), on lit la position
    # REELLE du gripper. Si la pince a bute sur un objet, elle ne pourra pas
    # atteindre la consigne et restera ouverte d'au moins X%. Si la pince
    # atteint (presque) la consigne, c'est qu'elle s'est fermee dans le vide
    # = saisie ratee.
    # Marge en pourcentage au-dessus de grip_close_pct pour conclure "saisi".
    # Ex : grip_close_pct=5, grasp_success_threshold_pct=15 -> on conclut OK
    # si la pince reelle reste >= 5 + 15 = 20% apres fermeture.
    # Pour un cube 30mm, la pince s'arrete typiquement vers 30-40% -> marge OK.
    grasp_success_threshold_pct: float = 15.0
    # Nombre max de RETRY apres echec (0 = pas de retry, 1 = 1 essai + 1 retry).
    max_grasp_retries: int = 1
    # Pause apres fermeture pince avant de LIRE la position (laisser le servo
    # se stabiliser sur l'objet). 0.3s est conservatif.
    grasp_settle_pause_s: float = 0.3


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
        # IK SPECIFIQUE A LA DEPOSE : poids de rotation reduit (0.05 vs 0.1
        # par defaut). Pour la depose on s'en moque que la pince soit pile
        # verticale (elle ouvre, le cube tombe). On privilegie donc la
        # position. Compromis 0.05 (vs 0.01 essaye precedemment) : evite que
        # l'IK accepte des poses tordues a 90deg qui provoquaient des
        # collisions du bras avec lui-meme. Avec 0.05 : rot < 15deg.
        self._ik_drop = IKSolver(rotation_weight=0.05)
        self._provider = RobotStateProvider()

    def _load_scene(self):
        """Charge la position de la boite de depose.

        CONVENTION : `center_base_m` est interprete comme le centre du **FOND**
        de la boite (= sa face posee sur la table). C'est plus naturel a
        mesurer au metre que le centre du dessus. Le dessus est calcule
        automatiquement = fond + box_height.

        Calcule deux poses cibles au-dessus de la boite :
          - drop_above   : 5 cm au-dessus du dessus -> approche avec marge.
          - drop_release : 2 cm au-dessus du dessus -> point de relachement.

        Emet un avertissement si la position semble incoherente (boite
        chevauche la base du robot ou hors workspace) -- typique d'un
        scene.json non mesure.
        """
        scene_path = self.config.scene_config_path or (REPO / "configs" / "scene.json")
        if not scene_path.exists():
            raise FileNotFoundError(f"scene.json manquant : {scene_path}")
        data = json.load(open(scene_path))
        box = data["drop_box"]
        # `center_base_m` = centre du FOND de la boite (convention)
        bottom_center = np.array(box["center_base_m"], dtype=np.float64)
        box_h = float(box["dimensions_m"][2])
        # Dessus de la boite = fond + hauteur
        top_z = float(bottom_center[2]) + box_h
        # drop_above : 5 cm au-dessus du dessus, drop_release : 2 cm au-dessus
        self.drop_position = np.array([bottom_center[0], bottom_center[1], top_z])
        self.drop_above   = self.drop_position + np.array([0.0, 0.0, 0.05])
        self.drop_release = self.drop_position + np.array([0.0, 0.0, 0.02])

        # AVERTISSEMENT si position incoherente
        x, y = float(bottom_center[0]), float(bottom_center[1])
        dist_to_base = np.hypot(x, y)
        warnings = []
        if dist_to_base < 0.12:
            warnings.append(
                f"boite tres proche de la base ({dist_to_base*100:.1f} cm) -- "
                f"chevauche probablement la zone d'exclusion du robot "
                f"(rayon 10cm). Le bras va se cogner."
            )
        ws = data.get("workspace_bounds_base_m", {})
        if ws:
            if not (ws.get("x_min", -1) <= x <= ws.get("x_max", 1)):
                warnings.append(f"X={x:.3f}m hors workspace [{ws.get('x_min')}, {ws.get('x_max')}]")
            if not (ws.get("y_min", -1) <= y <= ws.get("y_max", 1)):
                warnings.append(f"Y={y:.3f}m hors workspace [{ws.get('y_min')}, {ws.get('y_max')}]")

        # Log informatif des poses calculees
        print(f">> Boite de depose chargee depuis {scene_path.name} :")
        print(f"   center_fond = ({bottom_center[0]*1000:+6.1f}, {bottom_center[1]*1000:+6.1f}, "
              f"{bottom_center[2]*1000:+6.1f}) mm  (X devant, Y gauche+/droite-, Z table)")
        print(f"   dimensions  = {box['dimensions_m'][0]*100:.1f} x "
              f"{box['dimensions_m'][1]*100:.1f} x {box_h*100:.1f} cm")
        print(f"   dessus boite a Z = {top_z*1000:.1f} mm")
        print(f"   drop_above   = ({self.drop_above[0]*1000:+6.1f}, "
              f"{self.drop_above[1]*1000:+6.1f}, {self.drop_above[2]*1000:+6.1f}) mm")
        print(f"   drop_release = ({self.drop_release[0]*1000:+6.1f}, "
              f"{self.drop_release[1]*1000:+6.1f}, {self.drop_release[2]*1000:+6.1f}) mm")
        for w in warnings:
            print(f"   [WARN scene.json] {w}")
        if warnings:
            print(f"   --> Verifie configs/scene.json. Si tu changes la boite de place, "
                  f"remesure et mets a jour center_base_m.")

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
            # B0 : si demande, on ne garde que le prompt qui mappe vers target_label
            # Elimine les detections parasites (tissue_box, pen, robot_arm, etc.)
            if self.config.hf_restrict_to_target:
                target = self.config.target_label
                prompts_to_keep = [p for p in labels
                                   if self._label_mapping.get(p, p) == target]
                if prompts_to_keep:
                    n_before = len(labels)
                    labels = prompts_to_keep
                    self._label_mapping = {
                        p: m for p, m in self._label_mapping.items()
                        if p in prompts_to_keep
                    }
                    print(f">> HF restreint a {len(labels)}/{n_before} label(s) "
                          f"(cible='{target}') : {labels}")
                else:
                    print(f"[WARN] aucun prompt HF ne mappe vers '{target}', "
                          f"utilisation des {len(labels)} labels par defaut")
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
            # ARCHITECTURE THREADING (fix saccades) :
            # Probleme avant : la detection HF (~3-5s sur M4) etait appelee
            # depuis le callback execute_trajectory -> bloquait l'envoi des
            # commandes moteur pendant ces 3-5s -> bras s'arrete, puis envoi
            # en rafale pour rattraper le timing -> mouvements brusques.
            #
            # Solution : detection dans un THREAD WORKER separe.
            #   - Main loop (callback) : grab frames, push dans une queue,
            #     affiche les dernieres detections disponibles, retourne
            #     immediatement (< 50 ms). La trajectoire continue fluide.
            #   - Worker thread : prend les frames, fait la detection (3-5s),
            #     met a jour le cache des detections.
            # Les detections affichees peuvent etre legerement obsoletes
            # (delai de quelques secondes en HF) mais le display reste fluide.
            def make_live_callback(initial_dets=None):
                if not self.config.display:
                    return None
                import cv2
                import threading
                from queue import Queue, Empty, Full

                # Queue de taille 1 : on garde seulement les dernieres frames
                # (les anciennes sont perdues, c'est OK)
                frame_queue: Queue = Queue(maxsize=1)
                # Cache partage des dernieres detections (avec lock pour
                # synchronisation main <-> worker). On PRE-POPULE avec les
                # detections de la perception initiale -> les bbox vertes
                # sont visibles immediatement au lieu d'attendre le premier
                # cycle worker HF (~3-5 sec).
                dets_cache = {"value": initial_dets or {}}
                dets_lock = threading.Lock()
                stop_event = threading.Event()

                def detector_worker():
                    """Boucle worker : detecte sur les dernieres frames dispo."""
                    while not stop_event.is_set():
                        try:
                            frames = frame_queue.get(timeout=0.2)
                        except Empty:
                            continue
                        try:
                            dets = self._detector.detect_multi(frames)
                            if self._label_mapping:
                                for cam_dets in dets.values():
                                    for d in cam_dets:
                                        if d.label in self._label_mapping:
                                            d.label = self._label_mapping[d.label]
                            with dets_lock:
                                dets_cache["value"] = dets
                        except Exception as e:
                            print(f"[live worker] detection error: {e}")

                # Demarre le worker (daemon=True : se ferme avec le programme)
                worker = threading.Thread(target=detector_worker, daemon=True)
                worker.start()
                # Memorise pour pouvoir l'arreter proprement dans finally
                self._live_stop_event = stop_event

                def on_step(i, trajectory):
                    try:
                        rs = self._provider.read_live()
                    except Exception:
                        return
                    frames = mc.grab(robot_state=rs)
                    # Pousse les frames au worker (remplace si encore pleine
                    # = le worker n'a pas fini la precedente, on saute)
                    try:
                        frame_queue.put_nowait(frames)
                    except Full:
                        pass  # worker occupe, on ignore ce frame pour la detection
                    # Display avec les dernieres detections (potentiellement
                    # obsoletes de quelques sec en HF, instantanees en HSV)
                    with dets_lock:
                        cur_dets = dict(dets_cache["value"])
                    import numpy as np
                    tiles = [self._annotate_frame(frames.get(k),
                                                  cur_dets.get(k, []),
                                                  None)
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

            # Cree le callback live UNE SEULE FOIS pour eviter de lancer
            # plusieurs worker threads de detection. Le meme callback sera
            # passe a phase 1 et phase 2. Pre-popule le cache avec les
            # detections initiales pour que les bbox vertes soient visibles
            # immediatement (sans attendre 3-5 sec du premier cycle HF).
            live_callback = make_live_callback(initial_dets=dets_by_cam)

            if self.config.closed_loop and not self.config.dry_run:
                # --- Phase 1 : trajectoire courant -> approach ---
                print(">> Phase 1 : courant -> approach (boucle fermee Sprint 4)")
                traj_phase1 = self._build_phase1_trajectory(
                    q_current, r_app,
                    gripper_open_pct=grasp_pose.gripper_open_pct)
                print(f"   {len(traj_phase1)} points, duree {traj_phase1.duration_s:.1f}s")
                controller.execute_trajectory(traj_phase1, verbose=True,
                                              on_step=live_callback)
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

                # === A4 : Mini-descente + Refinement #2 (cam_2 plus pres) ===
                # Apres le 1er refinement, on descend a mi-chemin entre approach
                # et grasp (4cm au-dessus de l'objet), puis on refait un
                # refinement cam_2. A cette distance, cam_2 voit l'objet de
                # plus pres -> plus precis. Si l'objet a bouge (ex: pousse par
                # une petite vibration), on detecte et on recorrige.
                print(">> A4 : Mini-descente + Refinement #2")
                T_intermediate = grasp_pose.T_base_gripper_grasp.copy()
                T_intermediate[2, 3] += 0.04  # 4cm au-dessus du grasp
                r_intermediate = self._ik.solve(
                    T_intermediate, q_init=r_app.joint_angles_rad)
                # B4 (2026-05-19) : c'est PENDANT cette mini-descente que la
                # pince s'ouvre (de home_gripper_pct=5% a gripper_open_pct
                # adapte a la bbox de l'objet). Avant : la pince etait deja
                # ouverte depuis le debut de l'approche. Maintenant : reste
                # fermee jusqu'a approach, puis s'ouvre pile au moment de
                # descendre vers l'objet. Plus economique visuellement et
                # mecaniquement (moins de risque d'accrochage cables).
                mini_traj = quintic_trajectory(
                    rs_at_approach.joint_angles_rad,
                    r_intermediate.joint_angles_rad,
                    duration_s=estimate_duration_safe(
                        rs_at_approach.joint_angles_rad,
                        r_intermediate.joint_angles_rad,
                        max_velocity_rad_s=self.config.max_velocity_rad_s),
                    gripper_start=self.config.home_gripper_pct,    # B4 : ferme a l'arrivee approach
                    gripper_end=grasp_pose.gripper_open_pct,        # s'ouvre pendant la mini-descente
                )
                print(f"   Mini-descente : {len(mini_traj)} points, "
                      f"duree {mini_traj.duration_s:.1f}s")
                controller.execute_trajectory(mini_traj, verbose=False,
                                              on_step=live_callback)
                # Refinement #2
                rs_at_intermediate = self._provider.read_live()
                refinement2 = refine_grasp_with_cam2(
                    target_label=self.config.target_label,
                    detector=self._detector,
                    multi_camera=mc,
                    robot_state=rs_at_intermediate,
                    target_z_base_m=float(grasp_pose.T_base_gripper_grasp[2, 3]),
                    label_mapping=self._label_mapping,
                )
                print(f"   Refinement #2 : {refinement2.message}")
                # B2 (2026-05-19) : PLAFOND ABSOLU 30mm pour R2,
                # INDEPENDANT du score HF.
                #
                # Justification physique : entre R1 et R2 il s'ecoule la
                # mini-descente (~0.5s). L'objet ne peut PAS bouger de >30mm
                # en 0.5s sans qu'on le pousse activement. Une correction
                # R2 >30mm = forcement une fausse detection (faux positif HF
                # sur un autre objet, bbox sur bord d'image, etc.).
                #
                # Cas concret : essai 3 du 2026-05-19, R2 a propose +80mm
                # (cam_2 a vu l'objet a Y=-106mm alors qu'il etait a Y=-25mm
                # avec score 0.69) -> robot envoye au vide.
                #
                # Note : la logique P2 dynamique (60/80/100mm selon score)
                # reste active pour le REFINEMENT RETRY (apres saisie ratee),
                # ou l'objet PEUT vraiment avoir bouge suite au contact rate.
                R2_MAX_ABSOLUTE_MM = 30.0
                score = refinement2.confidence
                if refinement2.confidence > 0.1 and refinement2.delta_norm_mm < R2_MAX_ABSOLUTE_MM:
                    apply_correction_to_grasp_pose(grasp_pose,
                                                    refinement2.delta_base_m)
                    print(f"   Correction #2 appliquee "
                          f"(norme {refinement2.delta_norm_mm:.1f} mm, "
                          f"plafond R2={R2_MAX_ABSOLUTE_MM:.0f}mm @ score {score:.2f})")
                    # Re-IK grasp et retract uniquement (approach deja passe)
                    r_grp = self._ik.solve(
                        grasp_pose.T_base_gripper_grasp,
                        q_init=r_intermediate.joint_angles_rad)
                    r_ret = self._ik.solve(
                        grasp_pose.T_base_gripper_retract,
                        q_init=r_grp.joint_angles_rad)
                elif refinement2.delta_norm_mm >= R2_MAX_ABSOLUTE_MM:
                    print(f"   [WARN] correction #2 = {refinement2.delta_norm_mm:.1f}mm "
                          f"> plafond R2 {R2_MAX_ABSOLUTE_MM:.0f}mm "
                          f"(score {score:.2f}, probable fausse detection), ignoree")
                else:
                    print(f"   [INFO] objet stable depuis refinement #1 (pas de re-correction)")
                print()

                # --- Phase 2 : BOUCLE de tentatives de saisie avec FEEDBACK (P1) ---
                # Au lieu d'executer une trajectoire monolithique, on procede en
                # 3 etapes par tentative :
                #   1. SOUS-TRAJ : intermediate -> approach -> grasp -> ferme pince
                #   2. CHECK : lecture position reelle gripper
                #        - si position >> consigne -> pince a bute sur l'objet = SAISIE OK
                #        - sinon -> SAISIE RATEE (pince ferme dans le vide)
                #   3. Si OK -> sous-traj finish (retract + drop + home)
                #      Si rate ET retry possible -> remontee + refinement + boucle
                #      Si rate ET tous retries epuises -> abort vers home sans depose
                print(f">> Phase 2 : tentative(s) de saisie avec feedback pince + depot + retour ({home_origin})")
                q_at_intermediate = rs_at_intermediate.joint_angles_rad

                # Log structure pour campagne experimentale (P4)
                grasp_attempts_log = []
                grasp_succeeded = False
                q_start_attempt = q_at_intermediate  # depart de la 1ere descente

                attempt = 0
                while attempt <= self.config.max_grasp_retries:
                    attempt += 1
                    print()
                    print(f">> --- Tentative #{attempt}/{self.config.max_grasp_retries + 1} ---")

                    # Sous-traj : descente vers grasp + fermeture pince statique
                    traj_grasp = self._build_grasp_attempt_traj(
                        q_start_attempt,
                        r_app.joint_angles_rad,
                        r_grp.joint_angles_rad,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                        grip_close_pct=self.config.grip_close_pct,
                    )
                    print(f"   Descente + fermeture : {len(traj_grasp)} points, "
                          f"duree {traj_grasp.duration_s:.1f}s")
                    controller.execute_trajectory(traj_grasp, verbose=True,
                                                  on_step=live_callback)

                    # CHECK pince : laisse le servo se stabiliser puis lit la
                    # position reelle. Si la pince a bute sur l'objet, elle est
                    # restee ouverte d'au moins grasp_success_threshold_pct.
                    time.sleep(self.config.grasp_settle_pause_s)
                    gripper_now = controller.read_gripper_pct()
                    margin = gripper_now - self.config.grip_close_pct
                    success = margin > self.config.grasp_success_threshold_pct
                    grasp_attempts_log.append({
                        "attempt": attempt,
                        "gripper_pct": gripper_now,
                        "consigne_pct": self.config.grip_close_pct,
                        "marge_pct": margin,
                        "seuil_pct": self.config.grasp_success_threshold_pct,
                        "success": success,
                    })
                    tag = "SAISIE OK" if success else "SAISIE RATEE"
                    print(f"   [check pince] consigne={self.config.grip_close_pct:.0f}%, "
                          f"reel={gripper_now:.1f}%, marge={margin:+.1f}%, "
                          f"seuil={self.config.grasp_success_threshold_pct:.0f}%  -->  {tag}")

                    if success:
                        grasp_succeeded = True
                        break

                    # Echec : retry possible ?
                    if attempt > self.config.max_grasp_retries:
                        print(f"   >> ABANDON apres {attempt} tentative(s) (max retries atteint).")
                        break

                    print(f"   >> RETRY : remontee + refinement cam_2 + redescente")

                    # 1. Remontee a approach AVEC pince ouverte (libere ce qu'on aurait pu saisir partiellement)
                    rs_at_grasp = self._provider.read_live()
                    traj_lift = self._build_retry_lift_traj(
                        rs_at_grasp.joint_angles_rad,
                        r_app.joint_angles_rad,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                    )
                    print(f"   Remontee a approach : {len(traj_lift)} points, "
                          f"duree {traj_lift.duration_s:.1f}s")
                    controller.execute_trajectory(traj_lift, verbose=False,
                                                  on_step=live_callback)

                    # 2. Refinement cam_2 (l'objet peut avoir bouge a cause du contact rate)
                    rs_after_lift = self._provider.read_live()
                    print(f"   Refinement cam_2 retry...")
                    refinement_retry = refine_grasp_with_cam2(
                        target_label=self.config.target_label,
                        detector=self._detector,
                        multi_camera=mc,
                        robot_state=rs_after_lift,
                        target_z_base_m=float(grasp_pose.T_base_gripper_grasp[2, 3]),
                        label_mapping=self._label_mapping,
                    )
                    print(f"   {refinement_retry.message}")
                    # Seuil dynamique selon score (meme logique P2)
                    score_r = refinement_retry.confidence
                    if score_r >= 0.5:
                        seuil_r_mm = 100.0
                    elif score_r >= 0.3:
                        seuil_r_mm = 80.0
                    else:
                        seuil_r_mm = 60.0
                    if score_r > 0.1 and refinement_retry.delta_norm_mm < seuil_r_mm:
                        apply_correction_to_grasp_pose(grasp_pose, refinement_retry.delta_base_m)
                        print(f"   Correction retry appliquee "
                              f"(norme {refinement_retry.delta_norm_mm:.1f}mm, "
                              f"seuil={seuil_r_mm:.0f}mm @ score {score_r:.2f})")
                        r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                            grasp_pose, q_init=rs_after_lift.joint_angles_rad)
                    elif refinement_retry.delta_norm_mm >= seuil_r_mm:
                        print(f"   [WARN] correction retry = {refinement_retry.delta_norm_mm:.1f}mm "
                              f"> seuil {seuil_r_mm:.0f}mm (score {score_r:.2f}), ignoree")

                    # Le point de depart de la prochaine descente = position courante
                    q_start_attempt = rs_after_lift.joint_angles_rad
                    # (la boucle reprend : descente + fermeture + check)

                print()

                # --- Apres la boucle : depose si succes, retour direct si echec ---
                rs_after_grasp = self._provider.read_live()
                if grasp_succeeded:
                    print(f">> SAISIE REUSSIE en {attempt} tentative(s). "
                          f"Depose dans la boite + retour {home_origin}.")
                    traj_finish = self._build_finish_after_grasp_traj(
                        q_grp=rs_after_grasp.joint_angles_rad,
                        q_ret=r_ret.joint_angles_rad,
                        q_drop=r_drop_above.joint_angles_rad,
                        q_rel=r_drop_release.joint_angles_rad,
                        q_home=q_home_final,
                        grip_close_pct=self.config.grip_close_pct,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                    )
                else:
                    print(f">> ECHEC apres {attempt} tentative(s). "
                          f"Retour {home_origin} sans depose.")
                    traj_finish = self._build_abort_to_home_traj(
                        q_grp=rs_after_grasp.joint_angles_rad,
                        q_ret=r_ret.joint_angles_rad,
                        q_home=q_home_final,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                    )
                print(f"   {len(traj_finish)} points, duree {traj_finish.duration_s:.1f}s")
                controller.execute_trajectory(traj_finish, verbose=True,
                                              on_step=live_callback)

                # Log structure final pour la campagne (P4)
                self._grasp_attempts_log = grasp_attempts_log
                self._grasp_final_success = grasp_succeeded
                self._grasp_total_attempts = attempt
                self._print_grasp_summary(grasp_attempts_log, grasp_succeeded, attempt)

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
                    gripper_open_pct=grasp_pose.gripper_open_pct,
                )
                print(f"   {len(traj)} points, duree {traj.duration_s:.1f}s")
                print()
                if self.config.dry_run:
                    print(">> DRY RUN : pas d'execution sur le robot.")
                else:
                    print(f">> Execution sur le robot (jusqu'au retour {home_origin})...")
                    controller.execute_trajectory(traj, verbose=True,
                                                  on_step=live_callback)
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
            # Arrete proprement le worker thread du live display (s'il existe).
            # Sans cela, le thread tournerait jusqu'a la fin du programme
            # (daemon=True le ferme quand meme, mais c'est plus propre).
            stop_ev = getattr(self, "_live_stop_event", None)
            if stop_ev is not None:
                try:
                    stop_ev.set()
                except Exception:
                    pass
                self._live_stop_event = None
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
                                  ik_approach: IKResult,
                                  gripper_open_pct: Optional[float] = None,
                                  ) -> JointTrajectory:
        """Sprint 4 boucle fermee : trajectoire courant -> approach uniquement.

        B4 (2026-05-19) : pince reste FERMEE pendant tout le deplacement
        vers approach. Avant : la pince s'ouvrait progressivement de 5% a 80%
        durant ce segment, donc arrivait au-dessus de l'objet deja ouverte
        (et balayait visuellement la scene depuis le depart). Maintenant :
        la pince reste fermee pendant l'approche, et ne s'ouvre qu'au moment
        de la mini-descente (4cm au-dessus du grasp), pour ne plus etre
        ouverte que strictement quand necessaire (= pour saisir).
        Le parametre gripper_open_pct est ignore ici mais conserve dans la
        signature pour compatibilite.
        """
        c = self.config
        q_app = ik_approach.joint_angles_rad
        return quintic_trajectory(
            q_current, q_app,
            duration_s=estimate_duration_safe(q_current, q_app,
                                               max_velocity_rad_s=c.max_velocity_rad_s),
            gripper_start=c.home_gripper_pct,  # ferme au depart
            gripper_end=c.home_gripper_pct,    # B4 : reste fermee a l'arrivee
        )

    def _build_phase2_trajectory(self,
                                  q_at_approach: dict[str, float],
                                  ik_results: list[IKResult],
                                  q_home: Optional[dict] = None,
                                  gripper_open_pct: Optional[float] = None,
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
            gripper_open_pct : ouverture pince a utiliser (= calculee par
                TopDownGrasp selon la bbox de l'objet). Si None, fallback
                sur config.grip_open_pct (100%).
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad
        c = self.config
        gp_o = gripper_open_pct if gripper_open_pct is not None else c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

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
        # Pose intermediaire "safe" : bras releve haut, centre.
        # IMPORTANT : on HERITE wrist_roll et wrist_flex de q_home pour eviter
        # une rotation parasite finale (q_safe -> q_home). Sans ca, si
        # q_session_start a wrist_roll=0.5 et q_safe hardcode wrist_roll=0,
        # le poignet faisait 0.5 -> 0 (safe) -> 0.5 (home), inutile et bizarre.
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
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
            # 7.5 SORTIE VERTICALE de la boite : drop_release -> drop_above.
            #    Sans ce segment, le bras allait directement de l'interieur
            #    de la boite (drop_release) vers q_safe (centre, haut) -> la
            #    pince fixe cognait le bord interieur de la boite en chemin.
            #    En remontant d'abord verticalement, on sort proprement.
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 8. drop_above -> SAFE intermediaire (releve le bras AVANT
            #    le retour a home, pour eviter de traverser la zone du cube).
            #    On ferme PROGRESSIVEMENT la pince ici pour eviter qu'elle
            #    reste grand-ouverte (securite : pas d'accrochage sur cables).
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 9. SAFE -> HOME : transition RALENTIE pour atterrissage doux.
            #    Pince finit a home_gripper_pct (par defaut 5 = presque fermee).
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _build_full_trajectory(self,
                                q_current: dict[str, float],
                                ik_results: list[IKResult],
                                q_home: Optional[dict] = None,
                                gripper_open_pct: Optional[float] = None,
                                ) -> JointTrajectory:
        """Concatene les sous-trajectoires : current -> approach -> grasp ->
        (fermer pince) -> retract -> drop_above -> drop_release ->
        (ouvrir pince) -> safe -> home.

        On utilise quintic pour chaque segment (lisse) + pauses pour la
        fermeture/ouverture de pince.

        Args:
            q_home : pose finale a rejoindre. Si None, fallback sur
                config.home_angles_rad ou la pose hardcodee.
            gripper_open_pct : ouverture pince calculee par TopDownGrasp
                selon la bbox de l'objet. Si None, fallback config.
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad

        c = self.config
        gp_o = gripper_open_pct if gripper_open_pct is not None else c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

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
        # Pose safe intermediaire : on HERITE wrist_roll/wrist_flex de q_home
        # pour eviter une rotation parasite finale (cf _build_phase2_trajectory).
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )

        segs = [
            # 1. courant -> approach : pince s'OUVRE progressivement
            quintic_trajectory(q_current, q_app, duration_s=dur(q_current, q_app),
                                gripper_start=c.home_gripper_pct,  # ferme au depart
                                gripper_end=gp_o),                  # ouverte a l'arrivee
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
            # Sortie VERTICALE de la boite (drop_release -> drop_above) AVANT
            # safe -> evite que la pince fixe cogne le bord interieur.
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # Safe intermediate AVANT home (evite que la pince traverse la zone
            # du cube pendant le retour). Pince se referme progressivement.
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # Transition finale vers home : RALENTIE pour atterrissage doux.
            # Pince finit a home_gripper_pct (defaut 5 = quasi-fermee).
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    # ============================================================
    # P1 : Helpers SOUS-TRAJECTOIRES (pour boucle de tentatives + retry)
    # On split la phase 2 historique en sous-trajectoires individuelles,
    # appellees explicitement depuis run() avec un CHECK pince entre
    # la fermeture et le transport. Les helpers _build_phase2_trajectory
    # et _build_full_trajectory restent en place (utilises pour le mode
    # --no-closed-loop et comme fallback).
    # ============================================================

    def _build_grasp_attempt_traj(
        self, q_from: dict[str, float], q_app: dict[str, float],
        q_grp: dict[str, float],
        grip_open_pct: float, grip_close_pct: float,
    ) -> JointTrajectory:
        """Sous-traj : q_from -> approach -> grasp -> ferme pince (statique).

        Utilisee pour chaque TENTATIVE de saisie (1er essai + eventuels retries).
        A la sortie, la pince a recu la commande de fermeture (grip_close_pct)
        mais sa position REELLE depend de ce qu'elle a rencontre. C'est
        precisement ce que run() va lire via controller.read_gripper_pct().
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_o, gp_c = grip_open_pct, grip_close_pct
        segs = [
            quintic_trajectory(q_from, q_app, duration_s=dur(q_from, q_app),
                                gripper_start=gp_o, gripper_end=gp_o),
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            # Fermeture STATIQUE (le bras ne bouge pas, la pince ferme).
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
        ]
        return chain_trajectories(segs)

    def _build_retry_lift_traj(
        self, q_grp: dict[str, float], q_app: dict[str, float],
        grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj : remontee de grasp vers approach AVEC OUVERTURE PINCE.

        Sert au RETRY apres saisie ratee : on libere ce qu'on a (ou pas) saisi
        puis on remonte a approach pour relancer refinement + descente.
        """
        c = self.config
        return quintic_trajectory(
            q_grp, q_app,
            duration_s=estimate_duration_safe(q_grp, q_app,
                                              max_velocity_rad_s=c.max_velocity_rad_s),
            gripper_start=grip_open_pct, gripper_end=grip_open_pct,
        )

    def _build_finish_after_grasp_traj(
        self, q_grp: dict[str, float], q_ret: dict[str, float],
        q_drop: dict[str, float], q_rel: dict[str, float],
        q_home: dict[str, float],
        grip_close_pct: float, grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj SUCCES : grasp -> retract -> drop_above -> drop_release
        -> (ouvre pince) -> sortie verticale -> safe -> home.

        Pince fermee pendant le transport, ouvre au dessus de la boite, referme
        progressivement pour le retour home (home_gripper_pct, ~5%).
        Reprend la logique de _build_phase2_trajectory segments 4-9.
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_c, gp_o = grip_close_pct, grip_open_pct
        # q_safe = pose intermediaire haute. HERITE wrist_roll/wrist_flex de
        # q_home pour eviter rotation parasite finale (cf P3).
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )
        segs = [
            # 1. grasp -> retract (pince fermee, remonte avec objet)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 2. retract -> drop_above
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 3. drop_above -> drop_release (descend dans boite)
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 4. STATIQUE : relache (ouvre pince)
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # 5. SORTIE VERTICALE de la boite (drop_release -> drop_above)
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 6. drop_above -> SAFE (releve haut, ferme pince progressivement)
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 7. SAFE -> HOME (transition ralentie)
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _build_abort_to_home_traj(
        self, q_grp: dict[str, float], q_ret: dict[str, float],
        q_home: dict[str, float], grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj ECHEC : grasp -> retract (pince ouverte) -> safe -> home.

        Apres tous les retries epuises sans saisie reussie : on remonte sans
        rien transporter et on rentre. Pas de visite a la boite (rien a poser).
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_o = grip_open_pct
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )
        segs = [
            # 1. STATIQUE : pince OUVERTE (relache ce qu'on a partiellement saisi)
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 2. grasp -> retract (remonte, pince ouverte)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 3. retract -> safe (referme progressivement vers home_gripper_pct)
            quintic_trajectory(q_ret, q_safe, duration_s=dur(q_ret, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 4. safe -> home (ralenti)
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _print_grasp_summary(self, attempts_log: list[dict],
                              succeeded: bool, total_attempts: int):
        """Recap structure des tentatives de saisie (pour campagne P4).

        Imprime un tableau clair en fin de pipeline avec :
          - position pince reelle a chaque tentative
          - marge vs seuil de detection
          - resultat OK/RATE
          - resultat final REUSSIE / ECHEC
        Les attributs self._grasp_* sont aussi exposes pour qu'un script
        externe (scripts/experiment_campaign.py) puisse les lire.
        """
        print()
        print("=" * 70)
        print(" RECAP SAISIE")
        print("=" * 70)
        if not attempts_log:
            print("  Aucune tentative loggee (bug ?)")
        for entry in attempts_log:
            n = entry["attempt"]
            gp = entry["gripper_pct"]
            margin = entry["marge_pct"]
            sc = entry["success"]
            tag = "OK" if sc else "RATE"
            print(f"  Tentative #{n} : pince reelle = {gp:>5.1f}%  "
                  f"(marge {margin:+5.1f}% vs seuil {entry['seuil_pct']:.0f}%)  --> {tag}")
        final = "REUSSIE" if succeeded else "ECHEC"
        print(f"  --> Resultat final : {final}  ({total_attempts} tentative(s) total)")
        print("=" * 70)
