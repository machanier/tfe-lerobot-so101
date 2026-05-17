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
        controller = MotorController()
        if not self.config.dry_run:
            self._provider.connect_live(self.config.motor_port)
            controller.connect(self.config.motor_port)
            controller.enable_torque()

        try:
            # ============================================================
            # 2. Perception (UNE seule capture)
            # ============================================================
            print(">> Perception en cours...")
            with MultiCamera() as mc:
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
                # === DEBUG : detections brutes par camera ===
                print("   Detections brutes par camera :")
                for cam_key in ("cam_0", "cam_1", "cam_2"):
                    if cam_key in dets_by_cam:
                        dets = dets_by_cam[cam_key]
                        if dets:
                            labels = ", ".join(
                                f"{d.label}(s={d.score:.2f})" for d in dets[:5]
                            )
                            print(f"     {cam_key} : {len(dets)} detections -> {labels}")
                        else:
                            print(f"     {cam_key} : 0 detection")
                scene = self._estimator.build_scene(dets_by_cam, frames)
                # === DEBUG : si scene vide alors qu'on avait des dets ===
                if not scene.objects and any(dets_by_cam.get(k) for k in dets_by_cam):
                    print()
                    print("   [DIAGNOSTIC] Detections 2D presentes mais aucun objet 3D :")
                    # Affiche la raison de rejet par label (memorisee dans
                    # PoseEstimator._last_rejections)
                    rejections = getattr(self._estimator, "_last_rejections", {})
                    if rejections:
                        for label, reason in rejections.items():
                            print(f"     {label:<22} : {reason}")
                    # Re-essai sans filtres scene pour voir ce qui aurait passe sinon
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

            r_app, r_grp, r_ret = self._ik.solve_grasp_pose(grasp_pose, q_init=q_current)
            # IK pour la pose drop_above (pince ouverte, prete a relacher)
            from src.planning.grasp import _rotation_top_down, _se3
            T_drop_above = _se3(_rotation_top_down(0.0), self.drop_above)
            T_drop_release = _se3(_rotation_top_down(0.0), self.drop_release)
            r_drop_above = self._ik.solve(T_drop_above, q_init=r_ret.joint_angles_rad)
            r_drop_release = self._ik.solve(T_drop_release, q_init=r_drop_above.joint_angles_rad)

            for label, r in [("approach", r_app), ("grasp", r_grp),
                             ("retract", r_ret), ("drop_above", r_drop_above),
                             ("drop_release", r_drop_release)]:
                tag = "OK" if r.converged else "approx"
                print(f"   {label:<14} {tag:<6} "
                      f"trans={r.translation_err_mm:5.1f}mm rot={r.rotation_err_deg:5.1f}deg")
            print()

            # ============================================================
            # 6. Generation trajectoire complete
            # ============================================================
            print(">> Generation trajectoire...")
            traj = self._build_full_trajectory(
                q_current,
                [r_app, r_grp, r_ret, r_drop_above, r_drop_release],
            )
            print(f"   {len(traj)} points, duree {traj.duration_s:.1f}s")
            print()

            # ============================================================
            # 7. Execution
            # ============================================================
            if self.config.dry_run:
                print(">> DRY RUN : pas d'execution sur le robot.")
            else:
                print(">> Execution sur le robot...")
                controller.execute_trajectory(traj, verbose=True)
                print(">> Termine.")

        except KeyboardInterrupt:
            print("\nInterrompu par utilisateur.")
        finally:
            try:
                self._provider.disconnect_live()
            except Exception:
                pass
            try:
                controller.disconnect()
            except Exception:
                pass

    # ----- helpers --------------------------------------------------------

    def _build_full_trajectory(self,
                                q_current: dict[str, float],
                                ik_results: list[IKResult]
                                ) -> JointTrajectory:
        """Concatene les sous-trajectoires : current -> approach -> grasp ->
        (fermer pince) -> retract -> drop_above -> drop_release ->
        (ouvrir pince) -> rest (config zero).

        On utilise quintic pour chaque segment (lisse) + pauses pour la
        fermeture/ouverture de pince.
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

        segs = [
            # 1. courant -> approach (pince ouverte)
            quintic_trajectory(q_current, q_app, duration_s=dur(q_current, q_app),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 2. approach -> grasp (pince ouverte encore)
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 3. STATIQUE : ferme la pince (positions identiques, gripper change)
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
            # 4. grasp -> retract (pince fermee, objet saisi)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 5. retract -> drop_above
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 6. drop_above -> drop_release
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 7. STATIQUE : relache (ouvre la pince)
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # 8. drop_release -> rest (config zero)
            quintic_trajectory(q_rel, {j: 0.0 for j in ARM_JOINTS},
                                duration_s=dur(q_rel, {j: 0.0 for j in ARM_JOINTS}),
                                gripper_start=gp_o, gripper_end=gp_o),
        ]
        return chain_trajectories(segs)
