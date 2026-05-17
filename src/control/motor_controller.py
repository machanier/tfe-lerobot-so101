"""
motor_controller.py - Interface haut-niveau vers le bus moteur du SO-101.

Encapsule le bus Feetech pour exposer :
  - connect/disconnect propre
  - enable_torque / disable_torque
  - send_angles : envoie une commande d'angles + ouverture pince
  - execute_trajectory : suit une JointTrajectory en respectant les timestamps
  - read_state : lit la pose courante (delegue a RobotStateProvider)

Conversion angles_rad -> raw_encoder :
  LeRobot motors_bus.py:858 :
    raw = mid + angle_deg / 360 * 4095
  (avec mid = (range_min + range_max)/2 pour les joints non wraparound)

  Pour les joints "deroules" (wrist_roll), il faut utiliser le unwrap_center
  lu depuis encoder_unwrap.json. Si ce fichier n'existe pas (cas actuel),
  on utilise mid de la calibration.

SECURITE :
  - Avant toute commande : verifier que les angles sont dans les plages
    articulaires (sinon le servo refusera ou se mettra en erreur).
  - Avant deconnexion : ramener le bras a une pose "rest" sure
    (configuration zero ou pose memorisee).
  - Sur Ctrl+C : disable_torque pour eviter de laisser le bras sous tension.

Reference : LeRobot motors_bus.py (Feetech protocol implementation).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.calibration.forward_kinematics import ARM_JOINTS
from src.calibration.motor_to_angle import (
    ENCODER_FULL,
    STS3215_MAX_RESOLUTION,
    load_encoder_unwrap,
    load_motor_calibration,
)
from src.control.trajectory import JointTrajectory


class MotorController:
    """Wrapper haut-niveau autour du bus Feetech."""

    def __init__(self, calib_path: Optional[Path] = None,
                 unwrap_path: Optional[Path] = None):
        self.calib_path = calib_path or (REPO / "configs" / "calibration_follower.json")
        self.unwrap_path = unwrap_path or (REPO / "configs" / "encoder_unwrap.json")
        self.calib = load_motor_calibration(self.calib_path)
        self.unwrap = load_encoder_unwrap(self.unwrap_path, self.calib)
        self._bus = None
        self._torque_enabled = False

    # ----- connexion / deconnexion ----------------------------------------

    def connect(self, port: str, max_retries: int = 5):
        """Ouvre le bus Feetech (avec retry en cas de paquet corrompu au demarrage).

        Cas typique : "Failed to write 'Lock' on id_=1 [Incorrect status packet]"
        au tout debut. Cause : lancements consecutifs trop rapproches qui
        n'ont pas laisse le temps au bus serie de se liberer. Solution :
        retry avec un petit delai.
        """
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as e:
            raise ImportError(
                "LeRobot indisponible. Active le venv ou installe-le selon setup_env.sh."
            ) from e

        motors = {
            "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        import time
        last_err = None
        for attempt in range(max_retries):
            try:
                bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
                bus.connect()
                self._bus = bus
                self._torque_enabled = False
                if attempt > 0:
                    print(f"[motor_controller] Connexion reussie au {attempt + 1}eme essai.")
                return
            except (ConnectionError, RuntimeError) as e:
                last_err = e
                if attempt < max_retries - 1:
                    # Delai progressif : 0.5s, 1s, 2s, 4s, ...
                    delay = 0.5 * (2 ** attempt)
                    print(f"[motor_controller] Connexion {attempt + 1}/{max_retries} "
                          f"echouee ({type(e).__name__}), retry dans {delay:.1f}s...")
                    # Pause + cleanup partiel
                    try:
                        bus.disconnect()
                    except Exception:
                        pass
                    time.sleep(delay)
        # Tous les retries epuises
        raise ConnectionError(
            f"Impossible de connecter le bus moteur apres {max_retries} essais. "
            f"Causes a verifier dans l'ordre :\n"
            f"  1. ROBOT FOLLOWER ALIMENTE ? (verifier l'interrupteur et la LED).\n"
            f"  2. CABLE USB du follower bien branche (peut-etre debrancher/rebrancher) ?\n"
            f"  3. AUTRE PROCESS (python ou lerobot-teleoperate) tient le port ? Verifier avec :\n"
            f"     lsof | grep usbmodem\n"
            f"  4. Tous les MOTEURS detectes ? Tester :\n"
            f"     python scripts/check_motor_calibration.py\n"
            f"Erreur originale : {last_err}"
        ) from last_err

    def disconnect(self):
        """Libere le bus moteur. Recommande : disable_torque avant."""
        if self._bus is not None:
            try:
                if self._torque_enabled:
                    self._bus.disable_torque()
            finally:
                self._bus.disconnect()
                self._bus = None
                self._torque_enabled = False

    def enable_torque(self):
        """Active le couple : le bras maintient sa position et obeit aux commandes."""
        self._require_bus()
        self._bus.enable_torque()
        self._torque_enabled = True

    def disable_torque(self):
        """Coupe le couple : le bras peut etre manipule a la main."""
        self._require_bus()
        self._bus.disable_torque()
        self._torque_enabled = False

    # ----- conversion angles -> raw encoder -------------------------------

    def angles_to_raw(self, joint_angles_rad: dict[str, float],
                      gripper_pct: Optional[float] = None) -> dict[str, int]:
        """Convertit un dict {joint: rad} en valeurs encodeur raw a envoyer.

        Args:
            joint_angles_rad : pour ARM_JOINTS (5 joints).
            gripper_pct      : optionnel, 0-100. Sera converti dans la plage
                               encodeur du gripper.

        Returns:
            {joint: raw_int} pour les 5 arm + (optionnel) gripper.
        """
        out: dict[str, int] = {}
        for j in ARM_JOINTS:
            if j not in joint_angles_rad:
                raise KeyError(f"Angle manquant pour '{j}'")
            angle_rad = float(joint_angles_rad[j])
            angle_deg = np.degrees(angle_rad)
            # Securite : un angle |.| > 360deg est aberrant pour ces servos.
            if abs(angle_deg) > 360.0:
                raise ValueError(
                    f"Angle {angle_deg:.1f}deg pour '{j}' est aberrant "
                    f"(|.| > 360deg, hors plage physique). Verifie l'IK."
                )
            c = self.calib[j]
            # Centre encodeur = unwrap_center si dispo, sinon milieu plage
            center = self.unwrap.get(j, (c["range_min"] + c["range_max"]) / 2.0)
            # Inversion drive_mode
            if c["drive_mode"]:
                angle_deg = -angle_deg
            # raw_continu = center + delta_deg * (4095 / 360)  (peut sortir [0, 4095])
            raw_continu = center + angle_deg * STS3215_MAX_RESOLUTION / 360.0
            raw_wrapped = int(round(raw_continu)) % ENCODER_FULL

            r_min, r_max = c["range_min"], c["range_max"]
            TOL_COUNTS = 50  # ~4.4 deg

            # CAS 1 : plage normale (r_min < r_max, ne traverse pas 0/4095)
            # On verifie sur le delta wrappe pour rester correct.
            # CAS 2 : plage qui traverse la couture (r_min > r_max, exemple
            # wrist_roll deroule). Dans ce cas, "dans la plage" = raw >= r_min
            # OU raw <= r_max. C'est ce que LeRobot fait.
            if r_min <= r_max:
                # Plage normale
                in_range = (r_min <= raw_wrapped <= r_max)
            else:
                # Plage qui traverse 0/4095 (wraparound)
                in_range = (raw_wrapped >= r_min or raw_wrapped <= r_max)

            if in_range:
                out[j] = raw_wrapped
                continue

            # Hors plage : essaie de clip avec tolerance
            if r_min <= r_max:
                # Plage normale : distance a la borne la plus proche
                if raw_wrapped < r_min:
                    excess = r_min - raw_wrapped
                    target = r_min
                else:
                    excess = raw_wrapped - r_max
                    target = r_max
            else:
                # Plage wraparound : distance "circulaire" a la borne
                # la plus proche, peut wrap via 0/4095
                d_to_min = min((raw_wrapped - r_min) % ENCODER_FULL,
                               (r_min - raw_wrapped) % ENCODER_FULL)
                d_to_max = min((raw_wrapped - r_max) % ENCODER_FULL,
                               (r_max - raw_wrapped) % ENCODER_FULL)
                if d_to_min < d_to_max:
                    excess = d_to_min; target = r_min
                else:
                    excess = d_to_max; target = r_max

            if excess <= TOL_COUNTS:
                # Erreur d'arrondi acceptable : clip silencieusement
                out[j] = target
            else:
                excess_deg = excess * 360 / STS3215_MAX_RESOLUTION
                print(f"[motor_controller] WARN : {j} angle {angle_deg:+.1f}deg "
                      f"hors plage de {excess_deg:.1f}deg, clip a la butee.",
                      file=__import__('sys').stderr)
                out[j] = target

        if gripper_pct is not None:
            cg = self.calib["gripper"]
            pct = float(np.clip(gripper_pct, 0.0, 100.0))
            raw_g = int(round(cg["range_min"] + pct / 100.0 * (cg["range_max"] - cg["range_min"])))
            out["gripper"] = raw_g
        return out

    # ----- envoi de commandes ---------------------------------------------

    def send_angles(self, joint_angles_rad: dict[str, float],
                    gripper_pct: Optional[float] = None):
        """Envoie une commande de pose articulaire (avec optionnel gripper)."""
        self._require_bus()
        if not self._torque_enabled:
            raise RuntimeError("Couple desactive. Appelle enable_torque() d'abord.")
        raw = self.angles_to_raw(joint_angles_rad, gripper_pct)
        # sync_write Goal_Position en valeurs raw
        self._bus.sync_write("Goal_Position", raw, normalize=False)

    def execute_trajectory(self, trajectory: JointTrajectory,
                           dt_real_s: Optional[float] = None,
                           verbose: bool = True):
        """Suit une JointTrajectory en respectant les timestamps.

        Args:
            trajectory : sequence de poses + timestamps + gripper optionnel.
            dt_real_s  : si fourni, force ce dt entre commandes (override
                         les timestamps). Sinon respecte les timestamps.
            verbose    : log la progression.
        """
        self._require_bus()
        if not self._torque_enabled:
            raise RuntimeError("Couple desactive. Appelle enable_torque() d'abord.")
        if len(trajectory) == 0:
            return

        t0 = time.time()
        for i in range(len(trajectory)):
            pos = trajectory.position_at(i)
            grip = (float(trajectory.gripper_pct[i])
                    if trajectory.gripper_pct is not None else None)
            self.send_angles(pos, gripper_pct=grip)

            if verbose and i % max(1, len(trajectory) // 5) == 0:
                pct = 100 * i // max(1, len(trajectory) - 1)
                print(f"  [traj] {pct:3d}% ({i+1}/{len(trajectory)})")

            # Attendre le prochain timestamp
            if i < len(trajectory) - 1:
                if dt_real_s is not None:
                    time.sleep(dt_real_s)
                else:
                    next_t = float(trajectory.timestamps[i + 1])
                    elapsed = time.time() - t0
                    delay = next_t - elapsed
                    if delay > 0:
                        time.sleep(delay)

    # ----- lecture etat ---------------------------------------------------

    def read_raw_positions(self) -> dict[str, int]:
        """Lit les positions encodeur brutes des 6 moteurs."""
        self._require_bus()
        raw = self._bus.sync_read("Present_Position", normalize=False)
        return {k: int(v) for k, v in raw.items()}

    # ----- helpers --------------------------------------------------------

    def _require_bus(self):
        if self._bus is None:
            raise RuntimeError("Bus non connecte. Appelle connect(port) d'abord.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup propre meme si exception
        try:
            self.disable_torque()
        except Exception:
            pass
        self.disconnect()
        return False  # ne masque pas l'exception


# ============================================================
# Self-tests (lance avec : python -m src.control.motor_controller)
# Tests sans hardware : verifie juste la conversion + les erreurs gracieuses
# ============================================================
if __name__ == "__main__":
    print("Tests motor_controller.py")
    print()

    mc = MotorController()
    print(f"  {len(mc.calib)} moteurs charges")

    # 1. Conversion angles -> raw : config zero doit donner les milieux de plage
    zero_angles = {j: 0.0 for j in ARM_JOINTS}
    raw = mc.angles_to_raw(zero_angles)
    for j in ARM_JOINTS:
        c = mc.calib[j]
        expected_center = mc.unwrap.get(j, (c["range_min"] + c["range_max"]) / 2.0)
        # Tolerance 1 count pour les arrondis
        assert abs(raw[j] - expected_center) <= 1, \
            f"{j}: raw={raw[j]}, expected center={expected_center}"
    print(f"  [OK] angles_to_raw : config zero -> centres encodeur")

    # 2. Conversion angle non-zero : on doit retomber dessus avec raw_to_radians
    from src.calibration.motor_to_angle import raw_to_radians
    test_angles = {
        "shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 0.2,
        "wrist_flex": -0.1, "wrist_roll": 0.4,
    }
    raw2 = mc.angles_to_raw(test_angles)
    for j in ARM_JOINTS:
        recovered = raw_to_radians(raw2[j], mc.calib[j], mc.unwrap.get(j))
        err_deg = abs(np.degrees(recovered - test_angles[j]))
        # 360 / 4095 = 0.088 deg par count => roundtrip a < 0.1 deg
        assert err_deg < 0.15, f"{j}: roundtrip err = {err_deg:.3f} deg"
    print(f"  [OK] roundtrip angles -> raw -> angles : < 0.15 deg")

    # 3. gripper conversion
    raw_full = mc.angles_to_raw(zero_angles, gripper_pct=100.0)
    raw_zero = mc.angles_to_raw(zero_angles, gripper_pct=0.0)
    assert raw_full["gripper"] == mc.calib["gripper"]["range_max"]
    assert raw_zero["gripper"] == mc.calib["gripper"]["range_min"]
    print(f"  [OK] gripper 0% -> raw {raw_zero['gripper']}, 100% -> {raw_full['gripper']}")

    # 4. Angle hors plage -> erreur explicite
    bad_angles = dict(test_angles)
    bad_angles["shoulder_pan"] = 100.0  # 100 rad, hors plage
    try:
        mc.angles_to_raw(bad_angles)
        raise AssertionError("aurait du lever ValueError")
    except ValueError as e:
        msg = str(e)
        assert "shoulder_pan" in msg
        print(f"  [OK] angle hors plage detecte")

    # 5. send_angles sans bus -> RuntimeError
    try:
        mc.send_angles(zero_angles)
        raise AssertionError("aurait du lever RuntimeError")
    except RuntimeError:
        print(f"  [OK] send_angles sans bus -> RuntimeError")

    # 6. Context manager : pas de crash sur exit sans connexion
    with MotorController() as mc2:
        pass
    print(f"  [OK] context manager (with) propre")

    print()
    print("Tous les tests passent.")
