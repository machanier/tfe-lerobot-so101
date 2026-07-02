"""
robot_state.py - Lecture des moteurs SO-101 et calcul de la pose effecteur.

Encapsule la communication avec le bus Feetech pour fournir au pipeline de
perception :
  - les angles articulaires actuels (rad),
  - la pose de l'effecteur dans le repere base (T_base_gripper),
  - les T_base_cam des cameras eye-in-hand (par composition).

Ce module est volontairement le seul endroit ou la perception touche au
hardware moteur. Tout le reste du pipeline lit ses sorties (`RobotState`).

Mode hors-ligne : on peut construire un `RobotState` directement depuis une
configuration d'angles connue (utile pour les tests, le mode replay, et
n'importe quelle execution sans le robot branche).

Reference : convention "new_calib" du SO-ARM100 (cf src/calibration/motor_to_angle.py).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Repo root est deux niveaux au-dessus de ce fichier (src/perception/ -> repo)
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.calibration.forward_kinematics import ARM_JOINTS, KinematicChain
from src.calibration.motor_to_angle import (
    load_encoder_unwrap,
    load_motor_calibration,
    raw_to_radians,
)


@dataclass
class RobotState:
    """Etat instantane du robot, pret a etre consomme par la perception.

    Attributes:
        joint_angles_rad : {joint_name: rad}, contient au moins ARM_JOINTS.
        T_base_gripper   : pose 4x4 de l'effecteur dans le repere base (m).
        raw_positions    : {joint_name: int}, valeurs encodeur brutes (debug).
        timestamp        : horodatage de la lecture (epoch s).
    """

    joint_angles_rad: dict[str, float]
    T_base_gripper: np.ndarray
    raw_positions: dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RobotStateProvider:
    """Fournit des `RobotState` coherents au pipeline.

    Trois modes d'utilisation :

      1. `from_live(port)`           : se connecte au bus Feetech et lit en
                                        temps reel a chaque appel de `read()`.
      2. `from_angles(joint_angles)` : utilise une configuration fixe (tests).
      3. `from_raw(raw_positions)`   : utilise des valeurs encodeur (replay).

    Les calibrations moteur et l'unwrap d'encodeur sont charges une seule fois
    a l'init ; idem pour la chaine cinematique.
    """

    def __init__(self, urdf_path: Optional[Path] = None,
                 calib_path: Optional[Path] = None,
                 unwrap_path: Optional[Path] = None):
        calib_path = calib_path or (REPO / "configs" / "calibration_follower.json")
        unwrap_path = unwrap_path or (REPO / "configs" / "encoder_unwrap.json")
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration moteur introuvable: {calib_path}")
        self.calib = load_motor_calibration(calib_path)
        self.unwrap = load_encoder_unwrap(unwrap_path, self.calib)
        self.chain = KinematicChain(urdf_path) if urdf_path else KinematicChain()
        self._bus = None  # initialise par connect_live()

    # ----- helpers de calcul (purs, sans hardware) -------------------------

    def _angles_from_raw(self, raw_positions: dict[str, float]) -> dict[str, float]:
        """Convertit un dict de positions encodeur brutes en angles (rad)."""
        out = {}
        for j in ARM_JOINTS:
            if j not in raw_positions:
                raise KeyError(f"Position brute manquante pour '{j}'")
            out[j] = raw_to_radians(
                raw_positions[j], self.calib[j], self.unwrap.get(j)
            )
        return out

    def _state_from_angles(self, joint_angles_rad: dict[str, float],
                           raw_positions: Optional[dict[str, int]] = None,
                           timestamp: Optional[float] = None) -> RobotState:
        T = self.chain.fk(joint_angles_rad)
        return RobotState(
            joint_angles_rad=dict(joint_angles_rad),
            T_base_gripper=T,
            raw_positions=dict(raw_positions) if raw_positions else {},
            timestamp=timestamp if timestamp is not None else time.time(),
        )

    # ----- API factories ---------------------------------------------------

    def from_angles(self, joint_angles_rad: dict[str, float]) -> RobotState:
        """Construit un etat depuis des angles deja calcules (mode test/replay)."""
        return self._state_from_angles(joint_angles_rad)

    def from_raw(self, raw_positions: dict[str, float]) -> RobotState:
        """Construit un etat depuis des positions encodeur brutes (mode replay)."""
        angles = self._angles_from_raw(raw_positions)
        return self._state_from_angles(angles, raw_positions=raw_positions)

    # ----- mode live -------------------------------------------------------

    def connect_live(self, port: str):
        """Se connecte au bus Feetech du follower (mode live).

        Args:
            port : chemin du port USB (e.g. /dev/tty.usbmodem...).

        Raises:
            RuntimeError, FileNotFoundError, ImportError selon la cause.
        """
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as e:
            raise ImportError(
                "LeRobot indisponible. Activer le venv (source venv/bin/activate) "
                "ou l'installer selon setup_env.sh."
            ) from e

        motors = {
            "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)
        bus.connect()
        self._bus = bus

    def disconnect_live(self):
        """Libere le bus moteur s'il est ouvert."""
        if self._bus is not None:
            self._bus.disconnect()
            self._bus = None

    def read_live(self, max_retries: int = 3, fallback_to_last: bool = True
                  ) -> RobotState:
        """Lit l'etat courant du robot (necessite connect_live() au prealable).

        Tente jusqu'a `max_retries` fois en cas d'echec ponctuel du bus
        (typique : 'Incorrect status packet' quand un paquet est corrompu
        ou qu'un servo presente un defaut transitoire). Si tous les essais
        echouent et `fallback_to_last=True`, renvoie le dernier RobotState lu
        plutot que de lever une exception : pour la perception, c'est
        generalement acceptable (le robot bouge lentement, la pose change peu
        en 100 ms).
        """
        if self._bus is None:
            raise RuntimeError("Bus non connecte. Appeler connect_live(port) au prealable.")
        last_err = None
        for attempt in range(max_retries):
            try:
                raw = self._bus.sync_read("Present_Position", normalize=False)
                raw = {k: float(v) for k, v in raw.items()}
                angles = self._angles_from_raw(raw)
                state = self._state_from_angles(angles, raw_positions=raw)
                self._last_state = state
                return state
            except (ConnectionError, RuntimeError) as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(0.02)  # breve pause avant nouvel essai
                    continue
        # Tous les essais ont echoue
        if fallback_to_last and getattr(self, "_last_state", None) is not None:
            print(f"[robot_state] read_live en echec {max_retries}x, fallback derniere pose "
                  f"({last_err})", file=sys.stderr)
            return self._last_state
        raise ConnectionError(
            f"Lecture moteur impossible apres {max_retries} essais. {last_err}\n"
            "Verifier : (1) le robot est sous tension, (2) le cable USB n'est pas debranche, "
            "(3) aucun autre process n'utilise le bus."
        )

    # context manager pour usage propre : with RobotStateProvider() as p: ...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_live()
        return False  # ne supprime pas l'exception


# ============================================================
# Self-tests (execution : python -m src.perception.robot_state)
# ============================================================
if __name__ == "__main__":
    print("Tests robot_state.py")

    p = RobotStateProvider()
    print(f"  {len(p.calib)} moteurs charges, unwrap : {list(p.unwrap)}")

    # 1. from_angles : config zero -> T_base_gripper coherent avec FK
    zero = {j: 0.0 for j in ARM_JOINTS}
    s = p.from_angles(zero)
    assert s.T_base_gripper.shape == (4, 4)
    t = s.T_base_gripper[:3, 3]
    print(f"  Config zero -> effecteur ({t[0] * 1000:.1f}, {t[1] * 1000:.1f}, "
          f"{t[2] * 1000:.1f}) mm")
    assert 0.05 < float(np.linalg.norm(t)) < 0.6, "echelle implausible"
    print("  [OK] from_angles (config zero)")

    # 2. from_raw : centre encodeur de chaque joint -> config zero (idem FK)
    centers_raw = {}
    for j in ARM_JOINTS:
        ctr = p.unwrap.get(j, (p.calib[j]["range_min"] + p.calib[j]["range_max"]) / 2)
        centers_raw[j] = ctr
    s2 = p.from_raw(centers_raw)
    assert np.allclose(s2.T_base_gripper, s.T_base_gripper, atol=1e-6), \
        "centre encodeur != config zero"
    print("  [OK] from_raw (centres encodeur -> config zero)")

    # 3. from_raw : valeur manquante -> KeyError
    try:
        p.from_raw({"shoulder_pan": 2048})
        raise AssertionError("aurait du lever KeyError")
    except KeyError:
        print("  [OK] from_raw detecte les positions manquantes")

    # 4. read_live sans connect_live -> RuntimeError
    try:
        p.read_live()
        raise AssertionError("aurait du lever RuntimeError")
    except RuntimeError:
        print("  [OK] read_live sans connect -> RuntimeError")

    print("Tous les tests passent.")
