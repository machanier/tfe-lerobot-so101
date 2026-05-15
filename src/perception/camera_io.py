"""
camera_io.py - Capture synchronisee multi-cameras pour le SO-101.

Encapsule :
  - l'ouverture des 3 cameras USB declarees dans scripts/config.py,
  - la lecture intrinseques (configs/calibration_cam_*.json),
  - le chargement des extrinseques hand-eye (configs/handeye_cam_*.json),
  - la capture synchronisee (cv2.VideoCapture.grab puis retrieve),
  - la composition T_base_cam pour chaque camera, en fonction de son role :
       * eye-to-hand (cam_0, cam_1) : T_base_cam est constant.
       * eye-in-hand (cam_2)        : T_base_cam = T_base_gripper(t) @ T_gripper_cam.

Deux modes :

  MultiCamera (live)
      -> ouvre les peripheriques /dev/videoN, lit en temps reel.
      grab() retourne {cam_key: Frame} avec un timestamp commun proche.

  ReplayCamera (offline)
      -> lit depuis un dossier d'images preenregistrees + un manifest.json
      qui indique pour chaque snapshot les angles moteur (necessaires pour
      cam_2). Permet de developper et tester le pipeline sans avoir
      le robot branche.

Format de l'image : BGR uint8 (convention OpenCV), shape (H, W, 3).
Toutes les poses sont en METRES dans le repere base du robot.

Reference : OpenCV doc, VideoCapture.grab() est la primitive recommandee
pour synchroniser plusieurs cameras USB (decouple la capture du decodage).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# config.py est dans scripts/ ; on l'importe via un chemin absolu
sys.path.insert(0, str(REPO / "scripts"))
from config import CAMERAS  # noqa: E402

from src.perception.scene import Frame  # noqa: E402
from src.perception.robot_state import RobotState  # noqa: E402

EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"


# ============================================================
# Chargement des calibrations
# ============================================================


def load_intrinsics(cam_key: str, configs_dir: Optional[Path] = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Charge K (3x3) et dist_coeffs depuis configs/calibration_cam_N.json."""
    configs_dir = configs_dir or (REPO / "configs")
    cam_index = CAMERAS[cam_key]["index"]
    path = configs_dir / f"calibration_cam_{cam_index}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Intrinseque introuvable pour {cam_key} : {path}\n"
            f"Lance : python scripts/calibrate_intrinsic.py --index {cam_index}"
        )
    data = json.load(open(path))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def load_handeye(cam_key: str, configs_dir: Optional[Path] = None) -> dict:
    """Charge la calibration hand-eye d'une camera.

    Returns:
        dict {
            "configuration": "eye_to_hand" | "eye_in_hand",
            "transform": (4,4),                # T_base_cam OU T_gripper_cam
            "residuals": {...},                # stats hand-eye
        }
    """
    configs_dir = configs_dir or (REPO / "configs")
    cam_index = CAMERAS[cam_key]["index"]
    path = configs_dir / f"handeye_cam_{cam_index}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Hand-eye introuvable pour {cam_key} : {path}\n"
            f"Lance : python scripts/solve_handeye_cam.py --index {cam_index}"
        )
    data = json.load(open(path))
    return {
        "configuration": data["configuration"],
        "transform": np.array(data["transform"], dtype=np.float64),
        "transform_name": data["transform_name"],
        "residuals": data.get("residuals", {}),
    }


def compose_T_base_cam(cam_key: str, handeye: dict,
                       robot_state: Optional[RobotState] = None) -> np.ndarray:
    """Calcule T_base_cam (pose camera dans le repere base) pour une camera.

    eye-to-hand : T_base_cam vient directement du fichier hand-eye.
    eye-in-hand : T_base_cam = T_base_gripper(t) @ T_gripper_cam, donc depend
        de la position courante du robot (necessite RobotState).
    """
    cfg = handeye["configuration"]
    T_handeye = handeye["transform"]
    if cfg == EYE_TO_HAND:
        return T_handeye
    if cfg == EYE_IN_HAND:
        if robot_state is None:
            raise ValueError(
                f"{cam_key} est eye-in-hand : un RobotState est requis pour "
                "calculer T_base_cam."
            )
        return robot_state.T_base_gripper @ T_handeye
    raise ValueError(f"Configuration hand-eye inconnue: {cfg!r}")


# ============================================================
# Mode LIVE : capture synchronisee des cameras USB
# ============================================================


class MultiCamera:
    """Ouvre et synchronise les cameras declarees dans config.CAMERAS.

    Strategie de synchronisation :
        1. Pour chaque camera, on emet `grab()` (decoupe la capture du decodage).
        2. On enregistre un timestamp commun (juste apres le dernier grab).
        3. On appelle `retrieve()` pour decoder les frames.
        4. Si une seule frame a echoue, on la remplace par None et on logge.

    Cette approche est celle recommandee par OpenCV pour les ensembles de
    cameras USB (cf doc VideoCapture). L'erreur de synchronisation est
    typiquement < 1 frame (~33 ms a 30 FPS), ce qui est negligeable devant
    la vitesse du SO-101 (objet quasi-statique pendant la perception).
    """

    def __init__(self, cam_keys: Iterable[str] = ("cam_0", "cam_1", "cam_2"),
                 configs_dir: Optional[Path] = None):
        self.cam_keys = list(cam_keys)
        self.configs_dir = configs_dir or (REPO / "configs")
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._handeye: dict[str, dict] = {}
        self._opened = False

    # ----- contexte --------------------------------------------------------

    def open(self):
        """Ouvre les peripheriques et charge intrinseques/hand-eye."""
        if self._opened:
            return
        for k in self.cam_keys:
            cfg = CAMERAS[k]
            cap = cv2.VideoCapture(cfg["index"])
            if not cap.isOpened():
                # Cleanup partiel pour ne pas fuir les peripheriques
                self.close()
                raise RuntimeError(
                    f"Impossible d'ouvrir {k} (index {cfg['index']}). "
                    "Verifie l'autorisation camera macOS et que la camera "
                    "n'est pas utilisee par un autre processus."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg["width"]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg["height"]))
            cap.set(cv2.CAP_PROP_FPS, float(cfg["fps"]))
            # Reduit le buffer pour eviter d'accumuler du retard (latence basse).
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # pas critique
            self._caps[k] = cap
            self._intrinsics[k] = load_intrinsics(k, self.configs_dir)
            self._handeye[k] = load_handeye(k, self.configs_dir)
        self._opened = True

    def close(self):
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        self._opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ----- capture ---------------------------------------------------------

    def grab(self, robot_state: Optional[RobotState] = None
             ) -> dict[str, Optional[Frame]]:
        """Capture synchronisee des cameras ouvertes.

        Args:
            robot_state : etat courant du robot, OBLIGATOIRE si au moins une
                camera est eye-in-hand. Peut etre None si toutes sont eye-to-hand.

        Returns:
            dict {cam_key: Frame or None}. Les cameras qui ont echoue
            renvoient None (logge sur stderr).
        """
        if not self._opened:
            raise RuntimeError("MultiCamera n'est pas ouvert. Appelle open() d'abord.")

        # 1. emet grab() pour les N cameras de facon rapprochee
        grab_ok = {}
        for k in self.cam_keys:
            grab_ok[k] = self._caps[k].grab()
        ts = time.time()

        # 2. decode les frames qui ont reussi
        frames: dict[str, Optional[Frame]] = {}
        for k in self.cam_keys:
            if not grab_ok[k]:
                print(f"[camera_io] grab() KO pour {k}", file=sys.stderr)
                frames[k] = None
                continue
            ok, img = self._caps[k].retrieve()
            if not ok:
                print(f"[camera_io] retrieve() KO pour {k}", file=sys.stderr)
                frames[k] = None
                continue

            K, dist = self._intrinsics[k]
            T_base_cam = compose_T_base_cam(k, self._handeye[k], robot_state)
            frames[k] = Frame(
                cam_key=k, image=img, K=K, dist=dist,
                T_base_cam=T_base_cam, timestamp=ts,
            )
        return frames

    # ----- introspection ---------------------------------------------------

    def info(self) -> str:
        """Resume textuel pour le debug."""
        lines = []
        for k in self.cam_keys:
            cfg = CAMERAS[k]
            he = self._handeye.get(k)
            tag = f"({he['configuration']})" if he else "(?)"
            lines.append(f"  {k}  idx={cfg['index']:<2}  role={cfg['role']:<14} {tag}")
        return "\n".join(lines)


# ============================================================
# Mode REPLAY : lit des frames pre-enregistrees
# ============================================================


class ReplayCamera:
    """Joue un dataset de frames synchronisees enregistre par
    scripts/record_perception_frames.py.

    Structure attendue du dossier `root` :

        root/
            manifest.json      liste ordonnee de snapshots, chacun avec :
                               {
                                 "id": int,
                                 "timestamp": float,
                                 "robot_state": { joint_angles_rad: {...},
                                                   raw_positions: {...} },
                                 "frames": { "cam_0": "snap_001/cam_0.png", ... }
                               }
            snap_001/cam_0.png ...
            snap_001/cam_1.png ...
            snap_001/cam_2.png ...

    L'avantage du replay : on peut iterer sur le detecteur sans avoir le
    robot branche, ET on obtient des resultats EXACTEMENT reproductibles
    pour le memoire.
    """

    def __init__(self, root: Path, configs_dir: Optional[Path] = None):
        self.root = Path(root)
        self.configs_dir = configs_dir or (REPO / "configs")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json introuvable dans {self.root}")
        self.manifest = json.load(open(manifest_path))
        snapshots = self.manifest.get("snapshots") or self.manifest
        if not isinstance(snapshots, list):
            raise ValueError("manifest.json: clef 'snapshots' attendue (liste)")
        self.snapshots = snapshots
        self.cam_keys = self.manifest.get("cam_keys", ["cam_0", "cam_1", "cam_2"])
        self._intrinsics = {k: load_intrinsics(k, self.configs_dir) for k in self.cam_keys}
        self._handeye = {k: load_handeye(k, self.configs_dir) for k in self.cam_keys}

    def __len__(self):
        return len(self.snapshots)

    def __iter__(self):
        for i in range(len(self)):
            yield self.read(i)

    def read(self, index: int) -> tuple[dict[str, Optional[Frame]], Optional[RobotState]]:
        """Renvoie (frames, robot_state) pour le snapshot `index`."""
        from src.perception.robot_state import RobotStateProvider

        snap = self.snapshots[index]

        # --- reconstruit le RobotState s'il est present
        robot_state: Optional[RobotState] = None
        rs = snap.get("robot_state")
        if rs:
            provider = RobotStateProvider()
            if "joint_angles_rad" in rs:
                robot_state = provider.from_angles(rs["joint_angles_rad"])
            elif "raw_positions" in rs:
                robot_state = provider.from_raw(rs["raw_positions"])

        # --- charge les frames
        ts = float(snap.get("timestamp", index))
        frames: dict[str, Optional[Frame]] = {}
        for k in self.cam_keys:
            rel = snap.get("frames", {}).get(k)
            if rel is None:
                frames[k] = None
                continue
            path = self.root / rel
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                frames[k] = None
                continue
            K, dist = self._intrinsics[k]
            T_base_cam = compose_T_base_cam(k, self._handeye[k], robot_state)
            frames[k] = Frame(
                cam_key=k, image=img, K=K, dist=dist,
                T_base_cam=T_base_cam, timestamp=ts,
            )
        return frames, robot_state


# ============================================================
# Self-tests (lance avec : python -m src.perception.camera_io)
# ============================================================
if __name__ == "__main__":
    print("Tests camera_io.py")

    # 1. Chargement intrinseque
    K, dist = load_intrinsics("cam_0")
    assert K.shape == (3, 3)
    print(f"  [OK] load_intrinsics cam_0 : fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")

    # 2. Chargement hand-eye
    he = load_handeye("cam_0")
    assert he["transform"].shape == (4, 4)
    assert he["configuration"] in (EYE_TO_HAND, EYE_IN_HAND)
    print(f"  [OK] load_handeye cam_0 ({he['configuration']})")

    he2 = load_handeye("cam_2")
    assert he2["configuration"] == EYE_IN_HAND
    print(f"  [OK] load_handeye cam_2 (eye_in_hand)")

    # 3. compose_T_base_cam : eye-to-hand n'a pas besoin de RobotState
    T0 = compose_T_base_cam("cam_0", he)
    assert T0.shape == (4, 4)
    print(f"  [OK] compose_T_base_cam (eye-to-hand, sans robot_state)")

    # 4. compose_T_base_cam : eye-in-hand sans RobotState -> erreur explicite
    try:
        compose_T_base_cam("cam_2", he2)
        raise AssertionError("aurait du lever ValueError")
    except ValueError:
        print("  [OK] compose_T_base_cam (eye-in-hand sans robot_state -> ValueError)")

    # 5. compose_T_base_cam : eye-in-hand avec un RobotState fabrique
    from src.perception.robot_state import RobotStateProvider

    provider = RobotStateProvider()
    s = provider.from_angles({"shoulder_pan": 0.0, "shoulder_lift": 0.0,
                              "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0})
    T2 = compose_T_base_cam("cam_2", he2, robot_state=s)
    expected = s.T_base_gripper @ he2["transform"]
    assert np.allclose(T2, expected)
    print("  [OK] compose_T_base_cam (eye-in-hand, compose avec FK)")

    # 6. MultiCamera : on n'OUVRE pas les peripheriques en self-test (la cam
    #    physique peut etre absente). On se contente de l'introspection.
    mc = MultiCamera()
    assert mc.cam_keys == ["cam_0", "cam_1", "cam_2"]
    print("  [OK] MultiCamera: instanciation")

    print("Tous les tests passent.")
