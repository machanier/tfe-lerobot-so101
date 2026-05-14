"""
motor_to_angle.py - Conversion entre valeurs encodeur brutes et angles articulaires.

Les servomoteurs Feetech STS3215 retournent une "Present_Position" qui est un
entier sur 12 bits (0 a 4095). Pour la convertir en angle articulaire (en
radians), on a besoin de la calibration moteur stockee dans
configs/calibration_follower.json :

    {
        "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -1491,
                         "range_min": 1026, "range_max": 3151},
        ...
    }

LeRobot utilise la formule (cf motors_bus.py:858) :
    angle_deg = (raw - mid) * 360 / max_resolution
    avec mid = (range_min + range_max) / 2
    et max_resolution = 4095 pour STS3215

Le drive_mode (0 ou 1) inverse le sens de rotation. homing_offset n'intervient
PAS dans cette formule (il sert uniquement a recentrer le zero hardware lors
de la calibration initiale, deja applique aux range_min/max).

Reference :
- LeRobot motors_bus.py, _normalize() ligne 838-865
- Documentation Feetech STS3215 : 0-4095 = 0-360 deg
"""

import json
from pathlib import Path

import numpy as np

# STS3215 a un encodeur 12 bits => 4096 positions => max = 4095
STS3215_MAX_RESOLUTION = 4095


def load_motor_calibration(path):
    """Charge la calibration moteurs (Feetech) depuis un JSON LeRobot.

    Args:
        path: chemin vers configs/calibration_follower.json

    Returns:
        dict {motor_name: {id, drive_mode, homing_offset, range_min, range_max}}
    """
    with open(path) as f:
        return json.load(f)


def raw_to_radians(raw_value, motor_calib):
    """Convertit une valeur brute encodeur en angle (radians) pour un moteur.

    Args:
        raw_value: position brute (entier 0-4095)
        motor_calib: dict avec les cles drive_mode, range_min, range_max

    Returns:
        angle en radians (float)
    """
    range_min = motor_calib["range_min"]
    range_max = motor_calib["range_max"]
    drive_mode = motor_calib["drive_mode"]

    mid = (range_min + range_max) / 2
    angle_deg = (raw_value - mid) * 360.0 / STS3215_MAX_RESOLUTION

    if drive_mode:
        angle_deg = -angle_deg

    return np.deg2rad(angle_deg)


def raw_dict_to_radians(raw_positions, calibration, motor_order):
    """Convertit un dict {motor_name: raw} en vecteur d'angles ordonne.

    Args:
        raw_positions: dict {motor_name: raw_value}
        calibration: dict de calibration_follower.json
        motor_order: liste ordonnee des noms de moteurs (kinematic chain)

    Returns:
        np.ndarray (N,) : angles en radians dans l'ordre demande
    """
    angles = []
    for name in motor_order:
        if name not in raw_positions:
            raise KeyError(f"Motor '{name}' absent des raw_positions")
        if name not in calibration:
            raise KeyError(f"Motor '{name}' absent de la calibration")
        angle_rad = raw_to_radians(raw_positions[name], calibration[name])
        angles.append(angle_rad)
    return np.array(angles, dtype=np.float64)


# ============================================================
# Tests rapides (lance avec : python -m src.calibration.motor_to_angle)
# ============================================================
if __name__ == "__main__":
    print("Tests motor_to_angle.py")

    repo_root = Path(__file__).resolve().parents[2]
    calib_path = repo_root / "configs" / "calibration_follower.json"
    print(f"  Chargement {calib_path}")
    calib = load_motor_calibration(calib_path)
    print(f"  {len(calib)} moteurs : {list(calib.keys())}")

    # Test : valeur au milieu de la range = 0 rad
    for name, c in calib.items():
        mid = (c["range_min"] + c["range_max"]) / 2
        angle = raw_to_radians(mid, c)
        assert abs(angle) < 1e-9, f"{name} mid -> {angle} (devrait etre 0)"
        # Aux extremes
        a_min = raw_to_radians(c["range_min"], c)
        a_max = raw_to_radians(c["range_max"], c)
        amplitude_deg = np.rad2deg(a_max - a_min)
        print(f"  {name:14s}: range raw [{c['range_min']:>4d}, {c['range_max']:>4d}] "
              f"-> angles [{np.rad2deg(a_min):+7.2f}, {np.rad2deg(a_max):+7.2f}] "
              f"deg (amplitude {abs(amplitude_deg):.1f} deg)")

    # Test sur une vraie capture
    print()
    print("  Test sur la 1ere capture de configs/extrinsic_capture_cam_0.json :")
    capture_path = repo_root / "configs" / "extrinsic_capture_cam_0.json"
    if capture_path.exists():
        data = json.load(open(capture_path))
        first = data["captures"][0]
        raw = first["motor_positions_raw"]
        order = data["motor_names"]
        angles = raw_dict_to_radians(raw, calib, order)
        print(f"    raw    : {dict((k, int(v)) for k, v in raw.items())}")
        print(f"    radians: {angles}")
        print(f"    degres : {np.rad2deg(angles).round(2)}")

    print("[OK] tous les tests passent")
