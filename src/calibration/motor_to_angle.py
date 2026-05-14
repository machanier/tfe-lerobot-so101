"""
motor_to_angle.py - Conversion entre valeurs encodeur brutes et angles articulaires.

Les servomoteurs Feetech STS3215 retournent une "Present_Position" : un entier
sur 12 bits (0 a 4095) qui represente la position sur UN tour. L'encodeur est
CIRCULAIRE : apres 4095 on revient a 0 (comme une boussole).

LeRobot convertit en degres avec (cf motors_bus.py:858) :
    angle_deg = (raw - mid) * 360 / max_resolution
    avec mid = (range_min + range_max) / 2  et  max_resolution = 4095

Mais cette formule ignore le caractere circulaire de l'encodeur : si la course
d'un joint chevauche la couture 0/4095, deux positions physiquement voisines
(ex: raw=4090 et raw=5) donnent des angles a ~360 deg d'ecart. C'est le cas de
wrist_roll sur ce robot.

On corrige donc en "deroulant" la valeur autour d'un centre :
    angle_deg = wrap(raw - center) * 360 / 4095
    avec wrap(d) = ((d + 2048) % 4096) - 2048   (ramene dans (-180, 180] deg)

- Pour la plupart des joints, center = (range_min + range_max) / 2 et wrap()
  ne change rien (la plage ne touche pas la couture).
- Pour wrist_roll, dont la course chevauche la couture, center est lu depuis
  configs/encoder_unwrap.json (mesure par scripts/measure_wrist_roll.py).

Le drive_mode (0 ou 1) inverse le sens de rotation. homing_offset n'intervient
pas ici : il est applique cote hardware, donc deja inclus dans la "raw" lue.

Reference :
- LeRobot motors_bus.py, _normalize() ligne 838-865
- Documentation Feetech STS3215 : 0-4095 = 0-360 deg
"""

import json
from pathlib import Path

import numpy as np

# STS3215 : encodeur 12 bits => 4096 positions, valeur max 4095
ENCODER_FULL = 4096
STS3215_MAX_RESOLUTION = ENCODER_FULL - 1


def load_motor_calibration(path):
    """Charge la calibration moteurs (Feetech) depuis un JSON LeRobot.

    Args:
        path: chemin vers configs/calibration_follower.json

    Returns:
        dict {motor_name: {id, drive_mode, homing_offset, range_min, range_max}}
    """
    with open(path) as f:
        return json.load(f)


def load_encoder_unwrap(path, calibration=None):
    """Charge les centres de deroulage par joint depuis encoder_unwrap.json.

    Ce fichier (genere par scripts/measure_wrist_roll.py) ne concerne que les
    joints dont la course chevauche la couture 0/4095. S'il n'existe pas, on
    retourne un dict vide et tous les joints utilisent le milieu de leur plage.

    Args:
        path: chemin vers configs/encoder_unwrap.json (peut ne pas exister)
        calibration: si fourni, verifie que le homing_offset enregistre
            correspond toujours a la calibration courante (sinon la mesure est
            obsolete : il faut relancer measure_wrist_roll.py)

    Returns:
        dict {motor_name: unwrap_center}
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    centers = {}
    for joint, info in data.items():
        centers[joint] = info["unwrap_center"]
        if calibration is not None and joint in calibration:
            measured = info.get("homing_offset_when_measured")
            current = calibration[joint].get("homing_offset")
            if measured is not None and measured != current:
                print(
                    f"  AVERTISSEMENT : encoder_unwrap.json pour '{joint}' a ete mesure "
                    f"avec homing_offset={measured}, mais la calibration courante a "
                    f"homing_offset={current}.\n"
                    f"  -> le follower a ete recalibre, relance scripts/measure_wrist_roll.py"
                )
    return centers


def wrap_encoder_delta(d):
    """Ramene une difference d'encodeur dans (-2048, 2048] (encodeur circulaire)."""
    return ((d + ENCODER_FULL // 2) % ENCODER_FULL) - ENCODER_FULL // 2


def raw_to_radians(raw_value, motor_calib, unwrap_center=None):
    """Convertit une valeur brute encodeur en angle (radians) pour un moteur.

    L'encodeur etant circulaire, la valeur est deroulee autour d'un centre :
    le milieu de la plage calibree par defaut (le deroulage est alors sans
    effet), ou un centre mesure (unwrap_center) pour un joint dont la course
    chevauche la couture 0/4095.

    Args:
        raw_value: position brute (entier 0-4095)
        motor_calib: dict avec les cles drive_mode, range_min, range_max
        unwrap_center: centre de deroulage en counts encodeur
            (defaut: milieu de la plage calibree)

    Returns:
        angle en radians (float)
    """
    if unwrap_center is None:
        unwrap_center = (motor_calib["range_min"] + motor_calib["range_max"]) / 2

    delta = wrap_encoder_delta(raw_value - unwrap_center)
    angle_deg = delta * 360.0 / STS3215_MAX_RESOLUTION

    if motor_calib["drive_mode"]:
        angle_deg = -angle_deg

    return np.deg2rad(angle_deg)


def raw_dict_to_radians(raw_positions, calibration, motor_order, unwrap_centers=None):
    """Convertit un dict {motor_name: raw} en vecteur d'angles ordonne.

    Args:
        raw_positions: dict {motor_name: raw_value}
        calibration: dict de calibration_follower.json
        motor_order: liste ordonnee des noms de moteurs (kinematic chain)
        unwrap_centers: dict {motor_name: unwrap_center}, cf load_encoder_unwrap.
            Les joints absents utilisent le milieu de leur plage calibree.

    Returns:
        np.ndarray (N,) : angles en radians dans l'ordre demande
    """
    unwrap_centers = unwrap_centers or {}
    angles = []
    for name in motor_order:
        if name not in raw_positions:
            raise KeyError(f"Motor '{name}' absent des raw_positions")
        if name not in calibration:
            raise KeyError(f"Motor '{name}' absent de la calibration")
        angle_rad = raw_to_radians(
            raw_positions[name], calibration[name], unwrap_centers.get(name)
        )
        angles.append(angle_rad)
    return np.array(angles, dtype=np.float64)


# ============================================================
# Tests rapides (lance avec : python -m src.calibration.motor_to_angle)
# ============================================================
if __name__ == "__main__":
    print("Tests motor_to_angle.py")
    print()

    repo_root = Path(__file__).resolve().parents[2]
    calib = load_motor_calibration(repo_root / "configs" / "calibration_follower.json")
    unwrap = load_encoder_unwrap(repo_root / "configs" / "encoder_unwrap.json", calib)
    print(f"  {len(calib)} moteurs, centres de deroulage specifiques : {list(unwrap)}")
    print()

    # 1. Le centre de deroulage donne toujours 0 rad
    for name, c in calib.items():
        center = unwrap.get(name, (c["range_min"] + c["range_max"]) / 2)
        angle = raw_to_radians(center, c, unwrap.get(name))
        assert abs(angle) < 1e-9, f"{name}: centre -> {angle} (devrait etre 0)"
    print("  [OK] le centre de deroulage donne 0 rad pour chaque moteur")

    # 2. Continuite a la couture 0/4095 pour un joint deroule (wrist_roll)
    if "wrist_roll" in unwrap:
        c = calib["wrist_roll"]
        ctr = unwrap["wrist_roll"]
        a_4095 = np.rad2deg(raw_to_radians(4095, c, ctr))
        a_0 = np.rad2deg(raw_to_radians(0, c, ctr))
        # avec deroulage : 4095 et 0 sont voisins -> ~0.09 deg d'ecart
        assert abs(a_4095 - a_0) < 1.0, f"discontinuite a la couture : {a_4095} vs {a_0}"
        # sans deroulage : la formule naive ferait un saut de ~360 deg
        naive_4095 = (4095 - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095
        naive_0 = (0 - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095
        print(f"  [OK] wrist_roll continu a la couture : raw 4095->{a_4095:+.2f} deg, "
              f"raw 0->{a_0:+.2f} deg (ecart {abs(a_4095 - a_0):.2f} deg)")
        print(f"       sans deroulage la formule naive sauterait de "
              f"{abs(naive_4095 - naive_0):.0f} deg")

    # 3. wrist_roll : toute sa course reste dans une plage d'angles plausible
    if "wrist_roll" in unwrap:
        c = calib["wrist_roll"]
        ctr = unwrap["wrist_roll"]
        angles = [np.rad2deg(raw_to_radians(r, c, ctr)) for r in range(0, 4096, 16)]
        # course mesuree ~334 deg -> tous les angles atteignables dans +-170 deg
        # (les valeurs de la zone morte peuvent depasser, c'est normal)
        reachable = [a for a in angles if abs(a) <= 170]
        assert len(reachable) > 0.85 * len(angles), "trop d'angles hors plage plausible"
        print(f"  [OK] wrist_roll : course deroulee dans [{min(angles):+.0f}, "
              f"{max(angles):+.0f}] deg")

    # 4. Recapitulatif par moteur
    print()
    print(f"  {'moteur':<14} {'plage raw':<14} {'centre':>7}  {'amplitude angulaire'}")
    for name, c in calib.items():
        center = unwrap.get(name, (c["range_min"] + c["range_max"]) / 2)
        a_min = np.rad2deg(raw_to_radians(c["range_min"], c, unwrap.get(name)))
        a_max = np.rad2deg(raw_to_radians(c["range_max"], c, unwrap.get(name)))
        tag = " (deroule)" if name in unwrap else ""
        print(f"  {name:<14} [{c['range_min']:>4d},{c['range_max']:>4d}]   {center:>7.0f}  "
              f"[{a_min:+7.2f}, {a_max:+7.2f}] deg{tag}")

    print()
    print("[OK] tous les tests passent")
