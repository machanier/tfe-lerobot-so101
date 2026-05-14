"""
forward_kinematics.py - Cinematique directe du bras SO-101.

Calcule la pose de l'effecteur (gripper_frame_link) dans le repere de la base
du robot a partir des angles articulaires :

    T_base_gripper = fk_so101(joint_angles_rad)

Le modele geometrique est lu depuis un fichier URDF
(configs/so101_new_calib.urdf, genere par TheRobotStudio a partir du CAD
Onshape, recupere depuis le depot TheRobotStudio/SO-ARM100). L'URDF est la
SEULE source de verite pour la geometrie : pour corriger ou changer le modele,
on remplace le fichier URDF, pas le code.

Convention de calibration : l'URDF "new_calib" place le zero de chaque
articulation au MILIEU de sa course, ce qui correspond exactement a la
conversion faite par src/calibration/motor_to_angle.py.

Chaine cinematique (base -> effecteur), 5 articulations rotoides + 1 joint fixe :
    base_link --[shoulder_pan]--> shoulder_link --[shoulder_lift]--> upper_arm_link
      --[elbow_flex]--> lower_arm_link --[wrist_flex]--> wrist_link
      --[wrist_roll]--> gripper_link --[gripper_frame_joint, fixe]--> gripper_frame_link

L'articulation "gripper" (machoire) est une branche laterale : elle ne deplace
pas gripper_frame_link, donc elle n'intervient pas dans cette chaine.

Pour changer de methode de calcul (ex: passer a placo/pinocchio), il suffit de
reimplementer fk_so101() / KinematicChain.fk() en gardant la meme signature.

Reference : convention URDF (balises <origin xyz rpy>, <axis>),
http://wiki.ros.org/urdf/XML/joint

Utilise par : scripts/solve_handeye_*.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.utils.transforms import rotation_about_axis, xyz_rpy_to_matrix

# URDF par defaut : a la racine du depot, dans configs/
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "configs" / "so101_new_calib.urdf"

# Les 5 articulations rotoides entre la base et l'effecteur, dans l'ordre.
# (la 6e articulation "gripper" actionne la machoire et ne deplace pas
#  gripper_frame_link : elle est volontairement hors de cette chaine)
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

BASE_LINK = "base_link"
EE_LINK = "gripper_frame_link"


def _parse_joints(urdf_path):
    """Lit les articulations de l'URDF (enfants directs de <robot>).

    findall("joint") ne retourne que les enfants DIRECTS de <robot>, donc les
    <joint> imbriques dans les <transmission> sont naturellement ignores.

    Returns:
        dict {joint_name: {parent, child, type, origin (4x4), axis (3,)}}
    """
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis_el = joint.find("axis")
        axis = [float(v) for v in (axis_el.get("xyz") if axis_el is not None else "0 0 1").split()]
        joints[name] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "type": joint.get("type"),
            "origin": xyz_rpy_to_matrix(xyz, rpy),
            "axis": np.array(axis, dtype=np.float64),
        }
    return joints


def _build_chain(joints, base_link, ee_link):
    """Construit la liste ordonnee des articulations de base_link a ee_link.

    Remonte depuis ee_link vers base_link en suivant les liens parent/enfant.

    Returns:
        liste de noms d'articulations, ordonnee de la base vers l'effecteur
    """
    by_child = {j["child"]: name for name, j in joints.items()}
    chain = []
    link = ee_link
    while link != base_link:
        if link not in by_child:
            raise ValueError(
                f"Lien '{link}' non rattache a '{base_link}' dans l'URDF "
                f"(chaine cinematique cassee ou noms de liens incorrects)"
            )
        jname = by_child[link]
        chain.append(jname)
        link = joints[jname]["parent"]
    chain.reverse()
    return chain


class KinematicChain:
    """Chaine cinematique chargee depuis un URDF.

    Charge et met en cache la geometrie une seule fois, puis calcule la
    cinematique directe pour n'importe quelle configuration articulaire.
    """

    def __init__(self, urdf_path=DEFAULT_URDF, base_link=BASE_LINK, ee_link=EE_LINK):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF introuvable : {self.urdf_path}\n"
                "Recupere-le depuis TheRobotStudio/SO-ARM100 "
                "(Simulation/SO101/so101_new_calib.urdf)."
            )
        self.joints = _parse_joints(self.urdf_path)
        self.base_link = base_link
        self.ee_link = ee_link
        self.chain = _build_chain(self.joints, base_link, ee_link)
        # articulations actionnees de la chaine (non fixes), ordre base -> effecteur
        self.actuated = [j for j in self.chain if self.joints[j]["type"] != "fixed"]

    def fk(self, joint_angles_rad):
        """Cinematique directe : pose de l'effecteur dans le repere base.

        T_base_ee = produit, le long de la chaine, de :
            T_origin(articulation) @ Rot(axe, angle)   pour une articulation rotoide
            T_origin(articulation)                     pour un joint fixe

        Args:
            joint_angles_rad: dict {nom_articulation: angle_rad}. Doit contenir
                les 5 articulations du bras. Les cles supplementaires (ex:
                "gripper") sont ignorees.

        Returns:
            T_base_ee (4,4) : pose de l'effecteur dans le repere base (metres)

        Raises:
            KeyError: si un angle d'articulation actionnee est manquant.
        """
        T = np.eye(4)
        for jname in self.chain:
            joint = self.joints[jname]
            T = T @ joint["origin"]
            if joint["type"] != "fixed":
                if jname not in joint_angles_rad:
                    raise KeyError(f"Angle manquant pour l'articulation '{jname}'")
                T = T @ rotation_about_axis(joint["axis"], float(joint_angles_rad[jname]))
        return T


# Cache pour la chaine par defaut (evite de reparser l'URDF a chaque appel).
_DEFAULT_CHAIN = None


def fk_so101(joint_angles_rad, urdf_path=DEFAULT_URDF):
    """Cinematique directe du SO-101 (fonction de commodite).

    Args:
        joint_angles_rad: dict {nom_articulation: angle_rad}, voir KinematicChain.fk
        urdf_path: chemin de l'URDF (defaut: configs/so101_new_calib.urdf)

    Returns:
        T_base_gripper (4,4) en metres
    """
    global _DEFAULT_CHAIN
    if Path(urdf_path) == DEFAULT_URDF:
        if _DEFAULT_CHAIN is None:
            _DEFAULT_CHAIN = KinematicChain(urdf_path)
        return _DEFAULT_CHAIN.fk(joint_angles_rad)
    return KinematicChain(urdf_path).fk(joint_angles_rad)


# ============================================================
# Tests rapides (lance avec : python -m src.calibration.forward_kinematics)
# ============================================================
if __name__ == "__main__":
    import json

    print("Tests forward_kinematics.py")
    print()

    chain = KinematicChain()
    print(f"  URDF        : {chain.urdf_path.name}")
    print(f"  Chaine      : {chain.base_link} -> {chain.ee_link}")
    print(f"  Actionnees  : {chain.actuated}")
    assert chain.actuated == ARM_JOINTS, f"chaine inattendue : {chain.actuated}"
    print(f"  [OK] chaine cinematique conforme ({len(chain.actuated)} articulations)")
    print()

    # 1. FK a la configuration zero (toutes articulations au milieu de course)
    zero = {j: 0.0 for j in ARM_JOINTS}
    T0 = chain.fk(zero)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    assert np.allclose(R0 @ R0.T, np.eye(3), atol=1e-9), "R non orthonormale"
    assert abs(np.linalg.det(R0) - 1.0) < 1e-9, "det(R) != 1"
    dist0 = np.linalg.norm(t0)
    print(f"  Config zero -> effecteur ({t0[0] * 1000:.1f}, {t0[1] * 1000:.1f}, "
          f"{t0[2] * 1000:.1f}) mm, distance base = {dist0 * 1000:.1f} mm")
    assert 0.05 < dist0 < 0.6, f"distance effecteur implausible : {dist0:.3f} m"
    print(f"  [OK] FK config zero : SE(3) valide, echelle plausible")
    print()

    # 2. SE(3) valide sur des configurations aleatoires
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = {j: rng.uniform(-1.5, 1.5) for j in ARM_JOINTS}
        T = chain.fk(q)
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9
        assert np.linalg.norm(T[:3, 3]) < 0.6
    print(f"  [OK] FK valide sur 20 configurations aleatoires")
    print()

    # 3. Articulation manquante -> erreur explicite
    try:
        chain.fk({"shoulder_pan": 0.0})
        raise AssertionError("aurait du lever KeyError")
    except KeyError:
        print(f"  [OK] articulation manquante detectee (KeyError)")
    print()

    # 4. Integration avec motor_to_angle sur une vraie capture de calibration
    from src.calibration.motor_to_angle import load_motor_calibration, raw_to_radians

    repo_root = Path(__file__).resolve().parents[2]
    capture_path = repo_root / "configs" / "extrinsic_capture_cam_0.json"
    calib_path = repo_root / "configs" / "calibration_follower.json"
    if capture_path.exists() and calib_path.exists():
        calib = load_motor_calibration(calib_path)
        data = json.load(open(capture_path))
        first = data["captures"][0]
        raw = first["motor_positions_raw"]
        q = {j: raw_to_radians(raw[j], calib[j]) for j in ARM_JOINTS}
        T = chain.fk(q)
        t = T[:3, 3]
        angles_deg = ", ".join(f"{j}={np.rad2deg(a):.1f}" for j, a in q.items())
        print(f"  Capture 1 de extrinsic_capture_cam_0.json :")
        print(f"    angles (deg) : {angles_deg}")
        print(f"    effecteur    : ({t[0] * 1000:.1f}, {t[1] * 1000:.1f}, "
              f"{t[2] * 1000:.1f}) mm, distance base = {np.linalg.norm(t) * 1000:.1f} mm")
        assert 0.05 < np.linalg.norm(t) < 0.6, "pose effecteur implausible sur donnee reelle"
        print(f"  [OK] integration motor_to_angle -> FK coherente")
    else:
        print(f"  [SKIP] fichiers de capture absents")
    print()
    print("Tous les tests passent.")
