"""Utilitaires de manipulation des transformations rigides SE(3).

Une transformation SE(3) decrit la position et l'orientation d'un repere par
rapport a un autre. Elle est representee par une matrice 4x4 homogene :

    T = [R | t]      avec R = rotation 3x3, t = translation 3x1
        [0 | 1]

OpenCV travaille avec (rvec, tvec), ou rvec est un vecteur de Rodrigues
(axe multiplie par l'angle) ; des conversions dans les deux sens sont donc
fournies.

Utilise par la cinematique directe et les scripts de calibration main-oeil.
"""

import numpy as np
import cv2


def rvec_tvec_to_matrix(rvec, tvec):
    """Convertit (rvec, tvec) OpenCV en matrice 4x4 homogene.

    Args:
        rvec: vecteur de Rodrigues (3,) ou (3,1)
        tvec: vecteur translation (3,) ou (3,1) en mm

    Returns:
        T (4,4) : matrice homogene
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_rvec_tvec(T):
    """Convertit une matrice 4x4 en (rvec, tvec) OpenCV.

    Args:
        T (4,4) : matrice homogene

    Returns:
        rvec (3,), tvec (3,)
    """
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.flatten(), T[:3, 3].copy()


def se3_inverse(T):
    """Inverse d'une transformation SE(3).

    Si T = [R | t; 0 | 1], alors T^-1 = [R^T | -R^T @ t; 0 | 1].

    Args:
        T (4,4) : matrice homogene

    Returns:
        T_inv (4,4)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_compose(*Ts):
    """Compose plusieurs transformations : T = T1 @ T2 @ ... @ Tn."""
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def split_R_t(T):
    """Decoupe une matrice 4x4 en (R, t)."""
    return T[:3, :3].copy(), T[:3, 3].copy()


def rpy_to_matrix(rpy):
    """Convertit des angles roll-pitch-yaw (convention URDF) en rotation 3x3.

    URDF utilise la convention "fixed-axis XYZ" : rotation de `roll` autour de
    X, puis `pitch` autour de Y, puis `yaw` autour de Z, autour des axes FIXES.
    Cela equivaut a R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Args:
        rpy: (roll, pitch, yaw) en radians

    Returns:
        R (3,3)
    """
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def xyz_rpy_to_matrix(xyz, rpy):
    """Matrice 4x4 a partir d'une translation et d'angles rpy (convention URDF).

    Correspond a la balise URDF <origin xyz=... rpy=...> : la pose du repere
    enfant dans le repere parent, T = [R(rpy) | xyz ; 0 | 1].

    Args:
        xyz: translation (3,)
        rpy: (roll, pitch, yaw) en radians

    Returns:
        T (4,4)
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


def rotation_about_axis(axis, angle):
    """Matrice 4x4 d'une rotation d'`angle` radians autour d'un axe donne.

    Sert au mouvement d'une articulation rotoide : `axis` est l'axe de
    l'articulation (balise URDF <axis>), exprime dans son propre repere.

    Args:
        axis: axe de rotation (3,), normalise automatiquement
        angle: angle en radians

    Returns:
        T (4,4) : rotation pure (translation nulle)
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4)
    R, _ = cv2.Rodrigues(axis / norm * float(angle))
    T = np.eye(4)
    T[:3, :3] = R
    return T


# ============================================================
# Tests rapides (lance avec : python -m src.utils.transforms)
# ============================================================
if __name__ == "__main__":
    print("Tests transforms.py")

    # 1. Round-trip rvec/tvec <-> matrix
    rvec_in = np.array([0.1, -0.5, 0.3])
    tvec_in = np.array([100.0, -50.0, 200.0])
    T = rvec_tvec_to_matrix(rvec_in, tvec_in)
    rvec_out, tvec_out = matrix_to_rvec_tvec(T)
    assert np.allclose(rvec_in, rvec_out, atol=1e-9), "rvec round-trip casse"
    assert np.allclose(tvec_in, tvec_out, atol=1e-9), "tvec round-trip casse"
    print(f"  [OK] rvec/tvec <-> matrix round-trip")

    # 2. Inverse
    T_inv = se3_inverse(T)
    I = se3_compose(T, T_inv)
    assert np.allclose(I, np.eye(4), atol=1e-9), "inverse casse"
    print(f"  [OK] T @ T^-1 = I")

    # 3. Composition
    T2 = rvec_tvec_to_matrix([0.0, 0.0, np.pi / 2], [10.0, 0.0, 0.0])
    T3 = se3_compose(T, T2)
    assert T3.shape == (4, 4)
    print(f"  [OK] composition")

    # 4. rpy : rotation nulle, et coherence avec Rodrigues sur l'axe Z
    assert np.allclose(rpy_to_matrix([0, 0, 0]), np.eye(3)), "rpy zero casse"
    Rz_rpy = rpy_to_matrix([0.0, 0.0, 0.7])
    Rz_rod, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.7]))
    assert np.allclose(Rz_rpy, Rz_rod, atol=1e-9), "rpy yaw != rotation Z"
    print(f"  [OK] rpy_to_matrix")

    # 5. rotation_about_axis : equivaut a rpy pour l'axe Z, axe nul = identite
    Tz = rotation_about_axis([0, 0, 1], 0.7)
    assert np.allclose(Tz[:3, :3], Rz_rod, atol=1e-9), "rotation_about_axis Z casse"
    assert np.allclose(rotation_about_axis([0, 0, 0], 1.0), np.eye(4)), "axe nul casse"
    print(f"  [OK] rotation_about_axis")

    # 6. xyz_rpy_to_matrix : compose bien R et t
    T4 = xyz_rpy_to_matrix([1.0, 2.0, 3.0], [0.0, 0.0, 0.7])
    assert np.allclose(T4[:3, :3], Rz_rod, atol=1e-9)
    assert np.allclose(T4[:3, 3], [1.0, 2.0, 3.0])
    print(f"  [OK] xyz_rpy_to_matrix")

    print("Tous les tests passent.")
