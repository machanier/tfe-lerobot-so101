"""
transforms.py - Helpers pour manipuler les transformations rigides SE(3).

Une transformation SE(3) decrit la position + orientation d'un repere
par rapport a un autre. On la represente par une matrice 4x4 homogene :

    T = [R | t]      avec R = rotation 3x3, t = translation 3x1
        [0 | 1]

OpenCV travaille avec (rvec, tvec) ou rvec est un vecteur de Rodrigues
(axe * angle). On a donc besoin de conversions dans les deux sens.

Utilise par : src/calibration/forward_kinematics.py, scripts/solve_handeye.py
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

    print("Tous les tests passent.")
