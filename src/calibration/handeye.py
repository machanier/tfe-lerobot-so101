"""
handeye.py - Calibration hand-eye via cv2.calibrateHandEye.

Deux configurations :

  - eye-to-hand : la camera est fixe dans le monde, le damier est rigidement
    colle sur la pince. Resoudre donne `T_base_cam`, la pose de la camera dans
    le repere base du robot. C'est le cas de cam_0 et cam_1.

  - eye-in-hand : la camera est sur la pince, le damier est fixe dans le monde.
    Resoudre donne `T_gripper_cam`, la pose de la camera dans le repere pince.
    A composer ensuite avec la FK pour obtenir `T_base_cam(t)`. C'est le cas
    de cam_2.

Convention de noms : T_A_B = pose du repere B exprime dans A, ou de facon
equivalente la matrice qui transforme un point exprime en B vers A. Quand on
parle de cv2.calibrateHandEye, on garde la convention OpenCV "FROM2TO" pour
les noms d'argument (R_target2cam, etc.), mais on documente l'equivalence.

Astuce eye-to-hand : cv2.calibrateHandEye est concue pour eye-in-hand. Pour
l'eye-to-hand on inverse les poses gripper->base avant d'appeler (on substitue
la base au role du gripper). La sortie "R_cam2gripper" est alors R_cam2base.
Voir Tsai & Lenz 1989, Horaud & Dornaika 1995.
"""

from __future__ import annotations

import numpy as np

# OpenCV : meme convention partout (units quelconques tant qu'elles sont
# coherentes entre gripper2base et target2cam).
import cv2

METHODS = {
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
}


def _invert_rt(R_list, t_list):
    """Inverse une liste de poses : (R, t) -> (R.T, -R.T @ t)."""
    R_inv = [R.T for R in R_list]
    t_inv = [-R.T @ np.asarray(t).reshape(3) for R, t in zip(R_list, t_list)]
    return R_inv, t_inv


def _to_se3(R, t):
    """Construit une matrice homogene 4x4."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _se3_inverse(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def solve_eye_to_hand(R_gripper2base, t_gripper2base,
                       R_target2cam, t_target2cam,
                       method=cv2.CALIB_HAND_EYE_PARK):
    """Calibration eye-to-hand. Renvoie T_base_cam (4x4).

    Args:
        R_gripper2base: liste de R (3x3) - rotation gripper->base, une par pose
        t_gripper2base: liste de t (3,) ou (3,1) - en unite coherente avec t_target2cam
        R_target2cam, t_target2cam: idem pour la pose damier->camera
        method: une valeur de cv2.CALIB_HAND_EYE_*

    Returns:
        T_base_cam (4x4) : pose de la camera dans le repere base
    """
    # On inverse gripper2base : la sortie "R_cam2gripper" devient R_cam2base.
    R_b2g, t_b2g = _invert_rt(R_gripper2base, t_gripper2base)
    R_c2b, t_c2b = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_target2cam, t_target2cam, method=method
    )
    return _to_se3(R_c2b, t_c2b)


def solve_eye_in_hand(R_gripper2base, t_gripper2base,
                       R_target2cam, t_target2cam,
                       method=cv2.CALIB_HAND_EYE_PARK):
    """Calibration eye-in-hand. Renvoie T_gripper_cam (4x4)."""
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method
    )
    return _to_se3(R_c2g, t_c2g)


def _residual_stats(T_list, label):
    """Statistiques de variance d'une liste de SE(3) qui devrait etre constante.

    Returns: dict avec mean/max/median deviations en mm et deg.
    """
    t_arr = np.array([T[:3, 3] for T in T_list])  # (N, 3)
    t_mean = t_arr.mean(axis=0)
    t_dev_m = np.linalg.norm(t_arr - t_mean, axis=1)
    t_dev_mm = t_dev_m * 1000.0

    # Pour la rotation : prend la pose dont la translation est la plus proche
    # de la moyenne comme reference, et mesure l'angle entre R_i et R_ref.
    idx_ref = int(np.argmin(t_dev_m))
    R_ref = T_list[idx_ref][:3, :3]
    R_devs = []
    for T in T_list:
        R_rel = R_ref.T @ T[:3, :3]
        cos = (np.trace(R_rel) - 1.0) / 2.0
        R_devs.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    R_devs = np.array(R_devs)

    return {
        "label": label,
        "n_poses": len(T_list),
        "translation_mean_dev_mm": float(t_dev_mm.mean()),
        "translation_max_dev_mm": float(t_dev_mm.max()),
        "translation_median_dev_mm": float(np.median(t_dev_mm)),
        "rotation_mean_dev_deg": float(R_devs.mean()),
        "rotation_max_dev_deg": float(R_devs.max()),
        "rotation_median_dev_deg": float(np.median(R_devs)),
    }


def residuals_eye_to_hand(T_g2b_list, T_t2c_list, T_base_cam):
    """Calcule les T_gripper_target par pose et leur dispersion.

    Si la calibration est correcte, ces transformations doivent etre
    quasi-constantes (le damier est rigidement attache a la pince).
    """
    T_gt_list = []
    for T_g2b, T_t2c in zip(T_g2b_list, T_t2c_list):
        T_gt = _se3_inverse(T_g2b) @ T_base_cam @ T_t2c
        T_gt_list.append(T_gt)
    return _residual_stats(T_gt_list, "T_gripper_target")


def residuals_eye_in_hand(T_g2b_list, T_t2c_list, T_gripper_cam):
    """Calcule les T_base_target par pose et leur dispersion.

    Si la calibration est correcte, ces transformations doivent etre
    quasi-constantes (le damier est fixe dans le monde).
    """
    T_bt_list = []
    for T_g2b, T_t2c in zip(T_g2b_list, T_t2c_list):
        T_bt = T_g2b @ T_gripper_cam @ T_t2c
        T_bt_list.append(T_bt)
    return _residual_stats(T_bt_list, "T_base_target")
