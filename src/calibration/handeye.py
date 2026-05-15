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


def symmetric_board_corrections(square_size_m, rows, cols):
    """Transformations a appliquer pour corriger les detections ambigues d'un
    damier symetrique.

    cv2.findChessboardCorners detecte les coins en ordre 'row-major' depuis le
    coin 'top-left' de l'image. Pour un damier ou rows == cols, ce 'top-left'
    image peut correspondre a 4 coins physiques differents -> solvePnP donne
    alors un repere damier tourne de 0°, 90°, 180° ou 270° (autour de l'axe
    normal du damier) ET decale a un autre coin du damier. Cette fonction
    retourne les 4 corrections T_Fk_F0 a appliquer a droite de T_target2cam
    pour ramener toutes les detections sur l'orientation canonique k=0.

    Pour rows != cols, seules 2 orientations sont compatibles (0° et 180°).
    """
    D_x = (cols - 1) * square_size_m
    D_y = (rows - 1) * square_size_m

    def mk(theta, t):
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1.0],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    if rows == cols:
        return [
            mk(0,           np.zeros(3)),
            mk(-np.pi / 2,  np.array([0,   D_y, 0])),
            mk(np.pi,       np.array([D_x, D_y, 0])),
            mk(np.pi / 2,   np.array([D_x, 0,   0])),
        ]
    return [
        mk(0,     np.zeros(3)),
        mk(np.pi, np.array([D_x, D_y, 0])),
    ]


def _rot_angle_deg(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def solve_eye_to_hand_robust(R_gripper2base, t_gripper2base,
                              R_target2cam, t_target2cam,
                              corrections,
                              method=cv2.CALIB_HAND_EYE_HORAUD,
                              max_iter=10,
                              drop_pct=10,
                              min_poses=20):
    """Solveur eye-to-hand robuste pour damier symetrique.

    Pipeline :
    1. Resolution initiale avec les donnees brutes (toutes les poses).
    2. Identification du cluster majoritaire de T_gripper_target (poses qui
       sont mutuellement coherentes, c.-a-d. detectees dans la meme
       orientation).
    3. Realignement : pour chaque pose, choisit la correction parmi
       `corrections` qui rapproche le plus T_gripper_target du cluster.
    4. Rejet iteratif d'outliers (les `drop_pct` % de poses les plus
       eloignees a chaque iteration, jusqu'a stabilisation ou plancher
       `min_poses`).

    Returns:
        T_base_cam (4x4), indices_used (list[int]), stats (dict)
    """
    N = len(R_gripper2base)
    T_g2b_list = [_to_se3(R, t) for R, t in zip(R_gripper2base, t_gripper2base)]
    T_t2c_list = [_to_se3(R, t) for R, t in zip(R_target2cam, t_target2cam)]

    # 1. Solve initial
    T_bc0 = solve_eye_to_hand(R_gripper2base, t_gripper2base,
                               R_target2cam, t_target2cam, method=method)
    T_gt0 = [_se3_inverse(T_g2b_list[i]) @ T_bc0 @ T_t2c_list[i] for i in range(N)]

    # 2. Cluster majoritaire de T_gripper_target (utilise comme reference de pose
    # pour le realignement)
    counts = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if _rot_angle_deg(T_gt0[i][:3, :3], T_gt0[j][:3, :3]) < 5.0:
                counts[i] += 1
    seed = int(np.argmax(counts))
    T_gt_ref = T_gt0[seed]

    # 3. Realignement de chaque pose : choisit la correction qui rapproche
    # T_gripper_target du seed.
    T_t2c_corr = []
    for i in range(N):
        best_score = float("inf")
        best_T = T_t2c_list[i]
        for C in corrections:
            T_try = T_t2c_list[i] @ C
            T_gt_try = _se3_inverse(T_g2b_list[i]) @ T_bc0 @ T_try
            r_d = _rot_angle_deg(T_gt_ref[:3, :3], T_gt_try[:3, :3])
            t_d = float(np.linalg.norm(T_gt_try[:3, 3] - T_gt_ref[:3, 3])) * 1000.0
            score = r_d + 0.1 * t_d  # poids translation reduit (le solve initial peut decaler)
            if score < best_score:
                best_score, best_T = score, T_try
        T_t2c_corr.append(best_T)

    # 4. Solve iteratif avec rejet d'outliers
    indices = list(range(N))
    T_bc = T_bc0
    t_dev = R_devs = None
    for _ in range(max_iter):
        R_g2b_sub = [T_g2b_list[i][:3, :3] for i in indices]
        t_g2b_sub = [T_g2b_list[i][:3, 3] for i in indices]
        R_t2c_sub = [T_t2c_corr[i][:3, :3] for i in indices]
        t_t2c_sub = [T_t2c_corr[i][:3, 3] for i in indices]
        T_bc = solve_eye_to_hand(R_g2b_sub, t_g2b_sub, R_t2c_sub, t_t2c_sub, method=method)
        T_gt = [_se3_inverse(T_g2b_list[i]) @ T_bc @ T_t2c_corr[i] for i in indices]
        t_arr = np.array([T[:3, 3] for T in T_gt])
        t_mean = t_arr.mean(axis=0)
        t_dev = np.linalg.norm(t_arr - t_mean, axis=1) * 1000.0
        idx_min = int(np.argmin(t_dev))
        R_ref = T_gt[idx_min][:3, :3]
        R_devs = np.array([_rot_angle_deg(R_ref, T[:3, :3]) for T in T_gt])
        combined = t_dev + 10.0 * R_devs
        threshold = np.percentile(combined, 100 - drop_pct)
        new_indices = [indices[i] for i in range(len(indices)) if combined[i] <= threshold]
        if len(new_indices) < min_poses or len(new_indices) == len(indices):
            break
        indices = new_indices

    stats = {
        "label": "T_gripper_target (apres alignement + rejet outliers)",
        "n_poses": len(indices),
        "n_poses_total": N,
        "translation_mean_dev_mm": float(t_dev.mean()),
        "translation_max_dev_mm": float(t_dev.max()),
        "translation_median_dev_mm": float(np.median(t_dev)),
        "rotation_mean_dev_deg": float(R_devs.mean()),
        "rotation_max_dev_deg": float(R_devs.max()),
        "rotation_median_dev_deg": float(np.median(R_devs)),
    }
    return T_bc, indices, stats
