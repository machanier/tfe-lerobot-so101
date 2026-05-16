"""
test_pipeline.py - Test d'integration du pipeline perception complet, sans hardware.

Cree des images synthetiques (cubes colores projetes par deux cameras calibrees
artificiellement), fait passer le pipeline detection -> triangulation, et
verifie que la position 3D estimee est proche du ground truth synthetique.

Sert de filet de securite pour les refacto futures.

Lance via :
    python -m pytest tests/perception -v
ou directement :
    python tests/perception/test_pipeline.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.perception.detector import HSVDetector, ObjectSpec, HSVRange
from src.perception.pose_estimator import PoseEstimator, _projection_matrix
from src.perception.scene import Detection2D, Frame


def make_camera_pair(baseline_y_m=0.1, height_z_m=0.3):
    """Deux cameras eye-to-hand au-dessus du plan de travail, axe Z vers le bas."""
    K = np.array([[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]])
    dist = np.zeros(5)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])  # Z_cam = -Z_base
    T_L = np.eye(4); T_L[:3, :3] = R; T_L[:3, 3] = [0.10, +baseline_y_m / 2, height_z_m]
    T_R = np.eye(4); T_R[:3, :3] = R; T_R[:3, 3] = [0.10, -baseline_y_m / 2, height_z_m]
    return K, dist, T_L, T_R


def project_3d_to_pixel(X_base, K, T_base_cam):
    P = _projection_matrix(K, T_base_cam)
    uvw = P @ np.hstack([X_base, 1.0])
    return uvw[:2] / uvw[2]


def render_disk(img, center_uv, radius_px, color_bgr):
    cv2.circle(img, (int(center_uv[0]), int(center_uv[1])), int(radius_px),
               color_bgr, thickness=-1)


def make_red_cube_spec():
    """Spec rouge pour le test."""
    return ObjectSpec(
        label="red_cube",
        hsv=HSVRange(h_lo=0, h_hi=10, s_lo=100, s_hi=255, v_lo=60, v_hi=255,
                     hue_extra_lo=170, hue_extra_hi=179),
        min_area_px=200.0,
        max_area_px=200000.0,
    )


def test_red_cube_triangulated_under_5mm():
    """Avec une calibration parfaite, le cube doit etre triangule a < 5 mm
    d'erreur. Sub-mm sur le synthetique."""
    K, dist, T_L, T_R = make_camera_pair()
    X_true = np.array([0.12, 0.01, 0.04])

    uv_L = project_3d_to_pixel(X_true, K, T_L)
    uv_R = project_3d_to_pixel(X_true, K, T_R)

    img_L = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    img_R = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    render_disk(img_L, uv_L, radius_px=22, color_bgr=(0, 0, 220))
    render_disk(img_R, uv_R, radius_px=22, color_bgr=(0, 0, 220))

    frame_L = Frame(cam_key="cam_0", image=img_L, K=K, dist=dist, T_base_cam=T_L)
    frame_R = Frame(cam_key="cam_1", image=img_R, K=K, dist=dist, T_base_cam=T_R)

    det = HSVDetector([make_red_cube_spec()])
    dets_by_cam = det.detect_multi({"cam_0": frame_L, "cam_1": frame_R})
    assert dets_by_cam["cam_0"], "Detection L vide"
    assert dets_by_cam["cam_1"], "Detection R vide"

    estimator = PoseEstimator(load_scene_config=False)
    scene = estimator.build_scene(dets_by_cam, {"cam_0": frame_L, "cam_1": frame_R})
    assert len(scene.objects) == 1, f"Attendu 1 objet, recu {len(scene.objects)}"
    obj = scene.objects[0]
    err_mm = float(np.linalg.norm(obj.position_base_m - X_true) * 1000)
    assert err_mm < 5.0, f"Erreur {err_mm:.2f} mm > 5 mm"
    assert obj.meta["method"] == "stereo_triangulation"


def test_workspace_filter_rejects_aberrant_solutions():
    """Une detection coherente mais qui triangulerait hors workspace doit etre rejetee."""
    K, dist, T_L, T_R = make_camera_pair()
    # Place le cube tres loin (z = 0.80 m, au-dela du workspace SO-101)
    X_far = np.array([0.20, 0.0, 0.80])
    uv_L = project_3d_to_pixel(X_far, K, T_L)
    uv_R = project_3d_to_pixel(X_far, K, T_R)
    img_L = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    img_R = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    render_disk(img_L, uv_L, 22, (0, 0, 220))
    render_disk(img_R, uv_R, 22, (0, 0, 220))
    frame_L = Frame(cam_key="cam_0", image=img_L, K=K, dist=dist, T_base_cam=T_L)
    frame_R = Frame(cam_key="cam_1", image=img_R, K=K, dist=dist, T_base_cam=T_R)
    det = HSVDetector([make_red_cube_spec()])
    estimator = PoseEstimator(load_scene_config=False)
    scene = estimator.build_scene(det.detect_multi({"cam_0": frame_L, "cam_1": frame_R}),
                                   {"cam_0": frame_L, "cam_1": frame_R})
    assert len(scene.objects) == 0, "Position hors workspace devrait etre filtree"


def test_single_camera_no_stereo_no_pnp():
    """Detection sur 1 seule cam, sans contour exploitable -> aucune estimation."""
    K, dist, T_L, _ = make_camera_pair()
    frame_L = Frame(cam_key="cam_0", image=np.full((400, 600, 3), 80, dtype=np.uint8),
                    K=K, dist=dist, T_base_cam=T_L)
    # Detection bidon avec center_px mais SANS contour (pas de PnP possible)
    det = Detection2D(cam_key="cam_0", label="red_cube",
                      center_px=(300.0, 200.0), contour=None)
    estimator = PoseEstimator(load_scene_config=False)
    scene = estimator.build_scene({"cam_0": [det]}, {"cam_0": frame_L})
    assert len(scene.objects) == 0


def test_reprojection_error_is_used_to_reject():
    """Si on truque la detection R pour qu'elle soit incoherente, l'erreur de
    reprojection doit faire rejeter l'objet."""
    K, dist, T_L, T_R = make_camera_pair()
    X_true = np.array([0.12, 0.01, 0.04])
    uv_L = project_3d_to_pixel(X_true, K, T_L)
    # uv_R volontairement decale de 50 px (incoherence stereo)
    uv_R = project_3d_to_pixel(X_true, K, T_R) + np.array([50.0, 50.0])
    img_L = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    img_R = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    render_disk(img_L, uv_L, 22, (0, 0, 220))
    render_disk(img_R, uv_R, 22, (0, 0, 220))
    frame_L = Frame(cam_key="cam_0", image=img_L, K=K, dist=dist, T_base_cam=T_L)
    frame_R = Frame(cam_key="cam_1", image=img_R, K=K, dist=dist, T_base_cam=T_R)
    det = HSVDetector([make_red_cube_spec()])
    # max_reproj_error_px tres strict pour forcer le rejet
    from src.perception.pose_estimator import PoseEstimatorConfig
    estimator = PoseEstimator(PoseEstimatorConfig(max_reproj_error_px=5.0),
                              load_scene_config=False)
    scene = estimator.build_scene(det.detect_multi({"cam_0": frame_L, "cam_1": frame_R}),
                                   {"cam_0": frame_L, "cam_1": frame_R})
    assert len(scene.objects) == 0, "Erreur reprojection forte devrait rejeter"


if __name__ == "__main__":
    # Run tests directement (pas besoin de pytest)
    tests = [
        test_red_cube_triangulated_under_5mm,
        test_workspace_filter_rejects_aberrant_solutions,
        test_single_camera_no_stereo_no_pnp,
        test_reprojection_error_is_used_to_reject,
    ]
    for t in tests:
        try:
            t()
            print(f"  [OK] {t.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            raise
    print("Tous les tests d'integration passent.")
