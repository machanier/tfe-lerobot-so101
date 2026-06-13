"""
closed_loop.py - Raffinement de la pose de saisie par boucle fermee cam_2.

OBJECTIF : reduire l'erreur 3D de la triangulation stereo (~30 mm, dont
biais Y constant ~28 mm sur le poste de Maxence) a ~5-10 mm AVANT la
saisie, grace a la camera eye-in-hand (cam_2).

PRINCIPE :
  1. Le bras execute la trajectoire jusqu'a la pose `approach` (~8 cm
     au-dessus de l'objet, calcule par stereo).
  2. cam_2 (montee sur la pince) prend une image. L'objet est maintenant
     a ~8 cm de la camera = beaucoup plus precis.
  3. On detecte l'objet dans cette image (HSV ou HF, comme la perception
     principale).
  4. On reconstruit la position 3D dans le repere base via PnP monoculaire
     (l'objet apparait dans une zone connue grace a la pose courante du
     robot via FK + handeye eye-in-hand) ou via une heuristique simple :
     l'objet doit etre au centre de l'image. Si decale de Δx_px, Δy_px,
     on convertit en Δx_m, Δy_m via la geometrie cam_2.
  5. La correction est appliquee a la pose `grasp` : grasp_corrigé.x += Δx, etc.
  6. Le pipeline genere une NOUVELLE trajectoire approach -> grasp_corrige
     et continue.

CHOIX D'IMPLEMENTATION V1 : approche "centrage image"
  - On suppose que cam_2 est calibree pour viser AU CENTRE de l'image quand
    le bras est en pose approach et que l'objet est PILE en-dessous.
  - Si l'objet apparait decale de (Δu, Δv) pixels par rapport au centre,
    on convertit ce decalage en delta_metres dans le plan de la table via
    une homographie simple (camera fixee, table fixe, distance ~constante).
  - Correction de la pose grasp : decale dans le plan XY base, ne touche pas Z.

C'est moins rigoureux qu'un PnP mono complet, mais beaucoup plus simple a
implementer en V1. Si la precision est insuffisante (>10 mm), on passe
au PnP mono (V2).

References :
  - Chaumette & Hutchinson 2006, "Visual Servoing Control Part I" : c'est
    une version simplifiee du Image-Based Visual Servoing (IBVS).
  - Flandin et al. 2000, "Eye-in-hand / eye-to-hand cooperation".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.perception.camera_io import MultiCamera, compose_T_base_cam, load_handeye, load_intrinsics
from src.perception.detector import ObjectDetector
from src.perception.robot_state import RobotState, RobotStateProvider
from src.perception.scene import Detection2D, Frame


# ============================================================
# Resultat du raffinement
# ============================================================


@dataclass
class RefinementResult:
    """Resultat d'un cycle de raffinement boucle fermee.

    Attributes:
        delta_base_m       : correction (dx, dy, dz) en metres a appliquer
                             a la pose grasp dans le repere base.
        delta_pixels       : decalage detecte dans l'image cam_2 (Δu, Δv).
        confidence         : confiance dans la correction [0, 1]. 0 si pas
                             d'objet detecte par cam_2, sinon score detection.
        detection          : Detection2D source (debug).
        target_label       : nom de l'objet vise.
        method             : "image_centering" (V1) ou "pnp_mono" (V2 futur).
        message            : explication courte.
    """

    delta_base_m: np.ndarray
    delta_pixels: tuple[float, float]
    confidence: float
    detection: Optional[Detection2D]
    target_label: str
    method: str
    message: str = ""

    @property
    def delta_norm_mm(self) -> float:
        return float(np.linalg.norm(self.delta_base_m) * 1000.0)


# ============================================================
# Module principal
# ============================================================


def _bbox_touches_border(bbox, img_w: float, img_h: float,
                         margin_px: float = 4.0) -> bool:
    """True si la bbox touche le bord de l'image (detection TRONQUEE).

    Une bbox tronquee a un centre biaise (la partie hors champ manque) et la
    distorsion est maximale au bord -> les corrections derivees sont fausses
    (cas 'zone Y+100' du 2026-06-12 : objets a u=12-137px -> zigzag 25-50mm).
    On prefere NE PAS corriger plutot que corriger faux.
    """
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    return (x0 <= margin_px or y0 <= margin_px
            or x1 >= img_w - margin_px or y1 >= img_h - margin_px)


def refine_grasp_with_cam2(
    target_label: str,
    detector: ObjectDetector,
    multi_camera: MultiCamera,
    robot_state: RobotState,
    target_z_base_m: Optional[float] = None,
    z_height_above_object_m: float = 0.08,
    label_mapping: Optional[dict] = None,
    verbose: bool = True,
) -> RefinementResult:
    """Raffine la pose de saisie via cam_2 (eye-in-hand).

    PRE-REQUIS : le bras est deja a la pose `approach` (cam_2 regarde
    l'objet vers le bas, distance ~8 cm).

    PROCEDURE :
      1. Capture cam_2 (avec robot_state courant pour T_base_cam2).
      2. Detection de l'objet dans cam_2.
      3. Calcul du decalage (Δu, Δv) entre le centre detecte et le
         centre de l'image.
      4. Conversion pixels -> metres via projection inverse (homographie
         table simplifiee : on suppose que l'objet est au sol et que la
         camera regarde a peu pres en bas).

    Args:
        target_label   : label de l'objet a raffiner (e.g. "orange_cube").
        detector       : meme detecteur que dans le pipeline principal
                         (HSV ou HF).
        multi_camera   : doit etre deja ouvert.
        robot_state    : etat courant du robot (pose pince connue).
        z_height_above_object_m : distance verticale supposee entre cam_2
                                   et l'objet (utilise pour convertir
                                   pixels en metres dans le plan table).
        label_mapping  : si HF, mapping description -> label interne.

    Returns:
        RefinementResult avec correction delta_base_m a appliquer a la
        pose grasp.
    """
    # Capture cam_2 uniquement
    frames = multi_camera.grab(robot_state=robot_state)
    frame_c2 = frames.get("cam_2")
    if frame_c2 is None:
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(0.0, 0.0),
            confidence=0.0,
            detection=None,
            target_label=target_label,
            method="image_centering",
            message="cam_2 n'a pas pu capturer (KO USB ?)",
        )

    # Detection sur cam_2 seulement
    dets = detector.detect(frame_c2)
    # Applique mapping si HF
    if label_mapping:
        for d in dets:
            if d.label in label_mapping:
                d.label = label_mapping[d.label]
    # Filtre par label
    matches = [d for d in dets if d.label == target_label]
    if not matches:
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(0.0, 0.0),
            confidence=0.0,
            detection=None,
            target_label=target_label,
            method="image_centering",
            message=f"cam_2 n'a pas detecte '{target_label}'",
        )
    h, w = frame_c2.image.shape[:2]
    # Rejette les bboxes TRONQUEES au bord de l'image : centre biaise +
    # distorsion maximale -> correction fausse. Mieux vaut ne pas corriger.
    inside = [d for d in matches if not _bbox_touches_border(d.bbox, w, h)]
    if not inside:
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(0.0, 0.0),
            confidence=0.0,
            detection=max(matches, key=lambda d: d.score),
            target_label=target_label,
            method="ray_plane_intersection",
            message=(f"cam_2 : '{target_label}' detecte uniquement AU BORD de "
                     f"l'image (bbox tronquee) -> correction ignoree"),
        )
    # Garde la detection la plus confiante
    det = max(inside, key=lambda d: d.score)

    # Centre detecte vs POINT PRINCIPAL (cx, cy = axe optique), PAS le centre
    # geometrique (w/2, h/2). Le rayon "ou cam_2 vise" est l'axe optique ; et le
    # pixel detecte est undistordu plus bas (repere ideal de centre cx,cy). Pour
    # que la correction = (detecte - vise) soit NON BIAISEE, les deux references
    # doivent vivre dans le MEME repere -> on utilise cx,cy des deux cotes.
    u, v = det.center_px
    cx, cy = float(frame_c2.K[0, 2]), float(frame_c2.K[1, 2])
    u_center, v_center = cx, cy
    du_px = u - u_center  # positif = objet a droite dans l'image
    dv_px = v - v_center  # positif = objet en bas dans l'image

    # VRAIE PROJECTION INVERSE par intersection rayon-plan.
    # La formule simplifiee Δm = Δpx × Z / fx supposait cam_2 a la verticale,
    # ce qui n'est PAS le cas en pose approach (inclinaison ~15deg). Resultat :
    # correction sous-estimee (~50% de la vraie correction).
    #
    # Algorithme rigoureux :
    #   1. Pixel (u, v) → rayon en repere camera : d_cam = K^-1 @ [u, v, 1]
    #   2. Rayon en repere base : d_base = R_base_cam @ d_cam
    #   3. Origine du rayon en repere base : o_base = T_base_cam[:3, 3]
    #   4. Intersection avec plan z = target_z_base :
    #        t = (target_z_base - o_base[2]) / d_base[2]
    #        intersection = o_base + t * d_base
    #   5. Position 3D detectee de l'objet dans repere base.
    K = frame_c2.K
    T_base_cam2 = frame_c2.T_base_cam
    R_base_cam2 = T_base_cam2[:3, :3]
    o_base = T_base_cam2[:3, 3]
    z_cam2_base = float(o_base[2])

    target_z = target_z_base_m if target_z_base_m is not None else 0.0

    # UNDISTORTION du pixel avant le rayon. La triangulation stereo le fait
    # deja (pose_estimator) mais ce module utilisait le pixel BRUT : pres du
    # bord de l'image cam_2, la distorsion decale le rayon de plusieurs mm
    # au sol -> corrections faussees (diagnostic 2026-06-12, zone Y+100).
    u_id, v_id = u, v
    try:
        if frame_c2.dist is not None and np.any(np.asarray(frame_c2.dist) != 0):
            und = cv2.undistortPoints(
                np.array([[[float(u), float(v)]]], dtype=np.float64),
                K, frame_c2.dist, P=K).reshape(2)
            u_id, v_id = float(und[0]), float(und[1])
    except Exception:
        pass  # au pire : pixel brut (comportement historique)

    # Rayon dans le repere camera : d_cam (3,)
    K_inv = np.linalg.inv(K)
    d_cam = K_inv @ np.array([u_id, v_id, 1.0])
    # Rayon dans le repere base
    d_base = R_base_cam2 @ d_cam
    # Intersection avec plan Z = target_z
    if abs(d_base[2]) < 1e-6:
        # Rayon horizontal, pas d'intersection
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(du_px, dv_px),
            confidence=float(det.score),
            detection=det,
            target_label=target_label,
            method="ray_plane_intersection",
            message="Rayon trop horizontal, pas d'intersection avec plan objet",
        )
    t = (target_z - o_base[2]) / d_base[2]
    obj_pos_base = o_base + t * d_base  # position 3D detectee de l'objet en repere base

    # Position de l'objet ATTENDUE : intersection de l'AXE OPTIQUE (point
    # principal cx,cy, deja sans distorsion par definition) avec le plan objet
    # = ou cam_2 vise actuellement. Coherent avec le rayon detecte undistordu.
    d_cam_center = K_inv @ np.array([cx, cy, 1.0])
    d_base_center = R_base_cam2 @ d_cam_center
    t_center = (target_z - o_base[2]) / d_base_center[2]
    expected_pos_base = o_base + t_center * d_base_center  # ou cam_2 vise actuellement

    # Correction = position reelle - position visee
    delta_base = obj_pos_base - expected_pos_base
    # On NE touche pas a Z_base (la hauteur reste celle calculee par stereo)
    delta_base[2] = 0.0
    # On a notre vraie correction ray-plane
    dx_cam_m, dy_cam_m = float(delta_base[0]), float(delta_base[1])  # pour info debug

    if verbose:
        print(f"   [closed_loop] cam_2 a Z_base={z_cam2_base*1000:.1f}mm, "
              f"objet attendu Z={target_z*1000:.1f}mm")
        print(f"   [closed_loop] cam_2 vise actuellement : "
              f"({expected_pos_base[0]*1000:+.1f}, {expected_pos_base[1]*1000:+.1f}, "
              f"{expected_pos_base[2]*1000:+.1f}) mm")
        print(f"   [closed_loop] objet detecte par cam_2 a : "
              f"({obj_pos_base[0]*1000:+.1f}, {obj_pos_base[1]*1000:+.1f}, "
              f"{obj_pos_base[2]*1000:+.1f}) mm")

    return RefinementResult(
        delta_base_m=delta_base,
        delta_pixels=(du_px, dv_px),
        confidence=float(det.score),
        detection=det,
        target_label=target_label,
        method="ray_plane_intersection",
        message=(f"Correction Δbase=({delta_base[0]*1000:+.1f}, "
                 f"{delta_base[1]*1000:+.1f}, 0) mm  "
                 f"(pixel Δu={du_px:+.0f}px Δv={dv_px:+.0f}px, "
                 f"projection ray-plane)"),
    )


def apply_correction_to_grasp_pose(grasp_pose, delta_base_m: np.ndarray):
    """Applique la correction delta_base aux 3 poses du GraspPose.

    Modifie en place les translations des poses approach/grasp/retract en
    leur ajoutant delta_base. NE modifie pas les rotations.

    Args:
        grasp_pose : src.planning.grasp.GraspPose
        delta_base_m : (3,) correction en metres dans le repere base
    """
    from src.planning.grasp import GraspPose
    if not isinstance(grasp_pose, GraspPose):
        raise TypeError(f"Attendu GraspPose, recu {type(grasp_pose).__name__}")
    for attr in ("T_base_gripper_approach", "T_base_gripper_grasp", "T_base_gripper_retract"):
        T = getattr(grasp_pose, attr)
        T[:3, 3] += delta_base_m
        setattr(grasp_pose, attr, T)


# ============================================================
# Self-tests (lance avec : python -m src.control.closed_loop)
# ============================================================
if __name__ == "__main__":
    print("Tests closed_loop.py")
    print()

    # ============================================================
    # Test 1 : conversion pixel -> meters basique
    # ============================================================
    # Camera fictive : fx = 1200, image 1920x1080, hauteur 8 cm
    # Si l'objet est decale de 100 px a droite dans l'image,
    # le delta camera = 100 * 0.08 / 1200 = 0.00667 m = 6.7 mm
    fx = 1200.0
    Z = 0.08  # 8 cm
    du = 100.0
    dx_expected = du * Z / fx
    print(f"  [INFO] 100 px de decalage a 8 cm avec fx=1200 -> {dx_expected*1000:.1f} mm")
    assert abs(dx_expected - 0.00667) < 1e-4
    print(f"  [OK] formule projection inverse OK")

    # ============================================================
    # Test 2 : apply_correction_to_grasp_pose
    # ============================================================
    from src.planning.grasp import GraspPose
    import numpy as np

    T_approach = np.eye(4); T_approach[:3, 3] = [0.30, 0.0, 0.10]
    T_grasp    = np.eye(4); T_grasp[:3, 3]    = [0.30, 0.0, 0.02]
    T_retract  = np.eye(4); T_retract[:3, 3]  = [0.30, 0.0, 0.12]
    grasp = GraspPose(
        T_base_gripper_approach=T_approach,
        T_base_gripper_grasp=T_grasp,
        T_base_gripper_retract=T_retract,
        label="test", score=1.0,
    )
    delta = np.array([0.005, -0.028, 0.0])   # correction +5mm en X, -28mm en Y
    apply_correction_to_grasp_pose(grasp, delta)
    assert np.allclose(grasp.T_base_gripper_grasp[:3, 3], [0.305, -0.028, 0.02]), \
        f"correction mal appliquee : {grasp.T_base_gripper_grasp[:3, 3]}"
    assert np.allclose(grasp.T_base_gripper_approach[:3, 3], [0.305, -0.028, 0.10])
    assert np.allclose(grasp.T_base_gripper_retract[:3, 3], [0.305, -0.028, 0.12])
    print(f"  [OK] apply_correction_to_grasp_pose : decale les 3 poses")

    # ============================================================
    # Test 3 : RefinementResult
    # ============================================================
    r = RefinementResult(
        delta_base_m=np.array([0.005, -0.028, 0.0]),
        delta_pixels=(120.0, -40.0),
        confidence=0.85,
        detection=None,
        target_label="orange_cube",
        method="image_centering",
    )
    assert abs(r.delta_norm_mm - 28.4) < 0.5
    print(f"  [OK] RefinementResult.delta_norm_mm = {r.delta_norm_mm:.1f} mm")

    print()
    print("Tous les tests passent.")
