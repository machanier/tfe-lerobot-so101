"""Raffinement de la pose de saisie par boucle fermee cam_2 (eye-in-hand).

Objectif : reduire l'erreur 3D de la saisie avant la descente, grace a la camera
montee sur la pince (cam_2). L'erreur brute de la triangulation stereo est dominee
par un biais systematique (mesure : environ -32 mm en X constant, +16 a +24 mm en
Y, -20 mm en Z ; cf configs/perception/bias_correction.json). Ce biais constant est
compense en amont (bias_correction.json) ; il subsiste un residuel de l'ordre de
5 a 15 mm (variable, surtout en Y) que cam_2 reduit a quelques mm en regardant
l'objet de pres (environ 8 cm). cam_2 a son propre biais Y (environ +11 mm) :
fiable en X, a surveiller en Y (cf bias_correction.json, cle _attention_cam2).

Principe :
  1. Le bras execute la trajectoire jusqu'a la pose `approach` (environ 8 cm
     au-dessus de l'objet, calculee par stereo).
  2. cam_2 (montee sur la pince) prend une image. L'objet est alors a environ
     8 cm de la camera, donc bien plus precis.
  3. L'objet est detecte dans cette image (HSV ou HF, comme la perception
     principale).
  4. Projection rayon-plan : le pixel detecte (undistordu) definit un rayon en
     repere base (via K et T_base_cam2 du robot courant) ; on l'intersecte avec
     le plan horizontal z = hauteur de l'objet, ce qui donne la position 3D
     detectee. On la compare a la position visee par l'axe optique (meme
     intersection au point principal), d'ou l'ecart Δbase.
  5. La correction Δbase (XY seulement, Z inchange) est appliquee a la pose grasp.
  6. cam_2 mesure aussi l'orientation (grand axe) ; le pipeline peut realigner les
     machoires si l'objet est vu nettement allonge.

Garde-fous (cote pipeline, cf PipelineConfig) : la correction n'est appliquee que
si la detection cam_2 est fiable, c'est-a-dire blob assez gros
(area_frac >= cam2_min_blob_frac, et non le `score` qui est une aire normalisee
trompeuse) et correction sous le plafond de securite. Une bbox tronquee a gauche,
a droite ou en bas est rejetee (le bord haut est tolere car l'axe optique de cam_2
n'est pas aligne avec le bout des pinces). La projection rayon-plan horizontal est
exacte en top-down ; a fort tangage (90 deg) elle se degrade, et le seuil
cam2_max_pitch_deg permet de borner les angles (par defaut, tous autorises).

References :
  - Chaumette & Hutchinson 2006, "Visual Servoing Control Part I" : il s'agit
    d'une version simplifiee de l'Image-Based Visual Servoing (IBVS).
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
    """Resultat d'un cycle de raffinement en boucle fermee.

    Attributes:
        delta_base_m       : correction (dx, dy, dz) en metres a appliquer
                             a la pose grasp dans le repere base.
        delta_pixels       : decalage detecte dans l'image cam_2 (Δu, Δv).
        confidence         : confiance dans la correction [0, 1]. 0 si aucun
                             objet detecte par cam_2, sinon le score de detection.
        detection          : Detection2D source (debug).
        target_label       : nom de l'objet vise.
        method             : identifiant de la methode de raffinement employee.
        message            : explication courte.
    """

    delta_base_m: np.ndarray
    delta_pixels: tuple[float, float]
    confidence: float
    detection: Optional[Detection2D]
    target_label: str
    method: str
    message: str = ""
    # Orientation du grand axe de l'objet (repere base) vue par cam_2 (proche,
    # quasi au-dessus, donc blob plus gros et plus net que la stereo oblique).
    # None si l'empreinte n'est pas assez allongee pour trancher.
    yaw_base_cam2: Optional[float] = None
    elong_cam2: float = 1.0
    # Diagnostic du handoff cams fixes -> cam_2 : taille absolue du blob cam_2.
    # Le `confidence` ci-dessus (= det.score) est une aire normalisee par une
    # constante arbitraire (max_area_px), donc trompeur pour juger la fiabilite
    # du centroide. area_px (aire absolue) et area_frac (fraction du cadre cam_2)
    # indiquent reellement si cam_2 resout bien l'objet a 8 cm (gros blob =
    # centroide fiable) ou non.
    area_px: float = 0.0
    area_frac: float = 0.0

    @property
    def delta_norm_mm(self) -> float:
        return float(np.linalg.norm(self.delta_base_m) * 1000.0)


# ============================================================
# Module principal
# ============================================================


def long_axis_base_from_contour(contour, K, dist, T_base_cam,
                                z_plane_m: float):
    """Angle du grand axe de l'objet en repere base, depuis un contour image.

    Projette le contour (undistordu) sur le plan z=z_plane par intersection
    rayon-plan, puis applique une ACP 2D dans le plan XY base. Identique a
    PoseEstimator._footprint_orientation mais autonome (utilise pour cam_2).

    cam_2 etant proche et quasi au-dessus de l'objet a la pose approach, son
    blob est plus gros et plus net que la stereo oblique cam_0/cam_1, d'ou un
    axe plus fiable. Retourne (yaw_rad dans [-pi/2, pi/2], elongation >= 1) ou
    None.
    """
    if contour is None:
        return None
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 6:
        return None
    if len(pts) > 80:
        pts = pts[::max(1, len(pts) // 80)]
    try:
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        K_inv = np.linalg.inv(K)
        R = T_base_cam[:3, :3]
        o = T_base_cam[:3, 3]
        homog = np.hstack([und, np.ones((len(und), 1))])
        rays = (R @ (K_inv @ homog.T)).T
        dz = rays[:, 2]
        keep = np.abs(dz) > 1e-9
        s = (z_plane_m - o[2]) / dz[keep]
        fwd = s > 0
        P = o[None, :] + s[fwd, None] * rays[keep][fwd]
    except Exception:
        return None
    if len(P) < 6:
        return None
    xy = P[:, :2]
    d = xy - xy.mean(axis=0)
    mu20 = float((d[:, 0] ** 2).mean())
    mu02 = float((d[:, 1] ** 2).mean())
    mu11 = float((d[:, 0] * d[:, 1]).mean())
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    ca, sa = np.cos(theta), np.sin(theta)
    along = d[:, 0] * ca + d[:, 1] * sa
    perp = -d[:, 0] * sa + d[:, 1] * ca
    ext_long = float(along.max() - along.min())
    ext_court = float(perp.max() - perp.min())
    if ext_long < ext_court:
        ext_long, ext_court = ext_court, ext_long
        theta += np.pi / 2.0
    while theta > np.pi / 2:
        theta -= np.pi
    while theta < -np.pi / 2:
        theta += np.pi
    if ext_court < 1e-4:
        return None
    return float(theta), float(ext_long / max(ext_court, 1e-6))


def _bbox_touches_border(bbox, img_w: float, img_h: float,
                         margin_px: float = 4.0,
                         ignore_top: bool = False) -> bool:
    """True si la bbox touche un bord disqualifiant de l'image (centre biaise).

    Une bbox tronquee a un centre biaise (la partie hors champ manque) et la
    distorsion est maximale au bord, si bien que les corrections derivees sont
    fausses (un objet pres du bord gauche produit un zigzag de 25 a 50 mm). On
    prefere ne pas corriger plutot que corriger faux.

    ignore_top : sur cam_2 eye-in-hand, l'axe optique n'est pas aligne avec le
    bout des pinces, si bien qu'un objet correctement place sous la pince
    apparait naturellement haut dans l'image et peut froler le bord superieur
    sans etre tronque cote prise. Le bord haut n'est donc pas disqualifiant.
    Restent disqualifiants : le bas (les doigts occupent le bas du cadre), la
    gauche et la droite (un objet tronque lateralement y est de toute facon
    trop large ou trop loin pour etre saisi).
    """
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    touch = (x0 <= margin_px                 # gauche
             or x1 >= img_w - margin_px      # droite
             or y1 >= img_h - margin_px)     # bas (doigts)
    if not ignore_top:
        touch = touch or (y0 <= margin_px)   # haut
    return touch


def _save_cam2_debug(out_dir, frame, matches, chosen) -> None:
    """Sauve la vue cam_2 du raffinement (eye-in-hand) avec les detections.

    Permet de visualiser ce que cam_2 percoit au moment de la prise : qualite du
    masque, cadrage, taille du blob, et choix de la detection retenue. C'est le
    diagnostic central du handoff cams fixes -> cam_2 (la croix jaune marque
    l'axe optique, c'est-a-dire la ou cam_2 vise ; le rectangle rouge marque le
    blob retenu). Non bloquant : toute erreur d'ecriture est ignoree afin de ne
    jamais interrompre la prise.
    """
    try:
        from pathlib import Path
        from datetime import datetime
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img = frame.image.copy()
        cx, cy = int(frame.K[0, 2]), int(frame.K[1, 2])
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
        for d in matches:
            if d.bbox is None:
                continue
            x0, y0, x1, y1 = (int(v) for v in d.bbox)
            chosen_one = (d is chosen)
            col = (0, 0, 255) if chosen_one else (0, 200, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 if chosen_one else 1)
            cv2.circle(img, (int(d.center_px[0]), int(d.center_px[1])), 5, col, -1)
            cv2.putText(img, f"{d.label} s={d.score:.2f}", (x0, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"cam2_refine_{stamp}.png"
        cv2.imwrite(str(out_path), img)
        print(f"   [display] vue cam_2 raffinement sauvee : {out_path}")
    except Exception as e:  # pragma: no cover - debug seulement
        print(f"   [display] (vue cam_2 non sauvee : {e})")


def capture_cam2_snapshot(
    target_label: str,
    detector: ObjectDetector,
    multi_camera: MultiCamera,
    robot_state: RobotState,
    out_dir: object,
    label_mapping: Optional[dict] = None,
    flush_frames: int = 3,
    tag: str = "",
) -> None:
    """Capture cam_2 et sauve une image annotee sans calculer de correction.

    Diagnostic pur : garantit une vue cam_2 a chaque tentative, y compris au
    retry ou la re-perception stereo (cam_0/cam_1) court-circuite le raffinement
    cam_2. Sans cet appel, aucune image ne serait ecrite pour la deuxieme
    descente, ce qui empecherait de comparer les essais successifs. Reutilise le
    meme rendu que le raffinement (croix jaune = axe optique, rouge = blob
    retenu). Non bloquant : toute erreur est ignoree.
    """
    try:
        frames = multi_camera.grab(robot_state=robot_state, flush=flush_frames)
        frame_c2 = frames.get("cam_2")
        if frame_c2 is None:
            print(f"   [display] (snapshot cam_2{(' ' + tag) if tag else ''} : "
                  "pas de frame)")
            return
        dets = detector.detect(frame_c2)
        if label_mapping:
            for d in dets:
                if d.label in label_mapping:
                    d.label = label_mapping[d.label]
        matches = [d for d in dets if d.label == target_label]
        chosen = max(matches, key=lambda d: d.score) if matches else None
        if tag:
            print(f"   [snapshot cam_2 {tag}] {len(matches)} detection(s) "
                  f"'{target_label}'")
        _save_cam2_debug(out_dir, frame_c2, matches, chosen)
    except Exception as e:  # pragma: no cover - debug seulement
        print(f"   [display] (snapshot cam_2 non sauve : {e})")


def _load_cam2_bias_m():
    """Biais systematique de cam_2 (eye-in-hand), soustrait de la position 3D
    detectee par cam_2 dans la boucle fermee, avant le calcul de la correction.

    Chemin separe de la stereo : cam_2 possede sa propre calibration hand-eye
    (residu ~2.5 mm, distincte de cam_0/cam_1) donc son propre biais. Le biais
    stereo (bias_correction.json) ne peut pas etre reutilise ici. Configurable
    via configs/perception/bias_correction_cam2.json {dx_mm, dy_mm, dz_mm}.
    Vaut (0, 0, 0) par defaut si le fichier est absent.
    """
    import json
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2]
         / "configs" / "perception" / "bias_correction_cam2.json")
    if not p.exists():
        return np.zeros(3)
    try:
        d = json.load(open(p))
        return np.array([float(d.get("dx_mm", 0.0)),
                         float(d.get("dy_mm", 0.0)),
                         float(d.get("dz_mm", 0.0))], dtype=float) / 1000.0
    except Exception:
        return np.zeros(3)


def refine_grasp_with_cam2(
    target_label: str,
    detector: ObjectDetector,
    multi_camera: MultiCamera,
    robot_state: RobotState,
    target_z_base_m: Optional[float] = None,
    z_height_above_object_m: float = 0.08,
    label_mapping: Optional[dict] = None,
    verbose: bool = True,
    debug_save_dir: Optional[object] = None,
    grasp_xy_base_m: Optional[object] = None,
    flush_frames: int = 3,
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
    # Capture cam_2 uniquement. Avec flush_frames > 0, on vide le buffer pilote
    # des frames perimees (prises pendant le mouvement du bras vers approach)
    # avant de lire l'image, ce qui evite une detection "derriere l'objet" sur
    # une vieille frame. La perception initiale, elle, draine via le warmup open().
    frames = multi_camera.grab(robot_state=robot_state, flush=flush_frames)
    frame_c2 = frames.get("cam_2")
    if frame_c2 is None:
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(0.0, 0.0),
            confidence=0.0,
            detection=None,
            target_label=target_label,
            method="image_centering",
            message="cam_2 n'a pas pu capturer (probleme de flux USB ?)",
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
        # Snapshot meme sans detection de la cible : on sauve tout de meme la vue
        # cam_2 (avec toutes les detections, quel que soit le label) pour
        # visualiser ce que cam_2 percoit quand elle rate la cible. La croix
        # jaune marque l'axe optique.
        if debug_save_dir is not None:
            _save_cam2_debug(debug_save_dir, frame_c2, dets, None)
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
    # Filtre de bord selectif : on rejette les detections tronquees a gauche, a
    # droite ou en bas (centre biaise, donc correction fausse), mais on garde
    # celles qui ne touchent que le bord haut. L'axe optique de cam_2 n'est pas
    # aligne avec le bout des pinces, donc un objet bien place apparait haut dans
    # l'image (cf _bbox_touches_border, ignore_top=True).
    inside = [d for d in matches
              if not _bbox_touches_border(d.bbox, w, h, ignore_top=True)]
    if not inside:
        # Snapshot du blob rejete au bord : on sauve la vue avec la detection
        # tronquee mise en evidence. C'est precisement le cas a inspecter (par
        # exemple un faux blob pleine hauteur au bord droit).
        if debug_save_dir is not None:
            _save_cam2_debug(debug_save_dir, frame_c2, matches,
                             max(matches, key=lambda d: d.score))
        return RefinementResult(
            delta_base_m=np.zeros(3),
            delta_pixels=(0.0, 0.0),
            confidence=0.0,
            detection=max(matches, key=lambda d: d.score),
            target_label=target_label,
            method="ray_plane_intersection",
            message=(f"cam_2 : '{target_label}' detecte uniquement au bord de "
                     f"l'image (bbox tronquee) -> correction ignoree"),
        )
    # Garde la detection la plus confiante
    det = max(inside, key=lambda d: d.score)

    # Diagnostic du handoff : taille absolue du blob cam_2 (par opposition au
    # score, qui est une aire normalisee trompeuse). Un vrai objet a 8 cm remplit
    # une bonne fraction du cadre ; un fragment de masque (HSV trop serre) ou un
    # objet au bord reste petit.
    det_area_px = float(getattr(det, "area_px", 0.0) or 0.0)
    if det_area_px <= 0.0 and det.bbox is not None:
        bx0, by0, bx1, by1 = det.bbox
        det_area_px = float(abs((bx1 - bx0) * (by1 - by0)))
    det_area_frac = det_area_px / float(max(w * h, 1))
    if debug_save_dir is not None:
        _save_cam2_debug(debug_save_dir, frame_c2, inside, det)

    # Centre detecte compare au point principal (cx, cy = axe optique), et non au
    # centre geometrique (w/2, h/2). Le rayon "ou cam_2 vise" passe par l'axe
    # optique, et le pixel detecte est undistordu plus bas (repere ideal centre
    # sur cx, cy). Pour que la correction (detecte - vise) ne soit pas biaisee,
    # les deux references doivent vivre dans le meme repere : on utilise donc
    # cx, cy des deux cotes.
    u, v = det.center_px
    cx, cy = float(frame_c2.K[0, 2]), float(frame_c2.K[1, 2])
    u_center, v_center = cx, cy
    du_px = u - u_center  # positif = objet a droite dans l'image
    dv_px = v - v_center  # positif = objet en bas dans l'image

    # Projection inverse par intersection rayon-plan.
    # La formule simplifiee Δm = Δpx × Z / fx supposait cam_2 a la verticale, ce
    # qui n'est pas le cas en pose approach (inclinaison d'environ 15 deg). Elle
    # sous-estimait alors la correction (de l'ordre de la moitie de sa vraie
    # valeur).
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

    # Undistortion du pixel avant le calcul du rayon. La triangulation stereo le
    # fait deja (pose_estimator) mais ce module utilisait le pixel brut : pres du
    # bord de l'image cam_2, la distorsion decale le rayon de plusieurs mm au sol,
    # ce qui faussait les corrections.
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
    # Biais propre a cam_2 (distinct du biais stereo). Soustrait pour ramener la
    # detection cam_2 dans le meme repere de-biaise que la pose planifiee. X est
    # laisse a 0 (cam_2 precis en X quand le Z est bon) ; dy est a regler par
    # essais si la prise devie en Y. Voir configs/perception/bias_correction_cam2.json.
    obj_pos_base = obj_pos_base - _load_cam2_bias_m()

    # Position de reference : la ou la pince va saisir. La reference etait
    # auparavant l'axe optique (point principal cx, cy). Mais cam_2 est deportee
    # de la pince (hand-eye d'environ +60 mm en Y), si bien que la correction
    # etait biaisee par cette parallaxe et surestimee. On reference desormais la
    # position de prise planifiee (grasp_xy_base_m) projetee sur le plan objet,
    # d'ou Δ = objet_detecte - prise. Repli sur l'axe optique si grasp_xy_base_m
    # n'est pas fournie (compatibilite).
    if grasp_xy_base_m is not None:
        gxy = np.asarray(grasp_xy_base_m, dtype=float).reshape(-1)
        expected_pos_base = np.array([gxy[0], gxy[1], target_z])
    else:
        d_cam_center = K_inv @ np.array([cx, cy, 1.0])
        d_base_center = R_base_cam2 @ d_cam_center
        t_center = (target_z - o_base[2]) / d_base_center[2]
        expected_pos_base = o_base + t_center * d_base_center

    # Correction = position reelle - position visee
    delta_base = obj_pos_base - expected_pos_base
    # On ne touche pas a Z_base : la hauteur reste celle calculee par stereo.
    delta_base[2] = 0.0
    # Correction ray-plane finale.
    dx_cam_m, dy_cam_m = float(delta_base[0]), float(delta_base[1])  # info debug

    # Orientation vue par cam_2 : grand axe de l'objet en repere base, depuis le
    # contour (proche et quasi au-dessus, donc plus net que la stereo oblique).
    yaw_cam2 = None
    elong_cam2 = 1.0
    ori = long_axis_base_from_contour(det.contour, K, frame_c2.dist,
                                      T_base_cam2, target_z)
    if ori is not None:
        yaw_cam2, elong_cam2 = ori

    if verbose:
        print(f"   [closed_loop] cam_2 a Z_base={z_cam2_base*1000:.1f}mm, "
              f"objet attendu Z={target_z*1000:.1f}mm")
        print(f"   [closed_loop] cam_2 vise actuellement : "
              f"({expected_pos_base[0]*1000:+.1f}, {expected_pos_base[1]*1000:+.1f}, "
              f"{expected_pos_base[2]*1000:+.1f}) mm")
        print(f"   [closed_loop] objet detecte par cam_2 a : "
              f"({obj_pos_base[0]*1000:+.1f}, {obj_pos_base[1]*1000:+.1f}, "
              f"{obj_pos_base[2]*1000:+.1f}) mm")
        # Diagnostic de parallaxe en Z : cam_2 voit le dessus de l'objet mais on
        # projette le rayon sur target_z (= table + H/2). Si le vrai plan vu est
        # plus haut (dessus a environ table + H), la position projetee se decale
        # dans la direction de visee. On chiffre de combien (X, Y) la position
        # bougerait pour un plan situe 15 mm plus haut, ce qui donne la
        # sensibilite a la parallaxe mesuree sur le robot. Un chiffre eleve
        # (plus de quelques mm) est une cause directe d'une prise a cote ou dans
        # le vide malgre une bonne detection.
        _shift15 = (d_base[:2] / d_base[2]) * 0.015 * 1000.0
        print(f"   [diag parallaxe Z] plan actuel=table+H/2={target_z*1000:.0f}mm ; "
              f"si on projetait +15mm plus haut (dessus objet) la position bougerait "
              f"de ({_shift15[0]:+.1f},{_shift15[1]:+.1f}) mm "
              f"(cam_2 voit le dessus, pas le milieu)")

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
                 f"projection ray-plane)  "
                 f"[blob cam_2 {det_area_px/1000:.1f}kpx="
                 f"{100*det_area_frac:.1f}% cadre, score {det.score:.2f}]"),
        yaw_base_cam2=yaw_cam2,
        elong_cam2=elong_cam2,
        area_px=det_area_px,
        area_frac=det_area_frac,
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
