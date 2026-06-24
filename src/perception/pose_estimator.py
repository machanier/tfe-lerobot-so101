"""
pose_estimator.py - Reconstruction 3D des objets dans le repere base du robot.

Convertit les detections 2D (Detection2D, plan image) en objets 3D
(ObjectInstance, repere base) en exploitant les calibrations
intrinseque+hand-eye chargees par camera_io.py.

Deux strategies, complementaires :

  TRIANGULATION STEREO (cam_0 + cam_1)
      Quand un objet est vu par les deux cameras eye-to-hand simultanement,
      on triangule a partir des deux rayons.
      Algorithme : DLT lineaire (Hartley & Zisserman 2018, ch. 12) sur les
      coordonnees pixel normalisees (debarrassees de la distorsion via
      cv2.undistortPoints). Puis raffinement non-lineaire via cv2.solvePnP
      retroprojete sur les deux vues.

  PNP MONOCULAIRE (cam_2 eye-in-hand, fallback)
      Quand un objet n'est vu que par une seule camera mais qu'on connait
      sa TAILLE METRIQUE reelle (depuis ObjectSpec.meta), on resout le PnP
      a partir du contour 2D et d'un modele 3D simple (carre / disque /
      polygone planaire). Permet de couvrir les cas d'occlusion partielle.
      Reference : EPnP (Lepetit et al. 2009), utilise via cv2.solvePnP.

      LIMITE CONNUE : pour 4 points coplanaires (cube/rectangle vu de face),
      il existe une AMBIGUITE planaire (deux poses possibles) et le solveur
      IPPE_SQUARE peut choisir la branche flippee si l'objet est quasi
      parallele au plan image. En pratique le PnP eye-in-hand suppose une
      vue oblique. La validation experimentale du Sprint 2 reposera sur la
      triangulation stereo (cam_0 + cam_1), plus robuste.
      Cette limite est mentionnee dans Lepetit et al. 2009 (sec. "Planar case").

Toutes les positions retournees sont en METRES dans le repere base.
Toutes les covariances en mm^2.

Architecture : ce module est PURE (pas d'I/O, pas de hardware). On lui passe
les Detection2D et les Frame correspondantes, il rend les ObjectInstance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.perception.scene import Detection2D, Frame, ObjectInstance, Scene


# Offset horizontal MAX (m) entre le sommet triangule et le centroide pour qu'un
# objet a bbox image haute & fine soit accepte comme DEBOUT. Au-dela, le "sommet"
# de la bbox est en realite le BOUT LOINTAIN d'un objet allonge qui FUIT vers les
# cameras (cylindre couche // X) -> on REFUSE le classement "debout" (sinon hauteur
# = longueur -> prise bien trop haute). ~ rayon max plausible d'un dessus d'objet
# debout (nos objets <= 30mm de diametre -> rayon <= 15mm ; couche // X -> offset
# ~ demi-longueur ~ 30mm). Reglable. cf grasp-frame-bias-diagnostic.
DEBOUT_TOP_OFFSET_MAX_M = 0.025


# ============================================================
# Helpers de geometrie projective
# ============================================================


def _projection_matrix(K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Matrice de projection 3x4 P = K @ [R|t]_cam_base d'un point 3D
    exprime DANS LE REPERE BASE vers le plan image de la camera.

    On a T_base_cam (pose camera dans base), il faut [R|t]_cam_base pour
    projeter, donc on inverse :
        [R|t]_cam_base = T_cam_base[:3, :] = (T_base_cam^-1)[:3, :]
    """
    T_cam_base = _se3_inverse(T_base_cam)
    return K @ T_cam_base[:3, :]


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def _undistort_point(uv: tuple[float, float], K: np.ndarray, dist: np.ndarray
                     ) -> np.ndarray:
    """Annule la distorsion d'un pixel et renvoie sa coordonnee pixel ideale (sans distorsion).

    On garde la representation pixel (pas la representation normalisee
    "coordonnees camera") pour rester homogene avec la matrice de
    projection K@[R|t] utilisee par cv2.triangulatePoints.
    """
    pt = np.array([[uv]], dtype=np.float64)  # shape (1, 1, 2)
    out = cv2.undistortPoints(pt, K, dist, P=K)  # P=K -> reprojette en pixels
    return out.reshape(2)


# ============================================================
# Triangulation stereo (cam_0 + cam_1)
# ============================================================


def triangulate_stereo(det_left: Detection2D, det_right: Detection2D,
                       frame_left: Frame, frame_right: Frame) -> np.ndarray:
    """Triangule la position 3D d'un objet vu par 2 cameras eye-to-hand.

    Args:
        det_left, det_right : detections dans les deux cameras (centres pixel).
        frame_left, frame_right : frames associees (fournit K, dist, T_base_cam).

    Returns:
        position 3D (3,) dans le repere BASE, en metres.
    """
    # 1. Compense la distorsion sur les centres detectes
    uv_l = _undistort_point(det_left.center_px, frame_left.K, frame_left.dist)
    uv_r = _undistort_point(det_right.center_px, frame_right.K, frame_right.dist)

    # 2. Construit les matrices de projection P = K [R|t]_cam_base
    P_l = _projection_matrix(frame_left.K, frame_left.T_base_cam)
    P_r = _projection_matrix(frame_right.K, frame_right.T_base_cam)

    # 3. Triangulation lineaire (DLT)
    pts_4d = cv2.triangulatePoints(P_l, P_r,
                                    uv_l.reshape(2, 1), uv_r.reshape(2, 1))
    # pts_4d shape (4, 1) ; on normalise (X/W, Y/W, Z/W)
    w = pts_4d[3, 0]
    if abs(w) < 1e-12:
        raise ValueError("Triangulation degeneree (w = 0).")
    X = pts_4d[:3, 0] / w
    return X


def reproject_error(point_base_m: np.ndarray, det: Detection2D, frame: Frame
                    ) -> float:
    """Reprojette `point_base_m` dans la camera de `frame` et renvoie l'erreur
    en pixels par rapport au centre detecte (utile pour valider la triangulation).
    """
    P = _projection_matrix(frame.K, frame.T_base_cam)
    Xh = np.hstack([point_base_m, 1.0])
    uvw = P @ Xh
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    du = u - det.center_px[0]
    dv = v - det.center_px[1]
    return float(np.hypot(du, dv))


# ============================================================
# PnP monoculaire (cam_2 fallback, ou stereo indisponible)
# ============================================================


def _shape_object_points(spec_meta: dict) -> Optional[np.ndarray]:
    """Construit les points 3D de reference selon la forme attendue de l'objet.

    Utilise pour le PnP monoculaire : besoin de connaitre la taille metrique
    pour resoudre la profondeur. Les coordonnees sont en METRES, dans le
    repere local de l'objet (origine au centre, Z normal a la face avant).
    """
    shape = spec_meta.get("shape")
    if shape == "cube":
        side = float(spec_meta.get("side_mm", 30.0)) / 1000.0
        h = side / 2.0
        # 4 coins de la face avant
        return np.array([
            [-h, -h, 0.0],
            [+h, -h, 0.0],
            [+h, +h, 0.0],
            [-h, +h, 0.0],
        ], dtype=np.float64)
    if shape == "rect_prism":
        w = float(spec_meta.get("width_mm", 40.0)) / 1000.0
        h = float(spec_meta.get("height_mm", 25.0)) / 1000.0
        return np.array([
            [-w / 2, -h / 2, 0.0],
            [+w / 2, -h / 2, 0.0],
            [+w / 2, +h / 2, 0.0],
            [-w / 2, +h / 2, 0.0],
        ], dtype=np.float64)
    return None  # forme non geree -> pas de PnP


def estimate_pnp_mono(det: Detection2D, frame: Frame, spec_meta: dict
                      ) -> Optional[np.ndarray]:
    """Estime la position 3D par PnP monoculaire (necessite spec_meta.shape).

    Approxime le contour detecte par 4 coins (cv2.approxPolyDP) puis
    resout cv2.solvePnP avec les correspondances 2D-3D. Pour un cube vu de
    face, les 4 coins de la face avant suffisent.

    Returns:
        position 3D (3,) dans le repere BASE, en metres, ou None si echec.
    """
    obj_pts = _shape_object_points(spec_meta)
    if obj_pts is None or det.contour is None or len(det.contour) < 4:
        return None

    # Approxime le contour par un polygone et recupere ~4 coins
    contour = det.contour.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True).reshape(-1, 2)
    if approx.shape[0] != obj_pts.shape[0]:
        return None  # pas le bon nombre de coins -> PnP impossible
    # Ordonne les coins dans le meme sens que obj_pts (sens horaire depuis "bas-gauche")
    img_pts = _sort_quad_corners(approx).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, frame.K, frame.dist,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE
                                   if obj_pts.shape[0] == 4 else cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # tvec est la position du centre de l'objet dans le repere CAMERA.
    # On la transporte dans le repere BASE :  X_base = T_base_cam @ [X_cam ; 1]
    X_cam = tvec.flatten()
    X_base_h = frame.T_base_cam @ np.hstack([X_cam, 1.0])
    return X_base_h[:3]


def _sort_quad_corners(pts: np.ndarray) -> np.ndarray:
    """Trie 4 points 2D en ordre : bas-gauche, bas-droite, haut-droite, haut-gauche.

    Convention coherente avec _shape_object_points pour les formes a 4 coins.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],  # haut-gauche (min x+y)
        pts[np.argmin(d)],  # haut-droite (min y-x)
        pts[np.argmax(s)],  # bas-droite (max x+y)
        pts[np.argmax(d)],  # bas-gauche (max y-x)
    ])


# ============================================================
# Strategie complete : detections multi-cam -> Scene
# ============================================================


@dataclass
class PoseEstimatorConfig:
    """Hyperparametres du pose estimator.

    Attributes:
        stereo_keys              : ordre des deux cameras stereo (gauche, droite).
        max_reproj_error_px      : si l'erreur de reprojection moyenne depasse
                                   ce seuil, on rejette la triangulation.
        max_z_base_m             : limite haute Z dans le repere base (l'objet
                                   est forcement sur ou pres de la table).
        min_z_base_m             : limite basse Z (eviter les solutions degenerees
                                   negatives).
        enable_mono_pnp_fallback : si vrai et stereo indisponible/echec, tente
                                   un PnP monoculaire (sur cam_2 en priorite).
        scene_config_path        : chemin vers configs/scene.json (zones
                                   d'exclusion + bornes workspace). Si None,
                                   utilise les valeurs internes (min/max Z).
    """

    stereo_keys: tuple[str, str] = ("cam_0", "cam_1")
    # Compensation SYSTEMATIQUE du biais de calibration mesure
    # empiriquement (e.g. biais Y +28mm sur le poste de Maxence).
    # Chargee depuis configs/perception/bias_correction.json si present.
    # Soustraite a CHAQUE position triangulee : pos_corrigee = pos - bias.
    # Permet d'avoir une correction PERMANENTE sans modifier gt_test.json.
    bias_correction_m: Optional[object] = None  # ndarray (3,) ou None
    # Seuil reprojection : 60 px (a 40 px on rejettait des detections valides
    # juste au-dessus du seuil, ex: reproj=40.1px -> annulation tout le pipeline).
    # Historique :
    #   - 25 px : calcul theorique pur (7mm * 1225 / 500 + marge 1.5x)
    #   - 40 px : empirique avec HF, ne suffisait pas (cas du cube a Y=165mm
    #             ou la triangulation est juste au-dessus)
    #   - 60 px : assez permissif pour les detections HF a la marge, mais
    #             reste assez strict pour rejeter les mauvaises correspondances
    #             (qui donnent typiquement reproj > 200 px).
    max_reproj_error_px: float = 60.0
    max_z_base_m: float = 0.40
    min_z_base_m: float = -0.05
    enable_mono_pnp_fallback: bool = True
    scene_config_path: Optional[object] = None  # Path ou str


class PoseEstimator:
    """Construit une `Scene` 3D a partir des detections multi-cameras.

    Pipeline :
      1. Groupe les detections par label.
      2. Pour chaque label, essaie la triangulation stereo (cam_0 + cam_1).
      3. Si stereo echoue (l'objet n'est vu que par une cam, ou reprojection
         trop grande) et que l'option fallback est active, tente le PnP
         monoculaire sur cam_2 (eye-in-hand, le plus precis).
      4. Filtre les estimations dont z (base) est dehors de la plage attendue.
      5. Filtre les estimations qui tombent dans une **zone d'exclusion**
         (charge depuis configs/scene.json) : c'est ce qui evite de detecter
         le robot lui-meme comme un objet (cas du cube orange vs filament
         orange du robot).

    L'implementation reste SIMPLE et lisible : la sophistication sera ajoutee
    si necessaire au Sprint 3 (e.g. RANSAC sur N>2 vues, prior bayesien...).
    """

    def __init__(self, config: Optional[PoseEstimatorConfig] = None,
                 specs_by_label: Optional[dict[str, dict]] = None,
                 load_scene_config: bool = True):
        """
        Args:
            config           : hyperparametres (defaut : PoseEstimatorConfig()).
            specs_by_label   : {label: ObjectSpec.meta} pour le PnP monoculaire.
            load_scene_config : si True (defaut), charge configs/scene.json
                pour activer les zones d'exclusion + bornes workspace.
                Passe False pour les tests synthetiques qui veulent un
                comportement deterministe independant de la config reelle.
        """
        self.config = config or PoseEstimatorConfig()
        # specs_by_label : {label: ObjectSpec.meta} pour PnP monoculaire
        self.specs_meta = specs_by_label or {}
        # Zones d'exclusion + bornes workspace charges depuis scene.json (si dispo)
        self._exclusion_zones: list[dict] = []
        self._workspace_bounds: Optional[dict] = None
        if load_scene_config:
            self._load_scene_config(self.config.scene_config_path)
        # Compensation systematique du biais (cf D11+experimentation Maxence)
        self._bias_m: Optional[np.ndarray] = None
        self._load_bias_correction()

    def _load_scene_config(self, path):
        """Charge configs/scene.json si present. Defaut : pas de zones."""
        import json
        from pathlib import Path
        if path is None:
            # cherche le defaut a configs/scene.json relativement au repo
            default = Path(__file__).resolve().parents[2] / "configs" / "scene.json"
            if default.exists():
                path = default
            else:
                return
        path = Path(path)
        if not path.exists():
            return
        data = json.load(open(path))
        self._exclusion_zones = data.get("exclusion_zones_base_m", []) or []
        self._workspace_bounds = data.get("workspace_bounds_base_m")

    def _load_bias_correction(self):
        """Charge la compensation systematique depuis
        configs/perception/bias_correction.json si present.
        Format : {"dx_mm": ..., "dy_mm": ..., "dz_mm": ...}
        La valeur sera SOUSTRAITE a chaque position triangulee.
        """
        import json
        from pathlib import Path
        # Priorite : config.bias_correction_m si fourni explicitement
        if self.config.bias_correction_m is not None:
            self._bias_m = np.asarray(self.config.bias_correction_m, dtype=float).reshape(3)
            return
        # Sinon, fichier dans configs/perception/
        bias_path = Path(__file__).resolve().parents[2] / "configs" / "perception" / "bias_correction.json"
        if not bias_path.exists():
            self._bias_m = None
            return
        data = json.load(open(bias_path))
        self._bias_m = np.array([
            float(data.get("dx_mm", 0)) / 1000.0,
            float(data.get("dy_mm", 0)) / 1000.0,
            float(data.get("dz_mm", 0)) / 1000.0,
        ])

    # ----- API ------------------------------------------------------------

    def _estimate_bbox_3d_m(self, det, frame, X_base):
        """Estime l'extent 3D (dx, dy, dz) en metres a partir de la bbox 2D et
        de la profondeur objet->camera, pour alimenter les features
        adaptatives (ouverture pince = min(dx,dy) ; profondeur de prise =
        table + dz/2).

        Methode : taille_metrique ~= taille_pixel * profondeur / focale.
        APPROXIMATION (la bbox image melange empreinte au sol et hauteur selon
        l'angle de vue) ; suffisante pour adapter la prise, a raffiner avec les
        dimensions reelles connues des objets si besoin. dz est borne a
        [5, 150] mm. Renvoie None (=> comportement non-adaptatif) si indispo.
        """
        try:
            if det is None or getattr(det, "bbox", None) is None or frame is None:
                return None
            f = float(np.asarray(frame.K, dtype=float)[0, 0])
            T_cam_base = np.linalg.inv(np.asarray(frame.T_base_cam, dtype=float))
            Xh = np.hstack([np.asarray(X_base, dtype=float).reshape(3), 1.0])
            depth = abs(float((T_cam_base @ Xh)[2]))
            if f <= 1e-6 or depth <= 1e-6:
                return None
            x0, y0, x1, y1 = det.bbox
            dx = abs(float(x1 - x0)) * depth / f
            dy = abs(float(y1 - y0)) * depth / f
            dz = float(np.clip(dy, 0.005, 0.15))   # hauteur : proxy = extent vertical
            return (float(dx), float(dy), dz)
        except Exception:
            return None

    def _triangulate_bbox_top(self, det_L, det_R, f_L, f_R):
        """Triangule le point HAUT-CENTRE des deux bboxes -> sommet 3D de l'objet.

        Le haut de bbox des deux vues correspond approximativement au meme
        point physique (le sommet) pour deux cameras d'elevation voisine
        (paire stereo cam_0/cam_1) : erreur typique de quelques mm. C'est la
        hauteur FIABLE qui remplace le proxy 'hauteur pixels' de
        _estimate_bbox_3d_m, lequel melange longueur et hauteur quand l'objet
        pointe vers les cameras (bug 'cylindre // Y saisi trop haut',
        diagnostic 2026-06-12 : 53mm estimes pour un cylindre de 30mm).

        Returns: position 3D (3,) du sommet en repere base, ou None.
        """
        if (det_L is None or det_R is None or f_L is None or f_R is None
                or det_L.bbox is None or det_R.bbox is None):
            return None
        try:
            d_top_L = Detection2D(
                cam_key=det_L.cam_key, label=det_L.label,
                center_px=(0.5 * (det_L.bbox[0] + det_L.bbox[2]),
                           float(det_L.bbox[1])))
            d_top_R = Detection2D(
                cam_key=det_R.cam_key, label=det_R.label,
                center_px=(0.5 * (det_R.bbox[0] + det_R.bbox[2]),
                           float(det_R.bbox[1])))
            X_top = triangulate_stereo(d_top_L, d_top_R, f_L, f_R)
        except Exception:
            return None
        if not np.all(np.isfinite(X_top)):
            return None
        if self._bias_m is not None:
            X_top = X_top - self._bias_m
        return X_top

    def _footprint_orientation(self, det, frame, z_plane_m: float):
        """Orientation du grand axe de l'EMPREINTE de l'objet, en repere BASE.

        Projette les points du contour (ou les coins de bbox a defaut) sur le
        plan horizontal z=z_plane_m par intersection rayon-plan (pixels
        undistordus d'abord), puis ACP 2D dans le plan XY base.

        Contrairement a yaw_from_contour (angle dans le repere IMAGE), le
        resultat est independant de l'orientation de la camera -> les objets
        poses EN BIAIS donnent un yaw correct. Approximation connue : la
        silhouette inclut des points au-dessus du plan -> legere elongation
        parasite le long de l'axe de visee (quelques degres de biais au pire).

        Returns:
            (yaw_rad [-pi/2, pi/2], elongation >= 1, extent_long_m,
             extent_court_m) ou None si indisponible.
        """
        if det is None or frame is None:
            return None
        pts = None
        if det.contour is not None and len(det.contour) >= 6:
            pts = np.asarray(det.contour, dtype=np.float64).reshape(-1, 2)
            if len(pts) > 80:                      # sous-echantillonne (vitesse)
                pts = pts[::max(1, len(pts) // 80)]
        elif det.bbox is not None:
            x0, y0, x1, y1 = det.bbox
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                           dtype=np.float64)
        if pts is None or len(pts) < 4:
            return None
        try:
            und = cv2.undistortPoints(pts.reshape(-1, 1, 2), frame.K,
                                      frame.dist, P=frame.K).reshape(-1, 2)
            K_inv = np.linalg.inv(frame.K)
            R = frame.T_base_cam[:3, :3]
            o = frame.T_base_cam[:3, 3]
            homog = np.hstack([und, np.ones((len(und), 1))])
            rays = (R @ (K_inv @ homog.T)).T            # (N, 3) en base
            dz = rays[:, 2]
            keep = np.abs(dz) > 1e-9
            s = (z_plane_m - o[2]) / dz[keep]
            fwd = s > 0                                  # devant la camera
            P = o[None, :] + s[fwd, None] * rays[keep][fwd]
        except Exception:
            return None
        if len(P) < 4:
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
        if ext_long < ext_court:                         # garantit long >= court
            ext_long, ext_court = ext_court, ext_long
            theta += np.pi / 2.0
        while theta > np.pi / 2:
            theta -= np.pi
        while theta < -np.pi / 2:
            theta += np.pi
        if ext_court < 1e-4:
            return None
        return (float(theta), float(ext_long / max(ext_court, 1e-6)),
                ext_long, ext_court)

    def _estimate_geometry(self, det_L, det_R, f_L, f_R, X_base):
        """Geometrie 3D enrichie : bbox_3d corrige + classe de pose + yaw base.

        1. HAUTEUR par triangulation du sommet (fiable, remplace le proxy
           pixels qui surestimait quand l'objet pointe vers les cameras).
        2. CLASSE : "debout" si hauteur >> largeur d'empreinte horizontale,
           sinon "couche" (empreinte allongee) ou "compact".
        3. YAW repere base du grand axe (objets couches, biais inclus) par
           projection rayon-plan du contour a mi-hauteur.

        Returns:
            (bbox_3d_m ou None, meta_dict) — meta contient pose_class,
            yaw_base_rad (None = yaw libre), height_method,
            footprint_elongation.
        """
        meta: dict = {}
        bbox_old = self._estimate_bbox_3d_m(det_L, f_L, X_base)

        # --- 1. hauteur ---
        height = None
        X_top = self._triangulate_bbox_top(det_L, det_R, f_L, f_R)
        if X_top is not None and 0.005 <= float(X_top[2]) <= 0.25:
            height = float(X_top[2])               # table a Z=0 (REPERE_BASE.md)
            meta["height_method"] = "sommet_triangule"
        if height is None:
            if bbox_old is None:
                return None, meta
            height = float(bbox_old[2])
            meta["height_method"] = "proxy_bbox"

        # --- 1bis. OBJET HAUT detecte en IMAGE (bbox haute & fine, 2 vues) ---
        # Un objet DEBOUT (cylindre, boite verticale) a une bbox image nettement
        # plus HAUTE que LARGE. Le detecter ICI evite deux pieges qui cassaient
        # la hauteur des cylindres debout (-> Z de prise trop bas -> cam_2
        # sur-corrige X) :
        #   (a) le sommet triangule sous-estime un DESSUS DE CYLINDRE (cercle,
        #       pas un point : le haut de bbox des 2 vues n'est pas le meme point),
        #   (b) la projection de l'empreinte sur le plan table cree une fausse
        #       elongation -> classe 'couche' + h_cap qui rabote la hauteur.
        # Pour ces objets : classe 'debout', yaw libre (dessus rond), hauteur =
        # extent VERTICAL de la bbox image (proxy fiable pour un debout), SANS
        # passer par l'empreinte projetee ni h_cap.
        def _img_aspect(det):
            b = getattr(det, "bbox", None)
            if b is None:
                return None
            w, h = abs(float(b[2] - b[0])), abs(float(b[3] - b[1]))
            return (h / w) if w > 1e-6 else None
        a_L, a_R = _img_aspect(det_L), _img_aspect(det_R)
        # GARDE-FOU anti-FORESHORTENING (cylindre COUCHE pointant VERS les cameras,
        # ex // X) : une bbox image HAUTE & FINE peut venir d'un objet DEBOUT *ou*
        # d'un objet allonge qui S'ELOIGNE (le "haut" de bbox = le BOUT LOINTAIN,
        # pas un sommet a la verticale du centre). On les separe par la position 3D
        # du sommet triangule : un vrai DEBOUT a son sommet ~A LA VERTICALE du
        # centroide (offset horizontal ~ rayon du dessus, petit) ; un COUCHE qui fuit
        # a son "sommet" sur le bout lointain (offset horizontal ~ demi-longueur,
        # grand). L'offset est INVARIANT au biais (X_top et X_base le portent tous
        # deux). Sans ce garde-fou, un cylindre // X -> hauteur = longueur -> prise
        # ancree a table + longueur/2 = bien trop haut + cam_2 projette sur un plan
        # Z errone -> X ET Y faux (essai // X 2026-06-21 "trop haut + decale a droite").
        foreshortened = False
        if (X_top is not None and X_base is not None
                and a_L is not None and a_R is not None
                and a_L >= 1.4 and a_R >= 1.4):
            off_xy = float(np.hypot(X_top[0] - X_base[0], X_top[1] - X_base[1]))
            if off_xy > DEBOUT_TOP_OFFSET_MAX_M:
                foreshortened = True
                meta["debout_rejete_offset_mm"] = round(off_xy * 1000.0, 0)
        if (not foreshortened
                and a_L is not None and a_R is not None
                and a_L >= 1.4 and a_R >= 1.4 and bbox_old is not None):
            diam = float(bbox_old[0])              # largeur ~ diametre (dessus rond)
            h_up = float(bbox_old[2])              # extent vertical bbox = hauteur
            meta["height_method"] = "bbox_haute_debout"
            meta["pose_class"] = "debout"
            meta["yaw_base_rad"] = None
            meta["img_aspect"] = round(min(a_L, a_R), 2)
            return (diam, diam, h_up), meta

        # --- 2. classe debout / couche / compact ---
        # Largeur d'empreinte approx = extent HORIZONTAL image (peu pollue par
        # la hauteur), le min des deux vues.
        foot_w = None
        for det, frm in ((det_L, f_L), (det_R, f_R)):
            b = self._estimate_bbox_3d_m(det, frm, X_base)
            if b is not None:
                foot_w = b[0] if foot_w is None else min(foot_w, b[0])
        dx = float(bbox_old[0]) if bbox_old else 0.03
        dy = float(bbox_old[1]) if bbox_old else 0.03
        # MEME garde-fou foreshortening : un objet allonge qui FUIT vers les cameras
        # (couche // X) a une hauteur 3D (sommet triangule = bout lointain) qui peut
        # depasser 1.6x l'empreinte -> ne PAS le classer debout non plus. On le
        # laisse a la classification couche/compact (section 3), avec la hauteur =
        # diametre (sommet du bout lointain ~ table + diametre).
        if (not foreshortened and foot_w is not None
                and height > 1.6 * max(foot_w, 1e-3)):
            meta["pose_class"] = "debout"
            meta["yaw_base_rad"] = None            # empreinte ~circulaire vue du haut
            return (dx, dy, height), meta

        # --- 3. objet couche/compact : yaw du grand axe en repere base ---
        # DIAGNOSTIC : on calcule l'orientation vue par CHAQUE camera separement
        # et on la stocke (yaw_cam0_deg / yaw_cam1_deg). Si les deux cameras
        # donnent le meme angle ET qu'il correspond a l'objet reel -> perception
        # fiable. Si elles divergent ou collent toujours a ~0deg -> detection a
        # ameliorer. (Permet de distinguer 'perception fausse' de 'convention 90deg'.)
        ori = None
        det_ok, frm_ok = None, None
        for name, det, frm in (("yaw_cam0_deg", det_L, f_L),
                               ("yaw_cam1_deg", det_R, f_R)):
            o = self._footprint_orientation(det, frm, z_plane_m=height / 2.0)
            if o is not None:
                meta[name] = round(float(np.degrees(o[0])), 0)
                if ori is None:
                    ori = o
                    det_ok, frm_ok = det, frm
        if ori is None:
            meta["pose_class"] = "inconnu"
            meta["yaw_base_rad"] = None
            return (dx, dy, height), meta
        yaw_b, elong, ext_long, ext_court = ori

        # BORNE PHYSIQUE : un objet pose STABLEMENT repose sur sa plus grande
        # face -> sa hauteur <= largeur courte de l'empreinte. Le sommet
        # triangule surestime encore quand l'objet pointe PILE vers les
        # cameras (le haut de bbox = le bout lointain, pas le meme point dans
        # les 2 vues) ; l'empreinte, elle, reste fiable. On prend le min, et
        # on re-projette une fois l'empreinte au bon plan (hauteur corrigee).
        h_cap = min(height, ext_court)
        if h_cap < 0.8 * height:
            ori2 = self._footprint_orientation(det_ok, frm_ok,
                                               z_plane_m=h_cap / 2.0)
            if ori2 is not None:
                yaw_b, elong, ext_long, ext_court = ori2
                h_cap = min(height, ext_court)
            meta["height_method"] = (meta.get("height_method", "?")
                                     + "+borne_empreinte")
        height = max(0.005, h_cap)

        meta["footprint_elongation"] = round(elong, 2)
        # ORIENTATION (yaw_base) DECOUPLEE de la classe de pose. On fournit le
        # grand axe (yaw_base) des que l'empreinte a une orientation FIABLE :
        # nettement allongee (ratio > 1.25 ET difference absolue > 12mm). Sinon
        # (empreinte ~ronde/carree ou trop bruitee pour trancher) yaw_base=None
        # -> la pince saisira en yaw LIBRE (rotation minimale), ce qui est correct
        # pour un rond et sans risque quand l'orientation est incertaine.
        ax_diff = ext_long - ext_court
        has_axis = (ax_diff > 0.012) and (ext_long > 1.25 * ext_court)
        meta["yaw_base_rad"] = float(yaw_b) if has_axis else None
        # pose_class : info de pose (sert au Z de prise / aux logs). 'couche' =
        # franchement allonge et plat ; sinon 'compact'.
        elongated = (ext_long > 1.5 * max(height, 1e-3)
                     and ax_diff > 0.015)
        meta["pose_class"] = "couche" if elongated else "compact"
        # Empreinte projetee = meilleures dimensions au sol disponibles
        # (ext_court ~ largeur PERPENDICULAIRE a la prise -> ouverture pince).
        return (float(ext_long), float(ext_court), float(height)), meta

    def build_scene(self,
                    detections_by_cam: dict[str, list[Detection2D]],
                    frames: dict[str, Optional[Frame]]) -> Scene:
        """Construit une Scene a partir de detections + frames synchronisees.

        Si plusieurs detections du meme label sont presentes dans une meme
        camera (cas typique de OWL-ViTv2 qui donne plusieurs bboxes overlapping),
        on garde la PLUS CONFIANTE (score max).
        """
        # Groupe par label, en gardant la meilleure detection par (label, cam)
        by_label: dict[str, dict[str, Detection2D]] = {}
        for cam_key, dets in detections_by_cam.items():
            for d in dets:
                existing = by_label.setdefault(d.label, {}).get(cam_key)
                if existing is None or d.score > existing.score:
                    by_label[d.label][cam_key] = d

        objects: list[ObjectInstance] = []
        timestamps = [f.timestamp for f in frames.values() if f is not None]
        ts_scene = float(np.mean(timestamps)) if timestamps else 0.0

        for label, per_cam in by_label.items():
            inst = self._estimate_one(label, per_cam, frames)
            if inst is not None:
                objects.append(inst)

        return Scene(objects=objects, timestamp=ts_scene,
                     meta={"detector_labels": list(by_label.keys())})

    # ----- interne --------------------------------------------------------

    def _estimate_one(self, label: str, per_cam: dict[str, Detection2D],
                      frames: dict[str, Optional[Frame]]) -> Optional[ObjectInstance]:
        kL, kR = self.config.stereo_keys
        det_L = per_cam.get(kL)
        det_R = per_cam.get(kR)
        f_L = frames.get(kL)
        f_R = frames.get(kR)

        # Diagnostic : stocke la raison du rejet (utile pour _last_rejections)
        reject_reason = None

        # 1) STEREO si dispo
        if det_L is not None and det_R is not None and f_L is not None and f_R is not None:
            try:
                X = triangulate_stereo(det_L, det_R, f_L, f_R)
            except Exception as e:
                X = None
                reject_reason = f"triangulation exception: {e}"
            if X is not None:
                # IMPORTANT : reproj_error est calcule sur X NON COMPENSE,
                # pour valider que la triangulation initiale est coherente
                # avec les pixels detectes (sinon la compensation creerait
                # artificiellement un grand reproj_err). La compensation
                # est appliquee APRES validation, sur la position finale.
                err_L = reproject_error(X, det_L, f_L)
                err_R = reproject_error(X, det_R, f_R)
                err = 0.5 * (err_L + err_R)
                # COMPENSATION SYSTEMATIQUE : applique le biais empirique
                # mesure (e.g. -30mm en Y sur le poste de Maxence).
                # Calibrable via configs/perception/bias_correction.json.
                if self._bias_m is not None:
                    X = X - self._bias_m
                ws_reason = self._workspace_reject(X)
                in_ws = ws_reason is None
                pos_mm = X * 1000
                if not in_ws:
                    reject_reason = (
                        f"stereo OK (reproj {err:.1f}px) MAIS rejet : {ws_reason} "
                        f"-- position ({pos_mm[0]:+.0f},{pos_mm[1]:+.0f},{pos_mm[2]:+.0f}) mm"
                    )
                elif err > self.config.max_reproj_error_px:
                    reject_reason = (
                        f"stereo OK (pos {pos_mm[0]:+.0f},{pos_mm[1]:+.0f},{pos_mm[2]:+.0f} mm) "
                        f"MAIS reproj_err={err:.1f}px > seuil {self.config.max_reproj_error_px}px"
                    )
                else:
                    score = float(min(det_L.score, det_R.score) *
                                  np.exp(-err / 8.0))
                    # Geometrie enrichie : hauteur par sommet triangule,
                    # classe debout/couche, yaw du grand axe en repere base.
                    bbox3d, geo_meta = self._estimate_geometry(
                        det_L, det_R, f_L, f_R, X)
                    return ObjectInstance(
                        label=label,
                        position_base_m=X,
                        source_detections=[det_L, det_R],
                        score=score,
                        bbox_3d_m=bbox3d,
                        meta={
                            "method": "stereo_triangulation",
                            "reproj_error_px": err,
                            "reproj_error_per_cam_px": {kL: err_L, kR: err_R},
                            **geo_meta,
                        },
                    )
        elif det_L is None or det_R is None:
            present = [k for k in (kL, kR) if per_cam.get(k) is not None]
            reject_reason = f"detection presente seulement dans {present}, pas de stereo possible"

        # Memorise pour diagnostic externe
        self._last_rejections = getattr(self, "_last_rejections", {})
        if reject_reason:
            self._last_rejections[label] = reject_reason

        # 2) Fallback PnP monoculaire (priorite eye-in-hand cam_2)
        if self.config.enable_mono_pnp_fallback:
            for cam_key in ("cam_2", kL, kR):
                det = per_cam.get(cam_key)
                frm = frames.get(cam_key)
                if det is None or frm is None:
                    continue
                spec_meta = self.specs_meta.get(label, {})
                X = estimate_pnp_mono(det, frm, spec_meta)
                if X is not None and self._in_workspace(X):
                    return ObjectInstance(
                        label=label,
                        position_base_m=X,
                        source_detections=[det],
                        score=0.6 * det.score,  # confiance moindre que stereo
                        meta={"method": f"pnp_mono({cam_key})"},
                    )
        return None

    def _in_workspace(self, X: np.ndarray) -> bool:
        return self._workspace_reject(X) is None

    def _workspace_reject(self, X: np.ndarray):
        """Retourne None si la position est valide, sinon une raison PRECISE
        (bornes Z code / portee / bornes scene.json / zone d'exclusion <label>).
        Permet un diagnostic clair (avant : tout etait note 'hors workspace',
        meme un rejet par zone d'exclusion -> trompait le debug, cf essais 13/14).
        """
        x, y, z = float(X[0]), float(X[1]), float(X[2])
        # Bornes Z par defaut (de la config code)
        if not (self.config.min_z_base_m <= z <= self.config.max_z_base_m):
            return f"Z={z*1000:.0f}mm hors bornes code [{self.config.min_z_base_m*1000:.0f},{self.config.max_z_base_m*1000:.0f}]"
        # SO-101 a un bras de ~30 cm : tout ce qui est au-dela de 1 m est aberrant.
        if (x * x + y * y + z * z) > 1.0:
            return f"portee {np.linalg.norm([x,y,z])*1000:.0f}mm > 1m (aberrant)"
        # Bornes workspace de scene.json (si dispo)
        if self._workspace_bounds is not None:
            wb = self._workspace_bounds
            if not (wb["x_min"] <= x <= wb["x_max"]): return f"X={x*1000:.0f}mm hors bornes scene [{wb['x_min']*1000:.0f},{wb['x_max']*1000:.0f}]"
            if not (wb["y_min"] <= y <= wb["y_max"]): return f"Y={y*1000:.0f}mm hors bornes scene [{wb['y_min']*1000:.0f},{wb['y_max']*1000:.0f}]"
            # Borne Z INFERIEURE : tolerance vers le bas. Le Z du CENTROIDE stereo
            # est notoirement sous-estime sur les objets non plats (debout/couche) :
            # le centroide de la silhouette se triangule trop bas, jusqu'a passer
            # SOUS la table (ex. cylindre couche : Z=-23mm alors qu'il est pose
            # dessus). La PRISE n'utilise pas ce Z (elle ancre Z=table+H/2), donc
            # rejeter la detection sur ce Z faux fait perdre un objet bien vu en 2D
            # (// X "impossible", 2026-06-20). On tolere Z_UNDERSHOOT_TOL_M sous
            # z_min ; la borne HAUTE reste stricte (un objet trop HAUT est suspect).
            Z_UNDERSHOOT_TOL_M = 0.030
            if not (wb["z_min"] - Z_UNDERSHOOT_TOL_M <= z <= wb["z_max"]):
                return f"Z={z*1000:.0f}mm hors bornes scene [{wb['z_min']*1000:.0f},{wb['z_max']*1000:.0f}] (tol basse {Z_UNDERSHOOT_TOL_M*1000:.0f}mm)"
        # Zones d'exclusion : la position ne doit etre dans AUCUNE zone.
        # CARVE-OUT TABLE (2026-06-13) : la zone 'robot_arm_envelope' (box
        # englobant le bras replie) couvre x[0,0.20] z[0,0.30] et avalait donc
        # les objets reels POSES sur la table devant le robot (crash 'NON
        # DETECTE' essais 13/14 : cylindre a (199,44,35)mm rejete alors que la
        # triangulation etait excellente, reproj 4.2px). Un objet pose sur la
        # table (z bas) n'est jamais le bras (qui est sureleve quand replie) ->
        # on n'applique PAS cette enveloppe sous TABLE_OBJECT_Z_M. Les autres
        # zones (base robot) restent strictes.
        TABLE_OBJECT_Z_M = 0.06
        for zone in self._exclusion_zones:
            if zone.get("label") == "robot_arm_envelope" and z < TABLE_OBJECT_Z_M:
                continue
            if self._point_in_zone((x, y, z), zone):
                return f"dans zone d'exclusion '{zone.get('label', '?')}'"
        return None

    @staticmethod
    def _point_in_zone(p: tuple[float, float, float], zone: dict) -> bool:
        """Renvoie True si le point est dans la zone d'exclusion.

        Types geres : "cylinder" (axe Z) et "box" (AABB en metres).
        """
        x, y, z = p
        t = zone.get("type", "cylinder")
        c = zone.get("center_base_m", [0, 0, 0])
        if t == "cylinder":
            r = float(zone.get("radius_m", 0.10))
            h = float(zone.get("height_m", 0.30))
            return ((x - c[0]) ** 2 + (y - c[1]) ** 2 <= r * r
                    and c[2] - 0.01 <= z <= c[2] + h)
        if t == "box":
            d = zone.get("dimensions_m", [0.1, 0.1, 0.1])
            return (abs(x - c[0]) <= d[0] / 2
                    and abs(y - c[1]) <= d[1] / 2
                    and abs(z - c[2]) <= d[2] / 2)
        return False  # type inconnu : on ne bloque pas


# ============================================================
# Self-tests (lance avec : python -m src.perception.pose_estimator)
# ============================================================
if __name__ == "__main__":
    print("Tests pose_estimator.py")

    # Cadre synthetique : deux cameras eye-to-hand qui regardent vers l'origine,
    # baseline 100 mm le long de Y, axe optique vers -Z dans le repere base.
    K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    def make_eye2hand(t_base_cam, R_base_cam):
        T = np.eye(4)
        T[:3, :3] = R_base_cam
        T[:3, 3] = t_base_cam
        return T

    # Cameras a (0, +0.05, 0.30) et (0, -0.05, 0.30), regardant -Z
    # Convention OpenCV : axe Z de la camera = direction de regard
    # On choisit R tel que Z_cam = -Z_base, X_cam = X_base, Y_cam = -Y_base
    R_cam_base = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ], dtype=np.float64)

    T_L = make_eye2hand([0.0, +0.05, 0.30], R_cam_base)
    T_R = make_eye2hand([0.0, -0.05, 0.30], R_cam_base)

    # 1. Projection forward d'un point 3D et triangulation inverse
    rng = np.random.default_rng(0)
    err_max = 0.0
    for _ in range(20):
        X_true = np.array([
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(0.02, 0.10),
        ])
        P_L = _projection_matrix(K, T_L)
        P_R = _projection_matrix(K, T_R)
        uvw_L = P_L @ np.hstack([X_true, 1.0])
        uvw_R = P_R @ np.hstack([X_true, 1.0])
        uv_L = uvw_L[:2] / uvw_L[2]
        uv_R = uvw_R[:2] / uvw_R[2]

        # Construit les Frame + Detection2D et triangule
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        f_L = Frame(cam_key="cam_0", image=img, K=K, dist=dist, T_base_cam=T_L)
        f_R = Frame(cam_key="cam_1", image=img, K=K, dist=dist, T_base_cam=T_R)
        d_L = Detection2D(cam_key="cam_0", label="x", center_px=(uv_L[0], uv_L[1]))
        d_R = Detection2D(cam_key="cam_1", label="x", center_px=(uv_R[0], uv_R[1]))
        X_est = triangulate_stereo(d_L, d_R, f_L, f_R)
        err = float(np.linalg.norm(X_est - X_true) * 1000)
        err_max = max(err_max, err)
    print(f"  [OK] triangulation projet/inverse (20 points) : erreur max {err_max:.4f} mm")
    assert err_max < 0.1, f"triangulation trop imprecise : {err_max:.4f} mm"

    # 2. reproject_error doit etre quasi-zero pour le ground truth
    X_true = np.array([0.01, 0.02, 0.05])
    uvw_L = (_projection_matrix(K, T_L) @ np.hstack([X_true, 1.0]))
    uv_L = uvw_L[:2] / uvw_L[2]
    d = Detection2D(cam_key="cam_0", label="x", center_px=(uv_L[0], uv_L[1]))
    f = Frame(cam_key="cam_0", image=np.zeros((10, 10, 3), dtype=np.uint8),
              K=K, dist=dist, T_base_cam=T_L)
    err = reproject_error(X_true, d, f)
    assert err < 0.01, f"reproj_error devrait etre ~0, recu {err}"
    print("  [OK] reproject_error sur ground truth ~0")

    # 3. PoseEstimator.build_scene : pipeline complet
    # load_scene_config=False pour test independant des configs reelles
    spec_meta = {"x": {"shape": "cube", "side_mm": 30.0}}
    est = PoseEstimator(specs_by_label=spec_meta, load_scene_config=False)
    # Ce test valide la GEOMETRIE de triangulation, qui doit etre independante de
    # la compensation de biais empirique (bias_correction.json). On la desactive
    # ici, sinon le biais courant (~-32mm en X) fait echouer le seuil <1mm.
    est._bias_m = None
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frames = {
        "cam_0": Frame(cam_key="cam_0", image=img, K=K, dist=dist, T_base_cam=T_L),
        "cam_1": Frame(cam_key="cam_1", image=img, K=K, dist=dist, T_base_cam=T_R),
    }
    # Refait detections pour un point connu
    X_ref = np.array([0.01, 0.02, 0.05])
    P_L_ref = _projection_matrix(K, T_L)
    P_R_ref = _projection_matrix(K, T_R)
    uv_L_ref = (P_L_ref @ np.hstack([X_ref, 1.0]))[:2] / (P_L_ref @ np.hstack([X_ref, 1.0]))[2]
    uv_R_ref = (P_R_ref @ np.hstack([X_ref, 1.0]))[:2] / (P_R_ref @ np.hstack([X_ref, 1.0]))[2]
    d_L_ref = Detection2D(cam_key="cam_0", label="x", center_px=(uv_L_ref[0], uv_L_ref[1]))
    d_R_ref = Detection2D(cam_key="cam_1", label="x", center_px=(uv_R_ref[0], uv_R_ref[1]))
    dets_by_cam = {"cam_0": [d_L_ref], "cam_1": [d_R_ref]}
    scene = est.build_scene(dets_by_cam, frames)
    assert len(scene.objects) == 1
    o = scene.objects[0]
    err_mm = float(np.linalg.norm(o.position_base_m - X_ref) * 1000)
    assert err_mm < 1.0, f"erreur build_scene = {err_mm:.3f} mm"
    assert o.meta["method"] == "stereo_triangulation"
    print(f"  [OK] PoseEstimator.build_scene : {scene.objects[0].label} a "
          f"({o.position_base_m[0] * 1000:.2f}, {o.position_base_m[1] * 1000:.2f}, "
          f"{o.position_base_m[2] * 1000:.2f}) mm")

    # 4. _in_workspace : rejette les coordonnees aberrantes (sans scene config)
    cfg = PoseEstimatorConfig()
    e = PoseEstimator(cfg, load_scene_config=False)
    assert e._in_workspace(np.array([0.10, 0.0, 0.05]))
    assert not e._in_workspace(np.array([0.0, 0.0, 1.5]))   # z trop grand
    assert not e._in_workspace(np.array([10.0, 0, 0.05]))   # hors atteinte
    print("  [OK] _in_workspace filtre les positions aberrantes")

    # 4b. Zones d'exclusion : test sur scene.json reelle (la base du robot)
    e_with_scene = PoseEstimator()  # charge configs/scene.json automatiquement
    if e_with_scene._exclusion_zones:
        # un point pile sur la base du robot (0, 0, 0.10) doit etre rejete
        in_robot = np.array([0.02, 0.0, 0.10])
        assert not e_with_scene._in_workspace(in_robot), \
            f"point ({in_robot}) sur la base robot devrait etre rejete par exclusion zone"
        # Un point clairement devant le robot (cube typique 30cm devant
        # base_link, pose sur table donc Z=+15mm = mi-hauteur d'un cube 30mm
        # avec table a Z=0, cf docs/REPERE_BASE.md)
        in_front = np.array([0.30, 0.0, 0.015])
        assert e_with_scene._in_workspace(in_front), \
            f"point ({in_front}) devant le robot devrait etre accepte"
        print(f"  [OK] Zones d'exclusion (scene.json) : "
              f"{len(e_with_scene._exclusion_zones)} zone(s) actives, "
              f"point sur robot rejete, point cube (30cm devant) accepte")
    else:
        print("  [SKIP] scene.json absent : test des zones d'exclusion saute")

    # 5. PoseEstimator detecte les cas mono uniquement (fallback)
    est2 = PoseEstimator(specs_by_label=spec_meta, load_scene_config=False)
    # Construit une detection mono avec un contour (carre projete)
    side_half = 0.015
    obj_pts = np.array([
        [-side_half, -side_half, 0.0],
        [+side_half, -side_half, 0.0],
        [+side_half, +side_half, 0.0],
        [-side_half, +side_half, 0.0],
    ])
    obj_pts_h = np.hstack([obj_pts, np.ones((4, 1))])
    uvw = (_projection_matrix(K, T_L) @ obj_pts_h.T).T
    uv_corners = (uvw[:, :2] / uvw[:, 2:3])
    contour = uv_corners.reshape(-1, 2).astype(np.float32)
    d_mono = Detection2D(cam_key="cam_0", label="x",
                         center_px=(uv_corners.mean(axis=0).tolist()),
                         contour=contour, area_px=1000.0)
    scene2 = est2.build_scene({"cam_0": [d_mono]}, {"cam_0": frames["cam_0"]})
    # PnP doit donner une estimation proche de X_true (qui est 0,0,0 ici)
    assert len(scene2.objects) >= 1
    o2 = scene2.objects[0]
    assert o2.meta["method"].startswith("pnp_mono")
    err = float(np.linalg.norm(o2.position_base_m) * 1000)
    print(f"  [OK] Fallback PnP mono : objet a "
          f"({o2.position_base_m[0] * 1000:.1f}, {o2.position_base_m[1] * 1000:.1f}, "
          f"{o2.position_base_m[2] * 1000:.1f}) mm (erreur {err:.2f} mm)")

    # 6. Sommet triangule : cameras OBLIQUES (~50 deg) + objet vertical connu.
    # C'est la config realiste du poste (cam_0/cam_1 en plongee) ou le proxy
    # 'hauteur pixels' echoue quand l'objet pointe vers les cameras.
    def make_lookat(pos, target):
        pos = np.asarray(pos, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        z = target - pos
        z = z / np.linalg.norm(z)                      # axe optique (OpenCV)
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)                             # y image vers le bas
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
        return T

    T_L2 = make_lookat([0.05, +0.06, 0.30], [0.25, 0.0, 0.0])
    T_R2 = make_lookat([0.05, -0.06, 0.30], [0.25, 0.0, 0.0])
    f_L2 = Frame(cam_key="cam_0", image=img, K=K, dist=dist, T_base_cam=T_L2)
    f_R2 = Frame(cam_key="cam_1", image=img, K=K, dist=dist, T_base_cam=T_R2)

    def proj(T_cam, X3):
        uvw = _projection_matrix(K, T_cam) @ np.hstack([X3, 1.0])
        return uvw[:2] / uvw[2]

    X_center = np.array([0.25, 0.0, 0.03])
    X_top_true = np.array([0.25, 0.0, 0.06])
    dets_top = []
    for f2 in (f_L2, f_R2):
        uc = proj(f2.T_base_cam, X_center)
        ut = proj(f2.T_base_cam, X_top_true)
        dets_top.append(Detection2D(
            cam_key=f2.cam_key, label="x", center_px=(uc[0], uc[1]),
            bbox=(ut[0] - 20, ut[1], ut[0] + 20, ut[1] + 90)))
    est3 = PoseEstimator(load_scene_config=False)
    X_top_est = est3._triangulate_bbox_top(dets_top[0], dets_top[1], f_L2, f_R2)
    assert X_top_est is not None
    errz = abs(X_top_est[2] - X_top_true[2]) * 1000
    print(f"  [OK] _triangulate_bbox_top : Z sommet = {X_top_est[2]*1000:.1f} mm "
          f"(vrai 60.0, erreur {errz:.2f} mm)")
    assert errz < 2.0, f"Z sommet trop faux : {errz:.2f} mm"

    # 7. Yaw d'empreinte en repere BASE : rectangle 90x30 mm tourne de +30 deg
    # autour de Z, pose sur table (plan z=15mm a mi-hauteur). La projection
    # rayon-plan doit retrouver le yaw VRAI malgre la camera oblique.
    yaw_true = np.radians(30.0)
    c30, s30 = np.cos(yaw_true), np.sin(yaw_true)
    corners_local = np.array([
        [-0.045, -0.015], [0.0, -0.015], [+0.045, -0.015], [+0.045, 0.0],
        [+0.045, +0.015], [0.0, +0.015], [-0.045, +0.015], [-0.045, 0.0],
    ])
    contour_px = []
    for lx, ly in corners_local:
        wx = 0.25 + c30 * lx - s30 * ly
        wy = 0.00 + s30 * lx + c30 * ly
        contour_px.append(proj(T_L2, np.array([wx, wy, 0.015])))
    det_rect = Detection2D(cam_key="cam_0", label="r", center_px=(0, 0),
                           contour=np.array(contour_px))
    ori = est3._footprint_orientation(det_rect, f_L2, z_plane_m=0.015)
    assert ori is not None
    yaw_est, elong, ext_l, ext_c = ori
    print(f"  [OK] _footprint_orientation : yaw = {np.degrees(yaw_est):+.1f} deg "
          f"(vrai +30), elongation {elong:.2f}, empreinte "
          f"{ext_l*1000:.0f}x{ext_c*1000:.0f} mm (vraie 90x30)")
    assert abs(np.degrees(yaw_est) - 30.0) < 4.0
    assert elong > 2.0
    assert abs(ext_l - 0.090) < 0.012 and abs(ext_c - 0.030) < 0.010

    # 8. GARDE-FOU anti-FORESHORTENING : un cylindre DEBOUT reste 'debout' ; un
    #    cylindre COUCHE qui FUIT vers les cameras (// X, "sommet" de bbox = bout
    #    lointain) N'est PAS classe debout (sinon hauteur = longueur -> prise trop
    #    haute). Distinction = offset horizontal sommet-vs-centroide.
    est3._bias_m = None  # test dans un repere coherent (X_top et X_base non biaises)

    def _dets_haute(centroid, apex, halfw_px, h_px):
        out = []
        for f2 in (f_L2, f_R2):
            c = proj(f2.T_base_cam, centroid)
            a = proj(f2.T_base_cam, apex)
            out.append(Detection2D(
                cam_key=f2.cam_key, label="cyl", center_px=(c[0], c[1]),
                bbox=(a[0] - halfw_px, a[1], a[0] + halfw_px, a[1] + h_px)))
        return out
    # DEBOUT : sommet A LA VERTICALE du centroide -> garde 'debout'
    dL, dR = _dets_haute([0.25, 0.0, 0.03], [0.25, 0.0, 0.06], 50, 180)
    _, m_deb = est3._estimate_geometry(dL, dR, f_L2, f_R2, np.array([0.25, 0.0, 0.03]))
    assert m_deb.get("pose_class") == "debout", \
        f"un vrai debout doit rester debout, recu {m_deb.get('pose_class')}"
    # COUCHE // X : 'sommet' = bout lointain (offset ~30mm) -> PAS debout, hauteur ~ diametre
    dL, dR = _dets_haute([0.25, 0.0, 0.0125], [0.22, 0.0, 0.025], 50, 180)
    b_cou, m_cou = est3._estimate_geometry(dL, dR, f_L2, f_R2, np.array([0.25, 0.0, 0.0125]))
    assert m_cou.get("pose_class") != "debout", \
        f"un cylindre couche // X ne doit PAS etre classe debout, recu {m_cou.get('pose_class')}"
    assert b_cou[2] < 0.035, \
        f"hauteur couche // X attendue ~diametre (<35mm), recu {b_cou[2]*1000:.0f}mm"
    print(f"  [OK] garde-fou foreshortening : debout garde, couche // X reclasse "
          f"(hauteur {b_cou[2]*1000:.0f}mm = diametre, offset {m_cou.get('debout_rejete_offset_mm')}mm)")

    print("Tous les tests passent.")
