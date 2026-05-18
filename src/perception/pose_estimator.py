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
                in_ws = self._in_workspace(X)
                pos_mm = X * 1000
                if not in_ws:
                    reject_reason = (
                        f"stereo OK (reproj {err:.1f}px) MAIS position hors workspace "
                        f"({pos_mm[0]:+.0f},{pos_mm[1]:+.0f},{pos_mm[2]:+.0f}) mm"
                    )
                elif err > self.config.max_reproj_error_px:
                    reject_reason = (
                        f"stereo OK (pos {pos_mm[0]:+.0f},{pos_mm[1]:+.0f},{pos_mm[2]:+.0f} mm) "
                        f"MAIS reproj_err={err:.1f}px > seuil {self.config.max_reproj_error_px}px"
                    )
                else:
                    score = float(min(det_L.score, det_R.score) *
                                  np.exp(-err / 8.0))
                    return ObjectInstance(
                        label=label,
                        position_base_m=X,
                        source_detections=[det_L, det_R],
                        score=score,
                        meta={
                            "method": "stereo_triangulation",
                            "reproj_error_px": err,
                            "reproj_error_per_cam_px": {kL: err_L, kR: err_R},
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
        x, y, z = float(X[0]), float(X[1]), float(X[2])
        # Bornes Z par defaut (de la config code)
        if not (self.config.min_z_base_m <= z <= self.config.max_z_base_m):
            return False
        # SO-101 a un bras de ~30 cm : tout ce qui est au-dela de 1 m est aberrant.
        if (x * x + y * y + z * z) > 1.0:
            return False
        # Bornes workspace de scene.json (si dispo)
        if self._workspace_bounds is not None:
            wb = self._workspace_bounds
            if not (wb["x_min"] <= x <= wb["x_max"]): return False
            if not (wb["y_min"] <= y <= wb["y_max"]): return False
            if not (wb["z_min"] <= z <= wb["z_max"]): return False
        # Zones d'exclusion : la position ne doit etre dans AUCUNE zone.
        for zone in self._exclusion_zones:
            if self._point_in_zone((x, y, z), zone):
                return False
        return True

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

    print("Tous les tests passent.")
