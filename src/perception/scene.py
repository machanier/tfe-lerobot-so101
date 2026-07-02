"""
scene.py - Types de donnees de la perception.

Definit les `dataclasses` qui circulent dans le pipeline perception :

    Frame          : image RGB + intrinseques + pose camera-base + timestamp.
    Detection2D    : detection brute dans le plan image (classe, bbox, masque, score).
    ObjectInstance : objet reconstruit en 3D dans le repere base du robot.
    Scene          : etat instantane = liste d'instances + timestamp.

Tous les vecteurs 3D sont en metres, dans le repere base du SO-101.
Toutes les rotations sont des matrices 3x3 (orientees, det = +1).

Convention de nommage : T_A_B = pose du repere B exprimee dans le repere A
(matrice 4x4 SE(3)), coherente avec les modules calibration/.

Cette frontiere de types est consommee par le module de planification.
Modifier la signature de `ObjectInstance` casse la compatibilite avec les
modules aval ; preferer l'ajout de champs optionnels.

Reference : Bohg et al. 2014 "Data-Driven Grasp Synthesis - A Survey",
section 2 : taxonomie des representations d'objets pour le grasp planning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# Entrees de la perception : images calibrees synchronisees
# ============================================================


@dataclass
class Frame:
    """Image calibree d'une camera a un instant donne.

    Le champ `T_base_cam` est calcule par le module camera_io (eye-to-hand :
    constant, lu depuis handeye_cam_*.json ; eye-in-hand : compose avec la FK
    courante du robot). Le pipeline aval n'a donc pas a se preoccuper de la
    configuration optique de la camera.

    Attributes:
        cam_key       : identifiant logique ("cam_0", "cam_1", "cam_2").
        image         : image BGR (cv2 standard), shape (H, W, 3), uint8.
        K             : matrice intrinseque 3x3.
        dist          : coefficients de distorsion (5,) ou (8,) au format OpenCV.
        T_base_cam    : pose 4x4 de la camera dans le repere base du robot (m).
        timestamp     : temps de capture (epoch seconds, float).
    """

    cam_key: str
    image: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    T_base_cam: np.ndarray
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Frame.image doit etre HxWx3 BGR, recu {self.image.shape}")
        if self.K.shape != (3, 3):
            raise ValueError(f"Frame.K doit etre 3x3, recu {self.K.shape}")
        if self.T_base_cam.shape != (4, 4):
            raise ValueError(f"Frame.T_base_cam doit etre 4x4, recu {self.T_base_cam.shape}")


# ============================================================
# Sorties du detecteur (plan image, 2D)
# ============================================================


@dataclass
class Detection2D:
    """Une detection 2D dans le plan image d'une camera.

    Sortie du detecteur (HSV, YOLO, OWL-ViT, ...). Toutes les detections d'une
    meme image sont collectees puis passees au pose_estimator pour
    reconstruction 3D.

    Attributes:
        cam_key   : camera d'ou provient la detection.
        label     : nom de l'objet detecte ("red_cube", "pen", ...).
        center_px : centre du contour/bbox en pixels (u, v).
        bbox      : (u_min, v_min, u_max, v_max) ou None si pas de bbox.
        contour   : contour 2D (N, 2) en pixels, ou None si non disponible.
        mask      : masque binaire (H, W) booleen, ou None.
        area_px   : aire du contour/masque en pixels (utile pour rejeter le bruit).
        score     : confiance dans [0, 1].
        meta      : metadonnees libres (e.g. couleur dominante, descripteur).
    """

    cam_key: str
    label: str
    center_px: tuple[float, float]
    bbox: Optional[tuple[float, float, float, float]] = None
    contour: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    area_px: float = 0.0
    score: float = 1.0
    meta: dict = field(default_factory=dict)


# ============================================================
# Sortie du pipeline complet : etat de la scene en 3D
# ============================================================


@dataclass
class ObjectInstance:
    """Objet reconstruit en 3D, dans le repere base du robot.

    Sortie du pose_estimator. Les etapes suivantes (planification, saisie)
    consomment cette structure et ne reviennent jamais au plan image.

    Attributes:
        label              : nom de l'objet (meme convention que Detection2D.label).
        position_base_m    : centroide 3D (x, y, z) en metres, repere BASE.
        position_cov_mm    : covariance 3x3 (mm^2) de l'estimation. Permet
                             d'estimer une zone de fiabilite pour le planning.
        bbox_3d_m          : (dx, dy, dz) extent estime ou None. En metres,
                             aligne sur les axes du repere base (AABB approx).
        orientation_R      : rotation 3x3 de l'objet (repere base -> repere objet),
                             ou None si l'orientation n'est pas estimee.
        source_detections  : liste des Detection2D ayant permis l'estimation.
                             Tracable pour le debogage et pour le mode replay.
        score              : confiance globale dans [0, 1].
        timestamp          : moment de la detection.
        meta               : metadonnees libres.
    """

    label: str
    position_base_m: np.ndarray
    position_cov_mm: Optional[np.ndarray] = None
    bbox_3d_m: Optional[tuple[float, float, float]] = None
    orientation_R: Optional[np.ndarray] = None
    source_detections: list[Detection2D] = field(default_factory=list)
    score: float = 1.0
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.position_base_m = np.asarray(self.position_base_m, dtype=np.float64).reshape(3)


@dataclass
class Scene:
    """Etat instantane de la scene = liste d'objets + obstacles + timestamp.

    Bohg 2014 distingue cibles de saisie et obstacles ; on conserve ici la meme
    distinction pour preparer l'etape de planification.

    Attributes:
        objects    : liste des ObjectInstance dont au moins une estimation 3D est valide.
        obstacles  : liste d'ObjectInstance consideres comme non-saisissables (table,
                     boite de depose, distracteurs). Vide par defaut a ce stade.
        timestamp  : horodatage de la Scene (moyenne des timestamps des frames).
        meta       : metadonnees (configuration de capture, etc.).
    """

    objects: list[ObjectInstance] = field(default_factory=list)
    obstacles: list[ObjectInstance] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def pretty(self) -> str:
        """Representation textuelle compacte de la scene (utile pour le debogage)."""
        lines = [f"Scene @ t={self.timestamp:.3f}"]
        for o in self.objects:
            p = o.position_base_m * 1000.0
            lines.append(
                f"  {o.label:<14} pos=({p[0]:+7.1f}, {p[1]:+7.1f}, {p[2]:+7.1f}) mm  "
                f"score={o.score:.2f}  n_views={len(o.source_detections)}"
            )
        if self.obstacles:
            lines.append(f"  ({len(self.obstacles)} obstacles)")
        return "\n".join(lines)


# ============================================================
# Tests unitaires (lancer avec : python -m src.perception.scene)
# ============================================================
if __name__ == "__main__":
    print("Tests scene.py")

    # 1. Frame : validation des shapes
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    K = np.eye(3)
    dist = np.zeros(5)
    T = np.eye(4)
    f = Frame(cam_key="cam_0", image=img, K=K, dist=dist, T_base_cam=T)
    assert f.cam_key == "cam_0"
    print("  [OK] Frame: construction valide")

    try:
        Frame(cam_key="x", image=np.zeros((10, 10)), K=K, dist=dist, T_base_cam=T)
        raise AssertionError("aurait du lever ValueError")
    except ValueError:
        print("  [OK] Frame: shape image invalide -> ValueError")

    # 2. Detection2D + ObjectInstance + Scene
    d = Detection2D(cam_key="cam_0", label="red_cube", center_px=(100, 200), area_px=1500.0)
    o = ObjectInstance(
        label="red_cube",
        position_base_m=np.array([0.20, -0.05, 0.03]),
        source_detections=[d],
        score=0.92,
    )
    assert o.position_base_m.shape == (3,)
    s = Scene(objects=[o])
    txt = s.pretty()
    assert "red_cube" in txt
    assert "200.0" in txt  # 0.20 m -> 200.0 mm
    print("  [OK] Detection2D + ObjectInstance + Scene")

    print("Tous les tests passent.")
