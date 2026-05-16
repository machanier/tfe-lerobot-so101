"""
detector.py - Detection 2D d'objets dans les frames camera.

Definit l'interface `ObjectDetector` (ABC) et deux implementations :

    HSVDetector  (V1, deterministe)
        Detecte les primitives colorees par seuillage HSV + analyse de
        contours. Permet d'isoler la contribution de la geometrie (calibration
        + triangulation) avant d'introduire les incertitudes d'un detecteur
        appris. Reproductible, sans dependance ML.
        Reference : Forsyth & Ponce, "Computer Vision: A Modern Approach",
        ch. 6 (color-based segmentation).

    HFDetector   (V2, stub - extension prevue)
        Wrapper pour un detecteur open-vocabulary issu de la stack Hugging
        Face (OWL-ViT, Grounding-DINO). Coherent avec l'ecosysteme LeRobot
        sur lequel le projet est base. L'implementation est laissee pour
        plus tard : on a juste l'interface pour que le pipeline reste
        ouvert. Voir docs/PROJECT_STATUS.md, section "Roadmap".

Toutes les sorties sont des `Detection2D` (cf src/perception/scene.py),
dans le repere image de la camera d'origine.

Convention de label : un detecteur peut renvoyer des objets de plusieurs
classes. Les classes sont parametrees via `ObjectSpec`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.perception.scene import Detection2D, Frame


# ============================================================
# Specifications des objets a detecter
# ============================================================


@dataclass
class HSVRange:
    """Plage de seuillage HSV avec gestion des couleurs achromatiques.

    OpenCV : H in [0, 179], S et V dans [0, 255].

    color_mode :
      - "chromatic" : la couleur a une teinte significative (rouge, bleu,
        vert, violet, ...). On seuille H + S + V. Cas standard pour les
        primitives colorees.
      - "black"     : LA TEINTE N'A PAS DE SENS pour le noir (V proche de 0).
        On seuille uniquement V <= v_hi (typiquement < 60). H et S ignores.
      - "white"     : pas de saturation. On seuille S <= s_hi (typiquement
        < 30) ET V >= v_lo (typiquement > 200). H ignore.
      - "gray"      : peu de saturation, V intermediaire. On seuille S <= s_hi
        ET v_lo <= V <= v_hi.

    Cette distinction est CRITIQUE : sans elle, "noir" et "blanc" calibres
    en mode chromatic captent toutes les teintes a la fois, ce qui
    provoque des confusions massives entre objets sombres (noir/bleu/violet)
    ou clairs (blanc/objets pales). Voir docs/PROJECT_STATUS.md D8.

    Pour le rouge qui chevauche la couture H=0/179, on definit DEUX plages
    chromatic via `hue_extra_lo`/`hue_extra_hi`.

    Attributes:
        color_mode : voir ci-dessus ("chromatic" par defaut pour compat).
        h_lo, h_hi : bornes teinte (chromatic uniquement).
        s_lo, s_hi : bornes saturation. Pour "white"/"gray", on utilise s_hi.
        v_lo, v_hi : bornes valeur. Pour "black", on utilise v_hi. Pour
                     "white", v_lo.
        hue_extra_lo, hue_extra_hi : 2eme plage H pour rouge wrap-around.
    """

    h_lo: int = 0
    h_hi: int = 179
    s_lo: int = 0
    s_hi: int = 255
    v_lo: int = 0
    v_hi: int = 255
    hue_extra_lo: Optional[int] = None
    hue_extra_hi: Optional[int] = None
    color_mode: str = "chromatic"

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Renvoie un masque uint8 (H, W) avec 255 ou les pixels matchent."""
        if self.color_mode == "black":
            # V bas suffit. H et S ignores.
            lower = np.array([0, 0, 0], dtype=np.uint8)
            upper = np.array([179, 255, self.v_hi], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        if self.color_mode == "white":
            # S bas + V haut. H ignore.
            lower = np.array([0, 0, self.v_lo], dtype=np.uint8)
            upper = np.array([179, self.s_hi, 255], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        if self.color_mode == "gray":
            # S bas + V intermediaire. H ignore.
            lower = np.array([0, 0, self.v_lo], dtype=np.uint8)
            upper = np.array([179, self.s_hi, self.v_hi], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        # chromatic (cas par defaut)
        lower = np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8)
        upper = np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        if self.hue_extra_lo is not None and self.hue_extra_hi is not None:
            lower2 = np.array([self.hue_extra_lo, self.s_lo, self.v_lo], dtype=np.uint8)
            upper2 = np.array([self.hue_extra_hi, self.s_hi, self.v_hi], dtype=np.uint8)
            m2 = cv2.inRange(hsv, lower2, upper2)
            m = cv2.bitwise_or(m, m2)
        return m


@dataclass
class ObjectSpec:
    """Description d'une classe d'objets a detecter par HSV.

    Attributes:
        label        : nom logique ("red_cube", "blue_cylinder", ...).
        hsv          : plage HSV principale.
        min_area_px  : aire minimale du contour (rejet bruit). Ajuster selon
                       la distance camera-objet.
        max_area_px  : aire maximale (rejette les "blobs" qui prennent toute
                       l'image, e.g. fond colore mal eclaire).
        meta         : metadonnees libres (e.g. forme attendue, taille reelle
                       en mm pour PnP monoculaire).
    """

    label: str
    hsv: HSVRange
    min_area_px: float = 300.0
    max_area_px: float = 1.0e6
    meta: dict = field(default_factory=dict)


# ============================================================
# Interface abstraite
# ============================================================


class ObjectDetector(ABC):
    """Interface : prend une Frame, retourne une liste de Detection2D.

    Une instance d'`ObjectDetector` est *stateless cote scene* (elle peut
    avoir des params internes / poids) : appeler `detect(frame)` plusieurs
    fois doit donner exactement le meme resultat.

    Cette interface garantit que swap-er `HSVDetector` pour `HFDetector`
    plus tard ne casse aucun consommateur (run_perception.py, check_perception.py).
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection2D]:
        """Detecte tous les objets connus dans `frame`."""
        ...

    def detect_multi(self, frames: dict[str, Optional[Frame]]
                     ) -> dict[str, list[Detection2D]]:
        """Detecte sur plusieurs cameras. Cameras absentes -> liste vide."""
        out: dict[str, list[Detection2D]] = {}
        for k, f in frames.items():
            out[k] = self.detect(f) if f is not None else []
        return out

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom court du detecteur, pour le logging."""
        ...


# ============================================================
# Implementation V1 : HSV + contours
# ============================================================


class HSVDetector(ObjectDetector):
    """Detecteur deterministe basé sur seuillage HSV + analyse de contours.

    Pipeline par image :
      1. BGR -> HSV.
      2. Pour chaque ObjectSpec : masque HSV puis morphologie (open + close).
      3. cv2.findContours, on garde ceux dont l'aire est dans [min, max].
      4. Pour chaque contour : centre = moments, bbox, score = aire normalisée.
      5. Optionnel : retient seulement les `top_k` plus grands contours par
         classe (par defaut 1 = on suppose un exemplaire visible par classe).

    Choix de design :
      - On RETOURNE le contour entier (utile pour debug + pour passer a un
        estimateur de pose plus fin si on veut).
      - On NE renvoie PAS de masque par defaut (coute en memoire). Activable
        via `emit_mask=True` pour les besoins de debug.
      - L'image cible peut etre une crop downscaled : pour la V1 on opere
        plein resolution pour ne pas perdre les petits objets.

    Calibration des couleurs : Maxence enregistre des echantillons via
    `scripts/calibrate_hsv.py` (a venir) qui ecrit configs/perception/hsv_*.json.
    """

    def __init__(self, specs: list[ObjectSpec], *,
                 morph_kernel: int = 5,
                 top_k_per_label: int = 1,
                 emit_mask: bool = False):
        if not specs:
            raise ValueError("HSVDetector: au moins une ObjectSpec requise.")
        self.specs = specs
        self.morph_kernel = morph_kernel
        self.top_k_per_label = top_k_per_label
        self.emit_mask = emit_mask
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    @property
    def name(self) -> str:
        return f"HSVDetector(n={len(self.specs)})"

    # ----- API ------------------------------------------------------------

    def detect(self, frame: Frame) -> list[Detection2D]:
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        detections: list[Detection2D] = []
        for spec in self.specs:
            detections.extend(self._detect_one_spec(frame, hsv, spec))
        return detections

    # ----- interne --------------------------------------------------------

    def _detect_one_spec(self, frame: Frame, hsv: np.ndarray, spec: ObjectSpec
                         ) -> list[Detection2D]:
        mask = spec.hsv.mask(hsv)
        # Morphologie : open (retire petits points), close (bouche les trous)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [(c, cv2.contourArea(c)) for c in contours]
        valid = [(c, a) for c, a in valid if spec.min_area_px <= a <= spec.max_area_px]
        # tri decroissant par aire ; on garde les top_k
        valid.sort(key=lambda ca: -ca[1])
        valid = valid[: self.top_k_per_label]

        out: list[Detection2D] = []
        for cnt, area in valid:
            M = cv2.moments(cnt)
            if M["m00"] <= 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(cnt)
            # Score : aire normalisee par max_area, clip [0, 1]
            score = float(np.clip(area / max(spec.max_area_px, 1e-6), 0.05, 1.0))
            det = Detection2D(
                cam_key=frame.cam_key,
                label=spec.label,
                center_px=(float(cx), float(cy)),
                bbox=(float(x), float(y), float(x + w), float(y + h)),
                contour=cnt.reshape(-1, 2),
                mask=mask.astype(bool).copy() if self.emit_mask else None,
                area_px=float(area),
                score=score,
                meta={"detector": "HSVDetector"},
            )
            out.append(det)
        return out


# ============================================================
# Stub V2 : detecteur Hugging Face (extension prevue)
# ============================================================


class HFDetector(ObjectDetector):
    """Detecteur open-vocabulary base sur OWL-ViTv2 (Hugging Face).

    Strategie : on donne une LISTE de labels EN TEXTE NATUREL ("orange cube",
    "robot arm", ...) et le modele renvoie pour chaque label une liste de
    bboxes + scores. Le modele a ete entraine sur des millions d'images +
    legendes : il peut distinguer le cube orange du bras orange du robot
    par la FORME et le CONTEXTE, pas seulement la couleur.

    Difference cle avec HSVDetector :
      - HSVDetector : "il y a des pixels oranges ici" (sans savoir ce que c'est).
      - HFDetector  : "il y a un 'orange cube' ici (proba 0.92) ET un
        'robot arm' la (proba 0.88)" -- distingue les categories semantiques.

    Coherence stack : utilise la lib `transformers` de Hugging Face, qui
    est la meme que celle utilisee par LeRobot pour ses policies (SmolVLA,
    ACT, etc.). Pas de fragmentation de dependances.

    Sortie : Detection2D avec la meme convention que HSVDetector. Le pseudo-
    contour est constitue des 4 coins de la bbox (utilisable par le grasp
    planner top-down ; pour une orientation fine du wrist_roll, il faudra
    ajouter une segmentation a poste, mais ce n'est pas requis en V1).

    Reference : Minderer et al. 2023, "Scaling Open-Vocabulary Object
    Detection", NeurIPS, arxiv:2306.09683.
    """

    def __init__(self, prompt_labels: list[str], *,
                 model_name: str = "google/owlv2-base-patch16-ensemble",
                 score_threshold: float = 0.15,
                 device: Optional[str] = None,
                 verbose: bool = True):
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            import torch
        except ImportError as e:
            raise ImportError(
                "HFDetector necessite transformers, torch, pillow. "
                "Installe avec :\n  pip install transformers torch pillow\n"
                f"Erreur originale : {e}"
            ) from e

        if not prompt_labels:
            raise ValueError("HFDetector : au moins un label requis.")
        self.prompt_labels = list(prompt_labels)
        self.model_name = model_name
        self.score_threshold = float(score_threshold)

        # Detection device : MPS (Apple Silicon GPU) > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if verbose:
            print(f"[HFDetector] Chargement {model_name} sur {device}...")
        self._processor = Owlv2Processor.from_pretrained(model_name)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
        self._model.eval()
        self._torch = torch  # garde reference pour torch.no_grad()
        if verbose:
            print(f"[HFDetector] Pret. {len(self.prompt_labels)} labels actifs : "
                  f"{self.prompt_labels}")

    @property
    def name(self) -> str:
        return f"HFDetector({self.model_name})"

    def detect(self, frame: Frame) -> list[Detection2D]:
        from PIL import Image
        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        with self._torch.no_grad():
            inputs = self._processor(
                text=[self.prompt_labels], images=pil, return_tensors="pt"
            ).to(self.device)
            outputs = self._model(**inputs)

        # Post-process : convertit en bboxes (xyxy en pixels), scores, labels
        target_sizes = self._torch.tensor([(pil.height, pil.width)]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]

        detections: list[Detection2D] = []
        for score, label_idx, box in zip(results["scores"],
                                          results["labels"],
                                          results["boxes"]):
            label = self.prompt_labels[int(label_idx)]
            x0, y0, x1, y1 = [float(v) for v in box.cpu().numpy()]
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            # Pseudo-contour rectangulaire (4 coins de la bbox).
            # Permet au grasp planner d'estimer un yaw approximatif via
            # l'aspect ratio bbox, en attendant une segmentation reelle.
            contour = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32,
            )
            d = Detection2D(
                cam_key=frame.cam_key,
                label=label,
                center_px=(cx, cy),
                bbox=(x0, y0, x1, y1),
                contour=contour,
                area_px=float((x1 - x0) * (y1 - y0)),
                score=float(score),
                meta={
                    "detector": "HFDetector",
                    "model": self.model_name,
                    "device": self.device,
                },
            )
            detections.append(d)
        return detections


def load_hf_specs(path: Optional[Path] = None) -> dict:
    """Charge la config HFDetector depuis configs/perception/hf_specs.json."""
    path = path or (REPO / "configs" / "perception" / "hf_specs.json")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"hf_specs.json introuvable : {path}\n"
            "Cree-le ou utilise les valeurs par defaut de default_hf_labels()."
        )
    return json.load(open(path))


def default_hf_labels() -> list[str]:
    """Labels par defaut pour HFDetector (cas tes 9 objets + robot).

    Convention : en anglais (les modeles HF sont entraines majoritairement
    en anglais, meilleure precision).
    """
    return [
        "orange cube",
        "blue rectangular box",
        "purple cylinder",
        "black triangular prism",
        "white mug",
        "pen",
        "rubiks cube",
        "tissue box",
        "tall plastic cup",
        "robot arm",
    ]


# ============================================================
# Helpers de chargement (specs depuis JSON)
# ============================================================


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPECS_PATH = REPO / "configs" / "perception" / "hsv_specs.json"


def load_hsv_specs(path: Optional[Path] = None) -> list[ObjectSpec]:
    """Charge la liste d'ObjectSpec depuis un fichier JSON.

    Format attendu (`configs/perception/hsv_specs.json`) :

        {
          "specs": [
            {
              "label": "red_cube",
              "h_lo": 0, "h_hi": 10, "hue_extra_lo": 170, "hue_extra_hi": 179,
              "s_lo": 100, "s_hi": 255, "v_lo": 50, "v_hi": 255,
              "min_area_px": 500, "max_area_px": 200000,
              "meta": {"shape": "cube", "side_mm": 30.0}
            },
            ...
          ]
        }
    """
    path = path or DEFAULT_SPECS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Specs HSV introuvables : {path}\n"
            "Genere un fichier via scripts/calibrate_hsv.py (a venir), ou "
            "utilise default_hsv_specs() pour partir des plages standards."
        )
    data = json.load(open(path))
    out = []
    for s in data.get("specs", []):
        hsv = HSVRange(
            h_lo=int(s.get("h_lo", 0)), h_hi=int(s.get("h_hi", 179)),
            s_lo=int(s.get("s_lo", 0)), s_hi=int(s.get("s_hi", 255)),
            v_lo=int(s.get("v_lo", 0)), v_hi=int(s.get("v_hi", 255)),
            hue_extra_lo=s.get("hue_extra_lo"), hue_extra_hi=s.get("hue_extra_hi"),
            color_mode=str(s.get("color_mode", "chromatic")),
        )
        out.append(ObjectSpec(
            label=str(s["label"]), hsv=hsv,
            min_area_px=float(s.get("min_area_px", 300)),
            max_area_px=float(s.get("max_area_px", 1.0e6)),
            meta=s.get("meta", {}),
        ))
    return out


def default_hsv_specs() -> list[ObjectSpec]:
    """Plages HSV de depart pour les 4 primitives colorees attendues.

    Ces valeurs sont des MOYENNES indicatives ; la calibration finale doit
    se faire avec scripts/calibrate_hsv.py sous l'eclairage reel du poste.
    """
    return [
        # Rouge (chevauche H=0)
        ObjectSpec(
            label="red_cube",
            hsv=HSVRange(h_lo=0, h_hi=10, s_lo=100, s_hi=255, v_lo=60, v_hi=255,
                         hue_extra_lo=170, hue_extra_hi=179),
            min_area_px=500.0, max_area_px=200000.0,
            meta={"shape": "cube"},
        ),
        ObjectSpec(
            label="green_cylinder",
            hsv=HSVRange(h_lo=40, h_hi=85, s_lo=80, s_hi=255, v_lo=50, v_hi=255),
            min_area_px=500.0, max_area_px=200000.0,
            meta={"shape": "cylinder"},
        ),
        ObjectSpec(
            label="blue_triangle",
            hsv=HSVRange(h_lo=100, h_hi=130, s_lo=100, s_hi=255, v_lo=50, v_hi=255),
            min_area_px=500.0, max_area_px=200000.0,
            meta={"shape": "triangle_prism"},
        ),
        ObjectSpec(
            label="yellow_rectangle",
            hsv=HSVRange(h_lo=20, h_hi=35, s_lo=120, s_hi=255, v_lo=80, v_hi=255),
            min_area_px=500.0, max_area_px=200000.0,
            meta={"shape": "rect_prism"},
        ),
    ]


# ============================================================
# Self-tests (lance avec : python -m src.perception.detector)
# ============================================================
if __name__ == "__main__":
    print("Tests detector.py")

    # 1. Construit une image synthetique avec un rectangle rouge
    img = np.full((400, 600, 3), 80, dtype=np.uint8)  # fond gris
    cv2.rectangle(img, (150, 100), (250, 200), (0, 0, 220), thickness=-1)  # BGR rouge
    cv2.rectangle(img, (400, 250), (470, 320), (0, 200, 0), thickness=-1)  # BGR vert

    K = np.array([[600, 0, 300], [0, 600, 200], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    T = np.eye(4)
    frame = Frame(cam_key="cam_test", image=img, K=K, dist=dist, T_base_cam=T)

    # 2. Detecteur sur les 4 primitives par defaut
    det = HSVDetector(default_hsv_specs())
    detections = det.detect(frame)
    labels = sorted(d.label for d in detections)
    assert "red_cube" in labels, f"red_cube manquant: {labels}"
    assert "green_cylinder" in labels, f"green_cylinder manquant: {labels}"
    print(f"  [OK] HSVDetector sur image synthetique : detecte {labels}")

    # 3. Centre du rectangle rouge ~= (200, 150)
    red = next(d for d in detections if d.label == "red_cube")
    cx, cy = red.center_px
    assert abs(cx - 200) < 5 and abs(cy - 150) < 5, f"centre rouge incorrect: ({cx}, {cy})"
    print(f"  [OK] Centroide rouge ({cx:.1f}, {cy:.1f}) ~ (200, 150)")

    # 4. detect_multi : cameras manquantes -> liste vide
    out = det.detect_multi({"cam_a": frame, "cam_b": None})
    assert len(out["cam_a"]) >= 2
    assert out["cam_b"] == []
    print("  [OK] detect_multi gere les frames None")

    # 5. ABC : on ne peut pas instancier ObjectDetector
    try:
        ObjectDetector()  # type: ignore[abstract]
        raise AssertionError("aurait du lever TypeError")
    except TypeError:
        print("  [OK] ObjectDetector est abstrait (TypeError)")

    # 6. HFDetector : ne tente l'init que si transformers est installe
    try:
        import transformers  # noqa: F401
        # Si on est la, on peut tenter une mini init (mais on skip le modele
        # reel pour ne pas telecharger 600 Mo a chaque self-test).
        # On verifie juste que les imports fonctionnent + ValueError sur labels vides.
        try:
            HFDetector([])  # type: ignore[arg-type]
            raise AssertionError("aurait du lever ValueError")
        except ValueError:
            print("  [OK] HFDetector : ValueError sur labels vides")
        print("  [SKIP] HFDetector : import transformers OK, "
              "test live skippe (necessiterait 600 Mo de telechargement)")
    except ImportError:
        print("  [SKIP] HFDetector : transformers/torch non installes "
              "(pip install transformers torch pillow pour activer)")

    # 7. default_hsv_specs : 4 primitives
    specs = default_hsv_specs()
    assert len(specs) == 4
    assert {s.label for s in specs} == {"red_cube", "green_cylinder",
                                         "blue_triangle", "yellow_rectangle"}
    print("  [OK] default_hsv_specs : 4 primitives colorees")

    # 8. color_mode = "black" : ne capte que les pixels sombres, INDEPENDAMMENT de H
    img_blk = np.zeros((100, 100, 3), dtype=np.uint8)  # noir pur
    # BGR (40, 5, 5) : presque noir, legerement bleute. V_HSV ≈ 40
    img_almost_black_blue = np.full((100, 100, 3), [40, 5, 5], dtype=np.uint8)
    # BGR (5, 5, 40) : presque noir, legerement rouge. V_HSV ≈ 40
    img_almost_black_red = np.full((100, 100, 3), [5, 5, 40], dtype=np.uint8)
    # BGR (200, 50, 50) : bleu MOYEN, pas noir. V_HSV ≈ 200
    img_medium_blue = np.full((100, 100, 3), [200, 50, 50], dtype=np.uint8)
    hsv_blk = cv2.cvtColor(img_blk, cv2.COLOR_BGR2HSV)
    hsv_ab = cv2.cvtColor(img_almost_black_blue, cv2.COLOR_BGR2HSV)
    hsv_ar = cv2.cvtColor(img_almost_black_red, cv2.COLOR_BGR2HSV)
    hsv_mb = cv2.cvtColor(img_medium_blue, cv2.COLOR_BGR2HSV)
    rng_black = HSVRange(color_mode="black", v_hi=60)
    assert rng_black.mask(hsv_blk).mean() > 250, "noir pur doit etre capte"
    assert rng_black.mask(hsv_ab).mean() > 250, "presque-noir bleute aussi"
    assert rng_black.mask(hsv_ar).mean() > 250, "presque-noir rouge aussi (H ignore)"
    assert rng_black.mask(hsv_mb).mean() < 5,  "bleu moyen (V=200) ne doit PAS etre capte"
    print("  [OK] color_mode='black' capte V<=v_hi peu importe H (independance teinte)")

    # 9. color_mode = "white" : ne capte que pixels peu satures + clairs
    img_wht = np.full((100, 100, 3), [240, 240, 240], dtype=np.uint8)  # blanc cassé
    img_red_pure = np.full((100, 100, 3), [0, 0, 240], dtype=np.uint8)
    hsv_wht = cv2.cvtColor(img_wht, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(img_red_pure, cv2.COLOR_BGR2HSV)
    rng_white = HSVRange(color_mode="white", s_hi=30, v_lo=200)
    m_wht = rng_white.mask(hsv_wht)
    m_red = rng_white.mask(hsv_red)
    assert m_wht.mean() > 250, "blanc cassé doit etre capte"
    assert m_red.mean() < 5, "rouge pur ne doit PAS etre capte (S eleve)"
    print("  [OK] color_mode='white' capte S<=s_hi ET V>=v_lo (H ignore)")

    print("Tous les tests passent.")
