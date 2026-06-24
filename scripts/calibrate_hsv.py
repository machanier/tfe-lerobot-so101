#!/usr/bin/env python3
"""
calibrate_hsv.py - Genere les plages HSV de chaque primitive coloree par
echantillonnage de pixels.

Procedure :
    1. Lance le script en pointant la camera principale (cam_0 par defaut).
    2. Pose chaque objet primitif (cube rouge, cylindre vert, ...) bien en
       evidence sous l'eclairage definitif du poste.
    3. Pour chaque objet :
        - tape le nom (label) dans le terminal (ex: red_cube).
        - clique sur la fenetre video pour echantillonner des pixels DE
          L'OBJET. Aux endroits de couleur la plus pure. ~20 clics suffisent.
        - touche 'n' pour passer a l'objet suivant.
    4. Touche 's' a la fin pour sauvegarder.

Sortie :
    configs/perception/hsv_specs.json
        Liste d'ObjectSpec avec plages HSV (h_lo, h_hi, s_lo, ...) calculees
        comme [mean - 2.5*std, mean + 2.5*std] sur les pixels echantillonnes.
        Le rouge est detecte automatiquement (cluster autour de H~0/179) et
        on emet `hue_extra_lo/hi`.

Pas de hardware necessaire (juste la camera). On utilise la calibration
intrinseque pour eventuellement debruiter, mais HSV est lui-meme robuste.

Reference : OpenCV doc, cv2.cvtColor + COLOR_BGR2HSV. H in [0,179], S/V in [0,255].
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from config import CAMERAS  # noqa: E402

OUTPUT_PATH = REPO / "configs" / "perception" / "hsv_specs.json"
PATCH_HALF = 3  # demi-cote du patch echantillonne autour du clic (en pixels)


class HSVSampler:
    """Etat partage entre la fenetre OpenCV et la boucle principale."""

    def __init__(self):
        self.current_label: str | None = None
        self.samples: dict[str, list[np.ndarray]] = {}
        self.last_frame_hsv: np.ndarray | None = None

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.last_frame_hsv is None:
            return
        if self.current_label is None:
            print("  (clique apres avoir defini un label avec 'n')")
            return
        h, w = self.last_frame_hsv.shape[:2]
        y0, y1 = max(0, y - PATCH_HALF), min(h, y + PATCH_HALF + 1)
        x0, x1 = max(0, x - PATCH_HALF), min(w, x + PATCH_HALF + 1)
        patch = self.last_frame_hsv[y0:y1, x0:x1].reshape(-1, 3)
        self.samples.setdefault(self.current_label, []).append(patch)
        total = sum(p.shape[0] for p in self.samples[self.current_label])
        print(f"  + {patch.shape[0]} px ({total} cumules pour {self.current_label})")


def _detect_color_mode(samples: np.ndarray) -> str:
    """Detecte automatiquement le mode (chromatic / black / white / gray).

    Regle (basee sur les proprietes de l'espace HSV) :
      - V_median tres bas -> black (teinte n'a pas de sens quand V~0)
      - S_median bas + V_median tres haut -> white (pas de saturation, lumineux)
      - S_median bas + V_median intermediaire -> gray
      - sinon -> chromatic (teinte significative)

    Voir docs/PROJECT_STATUS.md decision D8 pour la justification.
    """
    H = samples[:, 0].astype(np.int32)
    S = samples[:, 1].astype(np.int32)
    V = samples[:, 2].astype(np.int32)
    s_med = float(np.median(S))
    v_med = float(np.median(V))
    if v_med < 70:
        return "black"
    if s_med < 50 and v_med > 180:
        return "white"
    if s_med < 50:
        return "gray"
    return "chromatic"


def _build_spec(label: str, samples: np.ndarray) -> dict:
    """Construit un dict ObjectSpec depuis les pixels echantillonnes.

    Adapte le mode (chromatic / black / white / gray) automatiquement.
    """
    H = samples[:, 0].astype(np.int32)
    S = samples[:, 1].astype(np.int32)
    V = samples[:, 2].astype(np.int32)
    mode = _detect_color_mode(samples)

    # --- Couleurs achromatiques : on ignore H ---
    if mode == "black":
        # IMPORTANT : pour le noir, on PRIVILEGIE un seuil conservateur (etroit)
        # pour eviter de capter le fond gris ou des ombres. Formule retenue :
        #   v_hi = max(45, mediane * 1.8)
        # plafonne a 100 (au-dela ce n'est plus du noir).
        # Le 95e percentile + marge donnait des seuils trop laxistes (>= 150)
        # quand l'utilisateur cliquait sur des bords avec reflets.
        v_med = float(np.median(V))
        v_hi = int(min(100, max(45, v_med * 1.8)))
        return {
            "label": label, "color_mode": "black", "v_hi": v_hi,
            "min_area_px": 500, "max_area_px": 200000, "meta": {},
            "_n_samples_px": int(samples.shape[0]),
            "_diagnostic": f"V median={int(v_med)}, max={int(np.max(V))}, "
                           f"v_hi retenu (median*1.8 clamp [45,100]) = {v_hi}",
        }
    if mode == "white":
        s_hi = int(min(255, np.percentile(S, 95) + 15))
        v_lo = int(max(0, np.percentile(V, 5) - 15))
        return {
            "label": label, "color_mode": "white", "s_hi": s_hi, "v_lo": v_lo,
            "min_area_px": 500, "max_area_px": 200000, "meta": {},
            "_n_samples_px": int(samples.shape[0]),
            "_diagnostic": f"S median={int(np.median(S))}, V median={int(np.median(V))}",
        }
    if mode == "gray":
        s_hi = int(min(255, np.percentile(S, 95) + 15))
        v_lo = int(max(0, np.percentile(V, 5) - 15))
        v_hi = int(min(255, np.percentile(V, 95) + 15))
        return {
            "label": label, "color_mode": "gray",
            "s_hi": s_hi, "v_lo": v_lo, "v_hi": v_hi,
            "min_area_px": 500, "max_area_px": 200000, "meta": {},
            "_n_samples_px": int(samples.shape[0]),
            "_diagnostic": f"S median={int(np.median(S))}, V median={int(np.median(V))}",
        }

    # --- Couleur chromatique : on utilise H + S + V ---
    # Detection automatique du rouge (cluster autour de 0/179)
    # On regarde quelle fraction des pixels est PROCHE de la couture 0/180.
    # Si >= 60 % sont a moins de 25 de la couture (cote 0 OU cote 179),
    # on considere que la teinte traverse 0.
    near_low = (H <= 25).mean()
    near_high = (H >= 155).mean()
    wraps_zero = (near_low + near_high) >= 0.6 and near_low > 0.05 and near_high > 0.05

    if wraps_zero:
        # On "deroule" H autour de 0 : si H > 90, on retranche 180.
        H_unwrapped = np.where(H > 90, H - 180, H)
        h_mean = float(np.mean(H_unwrapped))
        h_std = float(np.std(H_unwrapped))
        # Plage etroite autour du mean +/- 3*std (clip a [-180, 180])
        h_lo_raw = h_mean - 3.0 * h_std
        h_hi_raw = h_mean + 3.0 * h_std
        # Re-mappe : la plage principale est l'intersection avec [0, 179] ;
        # la plage extra est l'autre cote.
        if h_lo_raw < 0:
            # Partie cote bas : [0, h_hi_raw], extra : [180 + h_lo_raw, 179]
            h_lo, h_hi = 0, int(max(0, min(179, h_hi_raw)))
            extra_lo, extra_hi = int(max(0, min(179, 180 + h_lo_raw))), 179
        elif h_hi_raw > 179:
            # Partie cote haut : [h_lo_raw, 179], extra : [0, h_hi_raw - 180]
            h_lo, h_hi = int(max(0, min(179, h_lo_raw))), 179
            extra_lo, extra_hi = 0, int(max(0, min(179, h_hi_raw - 180)))
        else:
            h_lo, h_hi = int(max(0, h_lo_raw)), int(min(179, h_hi_raw))
            extra_lo, extra_hi = None, None
    else:
        h_lo = int(max(0, np.percentile(H, 2)))
        h_hi = int(min(179, np.percentile(H, 98)))
        extra_lo = extra_hi = None

    spec = {
        "label": label,
        "color_mode": "chromatic",
        "h_lo": h_lo,
        "h_hi": h_hi,
        "s_lo": int(max(20, np.percentile(S, 2))),
        "s_hi": 255,
        "v_lo": int(max(20, np.percentile(V, 2))),
        "v_hi": 255,
        "min_area_px": 500,
        "max_area_px": 200000,
        "meta": {},
        "_n_samples_px": int(samples.shape[0]),
        "_diagnostic": f"H median={int(np.median(H))}, S median={int(np.median(S))}, V median={int(np.median(V))}",
    }
    if extra_lo is not None:
        spec["hue_extra_lo"] = extra_lo
        spec["hue_extra_hi"] = extra_hi
    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Echantillonneur de couleurs HSV pour les primitives. "
                    "Par defaut : AJOUTE les nouvelles specs aux existantes "
                    "(remplace si meme label). --overwrite pour repartir de zero.",
    )
    parser.add_argument("--camera", type=int, default=CAMERAS["cam_0"]["index"],
                        help="Index de la camera a utiliser.")
    parser.add_argument("--width", type=int, default=CAMERAS["cam_0"]["width"])
    parser.add_argument("--height", type=int, default=CAMERAS["cam_0"]["height"])
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--overwrite", action="store_true",
                        help="Repart d'un fichier vide (efface toutes les specs existantes).")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # CHARGE LES SPECS EXISTANTES (mode append par defaut, voir --overwrite)
    existing_specs: list[dict] = []
    if output.exists() and not args.overwrite:
        try:
            existing_specs = json.load(open(output)).get("specs", [])
            existing_labels = [s["label"] for s in existing_specs]
            print(f"Specs existantes chargees : {existing_labels}")
            print("(Les labels que tu vas recalibrer seront REMPLACES, les autres conserves.")
            print(" Lance avec --overwrite pour repartir de zero.)")
            print()
        except Exception as e:
            print(f"AVERTISSEMENT : impossible de charger {output} ({e}). On repart de zero.")
            existing_specs = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la camera {args.camera}.")
        return
    # ALIGNEMENT RUNTIME : on ouvre la camera EXACTEMENT comme
    # src/perception/camera_io.py (codec MJPG force AVANT la resolution, puis
    # fps, puis warmup). But : que la calibration echantillonne les MEMES pixels
    # que le pipeline live, pour ne pas introduire de decalage de couleur
    # (chroma subsampling JPEG) entre calib et runtime. Comme le FOURCC n'est pas
    # observable sur macOS/AVFoundation (get renvoie 0), on garantit la parite en
    # faisant le meme appel des deux cotes, quel que soit l'effet reel du backend.
    # ORDRE IMPORTANT : codec d'abord, puis resolution, puis fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(CAMERAS["cam_0"]["fps"]))
    # Diagnostic (le FOURCC peut revenir 0 = non rapporte sur macOS, sans gravite).
    _f = int(cap.get(cv2.CAP_PROP_FOURCC))
    _codec = "".join(chr((_f >> 8 * i) & 0xFF) for i in range(4)).strip() or "(non rapporte)"
    print(f"[calib] codec demande=MJPG | rapporte={_codec!r} | "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    # Warmup : stabilise l'auto-exposition avant les premiers clics.
    for _ in range(10):
        cap.read()

    sampler = HSVSampler()
    win = "Calibration HSV"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, sampler.on_mouse)

    print()
    print("Procedure :")
    print("  'n' = nouveau label (te demande son nom)")
    print("  clic gauche dans la fenetre = echantillonne un patch HSV")
    print("  's' = sauvegarder et quitter")
    print("  'q' = quitter sans sauvegarder")
    print()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lecture echouee.")
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sampler.last_frame_hsv = hsv

            display = frame.copy()
            status = f"label: {sampler.current_label or '(aucun)'} | "
            status += f"echantillons: " + ", ".join(
                f"{k}={sum(p.shape[0] for p in v)}" for k, v in sampler.samples.items()
            )
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Annule.")
                return
            if key == ord("n"):
                cv2.destroyWindow(win)  # libere le focus terminal
                name = input("Nom du nouveau label (ex: red_cube, blue_triangle) : ").strip()
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(win, sampler.on_mouse)
                if name:
                    sampler.current_label = name
                    print(f"  -> label courant : {name}")
            if key == ord("s"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not sampler.samples:
        print("Aucun echantillon : rien a sauvegarder.")
        return

    new_specs = []
    for label, patches in sampler.samples.items():
        samples = np.concatenate(patches, axis=0)
        if samples.shape[0] < 20:
            print(f"  AVERTISSEMENT : {label} a {samples.shape[0]} pixels (< 20 recommandes)")
        new_specs.append(_build_spec(label, samples))

    # MERGE : on garde les anciennes specs, on remplace celles dont le label
    # vient d'etre recalibre. C'est ce qui evite que recalibrer UN objet
    # efface les autres (bug observe le 2026-05-16).
    new_labels = {s["label"] for s in new_specs}
    merged = [s for s in existing_specs if s["label"] not in new_labels] + new_specs
    if existing_specs and not args.overwrite:
        replaced = [s["label"] for s in new_specs if s["label"] in {e["label"] for e in existing_specs}]
        added = [s["label"] for s in new_specs if s["label"] not in {e["label"] for e in existing_specs}]
        if replaced:
            print(f"\n  Labels REMPLACES : {replaced}")
        if added:
            print(f"  Labels AJOUTES   : {added}")

    payload = {
        "_doc": "Genere par scripts/calibrate_hsv.py. Voir src/perception/detector.py.",
        "specs": merged,
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    specs = merged  # pour l'affichage final
    print()
    print(f"{len(specs)} specs HSV sauvegardees : {output}")
    print()
    print(f"  {'LABEL':<25} {'MODE':<10} CONDITION DE DETECTION")
    print(f"  {'-' * 70}")
    for s in specs:
        mode = s.get("color_mode", "chromatic")
        if mode == "black":
            cond = f"V <= {s['v_hi']}  (H et S ignores)"
        elif mode == "white":
            cond = f"S <= {s['s_hi']} ET V >= {s['v_lo']}  (H ignore)"
        elif mode == "gray":
            cond = f"S <= {s['s_hi']} ET V in [{s['v_lo']}, {s['v_hi']}]  (H ignore)"
        else:
            extra = ""
            if "hue_extra_lo" in s:
                extra = f" + H[{s['hue_extra_lo']}, {s['hue_extra_hi']}]"
            cond = (f"H[{s['h_lo']}, {s['h_hi']}]{extra}  "
                    f"S>={s['s_lo']}  V>={s['v_lo']}")
        print(f"  {s['label']:<25} {mode:<10} {cond}")
        if "_diagnostic" in s:
            print(f"  {'':<25} {'':<10} (diag: {s['_diagnostic']})")
    print()
    print("Conseils :")
    print("  - Si une couleur 'chromatic' detecte le ROBOT (orange physique), "
          "le pose_estimator filtrera via les zones d'exclusion (configs/scene.json).")
    print("  - Si une couleur 'black'/'white' a un v_hi/v_lo trop large, re-echantillonne "
          "en cliquant uniquement sur le centre de l'objet (pas les bords).")


if __name__ == "__main__":
    main()
