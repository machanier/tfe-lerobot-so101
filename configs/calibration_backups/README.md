# Calibrations hand-eye — calibration attitrée + backups

## La calibration attitrée (utilisée par défaut)

Le pipeline utilise **toujours** `configs/handeye_cam_0.json` + `handeye_cam_1.json`
(+ `handeye_cam_2.json` pour l'eye-in-hand). Comme avant, **il n'y a qu'UNE
calibration de référence** — pas besoin de flag, pas de système à gérer.

**Calibration attitrée actuelle : la stéréo conjointe B3b (anciennement "s1").**
- Méthode : `cv2.stereoCalibrate` conjointe + déduction de cam_1.
- cam_0 résidu 6.58 mm, cam_1 résidu 11.77 mm.
- **Cohérence stéréo 1.95 mm** (vs 21 mm pour l'ancienne méthode séparée).
- Biais Y en pratique : ~+14 mm (vs +40 mm avant) → corrigé par cam_2.

Pour travailler normalement :
```bash
python scripts/pick_and_place.py --target orange_cube --detector hf --display
```

## Ce dossier = souvenir / backup (hors flux normal)

On garde ici les calibrations passées « au cas où » et pour la comparaison
du mémoire. **Ce ne sont pas des profils à switcher au quotidien** — juste
des archives.

| Backup | Quoi | Pourquoi on le garde |
|---|---|---|
| `s1/` | Copie de la calibration attitrée (B3b session 1) | Pour y revenir si un test l'a écrasée |
| `s2/` | B3b session 2 (équivalente, R1 légèrement plus élevé) | Souvenir, comparaison |
| `legacy_separate/` | Ancienne méthode séparée (avant B3b) | **Comparaison mémoire** : avant/après la calibration stéréo conjointe |

## Ressortir un backup ponctuellement (exceptionnel)

Si tu veux re-tester une ancienne calibration (ex: pour le mémoire) :
```bash
python scripts/pick_and_place.py ... --calib-profile legacy_separate   # teste l'ancienne
python scripts/pick_and_place.py ... --calib-profile s1                # REVIENS à l'attitrée
```

⚠️ Ce flag **écrase** `configs/handeye_cam_*.json`. Pense à relancer avec
`--calib-profile s1` pour remettre la calibration attitrée après un test.

## Refaire une calibration de zéro

```bash
# Damier 9x6 22mm collé sur la pince fermée, les 2 cams en place :
python scripts/recalibrate_handeye_stereo.py
```
Cela produit une nouvelle calibration dans `configs/`. Pour la garder en
backup, copie-la dans un nouveau dossier ici (ex: `s3/`) et ajoute une entrée
dans `profiles_metadata.json`.

## Référence

Calibration stéréo conjointe : Hartley & Zisserman 2018, *Multiple View
Geometry*, ch.10. Motivation : les résidus hand-eye indépendants des 2 caméras
s'additionnent à la triangulation (biais Y +40 mm). L'optimisation conjointe
contraint la transformation entre caméras → cohérence 10× meilleure.
