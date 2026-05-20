# Profils de calibration hand-eye

Ce dossier stocke plusieurs **jeux de calibration hand-eye** (extrinsèques
cam_0 + cam_1) qu'on peut activer à la demande, sans avoir à re-scotcher le
damier sur la pince à chaque fois qu'on veut comparer.

## Le profil ACTIF

Le pipeline lit toujours `configs/handeye_cam_0.json` + `handeye_cam_1.json`
(+ `handeye_cam_2.json` pour l'eye-in-hand, qui ne change pas). Le profil
« actif » est simplement celui qui a été copié dans ces fichiers.

**Actif actuellement : `s1`** (B3b stéréo conjointe, session 1).

## Switcher de profil

```bash
# Charge un profil avant de lancer le pick-and-place :
python scripts/pick_and_place.py --target orange_cube --detector hf --display --calib-profile s1
python scripts/pick_and_place.py --target orange_cube --detector hf --display --calib-profile s2
python scripts/pick_and_place.py --target orange_cube --detector hf --display --calib-profile legacy_separate
```

Le flag `--calib-profile <nom>` copie `calibration_profiles/<nom>/handeye_cam_*.json`
vers `configs/` juste avant de lancer. Le dernier profil utilisé reste actif.

## Les 3 profils disponibles

| Profil | Méthode | cam_0 résidu | cam_1 résidu | Cohérence stéréo | R1 pratique | Verdict |
|---|---|---|---|---|---|---|
| **`s1`** ⭐ | Stéréo conjointe (B3b) | 6.58 mm | 11.77 mm | **1.95 mm** (max 7.81) | **~14.7 mm** | **ACTIF — recommandé** |
| `s2` | Stéréo conjointe (B3b) | 7.01 mm | 23.37 mm | 1.95 mm (max 24.72) | ~19.0 mm | Backup |
| `legacy_separate` | Séparée (avant B3b) | 5.82 mm | 6.80 mm | **21.46 mm** ⚠️ | ~40 mm | Ne PAS utiliser pour les prises |

### Lecture des métriques

- **Résidu cam_X** : erreur de la calibration hand-eye de chaque caméra prise
  isolément. Plus bas = mieux, mais c'est **trompeur** pour la stéréo (cf ci-dessous).
- **Cohérence stéréo** : à quel point cam_0 et cam_1 sont d'accord sur la position
  3D du même point. **C'est LA métrique qui compte pour la triangulation.**
  La méthode séparée a des résidus individuels plus bas (5.82 / 6.80) MAIS une
  cohérence catastrophique (21.46 mm) → biais Y +40 mm en pratique.
  La méthode stéréo conjointe (B3b) sacrifie un peu le résidu individuel pour
  gagner 10× sur la cohérence (1.95 mm).
- **R1 pratique** : correction moyenne que cam_2 applique en conditions réelles
  (= biais résiduel de la stéréo mesuré sur le robot). s1 est ~4 mm meilleur que s2.

### Pourquoi s1 plutôt que s2 ?

Les deux ont une cohérence stéréo moyenne identique (1.95 mm), mais s1 a :
- un R1 pratique plus faible (~14.7 vs ~19.0 mm) = stéréo plus précise,
- une cohérence plus homogène (max 7.81 vs 24.72 mm),
- cam_2 qui détecte toujours l'objet au départ (0 raté vs 2 pour s2).

La différence reste mince — les deux sont utilisables.

## Refaire une calibration (nouvelle session B3b)

```bash
# Damier 9x6 22mm collé sur la pince fermée, les 2 cams en place :
python scripts/recalibrate_handeye_stereo.py
```

Cela écrase `configs/handeye_cam_0/1.json` avec la nouvelle calibration.
Pour l'archiver comme nouveau profil, copie-la dans un nouveau dossier ici :

```bash
mkdir configs/calibration_profiles/s3
cp configs/handeye_cam_0.json configs/handeye_cam_1.json \
   configs/handeye_stereo_info.json configs/extrinsic_capture_stereo.json \
   configs/calibration_profiles/s3/
# Puis ajoute une entrée dans profiles_metadata.json
```

## Contenu de chaque profil

- `handeye_cam_0.json`, `handeye_cam_1.json` : les transformations T_base_cam (l'essentiel).
- `handeye_stereo_info.json` : T_cam0_cam1 + baseline (info, recalculé depuis les handeye).
- `extrinsic_capture_stereo.json` : les captures brutes du damier (pour re-solver si besoin).
- `legacy_separate/` n'a que les 2 handeye (pas de captures stéréo, c'était l'ancienne méthode).

## Historique

- **2026-05-19/20** : passage de la calibration séparée (legacy) à la calibration
  stéréo conjointe (B3b). Motivation : éliminer le biais Y +40 mm constaté en
  pratique, causé par l'addition des résidus indépendants des 2 caméras à la
  triangulation. Résultat : cohérence stéréo 21 mm → 2 mm, biais Y pratique
  40 mm → ~14 mm. Référence : Hartley & Zisserman 2018 ch.10 (stereo calibration).
