# Repère base du SO-101 — référence définitive

Ce document explique **où exactement** est l'origine (0, 0, 0) du repère
"base" du robot, **comment mesurer** des positions physiques cohérentes,
et **comment vérifier** expérimentalement.

À lire AVANT toute mesure physique pour le ground truth (`gt_test.json`,
`scene.json`, etc.).

## 1. Définition formelle (URDF)

L'origine du repère base est définie par le `<link name="base_link">` du
fichier `configs/so101_new_calib.urdf` (provenant du dépôt officiel
TheRobotStudio/SO-ARM100).

**`base_link` correspond au CENTRE GÉOMÉTRIQUE DE LA BASE MÉCANIQUE du
robot**, pas au centre du shoulder_pan motor comme je l'ai dit par erreur
plus tôt dans ce projet.

```
                      Z (vers le haut)
                      ▲
                      │
                      │     X (vers l'avant du robot,
                      │ ╱╱   vers les caméras)
                      │╱
       ╔══════════════╪═══════════════╗
       ║              ●               ║   ← shoulder_pan motor
       ║              │               ║      (axe Z vertical)
       ║          ╱╱  │               ║      à (X=+38.8, Y=0, Z=+62.4) mm
       ║              │               ║      depuis base_link
       ║              ▼               ║
       ╠══════════════╪═══════════════╣
       ║              ●               ║   ← base_link
       ║      (0, 0, 0) origine       ║      = ORIGINE URDF
       ║                              ║      = ZÉRO du repère robot
       ╚══════════════════════════════╝
       ───────────────────────────────────  table (sol physique)
```

### Position du shoulder_pan motor par rapport à base_link

Lue directement dans l'URDF (ligne du joint `shoulder_pan`) :
```
<origin xyz="0.0388353 -8.97657e-09 0.0624" rpy="..."/>
```

Donc le centre du shoulder_pan est à :
- **X = +38.8 mm** (légèrement devant le centre de la base)
- **Y = 0 mm** (centré)
- **Z = +62.4 mm** (au-dessus de la plaque de base)

### Validation par cinématique directe

Quand tous les moteurs sont à 0° (configuration "ZÉRO"), la cinématique
directe (FK) donne la position de la pince :

```
Config ZÉRO → effecteur à (X=+391.4, Y=0, Z=+226.5) mm depuis base_link
```

Donc en config zéro, la pince est à **39.1 cm devant** et **22.6 cm
au-dessus** du centre de la plaque de base.

## 2. Comment placer le base_link physiquement

Le `base_link` est, dans la réalité, le **centre de la plaque de base
inférieure** du robot — celle qui est posée sur la table. C'est le centre
géométrique de cette plaque (X=0, Y=0), avec Z=0 typiquement au **bas de
cette plaque** (au niveau de la table).

Pour visualiser :

```
Vue du dessus du robot :              Vue de profil :

      avant (X+)                          ↑ Z+
        ▲                                 │
        │                                 │     ╱╱ bras
   ┌────●────┐  base_link              ─●─── shoulder_pan
   │  (0,0)  │  (centre de la plaque)    │      (Z=+62.4)
   │         │                           │
   └─────────┘                         ──●── base_link Z=0
        ▲                                       (table ≈ Z=0)
        │
       Y+ (vue d'au-dessus, à gauche)
```

## 3. Procédure de mesure correcte d'un objet posé sur la table

Quand tu poses un cube (ou tout objet) sur la table devant le robot :

1. **Repère "base_link"** : son centre, à plat sur la table.
2. **X positif** = devant le robot (vers les caméras eye-to-hand).
3. **Y positif** = sur la gauche du robot, vu de derrière (= à droite du robot vu de face).
4. **Z = 0** = niveau de la table.
5. **Centre du cube** = (X_mesuré, Y_mesuré, hauteur_cube / 2).

Exemple : cube de 30 mm posé à 30 cm devant la base, centré :
```
position_base_mm = [300, 0, 15]
                    ↑    ↑   ↑
                    │    │   └── moitié hauteur cube (table à Z=0)
                    │    └────── centré sur l'axe Y
                    └─────────── 300 mm devant la plaque de base
```

**ATTENTION** : `Z = -17 mm` (que je t'ai dit auparavant) était **FAUX**.
La vraie valeur est Z = +15 mm (au-dessus de la table = au-dessus de
base_link).

## 4. Vérification expérimentale

Pour vérifier que **ta calibration est correcte** ET que tu mesures
depuis le bon point :

```bash
# 1. Branche le robot, active le venv
source venv/bin/activate

# 2. Lance la téléopération pour amener le bras en config zéro
python scripts/teleoperate.py
# Bouge le leader pour que tous les moteurs soient au milieu de leur plage
# (un peu fastidieux à la main, mais possible)

# OU (plus simple si tu as le code) : envoie directement la commande "zero"
python -c "
from src.control.motor_controller import MotorController
from src.calibration.forward_kinematics import ARM_JOINTS
from scripts.config import FOLLOWER_PORT
with MotorController() as mc:
    mc.connect(FOLLOWER_PORT)
    mc.enable_torque()
    mc.send_angles({j: 0.0 for j in ARM_JOINTS}, gripper_pct=50)
    import time
    time.sleep(2)
    input('Bras en config zero. Mesure la pince et appuie sur Entree.')
    mc.disable_torque()
"
```

3. Avec un mètre, mesure :
   - Distance horizontale (X) entre le centre de la plaque de base et le
     centre de la pince → **doit donner ~391 mm** (39 cm).
   - Hauteur (Z) entre la table et le centre de la pince → **doit donner
     ~227 mm** (23 cm).
   - Décalage latéral (Y) → **doit être ~0 mm**.

4. **Si les mesures correspondent à ±20 mm près** : ton repère mental et
   le repère URDF coïncident, ta calibration est OK.

5. **Si l'écart est plus grand** : soit ta calibration moteur est
   décalée (refaire `lerobot-calibrate`), soit ta calibration hand-eye est
   biaisée (refaire `scripts/solve_handeye_cam.py`), soit le robot n'est
   pas le `new_calib` (refaire la calibration LeRobot complète).

## 5. Position d'une caméra dans le repère base

Les positions des caméras eye-to-hand sont définies par la calibration
hand-eye (Sprint 1). On peut les lire :

```bash
python -c "
import json
import numpy as np
for i in (0, 1, 2):
    d = json.load(open(f'configs/handeye_cam_{i}.json'))
    T = np.array(d['transform'])
    pos_mm = T[:3, 3] * 1000
    print(f'cam_{i}  config={d[\"configuration\"]}  '
          f'position (mm) = ({pos_mm[0]:+7.1f}, {pos_mm[1]:+7.1f}, {pos_mm[2]:+7.1f})')
"
```

Résultat sur le poste actuel :
- `cam_0` (eye-to-hand) : (+647, +94, +235) mm — sur la barrière avant, à
  droite vue de face, ~24 cm de haut.
- `cam_1` (eye-to-hand) : (+667, -13, +230) mm — sur la barrière avant, à
  gauche vue de face.
- `cam_2` (eye-in-hand) : c'est `T_gripper_cam`, dépend de la pose du
  bras. Position absolue change en temps réel.

**IMPORTANT** : la position 3D détectée d'un objet (par exemple
`pos=(+340, +29, +20) mm` que le pipeline affiche) est **dans le repère
base** (= depuis base_link), **PAS depuis une caméra particulière**. La
triangulation stéréo prend les images des 2 caméras eye-to-hand et fournit
directement la position dans le repère robot grâce aux matrices
`T_base_cam` calibrées.

## 6. Récapitulatif des erreurs à ne PAS faire

| ❌ Erreur (j'ai dit avant) | ✅ Correct |
|---|---|
| "(0,0,0) = centre du shoulder_pan motor" | (0,0,0) = centre de la plaque de base inférieure |
| "Table à Z = -32 mm" | Table à Z = 0 mm (= niveau de base_link) |
| "Cube 30 mm sur table → centre à Z = -17 mm" | Centre à Z = +15 mm |
| "L'origine est en l'air, à 5 cm au-dessus de la table" | L'origine est AU NIVEAU de la table |

Désolé pour la confusion antérieure — c'est cette lecture rigoureuse de
l'URDF qui résout la question.

## 7. Pour ton mémoire

Cette section peut directement aller dans l'annexe ou la section
"Architecture > Repères". Tu peux citer :

> *« L'origine du repère base, telle que définie par le fichier URDF
> officiel `so101_new_calib.urdf` (TheRobotStudio/SO-ARM100), correspond
> au centre géométrique de la plaque inférieure du robot. Le centre du
> moteur shoulder_pan se trouve à (X=+38.8, Y=0, Z=+62.4) mm depuis cette
> origine. La cinématique directe en configuration nulle confirme cette
> géométrie : la pince est attendue à (+391.4, 0, +226.5) mm, valeur
> vérifiable au mètre. »*
