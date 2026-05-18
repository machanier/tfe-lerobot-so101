# Reprendre le projet — guide pour la prochaine session

Document à lire EN PREMIER quand tu reprends le projet (toi ou un nouvel
agent Claude Code dans une nouvelle conversation).

## 1. État du projet au 2026-05-18 (session 9)

✅ **Sprint 1 (Calibration)** : complet. Intrinsèques + hand-eye OK.
✅ **Sprint 2 (Perception)** : V1 HSV + V2 OWL-ViTv2 fonctionnels.
✅ **Sprint 3 (Planning + Contrôle)** : grasp top-down + IK + trajectoire.
✅ **Sprint 4 (Boucle fermée)** : raffinement cam_2 actif.
✅ **PREMIÈRE SAISIE RÉUSSIE** le 17 mai 2026 avec `pick_and_place.py`.

### Session 9 (2026-05-18) — fixes UX & dépose

Modifs dans `src/pipeline.py` :

1. **Dépose précise** : ajout d'un `IKSolver` dédié `_ik_drop` avec
   `rotation_weight=0.01` (vs 0.1 standard) pour les poses
   `drop_above`/`drop_release`. Justification : pour la dépose on s'en
   moque que la pince soit pile verticale (elle ouvre, le cube tombe), on
   veut surtout la position au mm près. Avant : `drop_above approx
   trans=47mm` (IK sacrifiait la position pour préserver l'orientation —
   sous-actuation 5/6 DDL). Après : `trans=0.1mm rot=11deg` (position
   parfaite, inclinaison négligeable pour un drop).

2. **Retour home = position de départ** : nouveau champ
   `PipelineConfig.home_from_session_start: bool = True` (défaut). Le
   robot revient EXACTEMENT à la pose dans laquelle il était au lancement
   du script. Maxence peut placer le robot dans une pose stable +
   caméra orientée vers la scène, lancer la commande, et le bras y
   revient. Plus de "robot qui part en cacahuètes" après la pose.

3. **Display fluide en HF** : le callback live n'appelle plus
   `detect_multi` à chaque rafraîchissement (~3-5s par appel en HF sur M4
   → bloquait la trajectoire). Il affiche juste les frames brutes →
   display fluide même avec `--display --detector hf`.

4. **Bug fix `_build_full_trajectory`** : `q_home` et `dur_home` étaient
   référencés sans être définis (NameError sur `--no-closed-loop` /
   `--dry-run`). Corrigé : utilise le param `q_home` ou fallback config.

🟡 **À continuer** : amélioration de la robustesse, recalibration HSV,
évitement obstacles (Sprint 4.5), SmolVLA comparatif (Sprint 5),
rédaction LaTeX.

## 2. Quick-start après git clone (ou nouvelle session)

```bash
git pull origin main
source venv/bin/activate

# Verifie que tout est OK
python scripts/check_calibration.py
# → doit afficher [OK] partout

# Voir où on en est sur le pipeline pick-and-place
python scripts/pick_and_place.py --target orange_cube --detector hf --dry-run
# (dry-run = simule, ne bouge pas le robot)

# Test LIVE (le robot bouge !)
python scripts/pick_and_place.py --target orange_cube --detector hf --display
# (--display = affiche les cameras pendant l'execution)
```

## 3. Documents-clés à lire pour comprendre

| Fichier | Quand le lire |
|---|---|
| [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md) | Document VIVANT. Toutes les décisions D1-D16. À lire en 1er. |
| [`docs/REPERE_BASE.md`](REPERE_BASE.md) | Référence sur l'origine du repère robot. À lire AVANT toute mesure physique. |
| [`docs/bachelor_chanier_25_26.pdf`](bachelor_chanier_25_26.pdf) | Cahier des charges Partie I. La référence ultime. |
| [`docs/memoire/`](memoire/) | Brouillon LaTeX du mémoire (33 pages déjà rédigées). |

## 4. Configuration spécifique du poste de Maxence

### Matériel
- SO-101 follower + leader, 6 moteurs Feetech STS3215 chacun (différents gear ratios).
- **Pince TPU XLRobot avec grip de tennis** ajouté pour la saisie (meilleure adhérence sur les surfaces lisses).
- 3 caméras USB 1920×1080 :
  - cam_0 + cam_1 : barrière avant, baseline ~109 mm en Y (stéréo eye-to-hand).
  - cam_2 : eye-in-hand sur la pince.
- 2 hubs USB séparés (D12) pour répartir cam_0+1 d'un côté, cam_2+robot de l'autre.
- MacBook Pro M4, macOS, Python 3.12.

### Repère base — origine (0, 0, 0)
Centre géométrique de la plaque inférieure (`base_so101_v2.stl`), à plat sur la table.
- shoulder_pan motor à `(+38.8, 0, +62.4)` mm depuis base_link.
- Pose home (config zéro FK) : effecteur attendu à `(+391, 0, +227)` mm.
- Voir `docs/REPERE_BASE.md` pour les détails.

### Compensation systématique du biais
- `configs/perception/bias_correction.json` : **dy = 30 mm** (mesuré expérimentalement).
- Soustrait à chaque détection 3D. Permet de ne pas modifier `gt_test.json` à chaque test.
- À réajuster si la calibration hand-eye est refaite.

### Objets utilisés (testés)
4 primitives colorées (HSV V1) :
- orange_cube (cube 30 mm) — **objet de référence pour les tests**
- blue_rectangular_box
- purple_cylinder
- black_triangular_prism (mode `color_mode: black`)

5 objets quotidien (HF V2, non calibrés HSV) :
- white_mug (imprimé 3D)
- pen (stylo Bic)
- rubiks_cube
- tissue_box (paquet mouchoirs)
- tall_plastic_cup (gobelet)

### Boîte de dépose
`configs/scene.json` → `drop_box.center_base_m`. À mesurer physiquement après chaque
déplacement de la boîte.

## 5. Bugs résolus et limites connues (à mentionner dans le mémoire)

### Résolus durant les itérations
1. HSV ne gère pas noir/blanc/gris (D8) → résolu par `color_mode`.
2. Faux positifs robot orange ↔ cube orange → résolu par exclusion_zones (D9).
3. 3 caméras 1080p saturent USB → résolu par MJPG (D10) + 2 hubs (D12).
4. Bus moteur "Lock id_=N" → retry 5x avec délai progressif.
5. **Repère base** mal compris (D11 corrigée) → docs/REPERE_BASE.md.
6. IK plages divisées par 2 (bug π vs 2π) → corrigé, +marge 3°.
7. Biais Y systématique stéréo +30 mm → compensation auto (D11 + bias_correction.json).
8. Compensation Y créait faux reproj_error → corrigé : reproj **avant** compensation.

### Limites identifiées (à documenter dans la discussion)
- OWL-ViTv2 = ~3-5 sec par image sur M4 (trop lent pour servoing temps réel).
- HSV échoue sur noir/blanc/transparent + sensibilité éclairage.
- Sous-actuation SO-101 5/6 DDL (toutes orientations pas atteignables).
- Pas de détection de collision (le robot frappe les obstacles s'il y en a).
- Faux positifs sémantiques OWL-ViTv2 (gourde → "purple_cylinder", essuie-tout → "white_mug").

## 6. Prochaines étapes (priorité décroissante)

### A) Validation expérimentale rigoureuse (Sprint 5 partiel)
20 essais de pick-and-place avec mesures :
- Taux de réussite (success/total)
- Précision finale (mm)
- Nombre de "retouches" (cas où le cube est déplacé puis ressaisi)
- Temps moyen par essai
→ Pour le tableau de résultats du chapitre 6 du mémoire.

### B) Évitement obstacles (Sprint 4.5, objectif 3 du cahier des charges)
Détecter et **éviter** les obstacles dans la trajectoire :
1. Le `pose_estimator` peut déjà détecter les objets autres que la cible.
2. Le grasp planner / trajectoire devrait les contourner.
3. Méthode simple : ajouter "marge" autour des autres détections, vérifier que
   la trajectoire ne traverse pas. Sinon : RRT (LaValle 2006) → c'est plus complexe.

### C) Recalibration HSV (5 min côté hardware)
Refaire `python scripts/calibrate_hsv.py` sous l'éclairage du jour pour avoir
des détections HSV fiables (utile pour la comparaison V1/V2 du mémoire).

### D) SmolVLA comparatif (Sprint 5 final)
1. Téléopérer le pick-and-place ~50 fois (datasets LeRobot).
2. `lerobot-train --policy=smolvla` (~quelques heures GPU).
3. Comparer SmolVLA vs notre pipeline modulaire sur les mêmes scènes.
4. Tableau dans le mémoire chapitre 6.

### E) Rédaction LaTeX
Compléter les sections vides de `docs/memoire/chapters/06_control_evaluation.tex`
et `07_discussion.tex` avec les chiffres expérimentaux obtenus.

## 7. Réponses aux questions récurrentes

**"Pourquoi le pipeline appelle l'IK et pas la FK ?"**
Les deux sont utilisées :
- **FK** : "j'ai les angles, où est la pince" → utilisé pour calculer `T_base_cam2` en temps réel (cam_2 eye-in-hand bouge avec le bras).
- **IK** : "j'ai la pose désirée, quels angles" → utilisé pour calculer les angles à envoyer aux moteurs pour atteindre approach/grasp/retract/drop.

**"Pourquoi `gt_test.json` si le robot détecte l'objet seul ?"**
- `pick_and_place.py` : **détecte** l'objet en live, n'utilise PAS gt_test.json.
- `check_perception.py` : **valide** la précision en comparant à un ground truth → utilise gt_test.json.

**"Comment le robot 'sait' où sont ses pinces ?"**
URDF définit `gripper_frame_link` = point virtuel au centre de la pince. FK calcule
sa position pour n'importe quelle config articulaire. Le grasp planner positionne ce
repère au centre du cube → pince attrape. **Pas de détection de collision physique**.

## 8. Comment poursuivre dans une nouvelle session Claude Code

Quand tu démarres une nouvelle conversation :
1. Le nouvel agent ne se souviendra pas des détails précédents.
2. **Donne-lui** le chemin `/Users/maxencechanier/Projects/tfe-lerobot-so101/docs/NEXT_SESSION.md` pour qu'il se mette à jour.
3. Il pourra lire automatiquement `PROJECT_STATUS.md`, `REPERE_BASE.md`, et le brouillon LaTeX.
4. La mémoire utilisateur (`~/.claude/projects/.../memory/project_tfe.md`) sera lue automatiquement et résumera la situation.

## 9. Fichiers tracés sur git à ne PAS oublier de commit

```bash
git status
# Si tu vois des modifs locales sur :
#   - scripts/run_perception.py (FPS counter perso)
#   - configs/perception/gt_test.json (ground truth de tes essais)
#   - configs/scene.json (boîte de dépose)
# → commit-les avec git add + git commit -m "..."
```
