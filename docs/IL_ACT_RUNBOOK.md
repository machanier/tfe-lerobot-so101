# Imitation Learning (ACT) — Runbook SO-101

2e méthode du TFE : apprendre la saisie par **imitation learning** via l'écosystème
officiel Hugging Face / LeRobot (policy **ACT**), en parallèle de la pipeline classique
(stéréo + détection + IK). 100 % officiel : `lerobot-*` + ACT + LeRobotDataset + HF Hub.

> Version utilisée : **lerobot 0.5.1** (venv). Note ce numéro dans le mémoire
> (`pip show lerobot`) : la lib bouge vite, c'est ta garantie de reproductibilité.

---

## 0. Principe et choix arrêtés

- **ACT** = réseau bout-en-bout `pixels (front + wrist) + état articulaire -> actions`.
  Pas de détecteur, pas de HSV, pas de stéréo/triangulation : la cible est apprise
  **implicitement** dans les démos. Un seul objet, figé dans les démos.
- **Caméras** : 2 vues RGB indépendantes — `wrist` (cam_2, eye-in-hand) + `front`
  (une des stéréo, eye-to-hand). En 640×480 (plus léger que le 1080p classique).
- **Pas de calibration caméra nécessaire pour ACT** : ni intrinsèque ni extrinsèque.
  ACT travaille sur les pixels bruts (cf. §3).

## 1. La règle d'or : COHÉRENCE record == eval

Cause de panne n°1. Entre l'enregistrement des démos et l'évaluation, **rien ne doit
bouger** : positions des caméras, éclairage, calibration moteurs, table, objet.
Les clés caméras `front`/`wrist` sont garanties identiques par `config.il_cameras_flag()`.

## 2. Choix de l'objet : **cube orange** (recommandé)

Parmi tes objets pipeline (orange_cube, purple_cylinder, blue_rectangular_box,
black_triangular_prism, light_blue_ball) :

- **orange_cube — recommandé.** Couleur très contrastée (aide la sélection implicite
  par apparence) + **cube = symétrie de rotation** => n'importe quel angle d'approche
  marche => le plus tolérant pour une 1re policy ACT.
- Les 3 cm ne sont **pas trop petits** : c'est dans la plage de la pince SO-101 et
  comparable aux cubes/lego des tutos officiels.
- Repli : `blue_rectangular_box` (axe de prise net car allongé, mais impose une
  orientation cohérente — moins tolérant). À garder si le cube se révèle capricieux.

## 3. Caméra de scène : cam_0 ou cam_1 ?

**Aucune préférence côté code** : la qualité de calibration n'a aucune importance pour
ACT (il n'utilise pas la calibration). Le choix est **purement physique**. Critères :

1. Voit **tout** l'espace de travail + la zone de prise.
2. **Occulte le moins** l'objet par le bras pendant l'approche.
3. Objet bien visible (contraste/éclairage homogène).
4. Angle légèrement **oblique/surélevé** > vue plate de côté (indices de profondeur).

Par défaut : `cam_0`. Pour comparer puis choisir :
```bash
python scripts/preview_camera.py --camera 0   # pose l'objet, observe le cadrage
python scripts/preview_camera.py --camera 1
```
Pour basculer sur cam_1 : dans `scripts/config.py`, mets `IL_SCENE_CAM = "cam_1"`.

## 4. Calibration moteurs : **déjà faite (oct. 2025)** — à tester d'abord

Les fichiers LeRobot existent déjà :
- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/mon_follower.json`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/mon_leader.json`

Elle date de ~8 mois. **Ne refais PAS forcément** : teste d'abord (§6.2). Si le follower
suit bien le leader (pas de décalage), garde-la. Refais seulement si ça dérive :
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT_F> --robot.id=mon_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<PORT_L> --teleop.id=mon_leader
```
> Procédure : tous les joints au milieu de leur course, Entrée, puis balaye chaque joint
> sur toute sa plage. Les `id` (`mon_follower`/`mon_leader`) doivent rester **identiques**
> partout (calibrate -> teleop -> record -> eval).

## 5. Montage physique

- **Leader ET follower fermement fixés à la table** (serre-joint). Le leader pas trop
  loin du follower = téléopération confortable et répétable.
- **Caméra de scène fixée** sur son support ; une fois les démos lancées, ne la bouge
  plus (règle §1). Pas besoin qu'elle soit pile au centre.

---

## 6. Procédure complète (commandes)

Toujours d'abord :
```bash
cd ~/Projects/tfe-lerobot-so101
source venv/bin/activate
```

### 6.1 Trouver ports & index caméras
```bash
lerobot-find-port                 # -> reporte les ports dans scripts/config.py
python scripts/detect_cameras.py  # -> confirme les index OpenCV (0,1,2)
```
> macOS réassigne ports et index à chaque rebranchement. Index 3/4 = webcam Mac/iPhone, à ignorer.

### 6.2 Vérifier robot + caméras (test à blanc)
```bash
python scripts/teleoperate.py
```
Tu dois voir le follower copier le leader, et les 2 flux `front` + `wrist` (rerun).
Bouge l'objet, vérifie le cadrage. Ctrl+C pour arrêter.

### 6.3 Enregistrer les démos (~1–2 h pour 50)
```bash
python scripts/record_dataset.py            # 50 épisodes, "Grab the orange cube", local
```
Pendant l'enregistrement : **→** épisode suivant · **←** ré-enregistrer (utilise-le dès
qu'une démo est ratée) · **Échap** stop+encodage.
Conseils qualité : gestes **lents et nets**, fermeture franche de la pince, et surtout
**varie la position de l'objet** entre épisodes (sinon sur-apprentissage d'une trajectoire).

### 6.4 Entraîner ACT
```bash
python scripts/train.py                     # ACT, 100k steps, MPS, batch 8
```
Checkpoints dans `outputs/train/act_so101_orange_cube/checkpoints/`.
Si MPS coince sur un opérateur :
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py
```
Si trop lent : notebook **Colab officiel ACT** (entraîne là-bas, récupère le checkpoint).

### 6.5 Évaluer sur le robot (même décor !)
```bash
python scripts/eval_policy.py               # 10 épisodes, dernier checkpoint
```
Compte les succès sur les 10 → ton taux de succès **sur la config entraînée**.

---

## 7. À quoi s'attendre

- **La 1re policy échoue souvent** (« woodpecker » : tape sans saisir). Normal.
  Prévois **2–3 cycles** record → train → eval. Compte ~1 semaine, pas un après-midi.
- Repères (à sourcer) : ~**50 démos** (doc ACT), **100 000 steps** (défaut LeRobot),
  ~**70 %** au 1er essai, ~**90 %** après ajout de diversité (retours communautaires).
- **Ne revendique pas de généralisation** : « X % sur la configuration entraînée ».
  ACT ne généralise pas hors du décor démontré.

## 8. Pour pousser sur le Hub (optionnel, repro mémoire)
```bash
hf auth login                               # token HF
python scripts/record_dataset.py --push-to-hub
python scripts/train.py --push-to-hub
```
