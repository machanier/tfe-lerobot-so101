# Imitation Learning (ACT) — Runbook SO-101

Seconde méthode du projet : apprendre la saisie par imitation learning via l'écosystème
officiel Hugging Face / LeRobot (policy ACT), en parallèle de la pipeline classique
(stéréo + détection + IK). Intégralement officiel : `lerobot-*` + ACT + LeRobotDataset + HF Hub.

> Version utilisée : lerobot 0.5.1 (venv). Le numéro exact s'obtient avec
> `pip show lerobot` : la bibliothèque évolue vite, c'est la référence de reproductibilité.

---

## 0. Principe et choix

- ACT = réseau bout-en-bout `pixels (front + wrist) + état articulaire -> actions`.
  Pas de détecteur, pas de HSV, pas de stéréo/triangulation : la cible est apprise
  implicitement dans les démos. Un seul objet, figé dans les démos.
- Caméras : 2 vues RGB indépendantes — `wrist` (cam_2, eye-in-hand) + `front`
  (une des stéréo, eye-to-hand). En 640×480 (plus léger que le 1080p classique).
- Pas de calibration caméra nécessaire pour ACT : ni intrinsèque ni extrinsèque.
  ACT travaille sur les pixels bruts (cf. §3).

## 1. Règle fondamentale : cohérence record == eval

Cause de panne principale. Entre l'enregistrement des démos et l'évaluation, rien ne doit
bouger : positions des caméras, éclairage, calibration moteurs, table, objet.
Les clés caméras `front`/`wrist` sont garanties identiques par `config.il_cameras_flag()`.

## 2. Choix de l'objet : cube orange (recommandé)

Parmi les objets de la pipeline (orange_cube, purple_cylinder, blue_rectangular_box,
black_triangular_prism, light_blue_ball) :

- orange_cube — recommandé. Couleur très contrastée (aide la sélection implicite
  par apparence) + cube = symétrie de rotation, donc n'importe quel angle d'approche
  fonctionne, ce qui en fait le choix le plus tolérant pour une première policy ACT.
- Les 3 cm ne sont pas trop petits : c'est dans la plage de la pince SO-101 et
  comparable aux cubes/lego des tutoriels officiels.
- Repli : `blue_rectangular_box` (axe de prise net car allongé, mais impose une
  orientation cohérente — moins tolérant). À conserver si le cube se révèle difficile.

## 3. Caméra de scène : cam_0 ou cam_1 ?

Aucune préférence côté code : la qualité de calibration n'a aucune importance pour
ACT (il n'utilise pas la calibration). Le choix est purement physique. Critères :

1. Voit tout l'espace de travail + la zone de prise.
2. Occulte le moins possible l'objet par le bras pendant l'approche.
3. Objet bien visible (contraste/éclairage homogène).
4. Angle légèrement oblique/surélevé plutôt qu'une vue plate de côté (indices de profondeur).

Par défaut : `cam_0`. Pour comparer puis choisir :
```bash
python scripts/preview_camera.py --camera 0   # pose l'objet, observe le cadrage
python scripts/preview_camera.py --camera 1
```
Pour basculer sur cam_1 : dans `scripts/config.py`, mettre `IL_SCENE_CAM = "cam_1"`.

## 4. Calibration moteurs : déjà réalisée — à tester d'abord

Les fichiers LeRobot existent déjà :
- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/mon_follower.json`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/mon_leader.json`

Il n'est pas nécessaire de la refaire systématiquement : tester d'abord (§6.2). Si le
follower suit bien le leader (pas de décalage), la conserver. La refaire seulement en cas
de dérive :
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT_F> --robot.id=mon_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<PORT_L> --teleop.id=mon_leader
```
> Procédure : tous les joints au milieu de leur course, Entrée, puis balayer chaque joint
> sur toute sa plage. Les `id` (`mon_follower`/`mon_leader`) doivent rester identiques
> partout (calibrate -> teleop -> record -> eval).

## 5. Montage physique

- Leader et follower fermement fixés à la table (serre-joint). Le leader pas trop
  loin du follower = téléopération confortable et répétable.
- Caméra de scène fixée sur son support ; une fois les démos lancées, ne plus la bouger
  (règle §1). Elle n'a pas besoin d'être exactement au centre.

---

## 6. Procédure complète (commandes)

Toujours d'abord :
```bash
cd ~/Projects/tfe-lerobot-so101
source venv/bin/activate
```

### 6.1 Trouver ports & index caméras
```bash
lerobot-find-port                 # -> reporter les ports dans scripts/config.py
python scripts/detect_cameras.py  # -> confirme les index OpenCV (0,1,2)
```
> macOS réassigne ports et index à chaque rebranchement. Index 3/4 = webcam Mac/iPhone, à ignorer.

### 6.2 Vérifier robot + caméras (test à blanc)
```bash
python scripts/teleoperate.py
```
Le follower doit copier le leader, et les 2 flux `front` + `wrist` doivent s'afficher (rerun).
Déplacer l'objet, vérifier le cadrage. Ctrl+C pour arrêter.

### 6.3 Enregistrer les démos (~1–2 h pour 50)
```bash
python scripts/record_dataset.py            # 50 épisodes, "Grab the orange cube", local
```
Pendant l'enregistrement : `→` épisode suivant · `←` ré-enregistrer (à utiliser dès
qu'une démo est ratée) · `Échap` stop + encodage.
Conseils qualité : gestes lents et nets, fermeture franche de la pince, et surtout
varier la position de l'objet entre épisodes (sinon sur-apprentissage d'une trajectoire).

### 6.4 Entraîner ACT
```bash
python scripts/train.py                     # ACT, 100k steps, MPS, batch 8
```
Checkpoints dans `outputs/train/act_so101_orange_cube/checkpoints/`.
Si MPS coince sur un opérateur :
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py
```
Si l'entraînement est trop lent, il peut être déporté sur un environnement GPU (par
exemple le notebook Colab officiel ACT), puis le checkpoint récupéré.

### 6.5 Évaluer sur le robot (même décor)
```bash
python scripts/eval_policy.py               # 10 épisodes, dernier checkpoint
```
Compter les succès sur les 10 pour obtenir le taux de succès sur la configuration entraînée.

---

## 7. À quoi s'attendre

- La première policy échoue souvent (« woodpecker » : tape sans saisir). C'est attendu.
  Prévoir 2–3 cycles record → train → eval, sur l'ordre d'une semaine plutôt qu'un après-midi.
- Repères indicatifs : ~50 démos (documentation ACT), 100 000 steps (défaut LeRobot),
  ~70 % au premier essai, ~90 % après ajout de diversité (retours communautaires).
- Ne pas revendiquer de généralisation : « X % sur la configuration entraînée ».
  ACT ne généralise pas hors du décor démontré.

## 8. Pousser sur le Hub (optionnel, reproductibilité)
```bash
hf auth login                               # token HF
python scripts/record_dataset.py --push-to-hub
python scripts/train.py --push-to-hub
```

## 9. Dépannage

### Patch local lerobot (compat transformers 5.8) — déjà appliqué
Le checkout `lerobot/` a un bug : `GR00TN15Config`
(`lerobot/src/lerobot/policies/groot/groot_n1.py`) plante à l'import sous
`transformers 5.8.1` (« non-default argument 'backbone_cfg' follows default
argument »), ce qui cassait toutes les commandes `lerobot-*`.

Correctif appliqué : `@dataclass` -> `@dataclass(init=False)` (la classe a déjà
un `__init__` personnalisé). N'affecte pas la pipeline classique (qui n'utilise que
`lerobot.motors`). `lerobot/` est un clone indépendant, hors du dépôt : en cas de
réinstallation ou de re-clonage de lerobot, réappliquer ce changement (ou faire un
`git -C lerobot stash` des modifications locales au préalable). Ne pas toucher à
transformers (lerobot exige `transformers>=5.3,<6`, et OWL-ViT en a besoin).

### Patches caméra basse-résolution lerobot — déjà appliqués (édits locaux)
Deux édits locaux dans `lerobot/src/lerobot/cameras/opencv/camera_opencv.py`
(clone hors dépôt, à réappliquer en cas de réinstallation ou de re-clonage de
lerobot). Aucun script de la pipeline classique n'utilise `OpenCVCamera`
(`grep OpenCVCamera scripts/` ne renvoie rien ; les scripts classiques font du
`cv2.VideoCapture` direct), donc ces deux patches sont propres au chemin IL/ACT
(`lerobot-record`).

1. Resize logiciel (capture native 640×480 → sortie 320×240). Les webcams
   ne capturent pas en 320×240 ; lerobot levait `RuntimeError: failed to set
   capture_width=320`. Patch : capturer en natif puis redimensionner la sortie
   (`_validate_width_and_height` + `_postprocess_image`, `cv2.resize INTER_AREA`),
   pour correspondre au modèle ACT basse-résolution. Repérable via les commentaires
   `PATCH` et le warning « capture native … != demande … -> resize de sortie ».

2. Tolérance de frame périmée 500 → 2000 ms (`read_latest`, ligne ~543 :
   `max_age_ms: int = 2000`). cam_2 (poignet, hub USB partagé) a parfois un
   hoquet de lecture (`read failed (status=False)`) qui fait vieillir la dernière
   image > 500 ms → `TimeoutError` qui interrompait l'épisode d'éval en plein grasp
   (observé : 528 ms). `so_follower.get_observation()` appelle `read_latest()`
   sans argument, donc ce défaut est le seul levier (aucun champ `OpenCVCameraConfig`
   ne le pilote). 2000 ms absorbe un hoquet transitoire ; un vrai blocage caméra
   reste détecté ailleurs (`failure_count > 10`).

> Note pince : à l'éval, un crash en plein transport peut faire apparaître
> `RuntimeError: Failed to write 'Torque_Enable' on id_=6 ... [RxPacketError]
> Overload error!` au `disconnect`. `id_=6` = la pince ; l'« Overload » est une
> protection couple/courant du servo, déclenchée car les mâchoires serraient le
> cube au moment de la coupure. C'est un symptôme de nettoyage, pas la cause : il
> disparaît dès que l'épisode va jusqu'au lâcher (la pince s'ouvre, plus de
> charge). Aucune action requise pour un run propre.

### Robustesse éval : décrochage cam_2 + Overload pince — patchs `so_follower.py` (déjà appliqués)
Pour une campagne de tests fiable (sans filmer), 3 édits locaux dans
`lerobot/src/lerobot/robots/so_follower/so_follower.py` (chemin IL uniquement —
la pipeline classique n'instancie pas `SOFollower` ; `scripts/calibrate.py` ne le
cite qu'en commentaire). À réappliquer en cas de réinstallation de lerobot.

1. `get_observation` — décrochage caméra non-fatal. cam_2 cale parfois > 2 s
   (`read failed (status=False)` répétés) → `read_latest()` levait `TimeoutError`
   (2021 ms observé) qui interrompait l'épisode en plein transport. Désormais la
   dernière image valide est réutilisée et l'exécution continue : l'épisode va
   toujours au bout (borné par `episode_time_s`) et se sauvegarde, sans run perdu.
   Seul cas qui lève encore : aucune image jamais reçue (caméra absente/mauvais index).
2. `disconnect` — teardown tolérant + port fermé. Si la pince (id=6) est en
   Overload à l'arrêt, `bus.disconnect()` levait dans `disable_torque()` avant
   `closePort()` → port série laissé ouvert → run d'éval suivant bloqué. L'erreur
   est rattrapée, journalisée, et le port est fermé malgré tout (best-effort).
3. `__init__` — deux dicts de cache (`_last_cam_frame`, `_cam_fail_count`) pour (1).

> Note seuil caméra : `read_latest` a aussi son défaut relevé 500→2000 ms (voir
> patch caméra ci-dessus). Combiné à (1) ci-dessus, cela forme une double protection :
> 2000 ms absorbe les hoquets courts sans bruit, et au-delà (1) prend le relais sans crash.

> Vérif rapide après réinstall : `grep -c "PATCH TFE" so_follower.py` doit valoir 3,
> et `grep -c "PATCH TFE" camera_opencv.py` ≥ 1.

### Warning `objc[...] AVFFrameReceiver ... implemented in both`
Bénin (cv2 et av embarquent tous deux libavdevice sur macOS). N'empêche pas
teleoperate/record. Si un crash caméra survient réellement, lancer avec une seule
des deux libs (par exemple `pip uninstall av`) ; en général ce warning est ignoré.
