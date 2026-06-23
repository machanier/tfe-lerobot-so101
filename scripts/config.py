"""
config.py – Configuration centralisée du robot SO-101

Tous les scripts importent leurs paramètres depuis ici.
Modifie CE SEUL FICHIER quand tu changes de ports USB, de caméra, etc.

Pour trouver tes ports :
    ls /dev/tty.usbmodem*
    ou : lerobot-find-port
"""

# === PORTS USB ===
# Les ports changent à chaque branchement sur macOS !
# Lance `ls /dev/tty.usbmodem*` pour trouver les bons.
FOLLOWER_PORT = "/dev/tty.usbmodem5A460830681"
LEADER_PORT = "/dev/tty.usbmodem5A460816001"

# === IDENTIFIANTS ===
FOLLOWER_ID = "mon_follower"
LEADER_ID = "mon_leader"

# === CAMERAS ===
# Lance `python scripts/preview_camera.py --camera <index>` pour verifier chaque camera.
# Les index peuvent changer si tu rebranches le hub USB !
#
# Pour trouver les index :
#   python scripts/detect_cameras.py
#
# Convention de nommage (les noms cam_X sont fixes, seul l'index OpenCV peut changer) :
#   cam_0 : stereo gauche (eye-to-hand, fixe sur la barriere avant)
#   cam_1 : stereo droite (eye-to-hand, fixe sur la barriere avant)
#   cam_2 : eye-in-hand (montee sur la tete du robot) — deja calibree intrinsequement
#
# /!\ Les index 3, 4 sont la webcam MacBook et l'iPhone (Continuity Camera) — ignores.

CAMERAS = {
    "cam_0": {
        "index": 0,           # A VERIFIER apres branchement
        "role": "stereo_left",
        "width": 1920,
        "height": 1080,
        "fps": 30,
    },
    "cam_1": {
        "index": 1,           # A VERIFIER apres branchement
        "role": "stereo_right",
        "width": 1920,
        "height": 1080,
        "fps": 30,
    },
    "cam_2": {
        "index": 2,           # A VERIFIER apres branchement
        "role": "eye_in_hand",
        "width": 1920,
        "height": 1080,
        "fps": 30,
    },
}

# Raccourcis pour compatibilite avec les anciens scripts
CAMERA_INDEX = CAMERAS["cam_2"]["index"]
CAMERA_WIDTH = CAMERAS["cam_2"]["width"]
CAMERA_HEIGHT = CAMERAS["cam_2"]["height"]
CAMERA_FPS = CAMERAS["cam_2"]["fps"]


# ============================================================================
# === IMITATION LEARNING (ACT) ===  [ajout 2026-06-23]
# Section dediee a la 2e methode du TFE : imitation learning via LeRobot/ACT.
# Purement ADDITIF : la pipeline classique n'utilise rien de ce qui suit.
# Voir docs/IL_ACT_RUNBOOK.md pour la procedure complete.
# ============================================================================
import glob as _glob
import json as _json

# Nom d'utilisateur Hugging Face (datasets + modeles). A adapter si besoin.
HF_USER = "maxence"

# --- Cameras utilisees pour l'IL (RGB uniquement : ACT n'utilise PAS la stereo) ---
# Une vue de scene (eye-to-hand, fixe) + la vue poignet (eye-in-hand).
# Pour changer la camera de scene, remplace "cam_0" par "cam_1" ci-dessous.
IL_SCENE_CAM = "cam_0"     # vue globale  -> cle LeRobot "front"
IL_WRIST_CAM = "cam_2"     # vue poignet  -> cle LeRobot "wrist"

# Resolution/fps dedies a l'IL : 640x480 (plus leger que le 1080p de la stereo).
# ACT redimensionne les images de toute facon ; 640x480 = entrainement plus
# rapide sur MPS, dataset plus petit, et 2 cameras tiennent sur le bus USB.
IL_CAM_WIDTH = 640
IL_CAM_HEIGHT = 480
IL_CAM_FPS = 30

# --- Dataset / tache ---
IL_TASK = "Grab the orange cube"            # phrase courte, verbe en tete
IL_REPO_ID = f"{HF_USER}/so101_orange_cube"
IL_NUM_EPISODES = 50

# --- Entrainement ACT ---
IL_POLICY_TYPE = "act"
IL_POLICY_DEVICE = "mps"     # Apple Silicon ; "cpu" en repli, "cuda" sur GPU NVIDIA
IL_BATCH_SIZE = 8
IL_STEPS = 100_000


def il_cameras_flag():
    """Flag --robot.cameras pour l'IL (2 cameras : "front" + "wrist").

    Les cles "front"/"wrist" DOIVENT rester identiques entre teleoperate,
    record et eval : LeRobot lie chaque observation au NOM de la camera.
    """
    scene_idx = CAMERAS[IL_SCENE_CAM]["index"]
    wrist_idx = CAMERAS[IL_WRIST_CAM]["index"]
    spec = {
        "front": {"type": "opencv", "index_or_path": scene_idx,
                  "width": IL_CAM_WIDTH, "height": IL_CAM_HEIGHT, "fps": IL_CAM_FPS},
        "wrist": {"type": "opencv", "index_or_path": wrist_idx,
                  "width": IL_CAM_WIDTH, "height": IL_CAM_HEIGHT, "fps": IL_CAM_FPS},
    }
    return "--robot.cameras=" + _json.dumps(spec)


def pick_ports():
    """Retourne (follower_port, leader_port), avec auto-detection en repli.

    Utilise FOLLOWER_PORT / LEADER_PORT s'ils sont presents, sinon retombe sur
    les ports /dev/tty.usbmodem* detectes (macOS change les ports a chaque
    branchement). NB : en auto, l'ordre follower/leader n'est pas garanti --
    fixe les vrais ports dans ce fichier via `lerobot-find-port` pour fiabilite.
    """
    detectes = _glob.glob("/dev/tty.usbmodem*")
    if not detectes:
        return None, None
    follower = FOLLOWER_PORT if FOLLOWER_PORT in detectes else None
    leader = LEADER_PORT if LEADER_PORT in detectes else None
    if follower and leader:
        return follower, leader
    if len(detectes) >= 2:
        return detectes[0], detectes[1]
    if len(detectes) == 1:
        return detectes[0], None
    return follower, leader
