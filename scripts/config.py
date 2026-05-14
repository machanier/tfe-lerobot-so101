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
