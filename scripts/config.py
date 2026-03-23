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

# === CAMERA ===
# Lance `python scripts/preview_camera.py` pour vérifier l'index
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
