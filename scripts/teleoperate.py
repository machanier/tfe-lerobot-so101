#!/usr/bin/env python3
"""
teleoperate.py – Téléopération du SO-101 (leader → follower)

Usage:
    python scripts/teleoperate.py

    Arrêter avec Ctrl+C.
"""

from lerobot.scripts.control_robot import control_robot
from lerobot.configs import parser


def main():
    # Configuration par défaut pour la téléopération SO-101
    # Adapte les ports USB si nécessaire (utilise `lerobot-find-port` pour les trouver)
    args = parser.parse_args()

    control_robot(
        robot_type="so101",
        robot_overrides=[
            "leader_arms.main.port=/dev/tty.usbmodem*",   # ← adapter si besoin
            "follower_arms.main.port=/dev/tty.usbmodem*", # ← adapter si besoin
        ],
        control_type="teleoperate",
    )


if __name__ == "__main__":
    main()
