# Matériel — pièces imprimées en 3D

Modèles 3D des pièces du poste (format **STL binaire** ; sources `.3mf` en option).

- `structure/` — structure porte-caméras (barrière avant pour la paire stéréo + support de la caméra poignet).
- `boite_depose/` — boîte de dépôt + son matelas TPU.
- `pince_tpu/` — pince compliante imprimée en TPU (mâchoires + cornes).
- `objets/` — objets manipulés imprimés : cube, cylindre, pavé, prisme triangulaire.
- `pieces_robot/` — pièces du SO-101 réimprimées (le cas échéant).

Deux objets testés ne sont pas fournis en STL :
- la **balle** en TPU est un modèle de la communauté Bambu Lab ;
- le **Rubik's Cube** est un objet du commerce.

Pour le montage du bras lui-même (pièces d'origine, câblage), voir les dépôts
officiels indiqués dans le [README principal](../README.md).

Les damiers de calibration ne sont pas des pièces imprimées : ils se génèrent
avec `scripts/generate_chessboard.py`.
