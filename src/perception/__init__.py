"""
src.perception - Module de perception multi-cameras pour le SO-101.

Pipeline du Sprint 2 : `objet visible -> position 3D dans le repere base du robot`.

Sous-modules :
    scene             : dataclasses (ObjectInstance, Scene, Detection2D, Frame).
    camera_io         : capture synchronisee des 3 cameras (live ou replay).
    detector          : ObjectDetector abstrait + HSVDetector + stub HFDetector.
    pose_estimator    : triangulation stereo + PnP monoculaire (fallback).
    robot_state       : lecture des moteurs + cinematique directe (FK).

L'architecture suit la separation perception/planning/control imposee par le
cahier des charges du TFE (Partie I). Toutes les positions 3D sont exprimees
dans le repere BASE du robot, en METRES.

References : Hartley & Zisserman 2018 (triangulation), Bohg et al. 2014 (grasp).
"""
