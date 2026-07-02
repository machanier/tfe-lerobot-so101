"""
src.planning - Module de planification de saisie pour le SO-101.

Sous-modules :
    grasp : strategies de grasp planning (approach, grasp, retract).

Le grasp planning traduit une ObjectInstance (position 3D) en une trajectoire
geometrique de la pince (3 poses + ouvertures/fermetures). La trajectoire
articulaire (IK + interpolation) est ensuite produite par les modules src/control/.

Reference : Bohg et al. 2014, "Data-Driven Grasp Synthesis - A Survey".
"""
