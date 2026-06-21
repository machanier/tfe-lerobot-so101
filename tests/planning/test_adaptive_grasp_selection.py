"""
test_adaptive_grasp_selection.py - Test d'integration de la SELECTION d'angle de
saisie adaptative, SANS hardware.

Construit le pipeline hors ligne (detecteur HSV, dry-run -> ni cameras ni moteurs
necessaires a l'instanciation) et appelle directement
`PickAndPlacePipeline._plan_and_solve_grasp` avec des ObjectInstance synthetiques,
pour verifier le coeur "generate -> filter(IK) -> rank" :
  - objet bas/plat        -> top-down (theta=0) retenu (non-regression) ;
  - objet haut            -> top-down exclu, prise inclinee retenue ;
  - objet hors d'atteinte -> repli BORNE -> aucune prise (pas d'execution forcee) ;
  - mode --top-down + objet haut -> aucune prise (comme avant).

Lance via :
    python -m pytest tests/planning -v
ou directement :
    python tests/planning/test_adaptive_grasp_selection.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from src.calibration.forward_kinematics import ARM_JOINTS  # noqa: E402
from src.perception.scene import ObjectInstance  # noqa: E402
from src.pipeline import PickAndPlacePipeline, PipelineConfig  # noqa: E402

_Q0 = {j: 0.0 for j in ARM_JOINTS}
# Depart REALISTE : le robot planifie une saisie depuis ~son home, ou wrist_roll
# vaut ~-100deg (raw 900, cf calibration_follower.json). Partir de wrist_roll=0
# (irrealiste) empeche le solveur de trouver la prise de cote (il reste coince au
# bord de la fenetre anti-flip) ; depuis le home il la trouve. cf memoire
# wrist-roll-side-grasp-diagnostic.
_Q_HOME = {**_Q0, "wrist_roll": np.radians(-100.8)}


def _pipeline(mode="adaptive"):
    cfg = PipelineConfig(target_label="orange_cube", detector_kind="hsv",
                         dry_run=True, grasp_mode=mode)
    return PickAndPlacePipeline(cfg)


def _obj(pos, bbox):
    return ObjectInstance(label="x", position_base_m=np.array(pos), bbox_3d_m=bbox)


def test_flat_object_picks_top_down():
    """Cube bas/plat -> candidat top-down (theta=0) retenu = TopDownGrasp."""
    gp, r_app, r_grp, r_ret = _pipeline("adaptive")._plan_and_solve_grasp(
        _obj([0.20, 0.0, 0.015], (0.03, 0.03, 0.03)), _Q0)
    assert gp is not None, "le cube plat devrait etre saisissable"
    assert gp.meta["pitch_deg"] == 0.0, f"attendu top-down, recu {gp.meta['pitch_deg']}"
    assert gp.meta["strategy"] == "TopDownGrasp"  # delegation en theta=0


def test_tall_object_picks_tilted():
    """Objet haut (18cm) a portee, DEPART du home -> top-down exclu, prise de cote.

    NB : on part de _Q_HOME (pas _Q0) car l'IK des prises de cote a besoin d'un
    depart realiste (cf wrist-roll-side-grasp-diagnostic). C'est le cas reel : le
    pipeline planifie depuis la pose courante du robot, proche du home.
    """
    gp, r_app, r_grp, r_ret = _pipeline("adaptive")._plan_and_solve_grasp(
        _obj([0.20, 0.0, 0.09], (0.04, 0.04, 0.18)), _Q_HOME)
    assert gp is not None, "l'objet haut devrait etre saisissable de biais/de face"
    assert gp.meta["pitch_deg"] != 0.0, "un objet de 18cm ne devrait pas etre top-down"
    assert r_grp.translation_err_mm <= 8.0, "la prise retenue doit etre atteignable"


def test_unreachable_object_returns_none():
    """Objet hors d'atteinte -> repli BORNE -> aucune prise (pas d'execution forcee)."""
    gp, r_app, r_grp, r_ret = _pipeline("adaptive")._plan_and_solve_grasp(
        _obj([0.60, 0.0, 0.09], (0.04, 0.04, 0.18)), _Q0)
    assert gp is None, "une pose hors d'atteinte ne doit PAS etre executee (repli borne)"


def test_top_down_mode_rejects_tall_object():
    """Mode --top-down + objet trop haut -> aucune prise (comportement historique)."""
    gp, r_app, r_grp, r_ret = _pipeline("top_down")._plan_and_solve_grasp(
        _obj([0.20, 0.0, 0.09], (0.04, 0.04, 0.18)), _Q0)
    assert gp is None, "le top-down doit refuser un objet de 18cm"


def test_preferred_pitch_zones():
    """L'angle prefere suit les ZONES (distance + hauteur), reglables."""
    from src.planning.grasp import preferred_pitch_deg
    # --- objet BAS : la distance decide ---
    # proche (d<=33cm) -> top-down
    assert preferred_pitch_deg(_obj([0.20, 0.0, 0.015], (0.05, 0.05, 0.02))) == 0.0
    # mi-distance (33<d<42cm) -> diagonale
    assert preferred_pitch_deg(_obj([0.37, 0.0, 0.015], (0.03, 0.03, 0.03))) == 45.0
    # loin (d>=42cm) ET BAS/PLAT -> diagonale 45, JAMAIS 90 (revision 2026-06-21,
    # essais cube : une 90 sur un objet plat ferme au-dessus / pousse l'objet ;
    # le 90 est reserve aux objets hauts/debout qui ont un vrai flanc a serrer).
    assert preferred_pitch_deg(_obj([0.43, 0.0, 0.015], (0.03, 0.03, 0.03))) == 45.0
    # --- objet HAUT (sommet>12cm) ou DEBOUT -> TOUJOURS face (90), jamais 45 ---
    # (Maxence 2026-06-20 : limite de hauteur sur le 45 ; un objet haut se prend
    # par le flanc. Si 90 inatteignable de pres, le pipeline retombe sur 45/0.)
    # haut + proche -> 90 (plus de 45 pour un objet haut)
    assert preferred_pitch_deg(_obj([0.20, 0.0, 0.09], (0.04, 0.04, 0.18))) == 90.0
    # haut + mi -> face (objet haut = angle plus grand des la mi-distance)
    assert preferred_pitch_deg(_obj([0.37, 0.0, 0.09], (0.04, 0.04, 0.18))) == 90.0
    # haut + loin -> face
    assert preferred_pitch_deg(_obj([0.43, 0.0, 0.09], (0.04, 0.04, 0.18))) == 90.0


def test_reorient_oriented_aligns_jaws_tilted():
    """replan_oriented (re-alignement cam_2 INCLINE) : a pitch 45 ET 90, l'axe des
    machoires u est PERPENDICULAIRE au grand axe mesure -> machoires en travers du
    PETIT cote, et le PITCH n'est PAS modifie. Verifie la geometrie (sans robot).
    (Le top-down reste gere par reorient_grasp_pose, chemin existant.)"""
    from src.planning.grasp import AdaptiveGrasp
    strat = AdaptiveGrasp()
    yaw = np.radians(30.0)                          # grand axe a +30deg
    long_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    obj = _obj([0.30, 0.0, 0.03], (0.07, 0.03, 0.03))   # objet allonge
    tested = 0
    for pitch in (45.0, 90.0):
        gp = strat.replan_oriented(obj, pitch, yaw)
        assert gp is not None, f"pitch {pitch}: replan_oriented infaisable (attendu faisable)"
        psi = float(gp.meta["azimuth_rad"])        # azimut d'approche retenu
        u = np.array([-np.sin(psi), np.cos(psi), 0.0])   # axe machoires physique
        assert abs(float(u @ long_axis)) < 1e-6, \
            f"pitch {pitch}: machoires pas perpendiculaires au grand axe (u.l={u@long_axis})"
        # pitch INCHANGE (on ne modifie que l'orientation des machoires)
        assert abs(abs(gp.meta["pitch_deg"]) - pitch) < 1e-6, \
            f"pitch {pitch}: le pitch a ete modifie ({gp.meta['pitch_deg']})"
        tested += 1
    assert tested == 2


if __name__ == "__main__":
    test_flat_object_picks_top_down()
    print("  [OK] objet plat -> top-down (theta=0) retenu")
    test_tall_object_picks_tilted()
    print("  [OK] objet haut -> prise inclinee retenue (top-down exclu)")
    test_unreachable_object_returns_none()
    print("  [OK] objet hors d'atteinte -> aucune prise (repli borne)")
    test_top_down_mode_rejects_tall_object()
    print("  [OK] mode top-down + objet haut -> aucune prise")
    test_preferred_pitch_zones()
    print("  [OK] angle prefere = f(zone) : proche->0, mi->45, loin/haut->90")
    test_reorient_oriented_aligns_jaws_tilted()
    print("  [OK] reorient cam_2 incline : machoires perpendiculaires au grand axe a 45/90 (pitch inchange)")
    print("Tous les tests de selection passent.")
