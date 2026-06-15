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
    """Objet haut (18cm) a portee -> top-down exclu, prise inclinee retenue."""
    gp, r_app, r_grp, r_ret = _pipeline("adaptive")._plan_and_solve_grasp(
        _obj([0.20, 0.0, 0.09], (0.04, 0.04, 0.18)), _Q0)
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


if __name__ == "__main__":
    test_flat_object_picks_top_down()
    print("  [OK] objet plat -> top-down (theta=0) retenu")
    test_tall_object_picks_tilted()
    print("  [OK] objet haut -> prise inclinee retenue (top-down exclu)")
    test_unreachable_object_returns_none()
    print("  [OK] objet hors d'atteinte -> aucune prise (repli borne)")
    test_top_down_mode_rejects_tall_object()
    print("  [OK] mode top-down + objet haut -> aucune prise")
    print("Tous les tests de selection passent.")
