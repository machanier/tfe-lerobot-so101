"""
grasp.py - Strategies de grasp planning : ObjectInstance -> GraspPose.

Une `GraspPose` decrit la sequence geometrique d'une saisie :

       T_base_gripper_approach    : pince au-dessus, ouverte
              ▼
       T_base_gripper_grasp       : pince sur l'objet, ferme ici
              ▲
       T_base_gripper_retract     : pince au-dessus, fermee

C'est le SEUL endroit ou la "strategie de saisie" est decidee. La
trajectoire articulaire (IK + interpolation) viendra dans des modules
ulterieurs (control/) ; la planification haut niveau (eviter les obstacles,
choisir un point de vue) viendra au Sprint 4.

Interface ABC `GraspStrategy` pour permettre l'ajout d'autres strategies
sans toucher au reste du pipeline (cf design Sprint 2 : meme pattern que
ObjectDetector).

Implementation V1 : TopDownGrasp. La pince descend verticalement (axe optique
de la pince = -Z_base) sur l'objet, avec wrist_roll aligne sur le grand axe
du contour 2D (utile pour un stylo, un rectangle allonge). Suffisant pour
les objets sur table accessibles par le dessus, ce qui couvre 7-8/9 objets
cibles du TFE.

Reference : Bohg et al. 2014, "Data-Driven Grasp Synthesis - A Survey",
section 2 (taxonomie heuristic / contact-based / data-driven). Top-down est
l'approche heuristique simple, deterministe, justifiable academiquement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.perception.scene import ObjectInstance


# ============================================================
# Types de donnees
# ============================================================


@dataclass
class GraspPose:
    """Sequence geometrique d'une saisie : approach -> grasp -> retract.

    Les 3 poses sont des matrices 4x4 SE(3) representant T_base_gripper
    (position + orientation de la pince dans le repere base du robot).

    Le repere "gripper" suit la convention SO-101 : Z_pince = axe de
    fermeture des doigts ; X_pince = axe "vue de face" de la pince ;
    Y_pince complete (regle de la main droite). En top-down, Z_pince
    pointe vers -Z_base (la pince regarde vers le bas).

    Attributes:
        T_base_gripper_approach : pose pince juste au-dessus, ouverte.
        T_base_gripper_grasp    : pose pince au moment de fermer.
        T_base_gripper_retract  : pose pince apres saisie (= approach
                                  typiquement, mais peut differer si on
                                  veut soulever plus haut pour eviter
                                  les obstacles).
        gripper_open_pct        : ouverture pince avant approach (0=ferme,
                                  100=ouvert).
        gripper_close_pct       : fermeture pince lors du grasp.
        label                   : nom de l'objet cible (debug / log).
        score                   : confiance dans la strategie [0, 1].
        meta                    : metadonnees (strategy, parametres...).
    """

    T_base_gripper_approach: np.ndarray
    T_base_gripper_grasp: np.ndarray
    T_base_gripper_retract: np.ndarray
    gripper_open_pct: float = 100.0
    gripper_close_pct: float = 0.0
    label: str = ""
    score: float = 1.0
    meta: dict = field(default_factory=dict)


# ============================================================
# Interface abstraite
# ============================================================


class GraspStrategy(ABC):
    """Interface : prend une ObjectInstance, retourne une GraspPose (ou None)."""

    @abstractmethod
    def plan(self, obj: ObjectInstance) -> Optional[GraspPose]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ============================================================
# Strategy V1 : top-down
# ============================================================


def _rotation_top_down(yaw_rad: float = 0.0) -> np.ndarray:
    """Construit la rotation R_base_gripper d'une saisie top-down.

    Convention : Z_pince pointe vers -Z_base (la pince regarde vers le bas),
    X_pince = (cos(yaw), sin(yaw), 0) (horizontal, ajustable via yaw),
    Y_pince = Z_pince x X_pince = (sin(yaw), -cos(yaw), 0).

    R a les vecteurs (X_pince, Y_pince, Z_pince) en colonnes :
        R = [[ cos(yaw),  sin(yaw),  0],
             [ sin(yaw), -cos(yaw),  0],
             [        0,         0, -1]]

    On verifie det(R) = +1 (rotation propre, pas reflexion). C'est
    equivalent a une rotation de pi autour de X suivi d'une rotation de
    yaw autour du nouveau Z.

    Note : inverser un seul axe (e.g. Z) sans corriger un autre donne une
    REFLEXION (det = -1), inutilisable comme rotation.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([
        [c,   s,   0.0],
        [s,  -c,   0.0],
        [0.0, 0.0, -1.0],
    ])
    return R


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def yaw_from_contour(contour: Optional[np.ndarray]) -> float:
    """Calcule l'angle du grand axe d'un contour 2D, dans [-pi/2, pi/2].

    Utilise les moments centraux du contour (equivalent ACP 2D) :
        theta = 0.5 * atan2(2 * mu11, mu20 - mu02)

    Si pas de contour ou contour degenere, renvoie 0.0 (alignement par
    defaut, axe X_pince = X_base).

    Note : pour un cube, le grand axe est mal defini (carre) -> theta
    proche de 0, ce qui est OK : la pince attaque selon X_base.

    Reference : OpenCV doc, image moments ; equivaut a la composante
    principale d'une ACP sur les pixels du contour.
    """
    if contour is None or len(contour) < 3:
        return 0.0
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy
    mu20 = float((x * x).sum())
    mu02 = float((y * y).sum())
    mu11 = float((x * y).sum())
    # Cas degenere : forme quasi-symetrique (carre, disque). L'axe principal
    # est mal defini. On renvoie 0 plutot qu'une valeur arbitraire.
    span = max(mu20 + mu02, 1e-12)
    if (abs(mu20 - mu02) / span < 0.05) and (abs(mu11) / span < 0.05):
        return 0.0
    # angle dans le repere image (Y vers le bas). On le retourne tel quel ;
    # la conversion image -> base est responsabilite du caller s'il y a
    # une ambiguite ; pour le top-down V1 on prend l'angle image comme
    # estimation grossiere du wrist_roll. C'est une approximation : un yaw
    # bien calcule demanderait de projeter le contour 3D du dessus de l'objet,
    # ce qui est hors scope V1.
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return float(theta)


class TopDownGrasp(GraspStrategy):
    """Saisie verticale par le haut. Strategie heuristique simple.

    Parametres :
        approach_height_m : hauteur de la pose d'approche au-dessus de l'objet (m).
        grasp_offset_m    : decalage Z entre le centroide de l'objet et la pose
                            grasp. Souvent on grasp AU CENTROIDE (offset 0), mais
                            pour les objets hauts (gobelet), on grasp un peu plus
                            bas que le centroide, donc grasp_offset_m < 0 ramene
                            la pince plus bas.
        retract_height_m  : hauteur du retract apres saisie (m).
        gripper_open_pct  : ouverture max de la pince (defaut 100 = grand ouvert).
        gripper_close_pct : fermeture pour saisir. Pour les objets fragiles,
                            ne pas serrer trop fort (~5-15) ; pour cube rigide, 0.
        align_wrist_roll  : si vrai, oriente wrist_roll selon le grand axe du
                            contour detecte (utile pour un stylo, un rectangle).
        max_object_height_m : objet plus haut que ca -> pas de top-down (retour None).
    """

    def __init__(self,
                 approach_height_m: float = 0.08,
                 grasp_offset_m: float = 0.0,
                 retract_height_m: float = 0.10,
                 gripper_open_pct: float = 100.0,
                 gripper_close_pct: float = 0.0,
                 align_wrist_roll: bool = True,
                 max_object_height_m: float = 0.12,
                 # --- A2 : decalage smart vers la pince fixe ---
                 # Le SO-101 a une pince ASYMETRIQUE : un doigt fixe, un
                 # doigt mobile qui ferme contre le fixe. Si on centre le
                 # grasp sur le centroide objet, le doigt fixe percute
                 # l'objet AVANT la fermeture -> l'objet bouge, saisie rate.
                 # Solution : decaler le grasp pour que l'objet finisse
                 # contre le doigt fixe (= le doigt fixe touche un bord
                 # de l'objet, le doigt mobile vient l'ecraser).
                 grasp_lateral_offset_mm: float = 8.0,
                 # Cote du doigt fixe dans le repere PINCE.
                 # SO-101 de Maxence : doigt fixe cote Y_base+ (gauche du
                 # robot, vu de derriere). Avec _rotation_top_down(yaw=0),
                 # Y_pince = -Y_base, donc doigt fixe a Y_pince-.
                 # Si on cherche un offset OPPOSE au doigt fixe : +Y_pince.
                 # En vecteur unitaire dans le repere PINCE :
                 fixed_finger_dir_gripper: tuple = (0, -1, 0),
                 # --- A3 : ouverture pince adaptative ---
                 # Si True ET que bbox_3d_m est fourni dans l'ObjectInstance,
                 # l'ouverture pince est calculee selon la largeur objet
                 # (plutot que d'ouvrir grand pour rien). Formule :
                 # pct = (largeur + 2*marge) / largeur_max_pince * 100.
                 adaptive_gripper_open: bool = True,
                 # marge d'ouverture de CHAQUE cote. 12mm pour ABSORBER l'erreur
                 # de position (~8mm d'IK + calibration) : sans ca, la pince
                 # ouvre trop juste et le doigt FIXE percute l'objet a la
                 # descente au lieu de le degager (#3 signale par Maxence).
                 gripper_open_margin_mm: float = 12.0,
                 gripper_max_opening_mm: float = 50.0,
                 # --- D-Z : ancrage de la profondeur de prise sur la table ---
                 table_z_m: float = 0.0,                 # plan table (repere base)
                 min_grasp_clearance_m: float = 0.005,   # ne jamais saisir sous +5mm
                 ):
        self.approach_height_m = approach_height_m
        self.grasp_offset_m = grasp_offset_m
        self.retract_height_m = retract_height_m
        self.gripper_open_pct = gripper_open_pct
        self.gripper_close_pct = gripper_close_pct
        self.align_wrist_roll = align_wrist_roll
        self.max_object_height_m = max_object_height_m
        self.grasp_lateral_offset_mm = grasp_lateral_offset_mm
        self.fixed_finger_dir_gripper = np.asarray(fixed_finger_dir_gripper,
                                                     dtype=np.float64)
        self.adaptive_gripper_open = adaptive_gripper_open
        self.gripper_open_margin_mm = gripper_open_margin_mm
        self.gripper_max_opening_mm = gripper_max_opening_mm
        self.table_z_m = table_z_m
        self.min_grasp_clearance_m = min_grasp_clearance_m

    @property
    def name(self) -> str:
        return "TopDownGrasp"

    def plan(self, obj: ObjectInstance) -> Optional[GraspPose]:
        x, y, z = float(obj.position_base_m[0]), float(obj.position_base_m[1]), float(obj.position_base_m[2])

        # Rejet : objet trop haut (impossible en top-down sans collision pince/objet)
        z_top_estime = z + 0.5 * (obj.bbox_3d_m[2] if obj.bbox_3d_m else 0.05)
        if z_top_estime > self.max_object_height_m:
            return None

        # Yaw : aligne la pince sur le grand axe de l'objet.
        # SOURCE PREFEREE : le yaw REPERE BASE calcule par la perception
        # (pose_estimator._estimate_geometry, projection rayon-plan du contour
        # sur le plan de l'objet). Correct quelle que soit l'orientation des
        # cameras -> les objets poses EN BIAIS sont geres. La perception
        # signale aussi la classe de pose :
        #   "debout"  -> empreinte au sol quasi circulaire, yaw libre (0).
        #      (Avant : le contour vu DE COTE d'un cylindre debout etait
        #      allonge verticalement dans l'image -> yaw fantome ~±90deg
        #      et rotation de poignet inutile. Diagnostic 2026-06-12.)
        #   "couche"  -> yaw_base_rad = grand axe de l'empreinte.
        #   "compact" -> pas de grand axe fiable, yaw 0.
        # FALLBACK (perception sans info de pose) : angle image brut du
        # contour cam_0/cam_1, comme en V1 (approximation documentee).
        yaw = 0.0
        meta_obj = obj.meta or {}
        if self.align_wrist_roll:
            if "pose_class" in meta_obj:
                yaw_base = meta_obj.get("yaw_base_rad")
                if yaw_base is not None:
                    yaw = float(yaw_base)
            elif obj.source_detections:
                # Legacy V1 : cherche un contour non degenere et prend
                # l'angle IMAGE tel quel (suppose cameras ~alignees base).
                for det in obj.source_detections:
                    if det.contour is not None and len(det.contour) >= 3:
                        if det.cam_key in ("cam_0", "cam_1"):
                            yaw = yaw_from_contour(det.contour)
                            break
                        # cam_2 : skip, on laisse yaw=0
        R = _rotation_top_down(yaw)

        # === A2 : decalage smart vers la pince fixe ===
        # On veut que l'objet finisse contre le doigt fixe (l'autre ferme
        # vers lui). Donc on decale le centre du grasp DANS LA DIRECTION
        # OPPOSEE au doigt fixe (en repere pince), puis on transforme dans
        # le repere base via R.
        opposite_dir_gripper = -self.fixed_finger_dir_gripper
        offset_base = R @ opposite_dir_gripper * (self.grasp_lateral_offset_mm / 1000.0)
        # On decale approach/grasp/retract pareillement (pour rester aligne)
        grasp_xy = np.array([x, y, 0.0]) + np.array([offset_base[0], offset_base[1], 0.0])
        gx, gy = float(grasp_xy[0]), float(grasp_xy[1])

        # === D-Z : profondeur de prise ANCREE SUR LA TABLE ===
        # Le Z stereo est l'axe le plus bruite (cf memoire, "sensibilite de la
        # profondeur a l'erreur de detection") : il peut tomber sous la table
        # -> la pince force dans le sol. Comme l'objet REPOSE sur la table
        # (Z=0, base_link au niveau de la table), on derive la profondeur de
        # prise du plan table + la moitie de la hauteur estimee (= milieu de
        # l'objet en hauteur), au lieu du Z detecte. Garde-fou : jamais sous
        # la table + une marge.
        if obj.bbox_3d_m is not None:
            obj_height_m = float(obj.bbox_3d_m[2])
            z_grasp = self.table_z_m + obj_height_m / 2.0    # milieu en hauteur
        else:
            z_grasp = z + self.grasp_offset_m                # fallback : Z detecte
        z_grasp = max(z_grasp, self.table_z_m + self.min_grasp_clearance_m)

        # 3 poses : approach, grasp, retract (meme (gx, gy), Z relatif au grasp)
        T_approach = _se3(R, [gx, gy, z_grasp + self.approach_height_m])
        T_grasp    = _se3(R, [gx, gy, z_grasp])
        T_retract  = _se3(R, [gx, gy, z_grasp + self.retract_height_m])

        # === A3 : ouverture pince adaptative selon bbox 3D ===
        # Si on a bbox_3d_m, on calcule l'ouverture optimale (plutot que
        # d'ouvrir grand). Formule : pct = (largeur + 2 * marge) / max_pince
        # On prend min(X, Y) car la pince ferme dans le plan XY.
        gripper_open = self.gripper_open_pct
        if self.adaptive_gripper_open and obj.bbox_3d_m is not None:
            obj_width_m = min(obj.bbox_3d_m[0], obj.bbox_3d_m[1])
            target_open_mm = obj_width_m * 1000 + 2 * self.gripper_open_margin_mm
            pct = (target_open_mm / self.gripper_max_opening_mm) * 100.0
            # Clip a [30, 100] : evite des ouvertures trop petites qui empechent
            # l'insertion + on plafonne a 100%
            gripper_open = float(np.clip(pct, 30.0, 100.0))

        return GraspPose(
            T_base_gripper_approach=T_approach,
            T_base_gripper_grasp=T_grasp,
            T_base_gripper_retract=T_retract,
            gripper_open_pct=gripper_open,
            gripper_close_pct=self.gripper_close_pct,
            label=obj.label,
            score=obj.score,
            meta={
                "strategy": "TopDownGrasp",
                "yaw_rad": yaw,
                "approach_height_m": self.approach_height_m,
                "grasp_offset_m": self.grasp_offset_m,
                "retract_height_m": self.retract_height_m,
                "lateral_offset_mm": self.grasp_lateral_offset_mm,
                "offset_base_xy_mm": (float(offset_base[0]*1000), float(offset_base[1]*1000)),
                "gripper_open_pct_computed": gripper_open,
            },
        )


# ============================================================
# Self-tests (lance avec : python -m src.planning.grasp)
# ============================================================
if __name__ == "__main__":
    print("Tests grasp.py")
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.perception.scene import Detection2D, ObjectInstance

    # 1. _rotation_top_down : Z_pince = -Z_base, det(R) = +1
    R = _rotation_top_down(0.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R non orthonormale"
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, "det(R) != 1 (rotation impropre)"
    Z_pince_base = R @ np.array([0, 0, 1.0])  # axe Z de la pince exprime en base
    assert np.allclose(Z_pince_base, [0, 0, -1]), f"Z_pince devrait pointer vers -Z_base, recu {Z_pince_base}"
    print(f"  [OK] _rotation_top_down (yaw=0) : Z_pince = -Z_base, R orthonormale")

    # 2. yaw_from_contour : rectangle horizontal -> theta ~ 0,
    #    rectangle vertical -> theta ~ +/-pi/2
    rect_horiz = np.array([[0, 0], [100, 0], [100, 10], [0, 10]], dtype=float)
    rect_vert  = np.array([[0, 0], [10, 0], [10, 100], [0, 100]], dtype=float)
    theta_h = yaw_from_contour(rect_horiz)
    theta_v = yaw_from_contour(rect_vert)
    assert abs(theta_h) < 0.05, f"rect horizontal : theta = {theta_h} attendu ~0"
    assert abs(abs(theta_v) - np.pi / 2) < 0.05, f"rect vertical : theta = {theta_v} attendu +/-pi/2"
    print(f"  [OK] yaw_from_contour : horiz={np.degrees(theta_h):+.1f} deg, vert={np.degrees(theta_v):+.1f} deg")

    # 3. TopDownGrasp.plan : cube en (0.15, 0, 0.03) -> 3 poses verticales
    obj = ObjectInstance(
        label="red_cube",
        position_base_m=np.array([0.15, 0.0, 0.03]),
        bbox_3d_m=(0.03, 0.03, 0.03),
    )
    # Test sans decalage smart ni decalage Z adaptatif pour valider la
    # logique de base : les 3 poses doivent etre verticalement alignees au
    # cube. Avec le decalage smart (defaut 8mm), le grasp est decale en Y
    # pour aligner avec la pince fixe -- teste plus bas. Idem pour Fix 7
    # (grasp 1/4 sous centre) : on desactive en passant bbox_3d_m=None
    # pour ce test puis on teste avec bbox plus bas.
    obj_no_bbox = ObjectInstance(
        label="red_cube",
        position_base_m=np.array([0.15, 0.0, 0.03]),
    )
    strategy = TopDownGrasp(approach_height_m=0.08, retract_height_m=0.10,
                              grasp_lateral_offset_mm=0.0)
    gp = strategy.plan(obj_no_bbox)
    assert gp is not None, "plan() devrait reussir pour un cube standard"
    # Verifie les 3 positions (alignees au cube vu que offset=0 et bbox=None)
    assert np.allclose(gp.T_base_gripper_approach[:3, 3], [0.15, 0, 0.11]), \
        f"approach Z attendu 0.11m, recu {gp.T_base_gripper_approach[:3, 3]}"
    assert np.allclose(gp.T_base_gripper_grasp[:3, 3], [0.15, 0, 0.03])
    assert np.allclose(gp.T_base_gripper_retract[:3, 3], [0.15, 0, 0.13])
    # Verifie l'orientation : Z_pince vers le bas
    Zp = gp.T_base_gripper_grasp[:3, :3] @ np.array([0, 0, 1.0])
    assert np.allclose(Zp, [0, 0, -1])
    print(f"  [OK] TopDownGrasp.plan (offset=0) : 3 poses verticales pour cube a (15, 0, 3) cm")

    # 3bis. TopDownGrasp avec decalage smart : le grasp est decale d'offset
    # mm dans la direction OPPOSEE au doigt fixe + grasp Z 1/4 sous centre.
    strategy_smart = TopDownGrasp(approach_height_m=0.08, retract_height_m=0.10,
                                    grasp_lateral_offset_mm=8.0)
    gp_smart = strategy_smart.plan(obj)  # cube 30mm bbox, centre Z=0.03
    # Avec yaw=0 et fixed_finger_dir_gripper=(0,-1,0), l'offset oppose dans
    # le repere base est (0, -0.008, 0). Donc grasp Y = -0.008.
    assert abs(gp_smart.T_base_gripper_grasp[1, 3] - (-0.008)) < 1e-6, \
        f"grasp Y attendu -0.008 (decalage smart), recu {gp_smart.T_base_gripper_grasp[1, 3]}"
    # Avec Fix 7 : grasp Z = centre - hauteur/4 = 0.03 - 0.03/4 = 0.0225
    assert abs(gp_smart.T_base_gripper_grasp[2, 3] - 0.0225) < 1e-6, \
        f"grasp Z attendu 0.0225 (1/4 sous centre), recu {gp_smart.T_base_gripper_grasp[2, 3]}"
    print(f"  [OK] TopDownGrasp.plan (offset smart 8mm + Z=1/4 sous centre)")

    # 4. Filtre objet trop haut : gobelet de 15 cm -> plan() = None
    obj_hi = ObjectInstance(
        label="tall_cup",
        position_base_m=np.array([0.15, 0.0, 0.07]),
        bbox_3d_m=(0.08, 0.08, 0.15),  # hauteur 15 cm
    )
    gp_hi = TopDownGrasp(max_object_height_m=0.12).plan(obj_hi)
    assert gp_hi is None, "objet de 15 cm devrait etre rejete par max_object_height_m=0.12"
    print(f"  [OK] TopDownGrasp.plan : objet > max_object_height_m -> None")

    # 5. Alignement wrist_roll sur le contour : rectangle vertical -> yaw ~ +/-pi/2
    det = Detection2D(cam_key="cam_0", label="yellow_rect",
                      center_px=(0, 0), contour=rect_vert)
    obj_rect = ObjectInstance(
        label="yellow_rect", position_base_m=np.array([0.10, 0.05, 0.02]),
        source_detections=[det],
    )
    gp_r = TopDownGrasp(align_wrist_roll=True).plan(obj_rect)
    assert gp_r is not None
    assert abs(abs(gp_r.meta["yaw_rad"]) - np.pi / 2) < 0.05, \
        f"yaw_rad attendu +/-pi/2, recu {gp_r.meta['yaw_rad']}"
    print(f"  [OK] align_wrist_roll : yaw = {np.degrees(gp_r.meta['yaw_rad']):+.1f} deg pour rect vertical")

    # 6. ABC : GraspStrategy abstraite
    try:
        GraspStrategy()  # type: ignore[abstract]
        raise AssertionError("aurait du lever TypeError")
    except TypeError:
        print("  [OK] GraspStrategy est abstrait")

    print("Tous les tests passent.")
