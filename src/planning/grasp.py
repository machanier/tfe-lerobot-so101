"""
grasp.py - Strategies de grasp planning : ObjectInstance -> GraspPose.

Une `GraspPose` decrit la sequence geometrique d'une saisie :

       T_base_gripper_approach    : pince au-dessus, ouverte
              ▼
       T_base_gripper_grasp       : pince sur l'objet, ferme ici
              ▲
       T_base_gripper_retract     : pince au-dessus, fermee

C'est le seul endroit ou la strategie de saisie est decidee. La trajectoire
articulaire (IK + interpolation) est produite par les modules de commande
(control/) ; la planification haut niveau (evitement d'obstacles, choix d'un
point de vue) reste hors du perimetre de ce module.

Interface ABC `GraspStrategy` pour permettre l'ajout d'autres strategies sans
toucher au reste du pipeline (meme motif que ObjectDetector).

Strategies disponibles :
  - TopDownGrasp : pince verticale (axe d'approche = -Z_base), wrist_roll aligne
    sur le grand axe de l'objet. Cas particulier de l'adaptatif (theta=0).
  - AdaptiveGrasp (defaut deploiement) : balaie le plan sagittal et propose
    plusieurs angles d'attaque -- top-down (0deg), diagonale (45deg), face avant
    (90deg) -- dans l'ordre de preference selon la zone d'usage (distance + hauteur,
    cf preferred_pitch_deg / GRASP_ZONE_*). Le pipeline filtre ensuite par
    atteignabilite IK et garde le premier faisable. Pour un objet allonge, l'azimut
    d'approche est aligne sur le grand axe, de sorte que les machoires serrent le
    petit cote.

Conventions de repere (deploiement, cf PipelineConfig) :
  - z=0 = la plaque ou reposent les objets (table_z_m=0) ; l'offset de serrage
    des machoires est gere a part (gripper_grab_offset_m).
  - convention pince : roll/yaw decale de grasp_yaw_offset_deg (=90, mesure terrain).
  - decalage de prise : offset lateral en repere pince (le long des machoires),
    applique apres le raffinement cam_2 et aligne sur l'orientation finale du grasp,
    cf pipeline (pince asymetrique SO-101).
  - reorientation cam_2 : reorient_grasp_pose (top-down) / AdaptiveGrasp.replan_oriented
    (incline) realignent les machoires sur le grand axe vu par cam_2.

Reference : Bohg et al. 2014, "Data-Driven Grasp Synthesis - A Survey",
section 2 (taxonomie heuristic / contact-based / data-driven). Approche
heuristique geometrique, deterministe, justifiable academiquement.
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

    Le repere gripper suit la convention SO-101 : Z_pince = axe de
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
    equivalent a une rotation de pi autour de X suivie d'une rotation de
    yaw autour du nouveau Z.

    Note : inverser un seul axe (par exemple Z) sans corriger un autre donne une
    reflexion (det = -1), inutilisable comme rotation.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([
        [c,   s,   0.0],
        [s,  -c,   0.0],
        [0.0, 0.0, -1.0],
    ])
    return R


def _rotation_grasp(azimuth_rad: float, pitch_rad: float,
                    roll_rad: float = 0.0) -> np.ndarray:
    """Rotation R_base_gripper d'une saisie a angle d'attaque quelconque dans le
    plan sagittal (le plan vertical base->objet que pointe le shoulder_pan).

    Generalise `_rotation_top_down` au tangage (pitch) : la pince balaie les
    180deg du plan sagittal, du sol cote face avant -> par-dessus -> sol cote
    face arriere. C'est l'axe naturel du SO-101 (pas de pan au poignet, donc pas
    de prise laterale gauche/droite).

    Conventions (colonnes X_pince, Y_pince, Z_pince) :
      - azimuth (phi) = atan2(y_obj, x_obj) : direction horizontale vers l'objet.
      - pitch (theta) signe : 0 = top-down ; +90deg = frontal (axe d'approche
        +r_hat, face avant) ; -90deg = face arriere. L'axe d'approche (vers
        l'objet) est :
            Z_pince = (sin th cos phi, sin th sin phi, -cos th)
        (theta=0 -> (0,0,-1) = top-down ; theta=+90,phi=0 -> (1,0,0) = frontal).
      - roll : rotation des machoires autour de l'axe d'approche. roll=0 met
        l'axe des machoires (Y_pince) perpendiculaire au plan sagittal (l_hat,
        horizontal). Le caller choisit roll pour aligner les machoires en
        travers du petit cote (cf AdaptiveGrasp).
      - X_pince = Y_pince x Z_pince (repere direct, det(R) = +1).

    A theta=0, en choisissant le roll qui aligne Y_pince sur (sin psi, -cos psi),
    R est identique a `_rotation_top_down(psi)` (verifie en self-test) : le
    top-down reste un cas particulier exact, donc pas de regression.
    """
    phi, th = float(azimuth_rad), float(pitch_rad)
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(th), np.sin(th)
    a = np.array([sth * cphi, sth * sphi, -cth])      # Z_pince (axe d'approche)
    u = np.array([-sphi, cphi, 0.0])                   # l_hat : horizontal, ⟂ plan
    v = np.cross(a, u)                                 # complete (a, u, v) direct
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    Y = cr * u + sr * v                                # axe des machoires
    X = np.cross(Y, a)                                 # X = Y x Z
    R = np.column_stack([X, Y, a])
    return R


def _extent_along(bbox_3d_m: tuple, direction: np.ndarray) -> float:
    """Largeur d'appui (support width) d'une bbox AABB le long d'une direction.

    Pour une boite alignee aux axes de dimensions (dx, dy, dz), l'extension le
    long d'un vecteur unitaire d est |dx*dx_comp| + |dy*dy_comp| + |dz*dz_comp|.
    Sert a estimer la dimension de l'objet serree entre les machoires (le long
    de Y_pince) pour un angle d'attaque donne.
    """
    d = np.asarray(direction, dtype=np.float64)
    dx, dy, dz = bbox_3d_m
    return float(abs(d[0]) * dx + abs(d[1]) * dy + abs(d[2]) * dz)


def _grasp_depth_z(obj, z_detected: float, table_z: float, grasp_offset: float,
                   stack_detect_m: float, min_clearance: float) -> float:
    """Profondeur (Z) du centre de prise : ancrage table conscient de l'empilement.

    - Pas de bbox 3D -> repli sur Z detecte + offset.
    - Objet pose sur la table (base estimee ~ table) -> table + hauteur/2 : plus
      robuste au bruit du Z stereo, car la base est connue (= table).
    - Objet pose sur un autre objet (base nettement au-dessus de la table) ->
      l'ancrage table viserait le support (trop bas) ; on fait alors confiance au
      centroide 3D detecte. Base estimee = centroide_Z - hauteur/2.
    Garde-fou : jamais sous table + min_clearance.
    """
    if obj.bbox_3d_m is None:
        z_grasp = z_detected + grasp_offset
    else:
        h = float(obj.bbox_3d_m[2])
        base_detected = z_detected - h / 2.0          # bas estime de l'objet
        if base_detected > table_z + stack_detect_m:  # objet sureleve (empile)
            z_grasp = z_detected                      # -> centroide detecte
        else:                                         # objet pose sur la table
            z_grasp = table_z + h / 2.0               # -> ancrage table (robuste)
    return max(z_grasp, table_z + min_clearance)


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
    # la conversion image -> base est de la responsabilite du caller s'il y a
    # une ambiguite ; en repli top-down on prend l'angle image comme estimation
    # grossiere du wrist_roll. C'est une approximation : un yaw bien calcule
    # demanderait de projeter le contour 3D du dessus de l'objet.
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return float(theta)


def reorient_grasp_pose(grasp_pose, new_yaw_rad: float,
                        fixed_finger_dir_gripper: tuple = (0, -1, 0)):
    """Reoriente une GraspPose top-down sur un nouveau yaw (machoires en travers
    du petit axe), en conservant la position de l'objet et le decalage lateral.

    Utilise quand cam_2 (vue proche, au-dessus) mesure l'orientation de l'objet
    plus fiablement que la stereo oblique : on remplace l'orientation de prise
    par celle vue par cam_2 avant de descendre. Modifie grasp_pose en place.
    """
    meta = grasp_pose.meta or {}
    offset_mm = float(meta.get("lateral_offset_mm", 0.0))
    old_off = meta.get("offset_base_xy_mm", (0.0, 0.0))
    old_off = np.array([float(old_off[0]) / 1000.0, float(old_off[1]) / 1000.0, 0.0])
    # Applique la meme correction de convention que la pose initiale.
    new_yaw_rad = float(new_yaw_rad) + float(meta.get("yaw_offset_rad", 0.0))
    # === Symetrie 180deg de la pince a machoires paralleles ===
    # cam_2 fournit un axe (defini mod 180deg) : yaw et yaw+-180 sont la meme prise
    # physique. On choisit le representant le plus proche du yaw courant pour que le
    # poignet fasse le plus petit mouvement. Sinon, si cam_2 mesure -85deg alors que
    # la prise etait a +75deg, la reorientation atteint ~160deg (demi-tour du
    # poignet), alors que +95deg (= -85 mod 180) est la meme prise a seulement +20deg.
    # Le decalage lateral est recalcule depuis R ci-dessous, donc le doigt fixe reste
    # du bon cote apres le pli.
    old_yaw_rad = float(meta.get("yaw_rad", new_yaw_rad))
    while new_yaw_rad - old_yaw_rad > np.pi / 2.0:
        new_yaw_rad -= np.pi
    while new_yaw_rad - old_yaw_rad < -np.pi / 2.0:
        new_yaw_rad += np.pi
    R = _rotation_top_down(float(new_yaw_rad))
    opp = -np.asarray(fixed_finger_dir_gripper, dtype=np.float64)
    new_off = R @ opp * (offset_mm / 1000.0)
    for attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                 "T_base_gripper_retract"):
        T = np.array(getattr(grasp_pose, attr), dtype=np.float64, copy=True)
        center_xy = T[:3, 3] - old_off          # retire l'ancien offset -> centre objet
        T[:3, :3] = R
        T[0, 3] = center_xy[0] + new_off[0]
        T[1, 3] = center_xy[1] + new_off[1]
        # T[2,3] (hauteur) inchange
        setattr(grasp_pose, attr, T)
    if grasp_pose.meta is not None:
        grasp_pose.meta["yaw_rad"] = float(new_yaw_rad)
        grasp_pose.meta["offset_base_xy_mm"] = (float(new_off[0] * 1000),
                                                float(new_off[1] * 1000))
        grasp_pose.meta["flipped_180"] = False  # on repart d'une orientation fraiche


class TopDownGrasp(GraspStrategy):
    """Saisie verticale par le haut. Strategie heuristique simple.

    Parametres :
        approach_height_m : hauteur de la pose d'approche au-dessus de l'objet (m).
        grasp_offset_m    : decalage Z entre le centroide de l'objet et la pose
                            grasp. Souvent on grasp au centroide (offset 0), mais
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
                 # Correction de convention (deg) ajoutee a l'angle de prise.
                 # 0 ici = convention nominale (tests). La valeur de deploiement
                 # est dans PipelineConfig.grasp_yaw_offset_deg = 90 (mesure
                 # terrain) : la pince SO-101 ferme a 90deg de la convention
                 # nominale (verifie : 90 -> saisies //X et //Y reussies
                 # d'emblee, couple 300+ ; 0 echouait systematiquement).
                 yaw_offset_deg: float = 0.0,
                 max_object_height_m: float = 0.12,
                 # --- Decalage lateral vers le doigt fixe ---
                 # Le SO-101 a une pince asymetrique : un doigt fixe, un doigt
                 # mobile qui ferme contre le fixe. Si on centre le grasp sur le
                 # centroide objet, le doigt fixe percute l'objet avant la
                 # fermeture, l'objet bouge et la saisie rate. On decale donc le
                 # grasp pour que l'objet finisse contre le doigt fixe (le doigt
                 # fixe touche un bord de l'objet, le doigt mobile vient l'ecraser).
                 # 8 ici = valeur nominale (tests). En deploiement, PipelineConfig
                 # met 0 par defaut (utile surtout pour les objets a faces plates
                 # type cube ; pour un objet rond le decalage ne sert pas et 8mm
                 # risque de faire rater un objet fin). Passer --grasp-lateral-offset
                 # 8 pour un cube si besoin.
                 grasp_lateral_offset_mm: float = 8.0,
                 # Cote du doigt fixe dans le repere pince.
                 # SO-101 : doigt fixe cote Y_base+ (gauche du robot, vu de
                 # derriere). Avec _rotation_top_down(yaw=0), Y_pince = -Y_base,
                 # donc doigt fixe a Y_pince-. Un decalage oppose au doigt fixe
                 # vaut donc +Y_pince. En vecteur unitaire dans le repere pince :
                 fixed_finger_dir_gripper: tuple = (0, -1, 0),
                 # --- Ouverture pince adaptative ---
                 # Si True et que bbox_3d_m est fourni dans l'ObjectInstance,
                 # l'ouverture pince est calculee selon la largeur objet (plutot
                 # que d'ouvrir au maximum sans raison). Formule :
                 # pct = (largeur + 2*marge) / largeur_max_pince * 100.
                 adaptive_gripper_open: bool = True,
                 # Marge d'ouverture de chaque cote. Absorbe l'erreur de position
                 # (~8mm d'IK + calibration) : sans elle, la pince ouvre trop juste
                 # et le doigt fixe percute l'objet a la descente au lieu de le
                 # degager.
                 gripper_open_margin_mm: float = 10.0,
                 # Ouverture max reelle de la pince SO-101 = 150mm (mesure terrain).
                 # Une valeur de 50mm (trop faible) faisait saturer la formule
                 # d'ouverture a 100% pour tout objet >= ~26mm. Avec 150 : un objet
                 # de 30mm -> (30+2*10)/150 = 33% ~= 50mm.
                 gripper_max_opening_mm: float = 150.0,
                 # --- Ancrage de la profondeur de prise sur la table ---
                 table_z_m: float = 0.0,               # z=0 = plaque (la ou les objets reposent). L'offset de pince est gere a part (gripper_grab_offset_m).
                 min_grasp_clearance_m: float = 0.0,   # plancher au niveau de la plaque (z=0). Le top-down prend a la hauteur reelle de l'objet ; le degagement plaque ne concerne que la prise 90 (tilted_grasp_center_min_m).
                 # Offset pince (Z, le long de l'axe d'approche). Defaut 0. Les
                 # cameras sont calibrees sur la plaque (z=0), donc un objet pose
                 # dessus a son centre a table + H/2 : on vise directement ce point,
                 # sans rien ajouter. Un offset constant (par exemple +14mm) ne
                 # compense qu'un cas particulier de sous-lecture de hauteur (grand
                 # cylindre debout) et devient faux en general : sur un objet
                 # court/couche il viserait au-dessus du sommet -> prise trop haute
                 # -> ratee. Reglable via --grab-offset (petite valeur si la pince
                 # butte la table sur les objets tres bas). Le vrai correctif des
                 # erreurs de hauteur est de fiabiliser H, pas un offset constant.
                 gripper_grab_offset_m: float = 0.0,
                 # Seuil de detection d'empilement : si la base estimee de l'objet
                 # (centroide - hauteur/2) depasse table + ce seuil, l'objet repose
                 # sur autre chose -> on vise le centroide detecte plutot que
                 # table+H/2 (qui viserait le support). 15mm = tolerance au bruit
                 # du Z stereo sur un objet pose a plat sur la table.
                 stack_detect_m: float = 0.015,
                 ):
        self.approach_height_m = approach_height_m
        self.grasp_offset_m = grasp_offset_m
        self.retract_height_m = retract_height_m
        self.gripper_open_pct = gripper_open_pct
        self.gripper_close_pct = gripper_close_pct
        self.align_wrist_roll = align_wrist_roll
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.max_object_height_m = max_object_height_m
        self.grasp_lateral_offset_mm = grasp_lateral_offset_mm
        self.fixed_finger_dir_gripper = np.asarray(fixed_finger_dir_gripper,
                                                     dtype=np.float64)
        self.adaptive_gripper_open = adaptive_gripper_open
        self.gripper_open_margin_mm = gripper_open_margin_mm
        self.gripper_max_opening_mm = gripper_max_opening_mm
        self.table_z_m = table_z_m
        self.min_grasp_clearance_m = min_grasp_clearance_m
        self.gripper_grab_offset_m = gripper_grab_offset_m
        self.stack_detect_m = stack_detect_m

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
        # Source preferee : le yaw en repere base calcule par la perception
        # (pose_estimator._estimate_geometry, projection rayon-plan du contour
        # sur le plan de l'objet). Correct quelle que soit l'orientation des
        # cameras, donc les objets poses en biais sont geres. La perception
        # signale aussi la classe de pose :
        #   "debout"  -> empreinte au sol quasi circulaire, yaw libre (0).
        #      (Sans cette classe, le contour vu de cote d'un cylindre debout est
        #      allonge verticalement dans l'image, ce qui produit un yaw fantome
        #      ~±90deg et une rotation de poignet inutile.)
        #   "couche"  -> yaw_base_rad = grand axe de l'empreinte.
        #   "compact" -> pas de grand axe fiable, yaw 0.
        # Repli (perception sans info de pose) : angle image brut du contour
        # cam_0/cam_1 (approximation documentee).
        # === Orientation de la pince (wrist_roll), continue et geometrique ===
        # Regle unique (pas de cas par objet) : on aligne les machoires en travers
        # du petit axe de l'empreinte detectee (= perpendiculaire au grand axe).
        # La perception fournit yaw_base_rad = angle du grand axe quand l'empreinte
        # a une orientation fiable (allongee) ; sinon None.
        #   - yaw_base connu  -> yaw = grand axe (machoires en travers du petit cote).
        #   - empreinte ronde / indeterminee (yaw_base None) -> yaw libre : l'IK
        #     choisit l'angle qui fait le moins tourner le poignet depuis la pose
        #     courante (pas de rotation imposee inutile). C'est correct pour un objet
        #     rond (tout angle saisit pareil) et sans risque pour un objet dont on
        #     ne connait pas l'orientation (on ne force pas un mauvais angle).
        #   - pas de geometrie 3D (repli) -> angle du contour image.
        yaw = 0.0
        yaw_free = False
        meta_obj = obj.meta or {}
        if self.align_wrist_roll:
            if "pose_class" in meta_obj:
                yaw_base = meta_obj.get("yaw_base_rad")
                if yaw_base is not None:
                    yaw = float(yaw_base)        # machoires en travers du petit cote
                else:
                    yaw_free = True              # rond/indetermine -> minimiser rotation
            elif obj.source_detections:
                for det in obj.source_detections:
                    if det.contour is not None and len(det.contour) >= 3:
                        if det.cam_key in ("cam_0", "cam_1"):
                            yaw = yaw_from_contour(det.contour)
                            break
        # Correction de convention : appliquee a un angle aligne seulement
        # (pas en yaw libre, ou l'angle n'a pas de sens).
        yaw_offset_rad = np.radians(self.yaw_offset_deg)
        if not yaw_free:
            yaw = yaw + yaw_offset_rad
        R = _rotation_top_down(yaw)

        # === Decalage lateral vers le doigt fixe ===
        # On veut que l'objet finisse contre le doigt fixe (l'autre ferme
        # vers lui). On decale donc le centre du grasp dans la direction
        # opposee au doigt fixe (en repere pince), puis on le transforme dans
        # le repere base via R.
        opposite_dir_gripper = -self.fixed_finger_dir_gripper
        offset_base = R @ opposite_dir_gripper * (self.grasp_lateral_offset_mm / 1000.0)
        # On decale approach/grasp/retract pareillement (pour rester aligne)
        grasp_xy = np.array([x, y, 0.0]) + np.array([offset_base[0], offset_base[1], 0.0])
        gx, gy = float(grasp_xy[0]), float(grasp_xy[1])

        # === Profondeur de prise (ancrage table, conscient de l'empilement) ===
        # Pour un objet pose sur la table, le Z stereo est bruite : on derive la
        # profondeur de table + hauteur/2 (milieu de l'objet), plus robuste au
        # bruit. Si l'objet repose sur un autre objet (base au-dessus de la table),
        # cet ancrage table viserait trop bas (au niveau du support) ; on detecte
        # ce cas (base = centroide - hauteur/2 nettement au-dessus de la table) et
        # on fait alors confiance au centroide detecte.
        z_object_center = _grasp_depth_z(obj, z, self.table_z_m, self.grasp_offset_m,
                                         self.stack_detect_m, self.min_grasp_clearance_m)
        # Remonte au point de serrage des machoires (z=0=plaque -> Z prise =
        # plaque + H/2 + offset pince). z_object_center (sans offset) = Z du centre
        # de l'objet ; il sert de plan de projection a cam_2 (meme reference que la
        # stereo, pas le Z de prise).
        z_grasp = z_object_center + self.gripper_grab_offset_m

        # 3 poses : approach, grasp, retract (meme (gx, gy), Z relatif au grasp)
        T_approach = _se3(R, [gx, gy, z_grasp + self.approach_height_m])
        T_grasp    = _se3(R, [gx, gy, z_grasp])
        T_retract  = _se3(R, [gx, gy, z_grasp + self.retract_height_m])

        # === Ouverture pince adaptative selon bbox 3D ===
        # Si on a bbox_3d_m, on calcule l'ouverture optimale (plutot que
        # d'ouvrir grand). Formule : pct = (largeur + 2 * marge) / max_pince.
        # On prend min(X, Y) car la pince ferme dans le plan XY.
        gripper_open = self.gripper_open_pct
        if self.adaptive_gripper_open and obj.bbox_3d_m is not None:
            obj_width_m = min(obj.bbox_3d_m[0], obj.bbox_3d_m[1])
            target_open_mm = obj_width_m * 1000 + 2 * self.gripper_open_margin_mm
            pct = (target_open_mm / self.gripper_max_opening_mm) * 100.0
            # Borne a [20, 100] : assez d'ouverture pour degager l'objet et l'erreur
            # de visee (~8mm IK + cam), sans ouvrir a fond inutilement. Avec la
            # max reelle (150mm) l'ouverture differencie les objets.
            gripper_open = float(np.clip(pct, 20.0, 100.0))

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
                "object_center_z_m": float(z_object_center),  # plan de projection cam_2 (Z objet, pas Z prise)
                "yaw_rad": yaw,
                "yaw_free": yaw_free,
                "pose_class": meta_obj.get("pose_class"),
                "object_center_xy_m": (float(x), float(y)),
                "yaw_offset_rad": float(yaw_offset_rad),
                "yaw_cam0_deg": meta_obj.get("yaw_cam0_deg"),
                "yaw_cam1_deg": meta_obj.get("yaw_cam1_deg"),
                "approach_height_m": self.approach_height_m,
                "grasp_offset_m": self.grasp_offset_m,
                "retract_height_m": self.retract_height_m,
                "lateral_offset_mm": self.grasp_lateral_offset_mm,
                "offset_base_xy_mm": (float(offset_base[0]*1000), float(offset_base[1]*1000)),
                "gripper_open_pct_computed": gripper_open,
                # Largeur serree par les machoires (petit cote de l'empreinte, mm).
                # Sert a l'offset lateral adaptatif : decaler le doigt fixe de
                # ~(largeur/2 + marge) pour qu'il tombe a fleur de l'arete, quelle
                # que soit la taille de l'objet (pas de constante codee en dur).
                # None si pas de bbox 3D.
                "jaw_width_mm": (float(min(obj.bbox_3d_m[0], obj.bbox_3d_m[1]) * 1000.0)
                                 if obj.bbox_3d_m is not None else None),
            },
        )


# ============================================================
# Strategy V2 : adaptative a l'angle d'attaque (balayage sagittal)
# ============================================================


class AdaptiveGrasp(GraspStrategy):
    """Saisie adaptative : choisit l'angle d'attaque dans le plan sagittal.

    Au lieu d'attaquer toujours par le dessus (top-down), la pince peut attaquer
    selon n'importe quel tangage sur les 180deg du plan vertical base->objet :
    du sol cote face avant -> par-dessus -> sol cote face arriere. La regle de
    serrage est conservee (machoires en travers du petit cote).

    Methode (motif standard *generate -> filter -> rank*, cf Bohg 2014 ;
    Miller/GraspIt! 2003 pour le jeu fini de prises canoniques par primitive) :
      1. Generer un petit jeu d'angles canoniques (orbites de symetrie des
         primitives, Pokorny et al. RSS 2013) : theta in {0, +45, +90} puis repli
         {-45, -90} (face arriere, seulement si proche).
      2. Filtrer (ici, geometrie) : ouverture pince suffisante ; degagement table.
         Le filtre d'atteignabilite IK (le plus important pour un bras 5 DDL : le
         top-down ne passe plus au-dela d'une certaine distance, idem -45/90) est
         fait par le pipeline qui dispose du solveur IK (reachability-aware
         grasping : Zacharias 2010, Vahrenkamp 2013, Lou 2020).
      3. Choisir le premier candidat faisable dans l'ordre de preference (top-down
         d'abord : plus stable et le plus sur cote collision ; frontal/diagonale
         quand le top-down est mauvais ou impossible). Fait par le pipeline.

    Le candidat theta=0 reproduit exactement TopDownGrasp (meme orientation, memes
    poses), donc aucune regression sur les objets deja bien saisis par le haut.
    """

    def __init__(self,
                 # Jeu d'angles canoniques (deg), dans l'ordre de preference.
                 # {0, +45, +90} uniquement. Les angles arriere (-45, -90)
                 # demanderaient d'attaquer la face arriere de l'objet (depuis
                 # l'autre cote, pince orientee vers la base) : structurellement
                 # hors d'atteinte sur le SO-101 (epaule-lift/coude/poignet-flex
                 # plient dans le meme plan, donc le poignet ne peut pas se replier
                 # derriere l'objet). Verifie sur les essais terrain (residus IK
                 # 40-110mm, jamais retenus), et un objet assez proche pour ces
                 # angles est de toute facon pris en top-down (0, prefere).
                 candidate_pitches_deg: tuple = (0.0, 45.0, 90.0),
                 approach_height_m: float = 0.08,
                 grasp_offset_m: float = 0.0,
                 retract_height_m: float = 0.10,
                 gripper_open_pct: float = 100.0,
                 gripper_close_pct: float = 0.0,
                 align_wrist_roll: bool = True,
                 yaw_offset_deg: float = 0.0,
                 # Le top-down (theta=0) reste refuse au-dela de cette hauteur
                 # (collision pince/objet par le haut, comme TopDownGrasp). Les
                 # candidats inclines, eux, sont justement la pour les objets
                 # hauts : pas de plafond de hauteur pour theta != 0.
                 max_object_height_m: float = 0.12,
                 grasp_lateral_offset_mm: float = 0.0,
                 fixed_finger_dir_gripper: tuple = (0, -1, 0),
                 adaptive_gripper_open: bool = True,
                 gripper_open_margin_mm: float = 10.0,
                 gripper_max_opening_mm: float = 150.0,
                 table_z_m: float = 0.0,  # z=0 = plaque (la ou les objets reposent). L'offset de pince est gere a part (gripper_grab_offset_m).
                 min_grasp_clearance_m: float = 0.0,   # plancher au niveau plaque (z=0). Seul le 90 garde un plancher (tilted_grasp_center_min_m = 7mm).
                 # Offset pince (cf TopDownGrasp) : defaut 0. On vise table + H/2
                 # directement (cameras calibrees sur la plaque). Un offset constant
                 # (par exemple +14mm) ne compense qu'un cas de sous-lecture de
                 # hauteur et devient faux en general.
                 gripper_grab_offset_m: float = 0.0,
                 # Seuil de detection d'empilement (cf TopDownGrasp.stack_detect_m).
                 stack_detect_m: float = 0.015,
                 # Degagement table pour une prise inclinee : a l'horizontale
                 # (theta=+/-90) les machoires sont a mi-hauteur de l'objet et il
                 # faut que le bas de la pince degage la table. Hauteur de prise
                 # mini requise = min_grasp_clearance + |sin(theta)| * ce terme.
                 # 25mm = hauteur du doigt a sa base (mesure terrain : doigt 10mm a
                 # la pointe, 25mm a la fixation imprimee) : au plus bas la pince
                 # occupe ~25mm sous la ligne de prise a l'horizontale. Reglable
                 # (--side-grasp-min-height). A theta=0 ce terme est nul, donc le
                 # top-down n'est jamais contraint par ce filtre.
                 side_grasp_min_height_m: float = 0.025,   # conserve pour compatibilite ; le filtre actif est tilted_grasp_center_min_m
                 # Degagement table (mesure terrain) : hauteur minimale du centre
                 # de prise au-dessus de la plaque pour une prise inclinee. Limite
                 # plate (meme pour 45 et 90), remplace l'ancienne formule
                 # min_clearance + |sin(theta)|*side_min_height (trop dure :
                 # exigeait H>=45mm a 45 / 60mm a 90). Geometrie de la pince : bout
                 # du doigt ~10mm, prise ~15mm, fond ~25mm ; un centre a >=7mm
                 # au-dessus de la plaque suffit, meme a 90 qui avance sur la plaque.
                 # Top-down (theta=0) jamais contraint (delegue plus haut). Reglable.
                 tilted_grasp_center_min_m: float = 0.007,   # 7mm au-dessus de la plaque. Repere : au runtime z=0 est la plaque (table_z_m=0), donc ce plancher vaut +7mm en repere base.
                 # Roll (deg) applique aux prises inclinees autour de l'axe
                 # d'approche pour la convention pince. None = reutilise
                 # yaw_offset_deg (mesure en top-down). Le signe n'est pas valide en
                 # incline ; reglable sans recompiler (PipelineConfig.grasp_tilt_roll_deg
                 # / --tilt-roll-offset) si les machoires ferment de travers au
                 # premier essai.
                 tilted_roll_deg: Optional[float] = None,
                 ):
        self.candidate_pitches_deg = tuple(float(t) for t in candidate_pitches_deg)
        self.approach_height_m = approach_height_m
        self.grasp_offset_m = grasp_offset_m
        self.retract_height_m = retract_height_m
        self.gripper_open_pct = gripper_open_pct
        self.gripper_close_pct = gripper_close_pct
        self.align_wrist_roll = align_wrist_roll
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.max_object_height_m = max_object_height_m
        self.grasp_lateral_offset_mm = grasp_lateral_offset_mm
        self.fixed_finger_dir_gripper = np.asarray(fixed_finger_dir_gripper,
                                                   dtype=np.float64)
        self.adaptive_gripper_open = adaptive_gripper_open
        self.gripper_open_margin_mm = gripper_open_margin_mm
        self.gripper_max_opening_mm = gripper_max_opening_mm
        self.table_z_m = table_z_m
        self.min_grasp_clearance_m = min_grasp_clearance_m
        self.gripper_grab_offset_m = gripper_grab_offset_m
        self.stack_detect_m = stack_detect_m
        self.side_grasp_min_height_m = side_grasp_min_height_m
        self.tilted_grasp_center_min_m = tilted_grasp_center_min_m
        self.tilted_roll_deg = tilted_roll_deg
        # Le candidat theta=0 delegue a TopDownGrasp (memes parametres) : le
        # comportement top-down reste identique bit pour bit (aucune regression).
        self._top_down = TopDownGrasp(
            stack_detect_m=stack_detect_m,
            approach_height_m=approach_height_m,
            grasp_offset_m=grasp_offset_m,
            retract_height_m=retract_height_m,
            gripper_open_pct=gripper_open_pct,
            gripper_close_pct=gripper_close_pct,
            align_wrist_roll=align_wrist_roll,
            yaw_offset_deg=yaw_offset_deg,
            max_object_height_m=max_object_height_m,
            grasp_lateral_offset_mm=grasp_lateral_offset_mm,
            fixed_finger_dir_gripper=fixed_finger_dir_gripper,
            adaptive_gripper_open=adaptive_gripper_open,
            gripper_open_margin_mm=gripper_open_margin_mm,
            gripper_max_opening_mm=gripper_max_opening_mm,
            table_z_m=table_z_m,
            min_grasp_clearance_m=min_grasp_clearance_m,
            gripper_grab_offset_m=gripper_grab_offset_m,
        )

    @property
    def name(self) -> str:
        return "AdaptiveGrasp"

    def _build_pose(self, obj: ObjectInstance,
                    theta_deg: float) -> Optional[GraspPose]:
        """Construit la GraspPose pour un tangage donne, ou None si infaisable
        geometriquement (ouverture pince ou degagement table).

        theta=0 -> delegation exacte a TopDownGrasp (non-regression). theta!=0 ->
        prise inclinee : machoires laterales (horizontales, perpendiculaires au
        plan sagittal) ; elles serrent une dimension horizontale de l'objet, ne
        risquent jamais la table et ne tentent pas de serrer la hauteur.
        L'alignement fin sur le petit axe en 3D pour les prises inclinees reste
        hors du perimetre de cette version.
        """
        # --- theta == 0 : top-down exact (delegation) ---
        if abs(theta_deg) < 1e-6:
            gp = self._top_down.plan(obj)
            if gp is not None and gp.meta is not None:
                gp.meta["pitch_rad"] = 0.0
                gp.meta["pitch_deg"] = 0.0
                gp.meta.setdefault(
                    "jaw_width_mm",
                    float(min(obj.bbox_3d_m[0], obj.bbox_3d_m[1]) * 1000.0)
                    if obj.bbox_3d_m is not None else 0.0)
            return gp

        # --- theta != 0 : prise inclinee (necessite la bbox 3D) ---
        if obj.bbox_3d_m is None:
            return None
        x, y, z = (float(obj.position_base_m[0]), float(obj.position_base_m[1]),
                   float(obj.position_base_m[2]))
        phi = float(np.arctan2(y, x))
        th = np.radians(float(theta_deg))
        obj_h = float(obj.bbox_3d_m[2])
        # profondeur de prise : ancrage table conscient de l'empilement (si l'objet
        # repose sur un autre, on vise le centroide detecte, pas table+H/2).
        z_center = _grasp_depth_z(obj, z, self.table_z_m, self.grasp_offset_m,
                                  self.stack_detect_m, self.min_grasp_clearance_m)

        # --- filtre degagement table : uniquement pour les prises quasi
        #     horizontales (~90deg). Elles avancent sur la plaque, donc le bout de
        #     pince risque de la taper. Le 45 (et le top-down) descendent en
        #     diagonale, sans raser la plaque -> aucune limite (le cas se pose
        #     surtout pour 90, pas pour 45). Seuil 67.5deg = a mi-chemin 45/90 :
        #     seul le 90 est contraint. Limite = centre de prise >=
        #     tilted_grasp_center_min_m au-dessus de la plaque ; (z_center -
        #     table_z) = demi-hauteur detectee. Le filtre s'appuie sur la hauteur
        #     detectee (potentiellement sous-lue), donc le vrai correctif est de
        #     fiabiliser la hauteur.
        if (abs(theta_deg) >= 67.5
                and (z_center - self.table_z_m) < self.tilted_grasp_center_min_m):
            return None

        # Azimut d'approche psi. Par defaut radial (base->objet) : l'axe des
        # machoires u (perpendiculaire a psi, horizontal) est alors fixe par la
        # direction du bras, independamment de l'orientation de l'objet. C'est
        # correct pour un objet rond/compact, mais faux pour un objet allonge
        # (couche) : si son grand axe est tangentiel (// Y), u tombe le long de la
        # longueur, les machoires s'ecartent sur la longueur et ferment a vide
        # (essais cylindre couche // Y). Correction : quand le grand axe base est
        # connu (yaw_base_rad, objet allonge), on approche le long du grand axe ;
        # u devient perpendiculaire au grand axe, donc en travers du petit cote
        # (prise correcte, coherente avec la convention top-down). Debout
        # (yaw_base None, dessus rond) et compact non concernes -> psi = phi
        # inchange.
        psi = phi
        _meta_obj = obj.meta or {}
        _yaw_base = _meta_obj.get("yaw_base_rad")
        _elong = float(_meta_obj.get("footprint_elongation", 1.0) or 1.0)
        if _yaw_base is not None and _elong >= 1.3:
            # 2 sens possibles (yb, yb+pi) : on retient celui dont le depart
            # d'approche est le plus proche de la base (le bras ne passe pas
            # au-dessus de l'objet pour l'attaquer).
            _best = None
            for _psi in (float(_yaw_base), float(_yaw_base) + np.pi):
                _a = np.array([np.sin(th) * np.cos(_psi),
                               np.sin(th) * np.sin(_psi), -np.cos(th)])
                _start = np.array([x, y, z_center]) - self.approach_height_m * _a
                _r = float(np.hypot(_start[0], _start[1]))
                if _best is None or _r < _best[0]:
                    _best = (_r, _psi)
            psi = _best[1]
        # axe d'approche a (vers l'objet) et axe lateral u (machoires physiques)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        a = np.array([np.sin(th) * cpsi, np.sin(th) * spsi, -np.cos(th)])
        u = np.array([-spsi, cpsi, 0.0])   # horizontal, perpendiculaire a l'approche

        # --- filtre ouverture : dimension serree = extension de l'objet le long
        #     de l'axe des machoires physique (u) ---
        jaw_width_m = _extent_along(obj.bbox_3d_m, u)
        need_mm = jaw_width_m * 1000.0 + 2.0 * self.gripper_open_margin_mm
        if need_mm > self.gripper_max_opening_mm:
            return None  # objet trop large lateralement pour la pince
        if self.adaptive_gripper_open:
            gripper_open = float(np.clip(
                need_mm / self.gripper_max_opening_mm * 100.0, 20.0, 100.0))
        else:
            gripper_open = self.gripper_open_pct

        # Convention pince : yaw_offset_deg (mesure en top-down) corrige le zero du
        # poignet, c.-a-d. une rotation autour de l'axe d'approche ; on l'applique
        # comme un roll. roll = +yaw_offset (derive de la convention top-down : la
        # pince ferme a yaw_offset du nominal) pour que les machoires physiques
        # finissent laterales (= u). La largeur serree est calculee sur u (axe
        # physique), donc independante du roll commande. Le sens du roll est logge
        # et reglable via tilted_roll_deg.
        roll_deg = (self.tilted_roll_deg if self.tilted_roll_deg is not None
                    else self.yaw_offset_deg)
        roll = float(np.radians(roll_deg))
        R = _rotation_grasp(psi, th, roll)

        # --- Decalage lateral vers le doigt fixe (defaut 0) ---
        offset_base = R @ (-self.fixed_finger_dir_gripper) * (
            self.grasp_lateral_offset_mm / 1000.0)
        center = np.array([x, y, z_center]) + offset_base
        center[2] = max(center[2], self.table_z_m + self.min_grasp_clearance_m)
        # Remonte au point de serrage des machoires le long de l'axe d'approche (a).
        # L'offset pince est la distance TCP->machoires le long de l'axe d'approche
        # du gripper (Z_pince = a), pas une translation verticale fixe.
        #   - top-down  : a = (0,0,-1) -> center -= offset*a = center[2] += offset
        #     (identique au comportement historique, aucune regression).
        #   - incline   : appliquer +Z fermerait les machoires trop haut. A 90deg,
        #     a est horizontal, donc un +offset en Z mettrait les machoires
        #     au-dessus du centre (au bord haut d'un cube bas, machoires plus hautes
        #     que le cube). En reculant le TCP le long de -a, les machoires
        #     (a +offset*a) retombent sur le centre objet, quel que soit le tangage.
        #     Le filtre de degagement table ci-dessus utilise z_center (centre
        #     objet, sans offset), ce qui reste correct.
        center = center - self.gripper_grab_offset_m * a

        # approche reculee le long de l'axe d'approche ; retract = levee verticale
        approach_pos = center - self.approach_height_m * a
        retract_pos = center + np.array([0.0, 0.0, self.retract_height_m])

        return GraspPose(
            T_base_gripper_approach=_se3(R, approach_pos),
            T_base_gripper_grasp=_se3(R, center),
            T_base_gripper_retract=_se3(R, retract_pos),
            gripper_open_pct=gripper_open,
            gripper_close_pct=self.gripper_close_pct,
            label=obj.label,
            score=obj.score,
            meta={
                "strategy": "AdaptiveGrasp",
                "object_center_z_m": float(z_center),  # plan de projection cam_2 (Z objet, pas Z prise)
                "pitch_rad": float(th),
                "pitch_deg": float(theta_deg),
                "azimuth_rad": float(psi),   # approche radiale OU le long du grand axe
                "azimuth_radial_rad": float(phi),
                "roll_rad": float(roll),
                "yaw_rad": float(psi),       # cap horizontal d'approche (log)
                "yaw_free": False,
                "pose_class": (obj.meta or {}).get("pose_class"),
                "jaw_width_mm": float(jaw_width_m * 1000.0),
                "object_center_xy_m": (float(x), float(y)),
                "yaw_offset_rad": float(roll),
                "approach_height_m": self.approach_height_m,
                "retract_height_m": self.retract_height_m,
                "lateral_offset_mm": self.grasp_lateral_offset_mm,
                "offset_base_xy_mm": (float(offset_base[0] * 1000),
                                      float(offset_base[1] * 1000)),
                "gripper_open_pct_computed": gripper_open,
                "table_clear_required_mm": float(self.tilted_grasp_center_min_m * 1000.0),
            },
        )

    def replan_oriented(self, obj: ObjectInstance, pitch_deg: float,
                        yaw_base_rad: float) -> Optional[GraspPose]:
        """Replanifie la prise au meme pitch en imposant le grand axe (yaw)
        mesure par cam_2 (plus fiable que la stereo oblique).

        Reutilise `_build_pose`, donc geometrie identique au planning : l'azimut
        d'approche psi est aligne sur le grand axe, donc l'axe des machoires (u,
        perpendiculaire a psi) tombe en travers du petit cote, quel que soit le
        pitch (0 / 45 / 90). C'est le re-alignement cam_2 pour les prises
        inclinees (le top-down passe par reorient_grasp_pose). Le pitch n'est pas
        modifie ; seule l'orientation des machoires change. La position est celle
        de `obj` (stereo) ; le pipeline re-applique ensuite la correction de
        position cam_2. None si infaisable.
        """
        from dataclasses import replace
        meta = dict(obj.meta or {})
        meta["yaw_base_rad"] = float(yaw_base_rad)
        # force la branche objet allonge pour aligner psi sur le grand axe
        meta["footprint_elongation"] = max(
            float(meta.get("footprint_elongation", 1.0) or 1.0), 1.3)
        obj2 = replace(obj, meta=meta)
        return self._build_pose(obj2, float(pitch_deg))

    def plan_candidates(self, obj: ObjectInstance) -> list[GraspPose]:
        """Liste des prises geometriquement faisables, dans l'ordre de preference.

        Le pipeline filtre ensuite par atteignabilite IK et garde la premiere.
        """
        out: list[GraspPose] = []
        for tdeg in self.candidate_pitches_deg:
            gp = self._build_pose(obj, tdeg)
            if gp is not None:
                out.append(gp)
        return out

    def plan(self, obj: ObjectInstance) -> Optional[GraspPose]:
        """Compat ABC : renvoie le candidat prefere (1er faisable geometriquement).

        Le choix definitif (atteignabilite IK) est fait par le pipeline via
        `plan_candidates`. Utile pour un usage geometrique direct / dry-run.
        """
        cands = self.plan_candidates(obj)
        return cands[0] if cands else None


# Zones d'utilisation des angles d'attaque, en metres. Reglables.
# Politique : le choix d'angle est pilote par la hauteur du sommet h_top
# (= z + H/2) et la distance d = sqrt(x^2+y^2) :
#                          proche (d<=32)        au-dela (d>32)
#   h_top < 12cm  (bas)    top-down 0            45  (best-effort jusqu'au bord)
#   12cm <= h_top < 18cm   45                    45
#   h_top >= 18cm (haut)   90 (face)             90 (face)
# Justification (essais terrain) :
#  - Objet bas : top-down de pres (le plus stable), diagonale 45 au-dela. Pas de
#    face 90 : elle n'etait choisie que loin (bord d'atteinte), or c'est
#    precisement la que le poignet et la tete eye-in-hand butent contre la table
#    (couple mesure 1004) ; l'objet est trop bas pour positionner la pince de face
#    sans forcer les moteurs. Le 45 garde la tete plus haute et reste le meilleur
#    pari longue portee. Au-dela de l'atteinte du 45 = limite de domaine assumee
#    (cf memoire), pas un cas a forcer.
#  - Objet haut (h_top >= face_m) : la face 90 redevient legitime (le sommet donne
#    la garde, la tete ne bute plus). En pratique aucun objet du banc d'essai n'est
#    aussi haut, donc le 90 ne se declenche pas : c'est volontaire.
GRASP_ZONE_NEAR_M = 0.32   # objet bas pris en top-down jusqu'ici ; au-dela -> 45. Reglable (--zone-topdown).
GRASP_ZONE_FAR_M = 0.45    # (conserve pour compat ; n'influe plus le choix d'angle)
GRASP_ZONE_TALL_M = 0.12   # sommet < : objet bas (top-down/45) ; >= : moyen/haut
GRASP_ZONE_FACE_M = 0.18   # sommet >= : objet haut -> face 90 (garde suffisante) ; entre TALL et FACE -> 45.


def preferred_pitch_deg(obj: ObjectInstance,
                        near_m: float = GRASP_ZONE_NEAR_M,
                        far_m: float = GRASP_ZONE_FAR_M,
                        tall_m: float = GRASP_ZONE_TALL_M,
                        face_m: float = GRASP_ZONE_FACE_M) -> float:
    """Angle d'attaque prefere, pilote par la hauteur du sommet (h_top = z + H/2)
    et la distance d = sqrt(x^2+y^2).

    Politique (cf bloc GRASP_ZONE_* ci-dessus) :
      - h_top < tall_m (12cm) : objet bas -> top-down si d<=near_m, sinon 45.
        Jamais 90 : la face 90 ne se declenchait que loin (bord d'atteinte), or la
        tete eye-in-hand y bute contre la table et force les moteurs. Au-dela de
        l'atteinte du 45 = limite de domaine assumee.
      - tall_m <= h_top < face_m : objet moyennement haut -> diagonale 45.
      - h_top >= face_m (18cm) : objet haut -> face 90 (le sommet donne la garde).

    Ce n'est pas un verrou : le pipeline filtre ensuite par atteignabilite IK et
    retient l'angle atteignable le plus proche de ce prefere (si le prefere n'est
    pas atteignable, on bascule sur le voisin). Bornes reglables ci-dessus.
    """
    x, y, z = (float(obj.position_base_m[0]), float(obj.position_base_m[1]),
               float(obj.position_base_m[2]))
    d = float(np.hypot(x, y))
    bbox = obj.bbox_3d_m
    H = float(bbox[2]) if bbox is not None else 0.0
    h_top = z + 0.5 * H
    # --- objet bas (sommet < 12cm) : top-down de pres, 45 au-dela ---
    # Pas de 90 : elle n'arriverait que loin (bord d'atteinte) ou la tete
    # eye-in-hand bute contre la table et force les moteurs (couple mesure 1004).
    # Le 45 garde la tete plus haute ; au-dela de son atteinte = limite de domaine
    # assumee (cf memoire). far_m conserve pour compat.
    if h_top < tall_m:
        if d <= near_m:
            return 0.0    # bas + proche -> top-down (le plus stable)
        return 45.0       # bas + au-dela -> diagonale 45 (best-effort longue portee)
    # --- objet moyen/haut : 45 puis face 90 au-dela de face_m (sommet haut = garde
    # suffisante, la tete ne bute plus). Top-down exclu (collision pince par le haut).
    if h_top >= face_m:
        return 90.0
    return 45.0


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
    # Test sans decalage lateral ni ancrage Z adaptatif pour valider la
    # logique de base : les 3 poses doivent etre verticalement alignees au
    # cube. Avec le decalage lateral (defaut 8mm), le grasp est decale en Y
    # pour aligner avec la pince fixe (teste plus bas). De meme pour l'ancrage
    # de profondeur : on le desactive en passant bbox_3d_m=None pour ce test,
    # puis on le teste avec bbox.
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

    # 3bis. TopDownGrasp avec decalage lateral : le grasp est decale d'offset
    # mm dans la direction opposee au doigt fixe ; Z = ancrage table + H/2.
    strategy_smart = TopDownGrasp(approach_height_m=0.08, retract_height_m=0.10,
                                    grasp_lateral_offset_mm=8.0)
    gp_smart = strategy_smart.plan(obj)  # cube 30mm bbox, centre Z=0.03
    # Avec yaw=0 et fixed_finger_dir_gripper=(0,-1,0), l'offset oppose dans
    # le repere base est (0, -0.008, 0). Donc grasp Y = -0.008.
    assert abs(gp_smart.T_base_gripper_grasp[1, 3] - (-0.008)) < 1e-6, \
        f"grasp Y attendu -0.008 (decalage lateral), recu {gp_smart.T_base_gripper_grasp[1, 3]}"
    # Ancrage profondeur : grasp Z = table_z_m + hauteur/2 = 0 + 0.03/2 = 0.015
    assert abs(gp_smart.T_base_gripper_grasp[2, 3] - 0.015) < 1e-6, \
        f"grasp Z attendu 0.015 (table + H/2), recu {gp_smart.T_base_gripper_grasp[2, 3]}"
    print(f"  [OK] TopDownGrasp.plan (offset lateral 8mm + ancrage table + H/2)")

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

    # ========================================================
    # 7-11. AdaptiveGrasp (saisie a angle adaptatif)
    # ========================================================

    # 7. _rotation_grasp : orthonormale, det=+1, axes d'approche corrects
    for phi in (0.0, 0.7, -1.2, np.pi):
        for thd in (-90, -45, 0, 30, 45, 90):
            Rg = _rotation_grasp(phi, np.radians(thd), 0.3)
            assert np.allclose(Rg @ Rg.T, np.eye(3), atol=1e-9), "R non orthonormale"
            assert abs(np.linalg.det(Rg) - 1.0) < 1e-9, "det(R) != 1"
    # theta=0 -> Z_pince = -Z_base ; theta=+90,phi=0 -> Z_pince = +X_base (frontal)
    assert np.allclose(_rotation_grasp(0.0, 0.0, 0.0)[:, 2], [0, 0, -1])
    assert np.allclose(_rotation_grasp(0.0, np.radians(90), 0.0)[:, 2], [1, 0, 0])
    print("  [OK] _rotation_grasp : orthonormale, det=+1, axes d'approche corrects")

    # 8. Non-regression : le candidat theta=0 d'AdaptiveGrasp == TopDownGrasp
    obj_cube = ObjectInstance(label="cube",
                              position_base_m=np.array([0.15, 0.0, 0.015]),
                              bbox_3d_m=(0.03, 0.03, 0.03))
    td = TopDownGrasp(grasp_lateral_offset_mm=0.0, yaw_offset_deg=0.0).plan(obj_cube)
    ad = AdaptiveGrasp(grasp_lateral_offset_mm=0.0, yaw_offset_deg=0.0)._build_pose(
        obj_cube, 0.0)
    for attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                 "T_base_gripper_retract"):
        assert np.allclose(getattr(td, attr), getattr(ad, attr), atol=1e-9), \
            f"regression top-down sur {attr}"
    assert abs(td.gripper_open_pct - ad.gripper_open_pct) < 1e-6
    print(f"  [OK] non-regression : AdaptiveGrasp(theta=0) identique a TopDownGrasp")

    # 9. Objet tres plat (centre de prise sous le degagement plaque 7mm) : le
    #    filtre table rejette la prise quasi horizontale 90 (elle raserait la
    #    plaque), le top-down reste. Le 45 n'est pas filtre (il descend en
    #    diagonale, sans raser la plaque). Un objet de 3cm (centre 15mm) passe le
    #    seuil 7mm et sort les 3 angles ; on teste donc le garde-fou reel : un objet
    #    sous 7mm de centre rejette le 90.
    obj_flat = ObjectInstance(label="flat",
                              position_base_m=np.array([0.20, 0.0, 0.005]),
                              bbox_3d_m=(0.05, 0.05, 0.01))   # centre de prise = 5mm < 7mm
    cands_flat = AdaptiveGrasp().plan_candidates(obj_flat)
    pitches_flat = [c.meta["pitch_deg"] for c in cands_flat]
    assert 90.0 not in pitches_flat, \
        f"objet tres plat : la prise 90 (skim plaque) aurait du etre rejetee, recu {pitches_flat}"
    assert 0.0 in pitches_flat, \
        f"objet tres plat : le top-down doit rester faisable, recu {pitches_flat}"
    print(f"  [OK] objet tres plat -> 90 rejete (degagement plaque), candidats {pitches_flat}")

    # 10. Objet haut (18 cm) -> top-down refuse (trop haut), inclines proposes
    obj_tall = ObjectInstance(label="tall",
                              position_base_m=np.array([0.20, 0.0, 0.09]),
                              bbox_3d_m=(0.04, 0.04, 0.18))
    cands_tall = AdaptiveGrasp().plan_candidates(obj_tall)
    pitches_tall = [c.meta["pitch_deg"] for c in cands_tall]
    assert 0.0 not in pitches_tall, "objet haut : le top-down aurait du etre refuse"
    assert 45.0 in pitches_tall and 90.0 in pitches_tall, \
        f"objet haut : attendait des candidats inclines, recu {pitches_tall}"
    # le candidat prefere (premier) est l'incline le plus proche du top-down
    assert cands_tall[0].meta["pitch_deg"] == 45.0
    print(f"  [OK] objet haut -> top-down refuse, candidats inclines {pitches_tall}")

    # 11. Geometrie d'une prise frontale : approche reculee le long de l'axe
    #     (horizontal), retract vertical
    gp_front = AdaptiveGrasp()._build_pose(obj_tall, 90.0)
    assert gp_front is not None
    c = gp_front.T_base_gripper_grasp[:3, 3]
    app = gp_front.T_base_gripper_approach[:3, 3]
    ret = gp_front.T_base_gripper_retract[:3, 3]
    # approche : recule en X (vers la base), meme Z que le grasp
    assert app[0] < c[0] - 0.05 and abs(app[2] - c[2]) < 1e-6, \
        f"approche frontale devrait reculer horizontalement, app={app} c={c}"
    # retract : pure levee verticale
    assert np.allclose(ret[:2], c[:2]) and ret[2] > c[2] + 0.05, \
        f"retract devrait etre une levee verticale, ret={ret} c={c}"
    # axe d'approche (Z_pince) horizontal -> +X
    assert np.allclose(gp_front.T_base_gripper_grasp[:3, 2], [1, 0, 0], atol=1e-9)
    assert _extent_along((0.03, 0.05, 0.10), [0, 0, 1]) == 0.10
    print(f"  [OK] prise frontale : approche horizontale, retract vertical")

    # 12. Non-regression avec la convention de deploiement (yaw_offset_deg=90) :
    #     theta=0 reste identique a TopDownGrasp, et les candidats inclines d'un
    #     objet haut restent faisables ; la largeur serree est calculee sur l'axe
    #     lateral (physique), pas sur la hauteur (sinon faux rejet du a la convention).
    td90 = TopDownGrasp(grasp_lateral_offset_mm=0.0, yaw_offset_deg=90.0).plan(obj_cube)
    ad90 = AdaptiveGrasp(grasp_lateral_offset_mm=0.0, yaw_offset_deg=90.0)
    ad90_0 = ad90._build_pose(obj_cube, 0.0)
    for attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                 "T_base_gripper_retract"):
        assert np.allclose(getattr(td90, attr), getattr(ad90_0, attr), atol=1e-9), \
            f"regression top-down (offset 90) sur {attr}"
    pitches90 = [c.meta["pitch_deg"] for c in ad90.plan_candidates(obj_tall)]
    assert 45.0 in pitches90 and 90.0 in pitches90, \
        f"objet haut (offset 90) : inclines doivent rester faisables, recu {pitches90}"
    fr90 = ad90._build_pose(obj_tall, 90.0)
    assert abs(fr90.meta["jaw_width_mm"] - 40.0) < 1.0, \
        f"largeur frontale attendue ~40mm (laterale), recu {fr90.meta['jaw_width_mm']}"
    print(f"  [OK] convention 90deg : top-down identique + inclines faisables")

    # 13. tilted_roll_deg : le roll des prises inclinees est reglable sans
    #     recompiler (pour corriger le sens de la convention au premier essai), et
    #     la largeur serree (axe lateral physique) en est independante.
    g_def = AdaptiveGrasp(yaw_offset_deg=90.0)._build_pose(obj_tall, 45.0)
    g_ovr = AdaptiveGrasp(yaw_offset_deg=90.0, tilted_roll_deg=-90.0)._build_pose(obj_tall, 45.0)
    assert abs(np.degrees(g_def.meta["roll_rad"]) - 90.0) < 1e-6
    assert abs(np.degrees(g_ovr.meta["roll_rad"]) - (-90.0)) < 1e-6
    assert abs(g_def.meta["jaw_width_mm"] - g_ovr.meta["jaw_width_mm"]) < 1e-6
    print("  [OK] tilted_roll_deg : roll incline reglable, largeur serree inchangee")

    # 14. Ancrage de profondeur conscient de l'empilement (_grasp_depth_z) :
    #     objet sur la table -> table + H/2 (robuste) ; objet sur un autre objet
    #     -> centroide detecte (sinon on viserait le support = grasp trop bas).
    obj_tbl = ObjectInstance(label="t", position_base_m=np.array([0.2, 0, 0.05]),
                             bbox_3d_m=(0.03, 0.03, 0.10))
    z_tbl = _grasp_depth_z(obj_tbl, 0.05, 0.0, 0.0, 0.015, 0.005)  # base=0 -> table
    assert abs(z_tbl - 0.05) < 1e-9, f"objet sur table -> table+H/2, recu {z_tbl}"
    obj_stk = ObjectInstance(label="s", position_base_m=np.array([0.2, 0, 0.076]),
                             bbox_3d_m=(0.03, 0.03, 0.10))
    z_stk = _grasp_depth_z(obj_stk, 0.076, 0.0, 0.0, 0.015, 0.005)  # base=26mm -> empile
    assert abs(z_stk - 0.076) < 1e-9, f"objet empile -> centroide detecte, recu {z_stk}"
    print("  [OK] _grasp_depth_z : table vs empile (anti grasp trop bas)")

    print("Tous les tests passent.")
