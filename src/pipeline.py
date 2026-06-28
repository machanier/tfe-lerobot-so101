"""
src.pipeline - Orchestration perception -> planning -> control pour pick-and-place.

C'est le module qui MET TOUT ENSEMBLE :
  1. Capture caméras + état robot.
  2. Détection 2D (HSV ou HF).
  3. Reconstruction 3D dans le repère base.
  4. Trouve l'objet cible dans la scène.
  5. Grasp planning (AdaptiveGrasp par défaut) : propose top-down/45/90 selon la
     ZONE (distance+hauteur), garde le 1er angle ATTEIGNABLE en IK.
  6. IK pour les 3 poses (approach/grasp/retract).
  7. Trajectoire jusqu'à approach, puis RAFFINEMENT cam_2 (boucle fermée) :
     recale la POSITION (résiduel stéréo) et réaligne les mâchoires sur le grand
     axe, sous garde-fous (taille du blob, plafond de correction).
  8. Offset latéral de prise en repère PINCE (le long des mâchoires) appliqué
     APRÈS cam_2 et aligné sur l'orientation finale du grasp (pince asymétrique).
  9. Descente + FERMETURE ASSERVIE au couple ; vérif après levée ; RETRY si la
     pince s'est fermée à vide (re-raffinement cam_2 + redescente).
 10. Si tenu : dépose (drop_above -> drop_release) puis retour ; sinon abandon.
 11. Exécution sur le robot via MotorController.

Usage via le CLI :
  python scripts/pick_and_place.py --target orange_cube --detector hf
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ============================================================
# Constantes display
# ============================================================
# Ordre des cameras dans le display horizontal. cam_1 a GAUCHE pour respecter
# la perspective de l'utilisateur derriere le robot : cam_1 est physiquement
# a gauche du robot vu de face (et donc a droite vu de derriere/par cam_0).
# cam_2 (eye-in-hand) toujours a droite.
CAM_DISPLAY_ORDER = ("cam_1", "cam_0", "cam_2")

# Taille de chaque tile dans la mosaic (640x360 = taille originale, 960x540
# = x1.5). Mosaic finale = 3 * tile_width x tile_height.
DISPLAY_TILE_SIZE = (960, 540)  # 1.5x la taille originale

from src.calibration.forward_kinematics import ARM_JOINTS
from src.control.ik import IKResult, IKSolver
from src.control.motor_controller import MotorController
from src.control.trajectory import (
    JointTrajectory,
    chain_trajectories,
    estimate_duration_safe,
    quintic_trajectory,
)
from src.perception.camera_io import MultiCamera
from src.perception.detector import (
    HFDetector,
    HSVDetector,
    default_hf_labels,
    default_hsv_specs,
    flatten_specs,
    load_hf_specs,
    load_hsv_specs,
)
from src.perception.pose_estimator import PoseEstimator
from src.perception.robot_state import RobotStateProvider
from src.perception.scene import Scene
from src.planning.grasp import AdaptiveGrasp, TopDownGrasp


# ============================================================
# Configuration du pipeline
# ============================================================


@dataclass
class PipelineConfig:
    """Hyperparametres du pipeline pick-and-place."""
    target_label: str = "orange_cube"
    detector_kind: str = "hsv"          # "hsv" ou "hf"
    hsv_specs_path: Optional[Path] = None
    hf_specs_path: Optional[Path] = None
    scene_config_path: Optional[Path] = None    # configs/scene.json
    motor_port: str = ""                # a fournir
    max_velocity_rad_s: float = 0.5     # vitesse articulaire max (prudent)
    grip_close_pct: float = 5.0         # fermer la pince a 5% pour grasper
    grip_open_pct: float = 100.0
    pause_grasp_s: float = 1.0          # attente apres fermeture pince
    pause_release_s: float = 0.5
    dry_run: bool = False               # True : pas d'envoi moteur, juste log
    closed_loop: bool = True            # Sprint 4 : raffinement cam_2 avant grasp
    # Plafond de la correction cam_2. PHILOSOPHIE (Maxence 2026-06-23) : cam_2 EST
    # le raffinement (proche, vue de dessus, calib plus fine) ; la stereo ne sert
    # qu'a amener le bras dans la zone. Une fois la, c'est cam_2 qui FAIT FOI. Un
    # GROS ecart cam_2<->stereo veut dire que la STEREO etait mauvaise = justement
    # le cas ou cam_2 est la plus utile -> on ne doit PAS jeter cam_2 pour ca. Le
    # vrai garde-fou de fiabilite, c'est la QUALITE du blob (cam2_min_blob_frac +
    # filtre de bord), PAS l'ampleur de l'ecart. Ce plafond ne sert donc plus qu'a
    # rejeter le cas RARISSIME ou cam_2 verrouille un objet a l'autre bout de la
    # plaque (> ~8cm = clairement un faux). 40->80mm (le cylindre couche, stereo
    # fausse de 47mm avec cam_2 juste, etait jete a tort). Reglable --closed-loop-max-correction.
    closed_loop_max_correction_mm: float = 80.0
    # Taille MINI du blob cam_2 (fraction du cadre) pour FAIRE CONFIANCE a son
    # centroide et appliquer la correction de position (Maxence 2026-06-20).
    # REMPLACE le verrou par `score` (= aire normalisee par max_area_px, une
    # constante arbitraire -> un petit objet net mais < ~10% du cadre etait
    # rejete a tort). Ici on juge la TAILLE ABSOLUE : un cube de 30mm a 8cm
    # remplit ~0.7-1.3% du cadre cam_2 (essais 2026-06-20) -> 0.3% laisse passer
    # une vraie detection tout en rejetant les fragments de masque / le bruit.
    cam2_min_blob_frac: float = 0.003
    # Plafond SERRE pour la correction cam_2 au RETRY (Maxence 2026-06-20). Le
    # refinement #1 (a l'approche, ~8cm, altitude sure) a deja centre la prise ;
    # le refinement du RETRY se fait PLUS BAS -> blob qui frole le bord/les doigts,
    # centroide bruite (essai 2026-06-20 : saut de 30mm a (273,4) loin du cube,
    # alors que 3 lectures concordaient a (297-302, 18-22)). Une vraie correction
    # de retry est un PETIT ajustement ; au-dela de ce plafond c'est un outlier
    # basse-altitude -> on garde la pose deja corrigee par le refinement #1.
    cam2_retry_max_correction_mm: float = 15.0
    # Tangage (pitch) AU-DELA duquel on n'utiliserait plus cam_2. DECISION MAXENCE
    # 2026-06-21 : cam_2 doit servir a TOUS les angles (0/45/90) -> defaut 180 =
    # jamais bloque. cam_2 sert a (1) recaler la POSITION et (2) aligner les
    # machoires sur le grand axe de l'objet, utile quel que soit l'angle. Garde-fou
    # restant : meme a 90 une correction position aberrante (>40mm) ou un blob trop
    # petit est deja rejete par les autres verrous. Si un jour le 90 derape a cause
    # de la projection rayon-plan, baisser CE seuil (ex 67.5) -- mais demander avant.
    cam2_max_pitch_deg: float = 180.0
    display: bool = False               # FENETRE LIVE cv2.imshow + grab continu des 3 cams pendant le mouvement (sature le bus -> cam_2 decroche ; cf display_grab_stride). N'a PLUS rien a voir avec la SAUVEGARDE des snapshots.
    # SAUVEGARDE des snapshots (.png) dans outputs/perception/ : snapshot perception
    # initial, vues cam_2 du raffinement, et surtout snapshots "PRISE" verite-terrain
    # (machoires + objet au moment de fermer). DECOUPLE de `display` (Maxence
    # 2026-06-23) : ces ecritures ne font que sauver une frame DEJA capturee (zero
    # grab continu) -> on peut les avoir SANS la fenetre live qui fait decrocher
    # cam_2. Actif par defaut ; --no-snapshots pour couper.
    save_snapshots: bool = True

    # Pose intermediaire "safe" entre drop_release et home (utile pour
    # eviter que le bras traverse la zone de l'objet pendant le retour).
    safe_intermediate_angles_rad: Optional[dict] = None

    # STRATEGIE DE RETOUR FINAL :
    # Si True (defaut) : le robot revient EXACTEMENT a la pose dans laquelle
    # il etait au lancement du script (capturee dans run()). Pratique :
    # Maxence place le robot en position stable + camera bien orientee, lance
    # la commande, et le robot y revient apres la pose.
    # Si False : utilise la pose fixe `home_angles_rad` ci-dessous.
    home_from_session_start: bool = True

    # Pose "home" fixe (utilisee uniquement si home_from_session_start=False).
    # Defaut : bras replie au-dessus de lui-meme, pince vers le bas.
    home_angles_rad: Optional[dict] = None

    # Vitesse RALENTIE pour la transition finale (home), pour eviter
    # l'impression de chute. Defaut : 0.3 rad/s (vs 0.5 par defaut).
    home_max_velocity_rad_s: float = 0.3

    # Ouverture pince a la fin (apres retour home). 0 = totalement fermee,
    # 100 = grand ouvert. Defaut 5 = presque fermee, pour eviter qu'une
    # pince ouverte accroche un cable / objet de la scene apres execution.
    home_gripper_pct: float = 5.0

    # Rafraichissement des detections HF en --display live. La detection HF
    # prend ~3-5s sur M4. Si on detecte a chaque rafraichissement du callback
    # (toutes les 30 frames de traj), le bras bloque pendant la detection.
    # Solution : detecter tous les N rafraichissements + cacher les dernieres
    # detections pour les afficher entre-temps. N=5 -> detection toutes les
    # ~5s, fluide pendant les intervalles. Mettre N=1 pour HSV (rapide).
    display_detect_every_n: int = 5

    # Compensation SYSTEMATIQUE des biais de calibration mesures
    # empiriquement (cf D11 : biais Y ~+28mm sur ce poste). Sera soustraite
    # a toutes les positions detectees par la stereo.
    # = np.array([dx, dy, dz]) en metres. None = pas de compensation.
    systematic_bias_correction_m: Optional[object] = None  # ndarray (3,)

    # ---------- B0 : restreindre HF a un seul label (la cible) ----------
    # Par defaut hf_specs.json contient 10 labels ("orange_cube", "tissue_box",
    # "pen", "robot_arm", ...). Sur le poste de Maxence, ces labels parasites
    # generent des fausses detections (e.g. tissue_box detecte a (+125,-243,+42)
    # alors qu'il n'y a rien la). Quand True (defaut), on ne charge que le
    # prompt qui mappe vers target_label, ce qui :
    #   - elimine les bbox parasites dans le display
    #   - reduit legerement le temps de detection HF (moins de comparaisons texte)
    #   - rend les logs lisibles (1 seul objet a discuter)
    # Mettre a False pour debugger ou comparer detections multi-objets.
    hf_restrict_to_target: bool = True

    # ---------- P1 : feedback de saisie + retry ----------
    # Apres la fermeture pince (consigne = grip_close_pct), on lit la position
    # REELLE du gripper. Si la pince a bute sur un objet, elle ne pourra pas
    # atteindre la consigne et restera ouverte d'au moins X%. Si la pince
    # atteint (presque) la consigne, c'est qu'elle s'est fermee dans le vide
    # = saisie ratee.
    # Marge en pourcentage au-dessus de grip_close_pct pour conclure "saisi".
    # Ex : grip_close_pct=5, grasp_success_threshold_pct=8 -> on conclut OK
    # si la pince reelle reste >= 5 + 8 = 13% apres fermeture.
    # CALIBRE EXPERIMENTALEMENT (2026-05-20) : sur le cube 30mm + pince TPU de
    # Maxence, la pince ferme a vide vers ~10-11% et sur le cube vers ~14-15%.
    # La marge utile est etroite (~4-5%). Seuil 8% = bon compromis : capte les
    # vraies saisies (marge ~9%) sans trop de faux positifs. Avant : 15% (trop
    # haut -> faux negatifs systematiques, vraies saisies a 14% classees RATE).
    # Ajustable via --grasp-threshold de pick_and_place.py.
    grasp_success_threshold_pct: float = 8.0
    # Seuil de COUPLE (Present_Load, magnitude brute Feetech 0-1023) au-dessus
    # duquel on considere la pince comme TENANT un objet, en COMPLEMENT (OR) de
    # la marge position. Signal fiable independant de la taille (cylindre fin
    # inclus), sans occlusion. DEFAUT 300 (cale terrain 2026-06-13) : sert de
    # plafond au seuil de CONTACT adaptatif de la fermeture asservie. Reglable
    # via --grasp-load-threshold (None = pas de plafond, contact 100%% adaptatif).
    grasp_load_threshold: Optional[float] = 300.0
    # Nombre max de RETRY apres echec (0 = pas de retry, 1 = 1 essai + 1 retry).
    max_grasp_retries: int = 1
    # Pause apres fermeture pince avant de LIRE la position (laisser le servo
    # se stabiliser sur l'objet). 0.3s est conservatif.
    grasp_settle_pause_s: float = 0.3

    # ---------- A2 : decalage lateral de la saisie (pince asymetrique) ----------
    # La pince SO-101 a un doigt FIXE et un doigt mobile. On decale le centre du
    # grasp pour que l'objet finisse contre le doigt fixe (le mobile vient
    # l'ecraser), sinon le doigt fixe percute l'objet et le pousse avant la
    # fermeture -> prise "de travers". Calibre a 8mm pour le cube 30mm.
    # Ce champ-ci (offset repere PINCE applique au moment du PLAN) reste a 0 :
    # applique au plan, cam_2 le GOMME au recentrage (verifie 2026-06-20). Le
    # decalage ACTIF est grasp_lateral_tool_offset_mm ci-dessous, applique APRES
    # cam_2 dans le MEME repere pince.
    grasp_lateral_offset_mm: Optional[float] = 0.0
    # OFFSET LATERAL DE PRISE -- repere IMAGE cam_2, "vers la GAUCHE de l'image"
    # (Maxence 2026-06-23, via _cam2_image_left_base). Applique APRES cam_2 pour
    # amener le doigt fixe a fleur de l'arete gauche de l'objet TEL QU'IL APPARAIT
    # dans le snapshot cam_2. Constant dans l'image -> reparti differemment sur X/Y
    # base selon l'orientation de la tete. But : doigt fixe au BORD, pas sur le
    # dessus. 10->15mm (Maxence 2026-06-23 : "pas assez decale a gauche"). Reglable
    # en direct via --grasp-lateral-offset (ex 18, 20) ; valeur NEGATIVE = inverse
    # le cote. 0 = desactive (centre sur l'objet).
    # NB : sert de VALEUR MANUELLE / fallback quand grasp_lateral_offset_auto=False
    # ou que la largeur d'objet est inconnue (cf ci-dessous).
    grasp_lateral_tool_offset_mm: float = 15.0
    # OFFSET LATERAL ADAPTATIF (Maxence 2026-06-23) : au lieu d'une constante, on
    # decale le doigt fixe de (DEMI-LARGEUR de l'objet + marge) pour qu'il tombe a
    # fleur de l'arete QUELLE QUE SOIT LA TAILLE -> exigence "le robot s'adapte a la
    # geometrie, pas de valeur par objet" (cf [[grasp-adaptatif-exigence]]). La
    # largeur vient de la prise (meta jaw_width_mm = petit cote de l'empreinte).
    # ATTENTION (honnete) : la STEREO sur-lit la largeur des objets ronds/obliques
    # (ex cylindre couche lu ~54mm pour un diametre ~25mm) -> l'auto peut sur-decaler
    # ces objets. Pour eux, override avec --grasp-lateral-offset <mm> (auto OFF).
    grasp_lateral_offset_auto: bool = True
    # Marge constante ajoutee a la demi-largeur (le "au cas ou la largeur est mal
    # calculee" de Maxence). Petite : juste de quoi degager l'arete. mm.
    grasp_lateral_offset_margin_mm: float = 5.0
    # OFFSET DE PROFONDEUR (Maxence 2026-06-23) -- repere IMAGE cam_2, "vers le BAS
    # de l'image". Les snapshots PRISE montrent (cube ET cylindre) que l'objet finit
    # TROP HAUT dans l'image / les machoires se ferment dans le vide AU-DESSUS de
    # lui : les doigts (rigides a cam_2) ferment a un endroit FIXE plus bas que l'axe
    # optique, mais le raffinement centre l'objet sur ~l'axe optique. On avance donc
    # la prise de cette distance le long de l'axe vertical-image (vers le bas image =
    # cote base) pour ramener les machoires SUR l'objet. CALIBRATION GEOMETRIQUE du
    # montage (machoires <-> axe optique cam_2) : CONSTANTE, identique pour TOUT
    # objet (pas un reglage par objet). Applique APRES cam_2, aux 3 sites, comme
    # l'offset lateral. Reglable --grasp-forward-offset ; NEGATIF = recule ; 0 = off.
    # 15 = estimation de depart (a affiner en regardant les snapshots PRISE).
    grasp_forward_offset_mm: float = 15.0
    # OFFSET DE SERRAGE PINCE (Z, le long de l'axe d'approche). Distance ajoutee
    # AU-DESSUS du centre de l'objet (table + H/2). None = garde le defaut de la
    # strategie, qui vaut DESORMAIS 0.0 (2026-06-21) : la pince vise le CENTRE
    # geometrique table + H/2, point. L'ancien +14mm etait une rustine qui
    # compensait une hauteur SOUS-LUE sur un grand cylindre debout ; faux en
    # general -> sur un objet court il visait AU-DESSUS du sommet (machoires dans
    # le vide -> "aucun contact"). Reglable via --grab-offset pour le reintroduire
    # ponctuellement (ex objet haut dont H est sous-estimee).
    grasp_gripper_grab_offset_m: Optional[float] = None

    # ---------- P1' : verification POST-LEVEE (anti faux positifs) ----------
    # Le couple A LA FERMETURE peut mentir : effleurement du sommet, morsure
    # de bord ou appui sur la table donnent un couple eleve SANS tenir l'objet
    # (faux positifs observes a couple=280, 328 et 500 le 2026-06-12, alors
    # que les vraies prises etaient a 356-380 -> AUCUN seuil ne separe).
    # Si True : apres une fermeture jugee OK, on monte d'abord a retract puis
    # on RELIT position+couple. Un objet tenu maintient les deux ; un objet
    # perdu retombe a la baseline -> le faux positif est attrape et on retry.
    lift_verify: bool = True

    # ---------- P5 : fermeture ASSERVIE AU COUPLE ----------
    # Si True : la fermeture n'est plus une consigne aveugle a grip_close_pct,
    # mais une rampe par pas qui LIT Present_Load et s'arrete au CONTACT
    # (+ grasp_squeeze_pct de pression de maintien). La pince s'adapte ainsi
    # a la taille de l'objet sans parametre par objet. Necessite un seuil
    # grasp_load_threshold ; sans seuil, la rampe va au plancher (equivalent
    # au comportement statique). False = fermeture statique historique.
    grasp_close_servo: bool = True
    # Serrage supplementaire apres contact (% de course pince). 4% = ferme
    # sans ecraser ; la consigne tenue reste relative a la largeur de l'objet.
    grasp_squeeze_pct: float = 4.0

    # ---------- Correction d'ORIENTATION par cam_2 ----------
    # Si True (defaut) : a la pose approach, cam_2 (proche, quasi au-dessus)
    # mesure le grand axe de l'objet et REORIENTE la prise (machoires en travers
    # du petit cote) avant de descendre. Plus fiable que la stereo oblique
    # cam_0/cam_1 dont le masque HSV est faible sur les objets fins. Repond au
    # probleme recurrent 'la pince n'est pas alignee avec l'objet'.
    cam2_reorient: bool = True
    # Elongation MIN vue par cam_2 pour faire confiance a son orientation (en
    # dessous = objet ~rond/ambigu, on ne reoriente pas).
    cam2_reorient_min_elong: float = 1.5

    # ---------- Ouverture adaptative (calibrage terrain) ----------
    # OUVERTURE MAX REELLE de la pince en mm (mesuree : 150mm sur le poste de
    # Maxence). Sert a convertir "largeur objet + marges" -> % d'ouverture.
    # None = defaut de TopDownGrasp (150). Calibrable via --gripper-max-opening.
    grasp_gripper_max_opening_mm: Optional[float] = None
    # Marge d'ouverture de CHAQUE cote (mm). Defaut TopDownGrasp = 10.
    grasp_gripper_open_margin_mm: Optional[float] = None
    # CORRECTION DE CONVENTION (deg) ajoutee a l'angle de prise. DEFAUT 90 :
    # mesure terrain 2026-06-13, la pince de Maxence ferme a 90deg de la convention
    # nominale du code (verifie sans ambiguite). Override via --grasp-yaw-offset
    # (ex: 0 pour la convention nominale, autre valeur si pince remontee autrement).
    grasp_yaw_offset_deg: Optional[float] = 90.0

    # ---------- P4' : stabilisation avant capture cam_2 ----------
    # execute_trajectory rend la main SANS pause finale -> la capture cam_2 du
    # raffinement se faisait pendant que le bras finissait de bouger -> l'objet
    # apparaissait systematiquement plus bas dans l'image (Dv>0 sur TOUS les
    # essais du 2026-06-12) -> correction Y parasite ~-15mm. On laisse le bras
    # SE STABILISER avant de lire l'etat + capturer.
    cam2_settle_s: float = 0.25

    # ---------- RECUL d'observation cam_2 (Maxence 2026-06-22) ----------
    # cam_2 (eye-in-hand) est RIGIDEMENT montee pres des doigts -> ceux-ci
    # occupent toujours ~le bas du cadre (v/H~0.82, INDEPENDANT de l'altitude).
    # A la pose approach standard (8cm), l'objet apparait juste AU-DESSUS des
    # doigts (v/H~0.57) : son bord PROCHE est mordu par les doigts -> le centroide
    # detecte est tire vers le bord ARRIERE (cote cams fixes) -> la prise tombe
    # sur ce bord. HAUTEUR D'OBSERVATION cam_2 = REFERENCE DE DETECTION (Maxence
    # 2026-06-25) : la capture cam_2 se fait a CETTE hauteur AU-DESSUS DE LA
    # POSITION STEREO de l'objet (12cm). On vise Z = Z_objet + 12cm DIRECTEMENT
    # (plus aucun "8cm + 4cm") : l'objet remonte vers le centre optique et S'ELOIGNE
    # des doigts (qui occupent le bas du cadre) -> objet entier visible, centroide
    # non biaise (verifie sim : a 12cm le blob fait encore ~1.3% du cadre > seuil
    # 0.3%). Puis raffinement, puis descente DIRECTE sur l'objet (qui passe par la
    # pose d'approche : simple point de passage, PAS un arret de detection). La
    # descente reste monotone, PAS de "descend/remonte/redescend". Reglable via
    # --cam2-observe-height (mm, defaut 120 = 12cm). Si le blob devient trop petit
    # (objet loin/petit) le gating area_frac retombe proprement sur la stereo.
    cam2_observe_height_m: float = 0.12

    # Distance (m) jusqu'a laquelle un objet BAS est pris en TOP-DOWN ; au-dela ->
    # diagonale 45 (cf preferred_pitch_deg / GRASP_ZONE_NEAR_M). Reglable via
    # --zone-topdown (mm) pour mesurer la frontiere top-down/45 en campagne.
    grasp_zone_topdown_m: float = 0.32

    # ---------- AFFICHAGE : throttle des captures pendant les trajectoires ----
    # Avec --display, on_step capturait les 3 cameras 1080p a CHAQUE pas de
    # trajectoire (~32/s sur 7s) -> SATURATION USB -> cam_2 (eye-in-hand, connecteur
    # fragile) decroche en plein run ("grab() KO", Maxence 2026-06-23, 4 runs/6
    # morts ainsi). On ne rafraichit l'affichage que tous les N pas -> le bus reste
    # libre pour les captures IMPORTANTES (refine/snapshot). Display SEUL, aucun
    # effet sur la prise. 1 = ancien comportement (capture chaque pas).
    display_grab_stride: int = 5

    # ---------- RETRY : stabilisation PLUS LONGUE avant re-capture ----------
    # Sur un RETRY (1re tentative ratee), l'objet a souvent BOUGE (effleure /
    # pousse) et le bras vient de remonter -> on prend le TEMPS de se stabiliser
    # avant de re-detecter, pour repartir d'une perception PROPRE (demande Maxence
    # 2026-06-21 : "qu'il ait le temps de recapter correctement l'objet"). Plus
    # long que cam2_settle_s (le settle normal en vol). Reglable.
    retry_settle_s: float = 1.0

    # ---------- RETRY : RE-PERCEPTION par les cameras FIXES ----------
    # Au RETRY, cam_2 (eye-in-hand) est parquee AU-DESSUS de l'ANCIENNE position ->
    # si l'objet a roule/bouge ailleurs, cam_2 ne le voit plus (ou voit du vide) et
    # on redescendait au meme endroit. Avec ce flag (defaut True), on relance une
    # PERCEPTION STEREO COMPLETE (cam_0 + cam_1, champ large) pour RE-LOCALISER la
    # cible dans toute la scene, puis on REPLANIFIE la prise depuis sa nouvelle
    # position (l'objet a pu changer de pose -> nouvel angle). Demande Maxence
    # 2026-06-21 : "lui associer les cams fixes, l'objet est decale la ou cam_2 ne
    # voit pas". Si la re-perception ne retrouve pas la cible -> on retombe sur le
    # raffinement cam_2 seul (comportement precedent). Re-plan SEULEMENT si la cible
    # a BOUGE de plus de ce seuil (sinon on garde la pose courante + cam_2).
    retry_reperceive: bool = True
    retry_reperceive_move_mm: float = 12.0

    # ---------- RAFFINEMENT SIMPLE vs DOUBLE ----------
    # Si False (defaut, 2026-06-13) : UN SEUL refinement cam_2 a la pose approach
    # (hauteur sure), puis le bras se REPOSITIONNE A 8cm (lateralement) et
    # DESCEND TOUT DROIT sur l'objet. Repond a la demande de Maxence : "analyse
    # avant de descendre", plus de mini-descente + recorrection a basse altitude
    # qui faisait "descendre un peu, remonter se repositionner, redescendre" et
    # frolait l'objet. Le 2e refinement etait de toute facon bruite (biais Y).
    # Si True : ancien comportement a deux etages (approach -> mini-descente 4cm
    # -> refinement #2 -> descente). Plus precis en theorie, mais source du
    # va-et-vient observe.
    closed_loop_two_stage: bool = False

    # ---------- MODE DE SAISIE : adaptatif (defaut) vs top-down ----------
    # "adaptive" : AdaptiveGrasp choisit l'angle d'attaque sur le balayage 180deg
    #   du plan sagittal (top-down / diagonale / frontal) en gardant la 1ere prise
    #   ATTEIGNABLE (filtre IK) dans l'ordre de preference. Etend la couverture aux
    #   objets hauts/lointains que le top-down seul ne peut pas saisir.
    # "top_down" : comportement historique (TopDownGrasp), force par --top-down.
    #   Le candidat top-down d'AdaptiveGrasp est de toute facon identique a
    #   TopDownGrasp -> pas de regression en mode adaptatif sur les objets bas.
    grasp_mode: str = "adaptive"

    # ---------- Saisie adaptative : seuils d'ATTEIGNABILITE IK ----------
    # Un candidat d'angle est retenu si sa pose GRASP est atteinte sous ces
    # residus (et non le flag `converged`, trop strict en 5 DDL). PROVENANCE des
    # valeurs (PROVISOIRES, a confirmer en campagne d'essais) :
    #   - 8mm = residu IK typique du grasp (~0.3mm en sim) + budget calibration
    #     stereo (~5-8mm, cf D17). A re-mesurer si la precision change.
    #   - 15deg = tolerance d'orientation acceptee vu la sous-actuation 5/6 DDL
    #     (une pince ne se referme pas exactement a l'angle vise). Indicatif.
    ik_tol_trans_mm: float = 8.0
    ik_tol_rot_deg: float = 15.0
    # Tolerance de position pour la pose APPROACH (plus laxiste : la precision
    # compte moins a l'approche). Sert a rejeter un approche HORS workspace.
    ik_tol_approach_mm: float = 15.0
    # BORNE DURE du repli : si aucun candidat n'est atteignable, on n'execute le
    # "moins mauvais" que s'il reste sous ces bornes ; au-dela on declare l'echec
    # (eviter de forcer le bras sur une pose desesperee). PROVISOIRE.
    grasp_fallback_max_trans_mm: float = 25.0
    grasp_fallback_max_rot_deg: float = 25.0
    # Hauteur de prise mini requise pour une prise INCLINEE (degagement table) :
    # required = min_clearance(5mm) + |sin(theta)| * ce terme. A theta=90 (face) :
    # required = 5 + 3 = 8mm -> un objet dont le CENTRE est a >=8mm du sol est
    # saisissable de face. Reglage 2026-06-17 (Maxence) : 25->3mm pour que des
    # objets bas comme le cube de 30mm (centre a 15mm) soient prenables de face
    # quand le top-down est hors de portee (machoires ~+/-5mm autour du centre,
    # ~3mm de marge au sol). A remonter si le bas des doigts TPU touche la table.
    grasp_side_min_height_m: Optional[float] = 0.003
    # Roll (deg) applique aux prises INCLINEES autour de l'axe d'approche, pour la
    # convention pince. Maxence 2026-06-27 : apres plus d'essais, 45 etait
    # "n'importe quoi" a 30-35cm (prises 45deg) -> RETOUR a 90. Le roll incline reste
    # imparfait par nature (la convention ignore le yaw reel de l'objet -> asymetrie
    # gauche/droite ; aucune valeur fixe n'est ideale des deux cotes) ; 90 = le moins
    # pire a l'usage. N'affecte QUE theta!=0 : le top-down delegue a TopDownGrasp
    # AVANT ce roll, donc proche/mi inchanges. Surchargeable via --tilt-roll-offset.
    grasp_tilt_roll_deg: Optional[float] = 90.0
    # Hauteur d'objet (m) au-dela de laquelle le candidat TOP-DOWN est refuse
    # (collision pince/objet par le haut). None = defaut strategie (0.12 = 12cm).
    # En adaptatif, au-dela l'incline (45/90) prend le relais. Reglable via
    # --max-top-down-height (utile pres du seuil ou la hauteur mesuree est bruitee).
    grasp_max_top_down_height_m: Optional[float] = None

    # ---------- SECURITE : rejet du demi-tour de poignet ----------
    # Une prise dont l'IK exige un saut de wrist_roll > ce seuil depuis la pose
    # COURANTE = la pince se RETOURNE (~180deg) et plonge vers la table, bras
    # contorsionne (observe + photos 2026-06-17 : a failli casser le support
    # cam_2). On REFUSE alors d'executer cette prise (echec propre) plutot que de
    # forcer mecaniquement. Reperes mesures : les prises SAINES sautent <=40deg ;
    # les retournements dangereux ~178deg -> 120deg les separe largement.
    # PROVISOIRE (a confirmer en campagne). Override CLI : --wrist-flip-max-deg.
    wrist_flip_max_deg: float = 120.0


# ============================================================
# Orchestrateur principal
# ============================================================


class PickAndPlacePipeline:
    """Pipeline complet pour saisir un objet et le poser dans la boite.

    Usage typique :
        config = PipelineConfig(target_label="orange_cube", motor_port="/dev/...")
        pipeline = PickAndPlacePipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._load_scene()
        self._build_perception()
        grasp_kwargs = {}
        if self.config.grasp_lateral_offset_mm is not None:
            grasp_kwargs["grasp_lateral_offset_mm"] = self.config.grasp_lateral_offset_mm
        if self.config.grasp_gripper_max_opening_mm is not None:
            grasp_kwargs["gripper_max_opening_mm"] = self.config.grasp_gripper_max_opening_mm
        if self.config.grasp_gripper_open_margin_mm is not None:
            grasp_kwargs["gripper_open_margin_mm"] = self.config.grasp_gripper_open_margin_mm
        if self.config.grasp_yaw_offset_deg is not None:
            grasp_kwargs["yaw_offset_deg"] = self.config.grasp_yaw_offset_deg
        if self.config.grasp_gripper_grab_offset_m is not None:
            grasp_kwargs["gripper_grab_offset_m"] = self.config.grasp_gripper_grab_offset_m
        if self.config.grasp_max_top_down_height_m is not None:
            grasp_kwargs["max_object_height_m"] = self.config.grasp_max_top_down_height_m
        # Strategie de saisie : adaptative (defaut) ou top-down (--top-down).
        # Les deux acceptent les memes kwargs de base (offset, ouverture, marge,
        # yaw) ; AdaptiveGrasp accepte en plus le degagement table et le roll
        # incline (specifiques aux prises inclinees).
        if self.config.grasp_mode == "top_down":
            self._grasp_strategy = TopDownGrasp(**grasp_kwargs)
        else:
            adaptive_kwargs = dict(grasp_kwargs)
            if self.config.grasp_side_min_height_m is not None:
                adaptive_kwargs["side_grasp_min_height_m"] = self.config.grasp_side_min_height_m
            if self.config.grasp_tilt_roll_deg is not None:
                adaptive_kwargs["tilted_roll_deg"] = self.config.grasp_tilt_roll_deg
            self._grasp_strategy = AdaptiveGrasp(**adaptive_kwargs)
        self._ik = IKSolver()
        # IK SPECIFIQUE A LA DEPOSE : poids de rotation reduit (0.05 vs 0.1
        # par defaut). Pour la depose on s'en moque que la pince soit pile
        # verticale (elle ouvre, le cube tombe). On privilegie donc la
        # position. Compromis 0.05 (vs 0.01 essaye precedemment) : evite que
        # l'IK accepte des poses tordues a 90deg qui provoquaient des
        # collisions du bras avec lui-meme. Avec 0.05 : rot < 15deg.
        self._ik_drop = IKSolver(rotation_weight=0.05)
        self._provider = RobotStateProvider()

    def _load_scene(self):
        """Charge la position de la boite de depose.

        CONVENTION : `center_base_m` est interprete comme le centre du **FOND**
        de la boite (= sa face posee sur la table). C'est plus naturel a
        mesurer au metre que le centre du dessus. Le dessus est calcule
        automatiquement = fond + box_height.

        Calcule deux poses cibles au-dessus de la boite :
          - drop_above   : 8 cm au-dessus du dessus -> transit, l'objet pendu
                           sous la pince degage le bord de la boite.
          - drop_release : 3 cm au-dessus du dessus -> point de relachement.

        Emet un avertissement si la position semble incoherente (boite
        chevauche la base du robot ou hors workspace) -- typique d'un
        scene.json non mesure.
        """
        scene_path = self.config.scene_config_path or (REPO / "configs" / "scene.json")
        if not scene_path.exists():
            raise FileNotFoundError(f"scene.json manquant : {scene_path}")
        data = json.load(open(scene_path))
        box = data["drop_box"]
        # `center_base_m` = centre du FOND de la boite (convention)
        bottom_center = np.array(box["center_base_m"], dtype=np.float64)
        box_h = float(box["dimensions_m"][2])
        # Dessus de la boite = fond + hauteur
        top_z = float(bottom_center[2]) + box_h
        # CLEARANCE BOITE (2026-06-17) : l'objet TENU depasse SOUS la pince (un
        # cylindre saisi pres d'un bout pend de plusieurs cm). A 5cm/2cm il
        # frolait/cognait le bord de la boite au transit (signale par Maxence).
        # On transite plus haut (drop_above) pour que l'objet pendu degage le
        # dessus, puis on descend DROIT dans l'ouverture. 8cm reste atteignable a
        # la position lointaine de la boite (Y=-225) ; 12cm risquait l'IK depose.
        self.drop_position = np.array([bottom_center[0], bottom_center[1], top_z])
        self.drop_above   = self.drop_position + np.array([0.0, 0.0, 0.08])
        self.drop_release = self.drop_position + np.array([0.0, 0.0, 0.03])

        # AVERTISSEMENT si position incoherente
        x, y = float(bottom_center[0]), float(bottom_center[1])
        dist_to_base = np.hypot(x, y)
        warnings = []
        if dist_to_base < 0.12:
            warnings.append(
                f"boite tres proche de la base ({dist_to_base*100:.1f} cm) -- "
                f"chevauche probablement la zone d'exclusion du robot "
                f"(rayon 10cm). Le bras va se cogner."
            )
        ws = data.get("workspace_bounds_base_m", {})
        if ws:
            if not (ws.get("x_min", -1) <= x <= ws.get("x_max", 1)):
                warnings.append(f"X={x:.3f}m hors workspace [{ws.get('x_min')}, {ws.get('x_max')}]")
            if not (ws.get("y_min", -1) <= y <= ws.get("y_max", 1)):
                warnings.append(f"Y={y:.3f}m hors workspace [{ws.get('y_min')}, {ws.get('y_max')}]")

        # Log informatif des poses calculees
        print(f">> Boite de depose chargee depuis {scene_path.name} :")
        print(f"   center_fond = ({bottom_center[0]*1000:+6.1f}, {bottom_center[1]*1000:+6.1f}, "
              f"{bottom_center[2]*1000:+6.1f}) mm  (X devant, Y gauche+/droite-, Z table)")
        print(f"   dimensions  = {box['dimensions_m'][0]*100:.1f} x "
              f"{box['dimensions_m'][1]*100:.1f} x {box_h*100:.1f} cm")
        print(f"   dessus boite a Z = {top_z*1000:.1f} mm")
        print(f"   drop_above   = ({self.drop_above[0]*1000:+6.1f}, "
              f"{self.drop_above[1]*1000:+6.1f}, {self.drop_above[2]*1000:+6.1f}) mm")
        print(f"   drop_release = ({self.drop_release[0]*1000:+6.1f}, "
              f"{self.drop_release[1]*1000:+6.1f}, {self.drop_release[2]*1000:+6.1f}) mm")
        for w in warnings:
            print(f"   [WARN scene.json] {w}")
        if warnings:
            print(f"   --> Verifie configs/scene.json. Si tu changes la boite de place, "
                  f"remesure et mets a jour center_base_m.")

    def _build_perception(self):
        """Construit detecteur + estimator selon la config."""
        if self.config.detector_kind == "hsv":
            specs_path = self.config.hsv_specs_path or (REPO / "configs/perception/hsv_specs.json")
            specs = load_hsv_specs(specs_path) if specs_path.exists() else default_hsv_specs()
            self._detector = HSVDetector(specs)
            # label->meta independant de la camera : union si specs par-camera.
            self._specs_meta = {s.label: s.meta for s in flatten_specs(specs)}
            self._label_mapping = {}  # HSV : labels = label internes directement
        elif self.config.detector_kind == "hf":
            hf_path = self.config.hf_specs_path or (REPO / "configs/perception/hf_specs.json")
            if hf_path.exists():
                cfg = load_hf_specs(hf_path)
                labels = cfg["labels"]
                model_name = cfg.get("model_name", "google/owlv2-base-patch16-ensemble")
                threshold = float(cfg.get("score_threshold", 0.15))
                self._label_mapping = cfg.get("_label_mapping") or {}
            else:
                labels = default_hf_labels()
                model_name = "google/owlv2-base-patch16-ensemble"
                threshold = 0.15
                self._label_mapping = {}
            # B0 : si demande, on ne garde que le prompt qui mappe vers target_label
            # Elimine les detections parasites (tissue_box, pen, robot_arm, etc.)
            if self.config.hf_restrict_to_target:
                target = self.config.target_label
                prompts_to_keep = [p for p in labels
                                   if self._label_mapping.get(p, p) == target]
                if prompts_to_keep:
                    n_before = len(labels)
                    labels = prompts_to_keep
                    self._label_mapping = {
                        p: m for p, m in self._label_mapping.items()
                        if p in prompts_to_keep
                    }
                    print(f">> HF restreint a {len(labels)}/{n_before} label(s) "
                          f"(cible='{target}') : {labels}")
                else:
                    print(f"[WARN] aucun prompt HF ne mappe vers '{target}', "
                          f"utilisation des {len(labels)} labels par defaut")
            self._detector = HFDetector(prompt_labels=labels,
                                        model_name=model_name,
                                        score_threshold=threshold)
            self._specs_meta = {}
        else:
            raise ValueError(f"detector_kind inconnu: {self.config.detector_kind}")
        self._estimator = PoseEstimator(specs_by_label=self._specs_meta)

    # ----- grasp planning + IK selection ----------------------------------

    @staticmethod
    def _wrist_jump_deg(q_ref: dict, ik_res) -> float:
        """Amplitude (deg) du saut de wrist_roll entre la pose de reference et la
        solution IK. > wrist_flip_max_deg = demi-tour dangereux (pince a l'envers)."""
        try:
            if ik_res is None or not getattr(ik_res, "joint_angles_rad", None):
                return 0.0
            wr0 = float(np.degrees(q_ref.get("wrist_roll", 0.0)))
            wr1 = float(np.degrees(ik_res.joint_angles_rad.get("wrist_roll", 0.0)))
            return abs(wr1 - wr0)
        except Exception:
            return 0.0

    def _plan_and_solve_grasp(self, target, q_current):
        """Planifie la saisie et resout l'IK ; renvoie (grasp_pose, r_app, r_grp,
        r_ret) ou (None, None, None, None) si rien de faisable.

        Mode top-down : comportement historique (1 plan + IK, branche yaw_free).
        Mode adaptatif : *generate -> filter(IK) -> rank*. On evalue les candidats
        d'angle d'attaque dans l'ordre de preference et on garde le PREMIER
        reellement ATTEIGNABLE (residus IK sous seuil). Tous les candidats sont
        loggues (utile pour les essais et les campagnes). Si aucun n'est pleinement
        atteignable, repli sur le moins mauvais (comportement degrade, comme avant)
        avec avertissement.
        """
        strat = self._grasp_strategy

        # ---- mode top-down : inchange ----
        if not isinstance(strat, AdaptiveGrasp):
            grasp_pose = strat.plan(target)
            if grasp_pose is None:
                return None, None, None, None
            print(f">> Grasp planifie ({grasp_pose.meta.get('strategy')}, "
                  f"yaw={np.degrees(grasp_pose.meta.get('yaw_rad', 0)):+.0f}deg, "
                  f"ouverture pince={grasp_pose.gripper_open_pct:.0f}%)")
            if grasp_pose.meta.get("yaw_free"):
                r_app, r_grp, r_ret = self._ik.solve_grasp_pose_free_yaw(
                    grasp_pose, q_init=q_current)
                yc = grasp_pose.meta.get("yaw_committed_deg")
                print(f"   (objet debout -> yaw LIBRE choisi pour minimiser la "
                      f"rotation du poignet : yaw={yc:+.0f}deg)" if yc is not None
                      else "   (objet debout -> yaw libre)")
            else:
                r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                    grasp_pose, q_init=q_current)
                if grasp_pose.meta.get("flipped_180"):
                    print(f"   (orientation de prise retournee de 180deg retenue, "
                          f"yaw={np.degrees(grasp_pose.meta.get('yaw_rad', 0)):+.0f}deg ; "
                          f"matrices persistees -> cibles coherentes pour la suite)")
            # NB : en TOP-DOWN la pince pointe toujours vers le bas ; un grand
            # wrist_roll n'est qu'une rotation des machoires (jamais un
            # retournement dangereux) -> pas de garde-fou ici. Le garde-fou
            # anti-retournement ne s'applique qu'aux prises INCLINEES (cf adaptatif).
            return grasp_pose, r_app, r_grp, r_ret

        # ---- mode adaptatif : balayage d'angles, choix par faisabilite IK ----
        cands = strat.plan_candidates(target)
        if not cands:
            print("!! Aucun candidat geometriquement faisable "
                  "(ouverture pince / degagement table).")
            return None, None, None, None

        # seuils d'ATTEIGNABILITE de la pose grasp. Le bras est 5 DDL (sous-
        # actionne) -> on tolere une petite erreur d'orientation, mais la POSITION
        # doit etre atteinte. cf reachability-aware grasping.
        c = self.config
        TRANS_OK_MM, ROT_OK_DEG = c.ik_tol_trans_mm, c.ik_tol_rot_deg
        APP_OK_MM = c.ik_tol_approach_mm
        print(f">> Saisie adaptative : {len(cands)} candidat(s) d'angle, choix par "
              f"atteignabilite IK + angle prefere selon la ZONE "
              f"(distance+hauteur : proche/bas->top-down, mi->45, loin/haut->face) :")
        results = []  # (gp, r_app, r_grp, r_ret, reachable)
        for gp in cands:
            if gp.meta.get("yaw_free"):
                r_app, r_grp, r_ret = self._ik.solve_grasp_pose_free_yaw(
                    gp, q_init=q_current)
            else:
                r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                    gp, q_init=q_current)
            # Atteignabilite = RESIDUS sous seuil (et non le flag `converged`,
            # trop strict en 5 DDL). On exige que la pose GRASP (position +
            # orientation) ET la pose APPROACH (position) soient atteintes : une
            # frontale peut avoir un grasp OK mais un approach recule de 8cm HORS
            # workspace -> a detecter ici. cf reachability-aware grasping.
            # SECURITE : une prise INCLINEE qui exige un demi-tour de poignet
            # (pince a l'envers, plonge vers la table) est traitee comme NON
            # atteignable. En top-down (theta=0) un grand wrist_roll = simple
            # rotation des machoires, jamais dangereux -> on N'applique PAS le
            # garde-fou (sinon on refuse a tort des prises top-down valides).
            tilted = abs(gp.meta.get("pitch_deg", 0.0)) > 1e-6
            wj = self._wrist_jump_deg(q_current, r_grp)
            flips = tilted and (wj > c.wrist_flip_max_deg)
            ik_ok = (r_grp.translation_err_mm <= TRANS_OK_MM
                     and r_grp.rotation_err_deg <= ROT_OK_DEG
                     and r_app.translation_err_mm <= APP_OK_MM)
            reachable = ik_ok and not flips
            if flips:
                verdict = f"rejete (SAUT poignet {wj:.0f}deg)"
            elif ik_ok:
                verdict = "atteignable"
            else:
                verdict = "rejete (IK)"
            print(f"   theta={gp.meta['pitch_deg']:+5.0f}deg  "
                  f"roll={np.degrees(gp.meta.get('roll_rad', 0.0)):+4.0f}deg  "
                  f"machoires={gp.meta['jaw_width_mm']:4.0f}mm  "
                  f"ouv={gp.gripper_open_pct:3.0f}%  "
                  f"IK: grasp {r_grp.translation_err_mm:4.1f}mm/"
                  f"{r_grp.rotation_err_deg:4.1f}deg  "
                  f"app {r_app.translation_err_mm:4.1f}mm  "
                  f"[{verdict}]")
            results.append((gp, r_app, r_grp, r_ret, reachable))

        # COHERENCE POLITIQUE (2026-06-26) : pour un objet BAS (sommet < tall_m, 12cm)
        # la prise de FACE 90 est EXCLUE meme si c'est le SEUL angle atteignable -> la
        # tete eye-in-hand bute contre la table (couple=1004 mesure). Deja retiree de
        # la PREFERENCE (preferred_pitch_deg) ; on la retire aussi ici de la SELECTION
        # ET du repli. Si plus rien n'est faisable -> on decline proprement (objet bas
        # + loin = hors domaine fiable). Le 90 reste autorise pour les objets HAUTS.
        from src.planning.grasp import GRASP_ZONE_TALL_M
        _bb = target.bbox_3d_m
        _Hm = float(_bb[2]) if _bb is not None else 0.0
        _h_top = float(target.position_base_m[2]) + 0.5 * _Hm
        if _h_top < GRASP_ZONE_TALL_M:
            _n_before = len(results)
            results = [r for r in results
                       if abs(r[0].meta.get("pitch_deg", 0.0)) < 67.5]
            if len(results) < _n_before:
                print(f"   [politique] objet BAS (sommet {_h_top*100:.0f}cm < "
                      f"{GRASP_ZONE_TALL_M*100:.0f}cm) -> prise de face 90 exclue "
                      f"(la tete buterait au sol).")
            if not results:
                print("!! Objet BAS : seule la prise de face 90 etait faisable, exclue "
                      "(tete au sol) -> hors domaine fiable, on N'EXECUTE PAS.")
                return None, None, None, None

        reachable_list = [r for r in results if r[4]]
        if reachable_list:
            # CHOIX = parmi les angles ATTEIGNABLES (IK), celui le plus proche de
            # l'angle PREFERE selon la ZONE d'usage (distance + hauteur), Maxence
            # 2026-06-18. Chaque angle a son domaine :
            #   - top-down (0)   : objet PROCHE et BAS ;
            #   - diagonale (45) : mi-distance / mi-hauteur (la plus polyvalente) ;
            #   - face (90)      : objet LOIN et/ou HAUT.
            # Si le prefere n'est pas atteignable -> on prend le voisin atteignable
            # (jamais d'echec "top-down force alors qu'une autre prise marchait").
            # Bornes de zone reglables dans grasp.py (GRASP_ZONE_*). Egalite ->
            # pitch le plus petit (poignet moins reoriente = plus sur).
            from src.planning.grasp import preferred_pitch_deg
            pref = preferred_pitch_deg(
                target, near_m=self.config.grasp_zone_topdown_m)
            best = min(reachable_list,
                       key=lambda r: (abs(abs(r[0].meta["pitch_deg"]) - pref),
                                      abs(r[0].meta["pitch_deg"])))
            chosen = best[:4]
            import numpy as _np
            _d = float(_np.hypot(target.position_base_m[0], target.position_base_m[1]))
            print(f"   -> zone : d={_d*100:.0f}cm -> angle prefere {pref:+.0f}deg ; "
                  f"retenu {best[0].meta['pitch_deg']:+.0f}deg parmi "
                  f"{len(reachable_list)} atteignable(s)")
        else:
            # repli : le moins mauvais (residu grasp), borne dure sinon echec propre.
            best = min(results, key=lambda r: r[2].translation_err_mm)
            gp, r_app, r_grp, r_ret, _ = best
            ftrans = r_grp.translation_err_mm
            frot = r_grp.rotation_err_deg
            fapp = r_app.translation_err_mm
            ftilted = abs(gp.meta.get("pitch_deg", 0.0)) > 1e-6
            fwrist = self._wrist_jump_deg(q_current, r_grp)
            # SECURITE : ne JAMAIS executer un repli INCLINE qui retourne le poignet
            # (pince a l'envers). En top-down un grand wrist_roll est inoffensif.
            if ftilted and fwrist > c.wrist_flip_max_deg:
                print(f"!! SECURITE : le moins mauvais candidat (incline) exige un "
                      f"demi-tour de poignet ({fwrist:.0f}deg > "
                      f"{c.wrist_flip_max_deg:.0f}deg) -> pince a l'envers. "
                      f"ON N'EXECUTE PAS.")
                return None, None, None, None
            if (ftrans > c.grasp_fallback_max_trans_mm
                    or frot > c.grasp_fallback_max_rot_deg
                    or fapp > c.grasp_fallback_max_trans_mm):
                print(f"!! Aucun candidat atteignable ; le moins mauvais "
                      f"(theta={gp.meta['pitch_deg']:+.0f}deg, grasp "
                      f"{ftrans:.1f}mm/{frot:.1f}deg, app {fapp:.1f}mm) DEPASSE les "
                      f"bornes ({c.grasp_fallback_max_trans_mm:.0f}mm/"
                      f"{c.grasp_fallback_max_rot_deg:.0f}deg) -> on N'EXECUTE PAS "
                      f"(eviter de forcer mecaniquement).")
                return None, None, None, None
            print(f"   [WARN] aucun candidat pleinement atteignable -> repli sur le "
                  f"moins mauvais DANS LES BORNES (theta={gp.meta['pitch_deg']:+.0f}deg, "
                  f"grasp {ftrans:.1f}mm, app {fapp:.1f}mm). A surveiller a l'essai.")
            chosen = (gp, r_app, r_grp, r_ret)

        gp, r_app, r_grp, r_ret = chosen
        print(f">> Prise retenue : {gp.meta['strategy']} "
              f"theta={gp.meta['pitch_deg']:+.0f}deg "
              f"roll={np.degrees(gp.meta.get('roll_rad', 0)):+.0f}deg "
              f"(ouverture pince={gp.gripper_open_pct:.0f}%)")
        return gp, r_app, r_grp, r_ret

    @property
    def _T_gripper_cam2(self):
        """Transform hand-eye cam_2 (eye-in-hand), charge une fois."""
        if getattr(self, "_t_gripper_cam2_cache", None) is None:
            from src.perception.camera_io import load_handeye
            self._t_gripper_cam2_cache = np.asarray(
                load_handeye("cam_2")["transform"], dtype=float)
        return self._t_gripper_cam2_cache

    def _cam2_image_left_base(self, T_base_gripper):
        """Direction "GAUCHE DANS L'IMAGE cam_2" en repere base, horizontale.

        ⚠️ Calcule depuis l'orientation du GRASP FINAL (T_base_gripper = pose de
        prise APRES toute reorientation cam_2), PAS la pose d'observation : cam_2 est
        rigide a la pince -> "gauche image" est une direction FIXE dans le repere
        pince ; l'offset doit donc utiliser l'orientation du poignet A LA PRISE pour
        caler le doigt FIXE au bon cote au moment OU les machoires ferment. Sinon, si
        le wrist tourne entre l'observation et la prise (objets allonges -> cam_2
        reoriente), l'offset pousse l'objet en BORD de pince (bug 2026-06-28, Maxence).

        Maxence (2026-06-23) : le decalage de degagement du doigt fixe se definit
        PAR RAPPORT AU SNAPSHOT cam_2 (l'avis propre de cam_2 sur l'objet), PAS en
        repere pince ni base. Le but : amener le doigt fixe a fleur de l'arete
        GAUCHE de l'objet TELLE QU'ELLE APPARAIT DANS L'IMAGE. "Gauche image" =
        -X camera (u decroissant). On la ramene en base via la pose REELLE de
        cam_2 = T_base_gripper(q) @ T_gripper_cam2, puis on PROJETTE sur
        l'horizontale (le decalage de prise est dans le plan table) et on
        normalise. Consequence (exactement ce que Maxence decrit) : un MEME
        "gauche image" -> direction base DIFFERENTE selon l'orientation de la tete,
        donc attribuee differemment a X/Y de la position calculee, mais le
        decalage reste CONSTANT du point de vue de la camera. Signe : +lat_mm =
        vers la gauche image ; passer une valeur NEGATIVE inverse le cote.
        """
        R_base_cam2 = (np.asarray(T_base_gripper, float)[:3, :3]
                       @ self._T_gripper_cam2[:3, :3])
        img_left = R_base_cam2 @ np.array([-1.0, 0.0, 0.0])  # -X cam = gauche image
        img_left[2] = 0.0                                    # projette sur la table
        n = float(np.linalg.norm(img_left))
        if n < 1e-9:
            return np.zeros(3)
        return img_left / n

    def _cam2_image_down_base(self, T_base_gripper):
        """Direction "VERS LE BAS DE L'IMAGE cam_2" en repere base, horizontale.

        ⚠️ Comme _cam2_image_left_base : calcule depuis l'orientation du GRASP FINAL
        (T_base_gripper), pas la pose d'observation -> coherent avec le wrist a la
        prise (apres reorientation cam_2).

        Jumeau de _cam2_image_left_base sur l'AUTRE axe image (+Y camera = v
        croissant = bas de l'image). Sert au DECALAGE DE PROFONDEUR : sur les
        snapshots PRISE, les machoires (rigides a cam_2) occupent un endroit FIXE
        plus BAS dans le cadre que l'axe optique, alors que le raffinement centre
        l'objet sur ~l'axe optique -> l'objet finit AU-DESSUS de la ligne de
        fermeture -> machoires dans le vide (Maxence 2026-06-23, vu sur cube ET
        cylindre). Avancer la prise de quelques mm le long de cette direction (vers
        le bas image = cote base, -X ici) ramene les machoires SUR l'objet. C'est
        une CALIBRATION GEOMETRIQUE du montage (decalage machoires<->axe optique
        cam_2), CONSTANTE et identique pour tout objet -- pas un reglage par objet.
        """
        R_base_cam2 = (np.asarray(T_base_gripper, float)[:3, :3]
                       @ self._T_gripper_cam2[:3, :3])
        img_down = R_base_cam2 @ np.array([0.0, 1.0, 0.0])   # +Y cam = bas image
        img_down[2] = 0.0                                    # projette sur la table
        n = float(np.linalg.norm(img_down))
        if n < 1e-9:
            return np.zeros(3)
        return img_down / n

    def _effective_lateral_offset_mm(self, grasp_pose) -> float:
        """Magnitude de l'offset lateral de prise (mm), ADAPTATIVE a la taille.

        Maxence 2026-06-23 : l'offset qui amene le doigt FIXE a fleur de l'arete
        doit valoir ~(DEMI-LARGEUR de l'objet + marge), pas une constante codee en
        dur (sinon trop sur les gros objets, trop peu sur les petits -> pas
        adaptatif). La largeur vient de la prise (meta `jaw_width_mm` = petit cote
        de l'empreinte = ce que les machoires serrent). Signe : + = gauche image.

        Fallback sur la valeur MANUELLE `grasp_lateral_tool_offset_mm` si l'auto est
        coupe (override `--grasp-lateral-offset`) ou si la largeur est inconnue.
        """
        manual = float(self.config.grasp_lateral_tool_offset_mm)
        if not self.config.grasp_lateral_offset_auto:
            return manual
        w = grasp_pose.meta.get("jaw_width_mm")
        if not w or w <= 0:
            return manual
        return 0.5 * float(w) + float(self.config.grasp_lateral_offset_margin_mm)

    def _reperceive_target(self, mc):
        """Re-capture cam_0/cam_1/cam_2 + build_scene -> ObjectInstance cible (ou None).

        Utilise au RETRY pour "reprendre conscience de la scene" : si l'objet a
        bouge hors du champ de cam_2 (eye-in-hand parquee au-dessus de l'ancienne
        position), les cameras FIXES cam_0/cam_1 (champ large) le retrouvent. Ne
        touche PAS la perception initiale (chemin teste) ; meme detecteur/estimateur.
        """
        rs = (self._provider.read_live() if not self.config.dry_run
              else self._provider.from_angles({j: 0.0 for j in ARM_JOINTS}))
        # warmup court (autoexposure) + capture
        for _ in range(2):
            mc.grab(robot_state=rs)
            time.sleep(0.05)
        frames = mc.grab(robot_state=rs)
        dets_by_cam = self._detector.detect_multi(frames)
        if self._label_mapping:
            for _ck, _dl in dets_by_cam.items():
                for _d in _dl:
                    if _d.label in self._label_mapping:
                        _d.label = self._label_mapping[_d.label]
        scene = self._estimator.build_scene(dets_by_cam, frames)
        return next((o for o in scene.objects
                     if o.label == self.config.target_label), None)

    # ----- main entry point -----------------------------------------------

    def run(self):
        """Execute un cycle complet de pick-and-place."""
        print("=" * 70)
        print(f" PICK-AND-PLACE — cible: '{self.config.target_label}', "
              f"detector: {self.config.detector_kind}")
        print("=" * 70)
        print()

        # ============================================================
        # 1. Connexion
        # ============================================================
        # IMPORTANT : initialiser controller AVANT le try, sinon le finally
        # referencera un nom non defini si MotorController() leve.
        controller = None
        try:
            controller = MotorController()
            if not self.config.dry_run:
                self._provider.connect_live(self.config.motor_port)
                controller.connect(self.config.motor_port)
                controller.enable_torque()
        except Exception as e:
            print(f"!! Connexion robot/moteur impossible : {e}")
            if not self.config.dry_run:
                print("   Bascule en mode --dry-run.")
                self.config.dry_run = True

        # BASELINE PINCE A VIDE : reference de la session pour interpreter le
        # couple. A vide les doigts TPU se compriment entre eux des ~10% ->
        # couple parasite ~200-230 meme sans objet. La logguer rend les
        # seuils interpretables (tenu = nettement au-dessus de la baseline).
        self._gripper_baseline = None
        if not self.config.dry_run and controller is not None and controller._bus is not None:
            try:
                base_pct = controller.read_gripper_pct()
                base_load = controller.read_gripper_load()
                self._gripper_baseline = (base_pct, base_load)
                print(f">> [baseline pince] au depart : position={base_pct:.1f}%, "
                      f"couple={base_load}  (reference 'a vide' de la session)")
            except Exception:
                pass

        # IMPORTANT : la MultiCamera DOIT rester ouverte pendant TOUT le
        # pipeline car Sprint 4 (closed_loop) a besoin de cam_2 APRES la
        # phase 1 d'execution. Sinon le 'with MultiCamera() as mc' se ferme
        # apres la perception initiale et le refinement crash avec
        # "MultiCamera n'est pas ouvert".
        try:
            mc = MultiCamera()
            mc.open()

            # ============================================================
            # 2. Perception (UNE seule capture initiale)
            # ============================================================
            print(">> Perception en cours...")
            rs = (self._provider.read_live() if not self.config.dry_run
                  else self._provider.from_angles({j: 0.0 for j in ARM_JOINTS}))
            # Warmup pour stabiliser autoexposure
            for _ in range(3):
                mc.grab(robot_state=rs)
                time.sleep(0.1)
            frames = mc.grab(robot_state=rs)
            dets_by_cam = self._detector.detect_multi(frames)
            # Si HF avec mapping : renomme les labels
            if self._label_mapping:
                for cam_key, dets in dets_by_cam.items():
                    for d in dets:
                        if d.label in self._label_mapping:
                            d.label = self._label_mapping[d.label]
            # === DEBUG : detections brutes par camera (TOUTES) ===
            print("   Detections brutes par camera :")
            for cam_key in ("cam_0", "cam_1", "cam_2"):
                if cam_key in dets_by_cam:
                    dets = dets_by_cam[cam_key]
                    if dets:
                        dets_sorted = sorted(dets, key=lambda d: -d.score)
                        print(f"     {cam_key} : {len(dets)} detections")
                        for d in dets_sorted:
                            bbox_size = ""
                            if d.bbox:
                                w = d.bbox[2] - d.bbox[0]
                                h = d.bbox[3] - d.bbox[1]
                                bbox_size = f"  bbox={int(w)}x{int(h)}px"
                            print(f"        {d.label:<22} s={d.score:.2f}"
                                  f"  center=({int(d.center_px[0])},{int(d.center_px[1])}){bbox_size}")
                    else:
                        print(f"     {cam_key} : 0 detection")
            scene = self._estimator.build_scene(dets_by_cam, frames)
            # === DEBUG : si scene vide alors qu'on avait des dets ===
            if not scene.objects and any(dets_by_cam.get(k) for k in dets_by_cam):
                print()
                print("   [DIAGNOSTIC] Detections 2D presentes mais aucun objet 3D :")
                rejections = getattr(self._estimator, "_last_rejections", {})
                if rejections:
                    for label, reason in rejections.items():
                        print(f"     {label:<22} : {reason}")
                from src.perception.pose_estimator import (
                    PoseEstimator, PoseEstimatorConfig)
                cfg_loose = PoseEstimatorConfig(max_reproj_error_px=200.0)
                est_no_filter = PoseEstimator(
                    config=cfg_loose,
                    specs_by_label=self._specs_meta,
                    load_scene_config=False,
                )
                scene_raw = est_no_filter.build_scene(dets_by_cam, frames)
                if scene_raw.objects:
                    print("   Avec seuils relaches (reproj 200 px, sans scene.json) :")
                    for o in scene_raw.objects:
                        p = o.position_base_m * 1000
                        err = o.meta.get("reproj_error_px", -1)
                        print(f"     {o.label:<22} pos=({p[0]:+6.1f},{p[1]:+6.1f},{p[2]:+6.1f}) mm  "
                              f"reproj_err={err:.1f} px")
            print(scene.pretty())
            print()

            # === SNAPSHOT PERCEPTION : sauve toujours (si save_snapshots), n'affiche
            # la fenetre live que si --display. Decouple save vs show (2026-06-23). ===
            if self.config.save_snapshots or self.config.display:
                self._show_perception_snapshot(frames, dets_by_cam, scene,
                                                title="Perception initiale (Sprint 2)")

            # ============================================================
            # 3. Trouve l'objet cible
            # ============================================================
            target = next((o for o in scene.objects if o.label == self.config.target_label), None)
            if target is None:
                print(f"!! Objet '{self.config.target_label}' NON DETECTE dans la scene. Annule.")
                # POURQUOI : la raison precise du rejet est DEJA calculee par le
                # PoseEstimator (stereo / reproj / workspace / zone d'exclusion).
                # On l'affichait seulement si la scene etait VIDE -> masquee quand
                # d'autres objets etaient fusionnes (cas du prisme drope alors que
                # 3 autres objets etaient la). On la sort systematiquement ici.
                rej = getattr(self._estimator, "_last_rejections", {}).get(
                    self.config.target_label)
                if rej:
                    print(f"   -> raison du rejet : {rej}")
                # Rappel des detections 2D brutes : souvent l'objet EST vu, mais les
                # meilleures bbox par camera (plus grande aire) ne se correspondent
                # pas en stereo (ex : tache noire du fond choisie a la place du prisme).
                seen = [f"{cam}:{n}" for cam, dl in dets_by_cam.items()
                        if (n := sum(1 for d in dl
                                     if d.label == self.config.target_label))]
                if seen:
                    print(f"   -> pourtant detecte en 2D dans {', '.join(seen)} "
                          f"(non fusionne en 3D)")
                return
            print(f">> Cible '{target.label}' a position "
                  f"({target.position_base_m[0]*1000:+.1f}, "
                  f"{target.position_base_m[1]*1000:+.1f}, "
                  f"{target.position_base_m[2]*1000:+.1f}) mm")
            # Geometrie enrichie (P3) : hauteur fiabilisee + classe + yaw base
            geo_bits = []
            if target.bbox_3d_m is not None:
                geo_bits.append(
                    f"empreinte ~{target.bbox_3d_m[0]*1000:.0f}x"
                    f"{target.bbox_3d_m[1]*1000:.0f}mm, "
                    f"hauteur={target.bbox_3d_m[2]*1000:.0f}mm "
                    f"({target.meta.get('height_method', '?')})")
            if target.meta.get("pose_class"):
                geo_bits.append(f"classe={target.meta['pose_class']}")
            if "pose_class" in target.meta:
                yb = target.meta.get("yaw_base_rad")
                geo_bits.append("yaw_base=libre" if yb is None
                                else f"yaw_base={np.degrees(yb):+.0f}deg")
            if geo_bits:
                print(f"   geometrie : {', '.join(geo_bits)}")
            # DIAGNOSTIC orientation : angle vu PAR CHAQUE camera. Si cam_0 et
            # cam_1 s'accordent ET correspondent a l'objet reel -> perception OK.
            # Si elles collent toujours a ~0deg quelle que soit la pose de l'objet
            # -> detection a ameliorer. (A comparer a l'orientation REELLE que tu vois.)
            yc0, yc1 = target.meta.get("yaw_cam0_deg"), target.meta.get("yaw_cam1_deg")
            if yc0 is not None or yc1 is not None:
                print(f"   [diag orientation] angle objet vu par cam_0={yc0}deg, "
                      f"cam_1={yc1}deg  (compare a l'angle REEL que tu observes)")
            print()

            # ============================================================
            # 4-5. Grasp planning + IK
            #   - mode adaptatif : genere des candidats d'angle d'attaque sur le
            #     balayage sagittal, filtre par ATTEIGNABILITE IK, garde le 1er
            #     faisable (top-down d'abord, frontal/diagonale en repli).
            #   - mode top-down : comportement historique inchange.
            # Toute la logique de selection + IK est dans _plan_and_solve_grasp.
            # ============================================================
            print(">> Cinematique inverse...")
            current_state = (self._provider.read_live() if not self.config.dry_run
                             else self._provider.from_angles({j: 0.0 for j in ARM_JOINTS}))
            q_current = current_state.joint_angles_rad
            # MEMORISE la pose de depart : c'est notre future "home" si
            # home_from_session_start=True. Le robot reviendra exactement la.
            q_session_start = dict(q_current)

            grasp_pose, r_app, r_grp, r_ret = self._plan_and_solve_grasp(
                target, q_current)
            if grasp_pose is None:
                print("!! Aucune prise faisable (objet trop haut / hors d'atteinte ?). "
                      "Annule.")
                return
            # Instrumentation campagne : residu IK de la prise RETENUE = "erreur de
            # pose atteinte" (planifiee) = ecart entre la pose de prise demandee et
            # ce que le bras 5-DDL peut realiser. INTERNE (pas de regle, pas de Y).
            # Lue par experiment_campaign pour le CSV ; None si jamais atteint.
            self._grasp_pose_error_mm = float(r_grp.translation_err_mm)
            self._grasp_pose_error_rot_deg = float(r_grp.rotation_err_deg)
            self._log_wrist("IK initiale", q_current, r_grp)
            # IK pour la pose drop_above et drop_release : on utilise l'IK
            # specialise self._ik_drop qui a un poids rotation reduit (0.01).
            # Cela permet de privilegier la POSITION (precision au mm pour
            # viser la boite) au prix d'une orientation pince approximative
            # (acceptable : la pince ouvre puis le cube tombe -- l'incli-
            # naison de ~10deg n'empeche pas le drop). Avec l'IK standard,
            # la pose drop ratait la cible de 5-13 cm (sous-actuation 5/6 DDL).
            from src.planning.grasp import _rotation_top_down, _se3
            # DEPOSE : yaw LIBRE -> on choisit l'orientation qui donne le
            # poignet le plus NATUREL (la pince ouvre pour lacher, le yaw n'a
            # aucune importance). Evite la contorsion / tete a l'envers a la
            # boite (la pose yaw=0 forcait un poignet retourne pour la boite
            # lointaine ; verifie au runtime : yaw libre -> wrist_roll ~6deg
            # au lieu de 156deg). drop_release reprend le meme yaw (transition
            # douce).
            r_drop_above, _yaw_drop = self._ik_drop.solve_topdown_free_yaw(
                self.drop_above, q_init=r_ret.joint_angles_rad)
            T_drop_release = _se3(_rotation_top_down(_yaw_drop), self.drop_release)
            r_drop_release = self._ik_drop.solve(
                T_drop_release, q_init=r_drop_above.joint_angles_rad)

            for label, r in [("approach", r_app), ("grasp", r_grp),
                             ("retract", r_ret), ("drop_above", r_drop_above),
                             ("drop_release", r_drop_release)]:
                tag = "OK" if r.converged else "approx"
                print(f"   {label:<14} {tag:<6} "
                      f"trans={r.translation_err_mm:5.1f}mm rot={r.rotation_err_deg:5.1f}deg")
            print()

            # ============================================================
            # 6. Generation trajectoire + execution
            # Si closed_loop : on EXECUTE jusqu'a approach, puis on raffine
            # avec cam_2, puis on EXECUTE la suite avec correction.
            # Sinon : trajectoire complete d'un coup.
            # ============================================================
            # Helper : callback display live qui rafraichit cv2.imshow
            # pendant l'execution moteur (toutes les ~30 frames de traj).
            #
            # ARCHITECTURE THREADING (fix saccades) :
            # Probleme avant : la detection HF (~3-5s sur M4) etait appelee
            # depuis le callback execute_trajectory -> bloquait l'envoi des
            # commandes moteur pendant ces 3-5s -> bras s'arrete, puis envoi
            # en rafale pour rattraper le timing -> mouvements brusques.
            #
            # Solution : detection dans un THREAD WORKER separe.
            #   - Main loop (callback) : grab frames, push dans une queue,
            #     affiche les dernieres detections disponibles, retourne
            #     immediatement (< 50 ms). La trajectoire continue fluide.
            #   - Worker thread : prend les frames, fait la detection (3-5s),
            #     met a jour le cache des detections.
            # Les detections affichees peuvent etre legerement obsoletes
            # (delai de quelques secondes en HF) mais le display reste fluide.
            def make_live_callback(initial_dets=None):
                if not self.config.display:
                    return None
                import cv2
                import threading
                from queue import Queue, Empty, Full

                # Queue de taille 1 : on garde seulement les dernieres frames
                # (les anciennes sont perdues, c'est OK)
                frame_queue: Queue = Queue(maxsize=1)
                # Cache partage des dernieres detections (avec lock pour
                # synchronisation main <-> worker). On PRE-POPULE avec les
                # detections de la perception initiale -> les bbox vertes
                # sont visibles immediatement au lieu d'attendre le premier
                # cycle worker HF (~3-5 sec).
                dets_cache = {"value": initial_dets or {}}
                dets_lock = threading.Lock()
                stop_event = threading.Event()

                def detector_worker():
                    """Boucle worker : detecte sur les dernieres frames dispo."""
                    while not stop_event.is_set():
                        try:
                            frames = frame_queue.get(timeout=0.2)
                        except Empty:
                            continue
                        try:
                            dets = self._detector.detect_multi(frames)
                            if self._label_mapping:
                                for cam_dets in dets.values():
                                    for d in cam_dets:
                                        if d.label in self._label_mapping:
                                            d.label = self._label_mapping[d.label]
                            with dets_lock:
                                dets_cache["value"] = dets
                        except Exception as e:
                            print(f"[live worker] detection error: {e}")

                # Demarre le worker (daemon=True : se ferme avec le programme)
                worker = threading.Thread(target=detector_worker, daemon=True)
                worker.start()
                # Memorise pour pouvoir l'arreter proprement dans finally
                self._live_stop_event = stop_event

                _grab_stride = max(1, int(self.config.display_grab_stride))

                def on_step(i, trajectory):
                    # THROTTLE USB (Maxence 2026-06-23) : capturer les 3 cams 1080p a
                    # CHAQUE pas saturait le bus -> cam_2 decrochait en plein run.
                    # On ne rafraichit l'affichage que tous les _grab_stride pas (le
                    # dernier pas est toujours affiche). waitKey garde la fenetre
                    # reactive. Aucun effet sur la prise (display only).
                    if (i % _grab_stride) != 0 and i != (len(trajectory) - 1):
                        try:
                            cv2.waitKey(1)
                        except Exception:
                            pass
                        return
                    try:
                        rs = self._provider.read_live()
                    except Exception:
                        return
                    frames = mc.grab(robot_state=rs)
                    # Pousse les frames au worker (remplace si encore pleine
                    # = le worker n'a pas fini la precedente, on saute)
                    try:
                        frame_queue.put_nowait(frames)
                    except Full:
                        pass  # worker occupe, on ignore ce frame pour la detection
                    # Display avec les dernieres detections (potentiellement
                    # obsoletes de quelques sec en HF, instantanees en HSV)
                    with dets_lock:
                        cur_dets = dict(dets_cache["value"])
                    import numpy as np
                    tiles = [self._annotate_frame(frames.get(k),
                                                  cur_dets.get(k, []),
                                                  None)
                             for k in CAM_DISPLAY_ORDER]
                    mosaic = np.hstack(tiles)
                    cv2.putText(mosaic, f"LIVE exec frame {i}/{len(trajectory)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Pipeline perception", mosaic)
                    cv2.waitKey(1)
                return on_step

            # Determine la pose home finale : soit la pose de depart de la
            # session (defaut, le robot revient ou Maxence l'avait place), soit
            # la pose fixe configuree dans home_angles_rad.
            if self.config.home_from_session_start:
                q_home_final = q_session_start
                home_origin = "pose de depart session"
            else:
                q_home_final = self.config.home_angles_rad
                home_origin = "pose fixe (home_angles_rad)"

            # Cree le callback live UNE SEULE FOIS pour eviter de lancer
            # plusieurs worker threads de detection. Le meme callback sera
            # passe a phase 1 et phase 2. Pre-popule le cache avec les
            # detections initiales pour que les bbox vertes soient visibles
            # immediatement (sans attendre 3-5 sec du premier cycle HF).
            live_callback = make_live_callback(initial_dets=dets_by_cam)

            if self.config.closed_loop and not self.config.dry_run:
                # --- OBSERVATION cam_2 = REFERENCE 12cm AU-DESSUS DE L'OBJET (Maxence) ---
                # cam2_observe_height_m (12cm, reglable via --cam2-observe-height) = la
                # hauteur d'observation/detection AU-DESSUS DE LA POSITION STEREO de
                # l'objet. On monte la pince a cette hauteur AVANT la capture cam_2 : a
                # 12cm l'objet est dans le champ et DEGAGE des doigts (qui occupent le
                # bas du cadre) -> blob propre, centroide non biaise. Puis raffinement,
                # puis descente DIRECTE sur l'objet (qui passe par la pose d'approche,
                # simple point de passage, PAS un arret de detection). On vise donc
                # Z = Z_objet + 12cm, DIRECTEMENT (aucun "8cm + 4cm").
                r_phase1_target = r_app
                observe_height_m = float(self.config.cam2_observe_height_m)
                object_z_m = float(grasp_pose.meta.get(
                    "object_center_z_m", grasp_pose.T_base_gripper_grasp[2, 3]))
                if observe_height_m > 1e-4:
                    T_observe = grasp_pose.T_base_gripper_approach.copy()  # XY + orient. approche
                    T_observe[2, 3] = object_z_m + observe_height_m        # 12cm au-dessus objet
                    r_obs = self._ik.solve(T_observe, q_init=r_app.joint_angles_rad)
                    # Gate = erreur de TRANSLATION sous le seuil approche (15mm), PAS
                    # r_obs.converged (qui exige ~1mm combine trans+rot -> jamais
                    # atteint en 5-DDL -> recul toujours rejete a tort). La geometrie
                    # ray-plan de cam_2 utilise la pose REELLE (FK), pas la cible IK.
                    if r_obs.translation_err_mm <= self.config.ik_tol_approach_mm:
                        r_phase1_target = r_obs
                        print(f"   [cam_2] observation/detection a "
                              f"{observe_height_m*1000:.0f}mm au-dessus de l'objet "
                              f"(Z pince={T_observe[2,3]*1000:.0f}mm, "
                              f"IK trans={r_obs.translation_err_mm:.1f}mm) "
                              f"pour eloigner l'objet des doigts dans l'image")
                    else:
                        print(f"   [cam_2] observation a "
                              f"{observe_height_m*1000:.0f}mm au-dessus de l'objet "
                              f"hors portee (IK trans={r_obs.translation_err_mm:.1f}mm > "
                              f"{self.config.ik_tol_approach_mm:.0f}mm) "
                              f"-> capture a la pose d'approche")
                # --- Phase 1 : trajectoire courant -> approach (ou observation) ---
                print(">> Phase 1 : courant -> approach (boucle fermee Sprint 4)")
                traj_phase1 = self._build_phase1_trajectory(
                    q_current, r_phase1_target,
                    gripper_open_pct=grasp_pose.gripper_open_pct)
                print(f"   {len(traj_phase1)} points, duree {traj_phase1.duration_s:.1f}s")
                controller.execute_trajectory(traj_phase1, verbose=True,
                                              on_step=live_callback)
                print()

                # --- Raffinement cam_2 ---
                print(">> Raffinement cam_2 (eye-in-hand)...")
                from src.control.closed_loop import (
                    apply_correction_to_grasp_pose,
                    refine_grasp_with_cam2,
                )
                # Re-lit l'etat robot (on est arrive a approach, qui peut differer
                # un peu de la cible IK a cause de la precision des moteurs).
                # SETTLE d'abord : le bras vient de s'arreter -> on le laisse se
                # stabiliser pour que la capture cam_2 ne soit pas prise en plein
                # mouvement (anti biais vertical, cf cam2_settle_s).
                time.sleep(self.config.cam2_settle_s)
                rs_at_approach = self._provider.read_live()
                # IMPORTANT : la MultiCamera est encore ouverte (cf 'with mc' plus haut)
                refinement = refine_grasp_with_cam2(
                    target_label=self.config.target_label,
                    detector=self._detector,
                    multi_camera=mc,
                    robot_state=rs_at_approach,
                    debug_save_dir=((REPO / "outputs" / "perception")
                                    if self.config.save_snapshots else None),
                    # IMPORTANT : on passe la hauteur ATTENDUE de l'objet
                    # (= Z de la pose grasp triangulée) pour que la formule
                    # de conversion pixel->m utilise la VRAIE distance
                    # cam_2 -> objet, pas une valeur en dur.
                    target_z_base_m=float(grasp_pose.meta.get(
                        "object_center_z_m", grasp_pose.T_base_gripper_grasp[2, 3])),
                    grasp_xy_base_m=(float(grasp_pose.T_base_gripper_grasp[0, 3]),
                                     float(grasp_pose.T_base_gripper_grasp[1, 3])),
                    label_mapping=self._label_mapping,
                )
                print(f"   {refinement.message}")
                # cam_2 corrige la POSITION par projection ray-plan, qui suppose une
                # vue ~top-down. Sur une prise INCLINEE (cote/diagonale) cette
                # projection est FAUSSE : le centroide du blob d'un objet DEBOUT se
                # projette systematiquement a cote du vrai centre -> la correction
                # DIVERGE (Δv ~ -200px constant d'une iteration a l'autre, X qui part
                # a +13mm au retry) et pousse la prise "devant et bas" -> ferme a vide
                # (essais cylindre 2026-06-20). On NE corrige donc PAS la position en
                # incline : la stereo fait foi (X fiable, Z = table + H/2). cam_2 ne
                # raffine que les prises TOP-DOWN, ou sa geometrie est valide.
                # cam_2 valide jusqu'a cam2_max_pitch_deg (0 et 45 OK ; 90 trop
                # horizontal -> rayon rase le plan). 45 RE-ACTIVE a la demande de
                # Maxence (a surveiller : le 45 oblique peut biaiser un peu).
                is_tilted = (abs(np.degrees(grasp_pose.meta.get("pitch_rad", 0.0)))
                             >= self.config.cam2_max_pitch_deg)
                if is_tilted:
                    print(f"   (prise theta="
                          f"{np.degrees(grasp_pose.meta.get('pitch_rad', 0.0)):+.0f}deg "
                          f">= {self.config.cam2_max_pitch_deg:.0f}deg -> correction cam_2 "
                          f"IGNOREE : axe optique trop horizontal, ray-plan invalide. "
                          f"On garde la pose stereo.)")
                elif refinement.detection is None:
                    print("   [WARN] cam_2 sans detection exploitable "
                          "-> correction NON appliquee (pose stereo seule)")
                elif refinement.area_frac < self.config.cam2_min_blob_frac:
                    print(f"   [WARN] blob cam_2 trop petit "
                          f"({100*refinement.area_frac:.2f}% < "
                          f"{100*self.config.cam2_min_blob_frac:.2f}% cadre) "
                          "-> correction NON appliquee (centroide peu fiable)")
                elif refinement.delta_norm_mm > self.config.closed_loop_max_correction_mm:
                    print(f"   [WARN] correction {refinement.delta_norm_mm:.1f} mm > seuil "
                          f"{self.config.closed_loop_max_correction_mm} mm. "
                          "Suspect, on NE corrige PAS (peut etre fausse detection).")
                else:
                    apply_correction_to_grasp_pose(grasp_pose, refinement.delta_base_m)
                    # Instrumentation campagne : amplitude de la correction cam_2
                    # appliquee = ecart entre l'estimation stereo et la position
                    # raffinee par cam_2 (precision PERCEPTION, Y-free). None si pas
                    # de correction (cam_2 absente / blob trop petit / hors seuil).
                    self._cam2_correction_mm = float(refinement.delta_norm_mm)
                    print(f"   Correction position appliquee "
                          f"(norme {refinement.delta_norm_mm:.1f} mm, "
                          f"blob {100*refinement.area_frac:.1f}% cadre)")
                    # CORRECTION D'ORIENTATION par cam_2 (2026-06-13, TOP-DOWN seul).
                    # cam_2 est proche et quasi au-dessus de l'objet -> son grand axe
                    # est plus FIABLE que la stereo oblique cam_0/cam_1. Si cam_2 voit
                    # un objet CLAIREMENT allonge, on REORIENTE la prise sur son petit
                    # axe. ⚠️ reorient_grasp_pose suppose le TOP-DOWN (il pose une
                    # rotation verticale) : l'appliquer a une prise 45/90 l'APLATIRAIT
                    # en top-down. Donc le re-alignement ne s'applique qu'au top-down ;
                    # la correction de POSITION (faite plus haut) reste valable a tous
                    # les angles. Re-alignement tilt-aware = TODO (a decider avec Maxence).
                    reoriented = False
                    _wants_reorient = (
                        self.config.cam2_reorient
                        and refinement.yaw_base_cam2 is not None
                        and refinement.elong_cam2 >= self.config.cam2_reorient_min_elong)
                    _is_top_down = abs(np.degrees(grasp_pose.meta.get("pitch_rad", 0.0))) < 1e-3
                    if _wants_reorient and _is_top_down:
                        from src.planning.grasp import reorient_grasp_pose
                        old_yaw = np.degrees(grasp_pose.meta.get("yaw_rad", 0.0))
                        reorient_grasp_pose(grasp_pose, refinement.yaw_base_cam2)
                        applied_yaw = np.degrees(grasp_pose.meta.get("yaw_rad", 0.0))
                        raw_yaw = np.degrees(refinement.yaw_base_cam2)
                        print(f"   cam_2 REORIENTE la prise : yaw {old_yaw:+.0f}deg "
                              f"-> {applied_yaw:+.0f}deg (axe cam_2 {raw_yaw:+.0f}deg replie "
                              f"sur l'equivalent +-180 le plus proche -> poignet bouge de "
                              f"{abs(applied_yaw-old_yaw):.0f}deg, PAS de demi-tour ; "
                              f"objet allonge x{refinement.elong_cam2:.1f})")
                        reoriented = True
                    elif _wants_reorient and not _is_top_down:
                        # PRISE INCLINEE (45/90) : on REPLANIFIE au MEME pitch avec le
                        # grand axe vu par cam_2 (reutilise _build_pose -> machoires en
                        # travers du petit cote, geometrie identique au planning). La
                        # correction de POSITION cam_2 (deja appliquee a grasp_pose
                        # ci-dessus) est RE-appliquee a la pose replanifiee.
                        new_gp = None
                        if hasattr(self._grasp_strategy, "replan_oriented"):
                            pitch_deg = np.degrees(grasp_pose.meta.get("pitch_rad", 0.0))
                            new_gp = self._grasp_strategy.replan_oriented(
                                target, pitch_deg, refinement.yaw_base_cam2)
                        if new_gp is not None:
                            apply_correction_to_grasp_pose(new_gp, refinement.delta_base_m)
                            grasp_pose = new_gp
                            print(f"   cam_2 REORIENTE la prise inclinee "
                                  f"(theta={np.degrees(grasp_pose.meta.get('pitch_rad', 0.0)):+.0f}deg) "
                                  f"sur le grand axe cam_2 "
                                  f"({np.degrees(refinement.yaw_base_cam2):+.0f}deg, "
                                  f"allonge x{refinement.elong_cam2:.1f}) -> machoires en "
                                  f"travers du petit cote, pitch inchange")
                            reoriented = True
                        else:
                            print(f"   (cam_2 voit l'objet allonge x{refinement.elong_cam2:.1f} "
                                  f"mais replan incline infaisable -> orientation conservee)")
                    # Re-IK. Si on a reoriente : NON verrouille (l'axe a change ->
                    # re-choisir la meilleure des 2 orientations 180 pour la
                    # continuite). Sinon verrouille (anti bascule).
                    r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                        grasp_pose, q_init=rs_at_approach.joint_angles_rad,
                        lock_orientation=(not reoriented))
                    print(f"   IK re-resolue avec poses corrigees.")
                    self._log_wrist("re-solve apres correction #1",
                                    rs_at_approach.joint_angles_rad, r_grp)
                print()

                # === Refinement #2 OPTIONNEL (closed_loop_two_stage) ===
                # Par defaut DESACTIVE (2026-06-13). La mini-descente + recorrection
                # a 4cm provoquait le "descend un peu, remonte se repositionner,
                # redescend" qui frole l'objet (signale par Maxence a repetition),
                # et le 2e refinement etait bruite (biais Y). On fait desormais UN
                # SEUL refinement (a approach, hauteur sure) ; la Phase 2 (attempt#1,
                # include_approach=True) repositionne lateralement A 8cm puis descend
                # TOUT DROIT sur l'objet -> "analyse avant de descendre".
                if not self.config.closed_loop_two_stage:
                    rs_at_intermediate = rs_at_approach
                    print(f">> Un seul refinement (detection a "
                          f"{self.config.cam2_observe_height_m*1000:.0f}mm = hauteur "
                          f"d'observation) -> repositionnement lateral en hauteur puis "
                          f"DESCENTE DROITE (pas de mini-descente basse altitude)")
                    print()
                else:
                    # --- A4 : Mini-descente + Refinement #2 (cam_2 plus pres) ---
                    print(">> A4 : Mini-descente + Refinement #2")
                    T_intermediate = grasp_pose.T_base_gripper_grasp.copy()
                    T_intermediate[2, 3] += 0.04  # 4cm au-dessus du grasp
                    r_intermediate = self._ik.solve(
                        T_intermediate, q_init=rs_at_approach.joint_angles_rad)
                    self._log_wrist("mini-descente",
                                    rs_at_approach.joint_angles_rad, r_intermediate)
                    mini_traj = quintic_trajectory(
                        rs_at_approach.joint_angles_rad,
                        r_intermediate.joint_angles_rad,
                        duration_s=estimate_duration_safe(
                            rs_at_approach.joint_angles_rad,
                            r_intermediate.joint_angles_rad,
                            max_velocity_rad_s=self.config.max_velocity_rad_s),
                        gripper_start=self.config.home_gripper_pct,
                        gripper_end=grasp_pose.gripper_open_pct,
                    )
                    print(f"   Mini-descente : {len(mini_traj)} points, "
                          f"duree {mini_traj.duration_s:.1f}s")
                    controller.execute_trajectory(mini_traj, verbose=False,
                                                  on_step=live_callback)
                    time.sleep(self.config.cam2_settle_s)
                    rs_at_intermediate = self._provider.read_live()
                    refinement2 = refine_grasp_with_cam2(
                        target_label=self.config.target_label,
                        detector=self._detector,
                        multi_camera=mc,
                        robot_state=rs_at_intermediate,
                        target_z_base_m=float(grasp_pose.meta.get(
                        "object_center_z_m", grasp_pose.T_base_gripper_grasp[2, 3])),
                        grasp_xy_base_m=(float(grasp_pose.T_base_gripper_grasp[0, 3]),
                                         float(grasp_pose.T_base_gripper_grasp[1, 3])),
                        label_mapping=self._label_mapping,
                        debug_save_dir=((REPO / "outputs" / "perception")
                                        if self.config.save_snapshots else None),
                    )
                    print(f"   Refinement #2 : {refinement2.message}")
                    # GATING par TAILLE ABSOLUE du blob (Maxence 2026-06-20),
                    # coherent avec les refinements #1 / retry (plus de `score`).
                    r2_cap_mm = self.config.closed_loop_max_correction_mm
                    # cam_2 ne corrige qu'en TOP-DOWN (ray-plan valide) ; incline -> stereo.
                    _is_tilted2 = (abs(np.degrees(grasp_pose.meta.get("pitch_rad", 0.0)))
                                   >= self.config.cam2_max_pitch_deg)
                    if (not _is_tilted2 and refinement2.detection is not None
                            and refinement2.area_frac >= self.config.cam2_min_blob_frac
                            and refinement2.delta_norm_mm < r2_cap_mm):
                        apply_correction_to_grasp_pose(grasp_pose,
                                                        refinement2.delta_base_m)
                        print(f"   Correction #2 appliquee "
                              f"(norme {refinement2.delta_norm_mm:.1f} mm, "
                              f"blob {100*refinement2.area_frac:.1f}% cadre)")
                        r_grp = self._ik.solve(
                            grasp_pose.T_base_gripper_grasp,
                            q_init=r_intermediate.joint_angles_rad)
                        r_ret = self._ik.solve(
                            grasp_pose.T_base_gripper_retract,
                            q_init=r_grp.joint_angles_rad)
                        self._log_wrist("re-solve apres correction #2",
                                        r_intermediate.joint_angles_rad, r_grp)
                    elif refinement2.area_frac < self.config.cam2_min_blob_frac:
                        print(f"   [INFO] blob cam_2 #2 trop petit "
                              f"({100*refinement2.area_frac:.2f}% < "
                              f"{100*self.config.cam2_min_blob_frac:.2f}% cadre) "
                              f"-> correction #2 ignoree")
                    elif refinement2.delta_norm_mm >= r2_cap_mm:
                        print(f"   [WARN] correction #2 = {refinement2.delta_norm_mm:.1f}mm "
                              f"> plafond {r2_cap_mm:.0f}mm (probable fausse detection), ignoree")
                    else:
                        print(f"   [INFO] objet stable depuis refinement #1")
                    print()

                # --- Phase 2 : BOUCLE de tentatives de saisie avec FEEDBACK ---
                # Chaque tentative :
                #   1. DESCENTE (pince ouverte) vers la pose grasp.
                #   2. FERMETURE : asservie au couple (P5, stop au contact +
                #      pression de maintien) ou statique (mode historique).
                #   3. CHECK FERMETURE : couple >= seuil (ou marge position).
                #   4. P1' VERIF POST-LEVEE : remontee a retract puis RE-LECTURE
                #      position+couple. Le couple a la fermeture peut mentir
                #      (effleurement du sommet / morsure de bord / appui table
                #      -> faux positifs a 280-500 observes) ; apres 10cm de
                #      levee, un objet perdu retombe a la baseline -> attrape.
                #   5. Si tenu -> depose ; sinon retry (refinement + redescente).
                print(f">> Phase 2 : tentative(s) de saisie avec feedback pince + depot + retour ({home_origin})")
                q_at_intermediate = rs_at_intermediate.joint_angles_rad

                # === OFFSET LATERAL DE PRISE -- REPERE IMAGE cam_2 (Maxence 2026-06-23) ===
                # But : degager la machoire FIXE (qui ne s'ecarte PAS a l'ouverture)
                # pour qu'elle arrive A FLEUR DE L'ARETE GAUCHE de l'objet TEL QU'IL
                # APPARAIT DANS LE SNAPSHOT cam_2. Le decalage est donc defini DANS
                # L'IMAGE cam_2 ("vers la gauche de l'image") : constant du point de
                # vue de la camera, mais correspondant a une direction base
                # DIFFERENTE selon l'orientation de la tete (donc reparti autrement
                # sur X/Y de la position calculee). C'est EXACTEMENT la demande de
                # Maxence -- et la raison pour laquelle un offset en repere
                # pince/base donnait des comportements incoherents selon l'objet.
                # _cam2_image_left_base = -X camera projete sur l'horizontale, calcule
                # depuis l'orientation du GRASP FINAL (apres reorientation cam_2) pour
                # que le doigt fixe se cale au bon cote QUAND les machoires ferment --
                # PAS depuis la pose d'observation (sinon decale en bord). +lat_mm = gauche
                # image ; valeur NEGATIVE = inverse le cote (si mauvais cote au robot).
                # APPLIQUE APRES cam_2 (sinon le recentrage l'annule) ; retry idem.
                lat_mm = self._effective_lateral_offset_mm(grasp_pose)
                fwd_mm = float(self.config.grasp_forward_offset_mm)
                if lat_mm != 0.0 or fwd_mm != 0.0:
                    offset_vec = (self._cam2_image_left_base(grasp_pose.T_base_gripper_grasp)
                                  * (lat_mm / 1000.0)
                                  + self._cam2_image_down_base(grasp_pose.T_base_gripper_grasp)
                                  * (fwd_mm / 1000.0))
                    for _attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                                  "T_base_gripper_retract"):
                        getattr(grasp_pose, _attr)[:3, 3] += offset_vec
                    r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                        grasp_pose, q_init=q_at_intermediate, lock_orientation=True)
                    _jw = grasp_pose.meta.get("jaw_width_mm")
                    _mode = (f"AUTO=½×{_jw:.0f}+{self.config.grasp_lateral_offset_margin_mm:.0f}"
                             if (self.config.grasp_lateral_offset_auto and _jw)
                             else "manuel")
                    print(f"   Offset prise : gauche-image {lat_mm:+.0f}mm [{_mode}] "
                          f"+ profondeur(bas-image) {fwd_mm:+.0f}mm "
                          f"-> base Δ=({offset_vec[0]*1000:+.0f},"
                          f"{offset_vec[1]*1000:+.0f},{offset_vec[2]*1000:+.0f})mm "
                          f"(orientation inchangee)")

                # Log structure pour campagne experimentale (P4)
                grasp_attempts_log = []
                grasp_succeeded = False
                q_start_attempt = q_at_intermediate  # depart de la 1ere descente
                # Consigne pince TENUE pendant le transport. En fermeture
                # asservie, c'est la consigne figee au contact (relative a la
                # largeur de l'objet), pas le plancher grip_close_pct.
                hold_cmd = self.config.grip_close_pct
                use_servo_close = bool(self.config.grasp_close_servo)
                load_thr = self.config.grasp_load_threshold

                attempt = 0
                while attempt <= self.config.max_grasp_retries:
                    attempt += 1
                    print()
                    print(f">> --- Tentative #{attempt}/{self.config.max_grasp_retries + 1} ---")

                    # 1. DESCENTE vers grasp (fermeture incluse seulement en
                    # mode statique ; en mode asservi elle vient juste apres).
                    traj_grasp = self._build_grasp_attempt_traj(
                        q_start_attempt,
                        r_app.joint_angles_rad,
                        r_grp.joint_angles_rad,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                        grip_close_pct=self.config.grip_close_pct,
                        include_close=not use_servo_close,
                        # include_approach=True : passe par la pose approach
                        # corrigee (lateral A 8cm, hauteur sure) PUIS descend tout
                        # droit sur l'objet -> repositionnement en hauteur, descente
                        # finale verticale (pas de reposition a basse altitude).
                        # En mode two_stage attempt#1 on part de la mini-descente
                        # (deja sous approach) -> descente directe (anti remontee).
                        include_approach=(attempt > 1
                                          or not self.config.closed_loop_two_stage),
                    )
                    label_traj = "Descente" if use_servo_close else "Descente + fermeture"
                    print(f"   {label_traj} : {len(traj_grasp)} points, "
                          f"duree {traj_grasp.duration_s:.1f}s")
                    controller.execute_trajectory(traj_grasp, verbose=True,
                                                  on_step=live_callback)

                    # SNAPSHOT VERITE-TERRAIN (Maxence 2026-06-23) : cam_2 AU MOMENT
                    # de la prise (basse altitude, machoires OUVERTES juste AVANT la
                    # fermeture). Montre OU les machoires finissent par rapport a
                    # l'objet -> on VOIT si l'objet est bien entre elles ou a cote,
                    # et dans quelle direction IMAGE c'est decale (au lieu de
                    # deviner). C'est le diagnostic qui tranche XY-image vs reste.
                    if self.config.save_snapshots:
                        from src.control.closed_loop import capture_cam2_snapshot
                        time.sleep(self.config.cam2_settle_s)
                        capture_cam2_snapshot(
                            target_label=self.config.target_label,
                            detector=self._detector,
                            multi_camera=mc,
                            robot_state=self._provider.read_live(),
                            out_dir=(REPO / "outputs" / "perception"),
                            label_mapping=self._label_mapping,
                            tag=f"PRISE-attempt{attempt}",
                        )

                    # 2. FERMETURE asservie au couple (P5)
                    close_info = None
                    hold_cmd = self.config.grip_close_pct
                    if use_servo_close:
                        close_info = controller.close_gripper_with_feedback(
                            start_pct=grasp_pose.gripper_open_pct,
                            floor_pct=self.config.grip_close_pct,
                            load_stop=load_thr,
                            squeeze_pct=self.config.grasp_squeeze_pct,
                        )
                        hold_cmd = close_info["stop_cmd_pct"]
                        if close_info["stopped_on_contact"]:
                            print(f"   [fermeture asservie] CONTACT a "
                                  f"{close_info['contact_pct']:.1f}% (~largeur objet), "
                                  f"maintien a {hold_cmd:.1f}% "
                                  f"(serrage +{self.config.grasp_squeeze_pct:.0f}%), "
                                  f"couple={close_info['final_load']}")
                        else:
                            print(f"   [fermeture asservie] aucun contact detecte, "
                                  f"plancher {self.config.grip_close_pct:.0f}% atteint "
                                  f"(couple={close_info['final_load']})")

                    # 3. CHECK FERMETURE.
                    # Mode asservi (defaut) : le signal FIABLE est BINAIRE =
                    # "stopped_on_contact" (machoires bloquees a la largeur de
                    # l'objet, > plancher). On N'UTILISE PLUS le couple statique
                    # compare a 300 : sur pince TPU compliante le couple tenu
                    # (~80) est tres en dessous du transitoire de contact (~300),
                    # donc un seuil unique declarait RATE des prises correctes
                    # (essais 3,4,5,6,8,11 du 2026-06-12). load_thr ne sert plus
                    # qu'a la DETECTION de contact dans la rampe (en OR du stall).
                    # Mode statique/legacy : ancienne marge couple/position.
                    time.sleep(self.config.grasp_settle_pause_s)
                    gripper_now = controller.read_gripper_pct()
                    gripper_load = controller.read_gripper_load()
                    margin = gripper_now - self.config.grip_close_pct
                    contact_ref = None
                    if use_servo_close and close_info is not None:
                        close_ok = bool(close_info["stopped_on_contact"])
                        contact_ref = close_info.get("contact_pct")
                    elif load_thr is not None and gripper_load >= 0:
                        close_ok = gripper_load >= load_thr
                    else:
                        close_ok = margin > self.config.grasp_success_threshold_pct
                    tag = "SAISIE OK" if close_ok else "SAISIE RATEE"
                    load_txt = f", couple={gripper_load}" if gripper_load >= 0 else ""
                    via = (close_info or {}).get("contact_via")
                    via_txt = f", contact={via}" if via else ""
                    print(f"   [check pince] consigne={hold_cmd:.1f}%, "
                          f"reel={gripper_now:.1f}%{load_txt}{via_txt}  -->  {tag}")

                    entry = {
                        "attempt": attempt,
                        "gripper_pct": gripper_now,
                        "gripper_load": gripper_load,
                        "consigne_pct": self.config.grip_close_pct,
                        "hold_cmd_pct": hold_cmd,
                        "marge_pct": margin,
                        "seuil_pct": self.config.grasp_success_threshold_pct,
                        "load_seuil": load_thr,
                        "servo_contact": (close_info or {}).get("stopped_on_contact"),
                        "contact_pct": (close_info or {}).get("contact_pct"),
                        "close_ok": close_ok,
                        "lift_pct": None,
                        "lift_load": None,
                        "held_after_lift": None,
                        "success": False,
                    }
                    grasp_attempts_log.append(entry)

                    failed_stage = None  # "close" | "lift" | None
                    if close_ok and self.config.lift_verify:
                        # 4. P1' : VERIF POST-LEVEE — remonte a retract, pince
                        # tenue a hold_cmd, puis re-lecture.
                        rs_g = self._provider.read_live()
                        traj_lift_v = quintic_trajectory(
                            rs_g.joint_angles_rad,
                            r_ret.joint_angles_rad,
                            duration_s=estimate_duration_safe(
                                rs_g.joint_angles_rad, r_ret.joint_angles_rad,
                                max_velocity_rad_s=self.config.max_velocity_rad_s),
                            gripper_start=hold_cmd,
                            gripper_end=hold_cmd,
                        )
                        print(f"   Levee de verification : {len(traj_lift_v)} points, "
                              f"duree {traj_lift_v.duration_s:.1f}s")
                        controller.execute_trajectory(traj_lift_v, verbose=False,
                                                      on_step=live_callback)
                        time.sleep(self.config.grasp_settle_pause_s)
                        lift_pct = controller.read_gripper_pct()
                        lift_load = controller.read_gripper_load()
                        # JUGEMENT POST-LEVEE par DEUX preuves (OR), 2026-06-13 :
                        #   (1) COUPLE : tenir un objet contre la gravite demande
                        #       un couple SOUTENU nettement au-dessus du couple a
                        #       VIDE (baseline ~24). Essai 3 du 2026-06-13 : tenu
                        #       -> couple 200, lache -> couple 24. Discriminant net.
                        #   (2) POSITION : les machoires restent BLOQUEES par
                        #       l'objet AU-DESSUS de la consigne de serrage hold_cmd
                        #       (si l'objet tombe, elles se referment jusqu'a hold_cmd).
                        # On compare a hold_cmd (consigne de maintien), PAS a
                        # contact_pct (premier contact) : un objet qui se comprime
                        # en etant serre descend SOUS contact_pct tout en restant
                        # tenu -> l'ancien critere (contact_pct-2.5) le declarait
                        # PERDU a tort (faux negatif essai 3).
                        baseline_load = (self._gripper_baseline[1]
                                         if getattr(self, "_gripper_baseline", None)
                                         else 24)
                        held_by_load = (lift_load >= 0
                                        and lift_load >= baseline_load + 90)
                        held_by_pos = (hold_cmd is not None
                                       and lift_pct >= hold_cmd + 3.0)
                        held = held_by_load or held_by_pos
                        why = []
                        if held_by_load: why.append("couple")
                        if held_by_pos: why.append("position")
                        why_txt = (" [" + "+".join(why) + "]") if held else ""
                        tag_l = ("OBJET TENU (saisie confirmee)" + why_txt if held
                                 else "OBJET PERDU -> faux positif attrape")
                        print(f"   [verif levee] pince={lift_pct:.1f}% (maintien "
                              f"{hold_cmd:.1f}%), couple={lift_load} (vide ~{baseline_load})"
                              f"  -->  {tag_l}")
                        entry["lift_pct"] = lift_pct
                        entry["lift_load"] = lift_load
                        entry["held_after_lift"] = held
                        if held:
                            entry["success"] = True
                            grasp_succeeded = True
                            break
                        failed_stage = "lift"
                    elif close_ok:
                        # Verif post-levee desactivee : on fait confiance au
                        # check fermeture (comportement historique).
                        entry["success"] = True
                        grasp_succeeded = True
                        break
                    else:
                        failed_stage = "close"

                    # Echec (fermeture a vide OU faux positif post-levee) :
                    # retry possible ?
                    if attempt > self.config.max_grasp_retries:
                        print(f"   >> ABANDON apres {attempt} tentative(s) (max retries atteint).")
                        break

                    reason = ("faux positif post-levee" if failed_stage == "lift"
                              else "fermeture a vide")
                    print(f"   >> RETRY ({reason}) : refinement cam_2 + redescente")

                    # retry_observe : True si la re-capture cam_2 se fait depuis la
                    # hauteur de RECUL (vue degagee des doigts) -> on peut alors faire
                    # plus confiance a une grosse correction (plafond elargi).
                    retry_observe = False
                    observe_height_m = float(self.config.cam2_observe_height_m)
                    object_z_m = float(grasp_pose.meta.get(
                        "object_center_z_m", grasp_pose.T_base_gripper_grasp[2, 3]))

                    if failed_stage == "lift":
                        # Deja a retract, pince fermee sur rien : OUVRIR en
                        # place (le bras ne bouge pas) puis refinement.
                        controller.set_gripper_pct(grasp_pose.gripper_open_pct)
                        time.sleep(0.6)
                        rs_after_lift = self._provider.read_live()
                    else:
                        # 1. Remontee a la pose d'OBSERVATION (approach + RECUL cam_2)
                        # AVEC pince ouverte (libere ce qu'on aurait pu saisir). On
                        # remonte au MEME niveau que la capture initiale (recul) pour
                        # que cam_2 voie l'objet DEGAGE des doigts au retry AUSSI ->
                        # sinon (run orange 2026-06-23) la re-capture se fait trop bas,
                        # la pince devant occulte le cube, la correction est bruitee
                        # (Maxence : "il n'a meme pas repris la hauteur pour mieux se
                        # replacer"). Fallback approche standard si le recul est hors IK.
                        rs_at_grasp = self._provider.read_live()
                        r_lift_target = r_app
                        if observe_height_m > 1e-4:
                            T_obs_r = grasp_pose.T_base_gripper_approach.copy()
                            T_obs_r[2, 3] = object_z_m + observe_height_m
                            r_obs_r = self._ik.solve(
                                T_obs_r, q_init=r_app.joint_angles_rad)
                            if (r_obs_r.translation_err_mm
                                    <= self.config.ik_tol_approach_mm):
                                r_lift_target = r_obs_r
                                retry_observe = True
                        traj_lift = self._build_retry_lift_traj(
                            rs_at_grasp.joint_angles_rad,
                            r_lift_target.joint_angles_rad,
                            grip_open_pct=grasp_pose.gripper_open_pct,
                        )
                        print(f"   Remontee a "
                              f"{'observation +recul' if retry_observe else 'approach'} : "
                              f"{len(traj_lift)} points, duree {traj_lift.duration_s:.1f}s")
                        controller.execute_trajectory(traj_lift, verbose=False,
                                                      on_step=live_callback)
                        rs_after_lift = self._provider.read_live()

                    # 2a. RE-PERCEPTION par cameras FIXES (l'objet a-t-il BOUGE ?)
                    # cam_2 (eye-in-hand) est parquee au-dessus de l'ANCIENNE
                    # position ; si l'objet a roule ailleurs elle ne le voit plus et
                    # on redescendait au meme endroit. On relance une perception
                    # STEREO complete (cam_0/cam_1, champ large) pour le RE-LOCALISER,
                    # puis on REPLANIFIE depuis sa nouvelle position avant de
                    # redescendre (demande Maxence 2026-06-21). Si non retrouve ou
                    # ~immobile -> on retombe sur le raffinement cam_2 (ci-dessous).
                    if self.config.retry_reperceive and failed_stage != "lift":
                        time.sleep(self.config.retry_settle_s)
                        try:
                            new_t = self._reperceive_target(mc)
                        except Exception as _e:
                            new_t = None
                            print(f"   [re-perception] echec capture ({_e}) "
                                  "-> raffinement cam_2")
                        if new_t is not None:
                            move_mm = float(np.hypot(
                                new_t.position_base_m[0] - target.position_base_m[0],
                                new_t.position_base_m[1] - target.position_base_m[1])
                                * 1000.0)
                            if move_mm >= self.config.retry_reperceive_move_mm:
                                p = new_t.position_base_m * 1000.0
                                print(f"   [re-perception cam_0/cam_1] cible RE-LOCALISEE "
                                      f"a ({p[0]:+.0f},{p[1]:+.0f},{p[2]:+.0f}) mm "
                                      f"(deplacement {move_mm:.0f}mm) -> REPLANIFICATION")
                                ng, na, ngr, nr = self._plan_and_solve_grasp(
                                    new_t, rs_after_lift.joint_angles_rad)
                                if ng is not None:
                                    grasp_pose, r_app, r_grp, r_ret = ng, na, ngr, nr
                                    target = new_t
                                    # Re-plan stereo (sans cam_2) -> re-applique
                                    # l'offset lateral pince le long des machoires
                                    # de la NOUVELLE orientation.
                                    lat_mm = self._effective_lateral_offset_mm(grasp_pose)
                                    fwd_mm = float(self.config.grasp_forward_offset_mm)
                                    if lat_mm != 0.0 or fwd_mm != 0.0:
                                        offset_vec = (
                                            self._cam2_image_left_base(grasp_pose.T_base_gripper_grasp)
                                            * (lat_mm / 1000.0)
                                            + self._cam2_image_down_base(grasp_pose.T_base_gripper_grasp)
                                            * (fwd_mm / 1000.0))
                                        for _attr in ("T_base_gripper_approach",
                                                      "T_base_gripper_grasp",
                                                      "T_base_gripper_retract"):
                                            getattr(grasp_pose, _attr)[:3, 3] += offset_vec
                                        r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                                            grasp_pose,
                                            q_init=rs_after_lift.joint_angles_rad,
                                            lock_orientation=True)
                                    # SNAPSHOT cam_2 de comparaison (Maxence
                                    # 2026-06-22) : cette branche court-circuite le
                                    # raffinement cam_2 -> sans capture explicite,
                                    # aucune image cam_2 ne serait sauvee pour la
                                    # tentative suivante (impossible de comparer #1/#2).
                                    if self.config.save_snapshots:
                                        from src.control.closed_loop import (
                                            capture_cam2_snapshot,
                                        )
                                        capture_cam2_snapshot(
                                            target_label=self.config.target_label,
                                            detector=self._detector,
                                            multi_camera=mc,
                                            robot_state=rs_after_lift,
                                            out_dir=(REPO / "outputs" / "perception"),
                                            label_mapping=self._label_mapping,
                                            tag="retry/replan",
                                        )
                                    # -> on redescend DIRECTEMENT sur la nouvelle
                                    # prise (pas de cam_2 : on a une perception fraiche).
                                    q_start_attempt = rs_after_lift.joint_angles_rad
                                    print()
                                    continue
                                print("   [re-perception] re-plan infaisable "
                                      "-> raffinement cam_2")
                            else:
                                print(f"   [re-perception cam_0/cam_1] cible ~au meme "
                                      f"endroit (deplacement {move_mm:.0f}mm < "
                                      f"{self.config.retry_reperceive_move_mm:.0f}mm) "
                                      f"-> raffinement cam_2")
                        else:
                            print("   [re-perception cam_0/cam_1] cible non retrouvee "
                                  "-> raffinement cam_2")

                    # 2b. Refinement cam_2 (l'objet peut avoir bouge a cause du
                    # contact rate) — SETTLE PLUS LONG avant re-capture : on prend
                    # le temps de se stabiliser pour re-detecter PROPREMENT l'objet
                    # (eventuellement deplace), au lieu de redescendre au meme
                    # endroit (demande Maxence 2026-06-21).
                    time.sleep(self.config.retry_settle_s)
                    rs_after_lift = self._provider.read_live()
                    print(f"   Stabilisation {self.config.retry_settle_s:.1f}s puis "
                          f"re-capture cam_2 (retry)...")
                    refinement_retry = refine_grasp_with_cam2(
                        target_label=self.config.target_label,
                        detector=self._detector,
                        multi_camera=mc,
                        robot_state=rs_after_lift,
                        target_z_base_m=float(grasp_pose.meta.get(
                        "object_center_z_m", grasp_pose.T_base_gripper_grasp[2, 3])),
                        grasp_xy_base_m=(float(grasp_pose.T_base_gripper_grasp[0, 3]),
                                         float(grasp_pose.T_base_gripper_grasp[1, 3])),
                        label_mapping=self._label_mapping,
                        debug_save_dir=((REPO / "outputs" / "perception")
                                        if self.config.save_snapshots else None),
                    )
                    print(f"   {refinement_retry.message}")
                    # GATING par TAILLE ABSOLUE du blob (Maxence 2026-06-20), plus
                    # par `score` (= aire normalisee par max_area_px, trompeur).
                    # Meme logique que le refinement initial : blob assez gros +
                    # correction sous le plafond de securite + top-down (ray-plan
                    # valide ; en incline on garde la pose stereo).
                    # Plafond de correction : SERRE (15mm) si la re-capture s'est
                    # faite bas (vue bruitee) ; ELARGI au plafond normal (40mm) si on
                    # a repris la hauteur de RECUL (vue degagee -> grosse correction
                    # fiable, ex objet pousse par le contact rate). Le motif du cap
                    # serre etait justement le blob bruite en re-capture basse.
                    seuil_r_mm = (self.config.closed_loop_max_correction_mm
                                  if retry_observe
                                  else self.config.cam2_retry_max_correction_mm)
                    is_tilted = (abs(np.degrees(grasp_pose.meta.get("pitch_rad", 0.0)))
                                 >= self.config.cam2_max_pitch_deg)
                    if is_tilted:
                        print("   (prise >= seuil pitch -> correction retry cam_2 ignoree : "
                              "axe optique trop horizontal ; pose stereo conservee)")
                    elif refinement_retry.detection is None:
                        print("   [INFO] cam_2 sans detection au retry -> pose stereo conservee")
                    elif refinement_retry.area_frac < self.config.cam2_min_blob_frac:
                        print(f"   [INFO] blob cam_2 retry trop petit "
                              f"({100*refinement_retry.area_frac:.2f}% < "
                              f"{100*self.config.cam2_min_blob_frac:.2f}% cadre) "
                              "-> correction retry ignoree")
                    elif refinement_retry.delta_norm_mm >= seuil_r_mm:
                        print(f"   [WARN] correction retry = {refinement_retry.delta_norm_mm:.1f}mm "
                              f"> seuil {seuil_r_mm:.0f}mm, ignoree (suspect)")
                    else:
                        apply_correction_to_grasp_pose(grasp_pose, refinement_retry.delta_base_m)
                        # RE-APPLIQUE les offsets pince (lateral + profondeur) : le
                        # recentrage cam_2 du retry vient de les annuler. Recalcule
                        # depuis l'orientation COURANTE (le long des machoires / bas-image).
                        lat_mm = self._effective_lateral_offset_mm(grasp_pose)
                        fwd_mm = float(self.config.grasp_forward_offset_mm)
                        if lat_mm != 0.0 or fwd_mm != 0.0:
                            offset_vec = (self._cam2_image_left_base(grasp_pose.T_base_gripper_grasp)
                                          * (lat_mm / 1000.0)
                                          + self._cam2_image_down_base(grasp_pose.T_base_gripper_grasp)
                                          * (fwd_mm / 1000.0))
                            for _attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                                          "T_base_gripper_retract"):
                                getattr(grasp_pose, _attr)[:3, 3] += offset_vec
                        print(f"   Correction retry appliquee "
                              f"(norme {refinement_retry.delta_norm_mm:.1f}mm, "
                              f"blob {100*refinement_retry.area_frac:.1f}% cadre ; "
                              f"offsets pince lateral {lat_mm:+.0f}mm + profondeur "
                              f"{fwd_mm:+.0f}mm re-appliques)")
                        # lock_orientation : conserver l'orientation deja
                        # engagee (anti demi-tour 180deg au retry, essai 10).
                        r_app, r_grp, r_ret = self._ik.solve_grasp_pose(
                            grasp_pose, q_init=rs_after_lift.joint_angles_rad,
                            lock_orientation=True)
                        self._log_wrist("re-solve retry",
                                        rs_after_lift.joint_angles_rad, r_grp)

                    # Le point de depart de la prochaine descente = position courante
                    q_start_attempt = rs_after_lift.joint_angles_rad
                    # (la boucle reprend : descente + fermeture + checks)

                print()

                # --- Apres la boucle : depose si succes, retour direct si echec ---
                rs_after_grasp = self._provider.read_live()
                if grasp_succeeded:
                    print(f">> SAISIE REUSSIE en {attempt} tentative(s). "
                          f"Depose dans la boite + retour {home_origin}.")
                    # NB : si la verif post-levee est active, on est DEJA a
                    # retract -> le 1er segment (grasp->retract) est ~statique.
                    traj_finish = self._build_finish_after_grasp_traj(
                        q_grp=rs_after_grasp.joint_angles_rad,
                        q_ret=r_ret.joint_angles_rad,
                        q_drop=r_drop_above.joint_angles_rad,
                        q_rel=r_drop_release.joint_angles_rad,
                        q_home=q_home_final,
                        grip_close_pct=hold_cmd,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                    )
                else:
                    print(f">> ECHEC apres {attempt} tentative(s). "
                          f"Retour {home_origin} sans depose.")
                    traj_finish = self._build_abort_to_home_traj(
                        q_grp=rs_after_grasp.joint_angles_rad,
                        q_ret=r_ret.joint_angles_rad,
                        q_home=q_home_final,
                        grip_open_pct=grasp_pose.gripper_open_pct,
                    )
                print(f"   {len(traj_finish)} points, duree {traj_finish.duration_s:.1f}s")
                controller.execute_trajectory(traj_finish, verbose=True,
                                              on_step=live_callback)

                # Log structure final pour la campagne (P4)
                self._grasp_attempts_log = grasp_attempts_log
                self._grasp_final_success = grasp_succeeded
                self._grasp_total_attempts = attempt
                self._print_grasp_summary(grasp_attempts_log, grasp_succeeded, attempt)

                print(f">> Termine. Robot revenu a la {home_origin}.")
                if self.config.display:
                    import cv2
                    cv2.destroyAllWindows()

            else:
                # Mode sans boucle fermee OU dry-run : trajectoire complete
                print(">> Generation trajectoire complete (sans boucle fermee)...")
                traj = self._build_full_trajectory(
                    q_current,
                    [r_app, r_grp, r_ret, r_drop_above, r_drop_release],
                    q_home=q_home_final,
                    gripper_open_pct=grasp_pose.gripper_open_pct,
                )
                print(f"   {len(traj)} points, duree {traj.duration_s:.1f}s")
                print()
                if self.config.dry_run:
                    print(">> DRY RUN : pas d'execution sur le robot.")
                else:
                    print(f">> Execution sur le robot (jusqu'au retour {home_origin})...")
                    controller.execute_trajectory(traj, verbose=True,
                                                  on_step=live_callback)
                    print(f">> Termine. Robot revenu a la {home_origin}.")
                    if self.config.display:
                        import cv2
                        cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\nInterrompu par utilisateur.")
            self._safe_stop_with_torque(controller)
        except Exception as e:
            print(f"\n!! EXCEPTION pendant le pipeline : {type(e).__name__}: {e}")
            self._safe_stop_with_torque(controller)
            raise
        finally:
            # Arrete proprement le worker thread du live display (s'il existe).
            # Sans cela, le thread tournerait jusqu'a la fin du programme
            # (daemon=True le ferme quand meme, mais c'est plus propre).
            stop_ev = getattr(self, "_live_stop_event", None)
            if stop_ev is not None:
                try:
                    stop_ev.set()
                except Exception:
                    pass
                self._live_stop_event = None
            try:
                mc.close()
            except Exception:
                pass
            try:
                self._provider.disconnect_live()
            except Exception:
                pass
            if controller is not None:
                try:
                    controller.disconnect()
                except Exception:
                    pass

    def _annotate_frame(self, frame, dets, scene):
        """Helper : annote une frame avec detections + reprojections 3D.

        Taille des tuiles fixee par DISPLAY_TILE_SIZE (constante au top du module).
        """
        import cv2
        import numpy as np
        from src.perception.pose_estimator import _projection_matrix
        tw, th = DISPLAY_TILE_SIZE
        if frame is None:
            return np.zeros((th, tw, 3), dtype=np.uint8)
        img = frame.image.copy()
        # Bandeau noir avec le nom de la cam (utile vu que l'ordre n'est plus
        # alphabetique : cam_1 a gauche maintenant).
        cv2.rectangle(img, (0, 0), (260, 50), (0, 0, 0), -1)
        cv2.putText(img, frame.cam_key, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        for d in dets:
            if d.bbox:
                x0, y0, x1, y1 = (int(v) for v in d.bbox)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cx, cy = int(d.center_px[0]), int(d.center_px[1])
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{d.label} ({d.score:.2f})", (cx + 8, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Reprojection des objets 3D estimes (rouge)
        if scene is not None:
            P = _projection_matrix(frame.K, frame.T_base_cam)
            for obj in scene.objects:
                X = np.hstack([obj.position_base_m, 1.0])
                uvw = P @ X
                if uvw[2] > 0:
                    u, v = uvw[0]/uvw[2], uvw[1]/uvw[2]
                    cv2.circle(img, (int(u), int(v)), 8, (0, 0, 255), 2)
        return cv2.resize(img, (tw, th))

    def _show_perception_snapshot(self, frames, dets_by_cam, scene, title=""):
        """Affiche les 3 frames camera avec detections, sauvegarde aussi
        dans outputs/perception/. Non-bloquant : utilise cv2.waitKey(1500) au
        lieu de input().
        """
        import cv2
        import numpy as np
        from datetime import datetime

        tiles = []
        for k in CAM_DISPLAY_ORDER:
            tiles.append(self._annotate_frame(frames.get(k), dets_by_cam.get(k, []), scene))
        mosaic = np.hstack(tiles)
        cv2.putText(mosaic, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Sauvegarde
        out_dir = REPO / "outputs" / "perception"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"pipeline_snapshot_{stamp}.png"
        cv2.imwrite(str(out_path), mosaic)
        print(f"   [snapshot] perception sauve : {out_path}")

        # Affichage NON BLOQUANT seulement si --display (la fenetre live + ses
        # grabs sont ce qui fait decrocher cam_2 ; la sauvegarde ci-dessus, non).
        if self.config.display:
            cv2.imshow("Pipeline perception", mosaic)
            cv2.waitKey(1500)   # 1.5s = visualisation suffisante
            # La fenetre reste ouverte ; sera mise a jour ou fermee plus tard

    def _live_display_during_execution(self, mc, robot_state_provider,
                                        every_n_frames: int = 30):
        """Generateur de callback : a appeler periodiquement pendant
        execute_trajectory pour rafraichir l'affichage des cameras.

        Utilise comme : passer ce callable a controller.execute_trajectory
        si on veut un display live. Sinon ne fait rien.
        """
        # Pour l'instant on n'integre pas dans execute_trajectory (trop
        # invasif). On laisse l'utilisateur lancer pick_and_place puis
        # voir le snapshot initial + le robot bouge a vue d'oeil.
        # TODO : ajouter un thread separe qui rafraichit la fenetre.
        pass

    def _safe_stop_with_torque(self, controller):
        """En cas d'exception ou Ctrl+C : MAINTIENT le torque pour eviter
        que le bras tombe sous son poids. Demande a l'utilisateur de
        placer le bras en securite manuellement.

        Sans cette procedure, le `finally` appellerait disconnect() qui
        coupe le torque -> le bras tombe.
        """
        if controller is None or controller._bus is None:
            return
        if not controller._torque_enabled:
            return
        print()
        print("=" * 70)
        print(" SECURITE : torque MAINTENU pour eviter que le bras tombe.")
        print(" 1. Soutiens manuellement le bras avec ta main libre.")
        print(" 2. Appuie sur ENTREE quand tu maintiens le bras et qu'il")
        print("    est en securite (pas suspendu en l'air).")
        print(" 3. Le torque sera ALORS coupe.")
        print("=" * 70)
        try:
            input("Pret a couper le torque ? Appuie ENTREE : ")
        except (EOFError, KeyboardInterrupt):
            pass
        try:
            controller.disable_torque()
            print("Torque coupe. Tu peux relacher le bras.")
        except Exception as e:
            print(f"[WARN] Echec disable_torque : {e}. Coupe l'alim manuellement.")

    # ----- helpers --------------------------------------------------------

    def _log_wrist(self, tag: str, q_ref: dict, ik_res) -> None:
        """Trace la continuite du wrist_roll a chaque (re-)resolution IK.

        Un ecart > 90deg = la cible demande un demi-tour de poignet : c'est la
        signature des 'tours' observes pendant mini-descente/descente. Avec la
        persistance d'orientation (solve_grasp_pose), cela ne devrait plus se
        produire ; ce log le verifie run apres run (ou pinpointe le cas restant).
        """
        try:
            if ik_res is None or not getattr(ik_res, "joint_angles_rad", None):
                return
            wr0 = float(np.degrees(q_ref.get("wrist_roll", 0.0)))
            wr1 = float(np.degrees(ik_res.joint_angles_rad.get("wrist_roll", 0.0)))
            flag = "  <<< SAUT (tour)" if abs(wr1 - wr0) > 90.0 else ""
            print(f"   [wrist] {tag} : {wr0:+.0f}deg -> {wr1:+.0f}deg{flag}")
        except Exception:
            pass

    def _build_phase1_trajectory(self,
                                  q_current: dict[str, float],
                                  ik_approach: IKResult,
                                  gripper_open_pct: Optional[float] = None,
                                  ) -> JointTrajectory:
        """Sprint 4 boucle fermee : trajectoire courant -> approach uniquement.

        B4 (2026-05-19) : pince reste FERMEE pendant tout le deplacement
        vers approach. Avant : la pince s'ouvrait progressivement de 5% a 80%
        durant ce segment, donc arrivait au-dessus de l'objet deja ouverte
        (et balayait visuellement la scene depuis le depart). Maintenant :
        la pince reste fermee pendant l'approche, et ne s'ouvre qu'au moment
        de la mini-descente (4cm au-dessus du grasp), pour ne plus etre
        ouverte que strictement quand necessaire (= pour saisir).
        Le parametre gripper_open_pct est ignore ici mais conserve dans la
        signature pour compatibilite.
        """
        c = self.config
        q_app = ik_approach.joint_angles_rad
        return quintic_trajectory(
            q_current, q_app,
            duration_s=estimate_duration_safe(q_current, q_app,
                                               max_velocity_rad_s=c.max_velocity_rad_s),
            gripper_start=c.home_gripper_pct,  # ferme au depart
            gripper_end=c.home_gripper_pct,    # B4 : reste fermee a l'arrivee
        )

    def _build_phase2_trajectory(self,
                                  q_at_approach: dict[str, float],
                                  ik_results: list[IKResult],
                                  q_home: Optional[dict] = None,
                                  gripper_open_pct: Optional[float] = None,
                                  ) -> JointTrajectory:
        """Sprint 4 phase 2 : approach -> grasp -> retract -> drop_above
        -> drop_release -> safe -> home.

        Identique a _build_full_trajectory mais part de q_at_approach
        (pas q_current) et n'inclut pas le premier segment.

        Args:
            q_home : pose finale a rejoindre apres la depose. Si None,
                fallback sur self.config.home_angles_rad ou la pose hardcodee.
                Typiquement, le pipeline appelle avec q_home=q_session_start
                pour que le robot retourne ou l'utilisateur l'avait place.
            gripper_open_pct : ouverture pince a utiliser (= calculee par
                TopDownGrasp selon la bbox de l'objet). Si None, fallback
                sur config.grip_open_pct (100%).
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad
        c = self.config
        gp_o = gripper_open_pct if gripper_open_pct is not None else c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

        # Pose "home" finale STABLE : priorite au parametre fourni (= pose de
        # depart session typiquement), fallback sur config, fallback sur
        # pose hardcodee "bras replie".
        q_home = q_home or c.home_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.0,
            "wrist_flex": -0.7,    # poignet replie pour pince vers le bas
            "wrist_roll": 0.0,
        }
        # Pose intermediaire "safe" : bras releve haut, centre.
        # IMPORTANT : on HERITE wrist_roll et wrist_flex de q_home pour eviter
        # une rotation parasite finale (q_safe -> q_home). Sans ca, si
        # q_session_start a wrist_roll=0.5 et q_safe hardcode wrist_roll=0,
        # le poignet faisait 0.5 -> 0 (safe) -> 0.5 (home), inutile et bizarre.
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        # Vitesse RALENTIE pour la transition vers home (anti-chute visuel)
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )

        segs = [
            # 1. q_at_approach -> approach corrige (petit deplacement si correction)
            quintic_trajectory(q_at_approach, q_app, duration_s=dur(q_at_approach, q_app),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 2. approach -> grasp
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 3. STATIQUE : ferme la pince
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
            # 4. grasp -> retract (pince fermee, bras remonte avec l'objet)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 5. retract -> drop_above (deplace au-dessus de la boite)
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 6. drop_above -> drop_release (descend dans la boite)
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 7. STATIQUE : relache (ouvre la pince)
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # 7.5 SORTIE VERTICALE de la boite : drop_release -> drop_above.
            #    Sans ce segment, le bras allait directement de l'interieur
            #    de la boite (drop_release) vers q_safe (centre, haut) -> la
            #    pince fixe cognait le bord interieur de la boite en chemin.
            #    En remontant d'abord verticalement, on sort proprement.
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 8. drop_above -> SAFE intermediaire (releve le bras AVANT
            #    le retour a home, pour eviter de traverser la zone du cube).
            #    On ferme PROGRESSIVEMENT la pince ici pour eviter qu'elle
            #    reste grand-ouverte (securite : pas d'accrochage sur cables).
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 9. SAFE -> HOME : transition RALENTIE pour atterrissage doux.
            #    Pince finit a home_gripper_pct (par defaut 5 = presque fermee).
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _build_full_trajectory(self,
                                q_current: dict[str, float],
                                ik_results: list[IKResult],
                                q_home: Optional[dict] = None,
                                gripper_open_pct: Optional[float] = None,
                                ) -> JointTrajectory:
        """Concatene les sous-trajectoires : current -> approach -> grasp ->
        (fermer pince) -> retract -> drop_above -> drop_release ->
        (ouvrir pince) -> safe -> home.

        On utilise quintic pour chaque segment (lisse) + pauses pour la
        fermeture/ouverture de pince.

        Args:
            q_home : pose finale a rejoindre. Si None, fallback sur
                config.home_angles_rad ou la pose hardcodee.
            gripper_open_pct : ouverture pince calculee par TopDownGrasp
                selon la bbox de l'objet. Si None, fallback config.
        """
        q_app  = ik_results[0].joint_angles_rad
        q_grp  = ik_results[1].joint_angles_rad
        q_ret  = ik_results[2].joint_angles_rad
        q_drop = ik_results[3].joint_angles_rad
        q_rel  = ik_results[4].joint_angles_rad

        c = self.config
        gp_o = gripper_open_pct if gripper_open_pct is not None else c.grip_open_pct
        gp_c = c.grip_close_pct

        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)

        # Pose finale : parametre fourni > config > defaut hardcode.
        # FIX (etait q_rest=config_zero precedemment, ce qui en plus pouvait
        # crasher sur q_home/dur_home non definis -- NameError).
        q_home = q_home or c.home_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.0,
            "wrist_flex": -0.7,
            "wrist_roll": 0.0,
        }
        # Pose safe intermediaire : on HERITE wrist_roll/wrist_flex de q_home
        # pour eviter une rotation parasite finale (cf _build_phase2_trajectory).
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0, "shoulder_lift": -0.6, "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )

        segs = [
            # 1. courant -> approach : pince s'OUVRE progressivement
            quintic_trajectory(q_current, q_app, duration_s=dur(q_current, q_app),
                                gripper_start=c.home_gripper_pct,  # ferme au depart
                                gripper_end=gp_o),                  # ouverte a l'arrivee
            quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                gripper_start=gp_o, gripper_end=gp_o),
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                gripper_start=gp_o, gripper_end=gp_c),
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # Sortie VERTICALE de la boite (drop_release -> drop_above) AVANT
            # safe -> evite que la pince fixe cogne le bord interieur.
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # Safe intermediate AVANT home (evite que la pince traverse la zone
            # du cube pendant le retour). Pince se referme progressivement.
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # Transition finale vers home : RALENTIE pour atterrissage doux.
            # Pince finit a home_gripper_pct (defaut 5 = quasi-fermee).
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    # ============================================================
    # P1 : Helpers SOUS-TRAJECTOIRES (pour boucle de tentatives + retry)
    # On split la phase 2 historique en sous-trajectoires individuelles,
    # appellees explicitement depuis run() avec un CHECK pince entre
    # la fermeture et le transport. Les helpers _build_phase2_trajectory
    # et _build_full_trajectory restent en place (utilises pour le mode
    # --no-closed-loop et comme fallback).
    # ============================================================

    def _build_grasp_attempt_traj(
        self, q_from: dict[str, float], q_app: dict[str, float],
        q_grp: dict[str, float],
        grip_open_pct: float, grip_close_pct: float,
        include_close: bool = True,
        include_approach: bool = True,
    ) -> JointTrajectory:
        """Sous-traj : q_from (-> approach) -> grasp (-> ferme pince statique).

        Utilisee pour chaque TENTATIVE de saisie (1er essai + eventuels retries).
        include_approach=False : descente DIRECTE q_from -> grasp, sans repasser
        par la pose approach. INDISPENSABLE a la 1ere tentative : on part de la
        mini-descente (grasp+4cm), qui est DEJA SOUS approach (grasp+8cm) ; passer
        par approach ferait REMONTER le bras de 4cm puis redescendre — c'est la
        remontee parasite "il amorce la descente, remonte se replacer, redescend"
        observee par Maxence (essai 11), qui frole parfois l'objet. Aux retries,
        le bras est a/au-dessus de approach -> include_approach=True (descente
        monotone).
        include_close=False (mode fermeture ASSERVIE P5) : la trajectoire
        s'arrete pince OUVERTE a la pose grasp ; la fermeture est ensuite
        pilotee par controller.close_gripper_with_feedback() (stop au contact).
        include_close=True (mode statique historique) : fermeture aveugle a
        grip_close_pct en fin de trajectoire ; run() lit la position apres.
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_o, gp_c = grip_open_pct, grip_close_pct
        if include_approach:
            segs = [
                quintic_trajectory(q_from, q_app, duration_s=dur(q_from, q_app),
                                    gripper_start=gp_o, gripper_end=gp_o),
                quintic_trajectory(q_app, q_grp, duration_s=dur(q_app, q_grp),
                                    gripper_start=gp_o, gripper_end=gp_o),
            ]
        else:
            # Descente directe depuis la position courante (mini-descente) ->
            # grasp, sans remonter a approach.
            segs = [
                quintic_trajectory(q_from, q_grp, duration_s=dur(q_from, q_grp),
                                    gripper_start=gp_o, gripper_end=gp_o),
            ]
        if include_close:
            # Fermeture STATIQUE (le bras ne bouge pas, la pince ferme).
            segs.append(
                quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_grasp_s, 0.5),
                                    gripper_start=gp_o, gripper_end=gp_c))
        return chain_trajectories(segs)

    def _build_retry_lift_traj(
        self, q_grp: dict[str, float], q_app: dict[str, float],
        grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj : remontee de grasp vers approach AVEC OUVERTURE PINCE.

        Sert au RETRY apres saisie ratee : on libere ce qu'on a (ou pas) saisi
        puis on remonte a approach pour relancer refinement + descente.
        """
        c = self.config
        return quintic_trajectory(
            q_grp, q_app,
            duration_s=estimate_duration_safe(q_grp, q_app,
                                              max_velocity_rad_s=c.max_velocity_rad_s),
            gripper_start=grip_open_pct, gripper_end=grip_open_pct,
        )

    def _build_finish_after_grasp_traj(
        self, q_grp: dict[str, float], q_ret: dict[str, float],
        q_drop: dict[str, float], q_rel: dict[str, float],
        q_home: dict[str, float],
        grip_close_pct: float, grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj SUCCES : grasp -> retract -> drop_above -> drop_release
        -> (ouvre pince) -> sortie verticale -> safe -> home.

        Pince fermee pendant le transport, ouvre au dessus de la boite, referme
        progressivement pour le retour home (home_gripper_pct, ~5%).
        Reprend la logique de _build_phase2_trajectory segments 4-9.
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_c, gp_o = grip_close_pct, grip_open_pct
        # q_safe = pose intermediaire haute. HERITE wrist_roll/wrist_flex de
        # q_home pour eviter rotation parasite finale (cf P3).
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )
        segs = [
            # 1. grasp -> retract (pince fermee, remonte avec objet)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 2. retract -> drop_above
            quintic_trajectory(q_ret, q_drop, duration_s=dur(q_ret, q_drop),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 3. drop_above -> drop_release (descend dans boite)
            quintic_trajectory(q_drop, q_rel, duration_s=dur(q_drop, q_rel),
                                gripper_start=gp_c, gripper_end=gp_c),
            # 4. STATIQUE : relache (ouvre pince)
            quintic_trajectory(q_rel, q_rel, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_c, gripper_end=gp_o),
            # 5. SORTIE VERTICALE de la boite (drop_release -> drop_above)
            quintic_trajectory(q_rel, q_drop, duration_s=dur(q_rel, q_drop),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 6. drop_above -> SAFE (releve haut, ferme pince progressivement)
            quintic_trajectory(q_drop, q_safe, duration_s=dur(q_drop, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 7. SAFE -> HOME (transition ralentie)
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _build_abort_to_home_traj(
        self, q_grp: dict[str, float], q_ret: dict[str, float],
        q_home: dict[str, float], grip_open_pct: float,
    ) -> JointTrajectory:
        """Sous-traj ECHEC : grasp -> retract (pince ouverte) -> safe -> home.

        Apres tous les retries epuises sans saisie reussie : on remonte sans
        rien transporter et on rentre. Pas de visite a la boite (rien a poser).
        """
        c = self.config
        def dur(q1, q2):
            return estimate_duration_safe(q1, q2, max_velocity_rad_s=c.max_velocity_rad_s)
        gp_o = grip_open_pct
        q_safe = c.safe_intermediate_angles_rad or {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 1.0,
            "wrist_flex": q_home.get("wrist_flex", 0.0),
            "wrist_roll": q_home.get("wrist_roll", 0.0),
        }
        dur_home = estimate_duration_safe(
            q_safe, q_home,
            max_velocity_rad_s=min(c.home_max_velocity_rad_s, c.max_velocity_rad_s),
        )
        segs = [
            # 1. STATIQUE : pince OUVERTE (relache ce qu'on a partiellement saisi)
            quintic_trajectory(q_grp, q_grp, duration_s=max(c.pause_release_s, 0.3),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 2. grasp -> retract (remonte, pince ouverte)
            quintic_trajectory(q_grp, q_ret, duration_s=dur(q_grp, q_ret),
                                gripper_start=gp_o, gripper_end=gp_o),
            # 3. retract -> safe (referme progressivement vers home_gripper_pct)
            quintic_trajectory(q_ret, q_safe, duration_s=dur(q_ret, q_safe),
                                gripper_start=gp_o, gripper_end=c.home_gripper_pct),
            # 4. safe -> home (ralenti)
            quintic_trajectory(q_safe, q_home, duration_s=dur_home,
                                gripper_start=c.home_gripper_pct,
                                gripper_end=c.home_gripper_pct),
        ]
        return chain_trajectories(segs)

    def _print_grasp_summary(self, attempts_log: list[dict],
                              succeeded: bool, total_attempts: int):
        """Recap structure des tentatives de saisie (pour campagne P4).

        Imprime un tableau clair en fin de pipeline avec :
          - position pince reelle a chaque tentative
          - marge vs seuil de detection
          - resultat OK/RATE
          - resultat final REUSSIE / ECHEC
        Les attributs self._grasp_* sont aussi exposes pour qu'un script
        externe (scripts/experiment_campaign.py) puisse les lire.
        """
        print()
        print("=" * 70)
        print(" RECAP SAISIE")
        print("=" * 70)
        if not attempts_log:
            print("  Aucune tentative loggee (bug ?)")
        for entry in attempts_log:
            n = entry["attempt"]
            gp = entry["gripper_pct"]
            load = entry.get("gripper_load", -1)
            sc = entry["success"]
            if sc:
                tag = "OK"
            elif entry.get("held_after_lift") is False:
                tag = "FAUX POSITIF (attrape a la levee)"
            else:
                tag = "RATE"
            load_txt = f", couple {load}" if load is not None and load >= 0 else ""
            print(f"  Tentative #{n} : fermeture pince {gp:>5.1f}%{load_txt}  --> {tag}")
            if entry.get("held_after_lift") is not None:
                ll = entry.get("lift_load")
                lp = entry.get("lift_pct")
                verdict = "tenu" if entry["held_after_lift"] else "PERDU"
                print(f"                 apres levee : pince {lp:.1f}%, "
                      f"couple {ll}  --> {verdict}")
        final = "REUSSIE" if succeeded else "ECHEC"
        print(f"  --> Resultat final : {final}  ({total_attempts} tentative(s) total)")
        print("=" * 70)
