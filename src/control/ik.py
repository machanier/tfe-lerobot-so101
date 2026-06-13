"""
ik.py - Cinematique inverse (Inverse Kinematics) du SO-101.

Convertit une pose cartesienne desiree de l'effecteur (T_base_gripper en SE(3))
en un vecteur d'angles articulaires (5 joints rotoides).

ALGORITHME : Gauss-Newton avec Jacobien numerique (differences finies).
Pure numpy, zero dependance externe au-dela de ce qui est deja utilise pour
la FK (src/calibration/forward_kinematics.py).

PRINCIPE :
  On formule l'IK comme un probleme d'optimisation non-lineaire :
    minimiser  || r(q) ||^2
  ou r(q) est le residu 6D (3 translation + 3 rotation) entre FK(q) et la
  pose cible.

  A chaque iteration :
    1. Calcul du residu r(q)
    2. Calcul du Jacobien J = dr/dq (matrice 6x5) par differences finies
    3. Resolution du systeme lineaire : delta_q = -(J^T J + lambda I)^-1 J^T r
       (Levenberg-Marquardt, lambda regule la stabilite)
    4. q <- q + delta_q (avec clip dans les plages articulaires)
    5. Arret si ||r|| < tol ou nombre d'iterations max atteint.

SOUS-ACTUATION SO-101 :
  Le SO-101 a 5 DDL (5 articulations rotoides utiles) pour 6 DDL d'espace SE(3).
  Toutes les poses ne sont donc pas exactement atteignables ; l'IK trouve
  la meilleure approximation. Pour le top-down grasp (4 contraintes :
  position xyz + yaw), le systeme est bien pose et l'IK converge.

PONDERATION TRANS/ROT :
  Les unites sont differentes (m vs rad). On utilise alpha=0.1 : 1 rad de
  rotation pese comme 0.1 m de translation. Coherent avec l'echelle du
  bras SO-101 (~30 cm).

References :
  - Sciavicco & Siciliano 2000, "Modelling and Control of Robot Manipulators"
    (chapitre 3 : Differential Kinematics ; chapitre 5 : Inverse Kinematics)
  - Levenberg 1944 / Marquardt 1963 (algorithme original)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.calibration.forward_kinematics import ARM_JOINTS, KinematicChain
from src.calibration.motor_to_angle import load_motor_calibration
from src.utils.transforms import matrix_to_rvec_tvec


# ============================================================
# Resultat d'un solve IK
# ============================================================


@dataclass
class IKResult:
    """Resultat de l'IK pour une pose cible.

    Attributes:
        joint_angles_rad : {nom_joint: angle_rad} si converge, sinon None.
        converged        : True si l'IK a converge dans la tolerance.
        residual_norm    : norme du residu final (combine m + 0.1*rad).
        translation_err_mm : erreur de position en mm.
        rotation_err_deg : erreur d'orientation en degres.
        n_iterations     : nombre d'iterations effectuees.
        message          : message explicatif (succes ou cause d'echec).
    """

    joint_angles_rad: Optional[dict[str, float]]
    converged: bool
    residual_norm: float
    translation_err_mm: float
    rotation_err_deg: float
    n_iterations: int
    message: str = ""


# ============================================================
# Solveur IK
# ============================================================


class IKSolver:
    """Solveur IK numerique (Gauss-Newton/Levenberg-Marquardt) pour le SO-101.

    Le solveur opere SANS hardware : on lui passe une pose cible (SE(3)) et
    une configuration de depart (optionnelle), il retourne les angles.

    Limites articulaires : chargees depuis configs/calibration_follower.json
    et appliquees comme clip a chaque iteration. Sans cela, l'IK pourrait
    converger vers des angles physiquement irrealisables (-pi, +pi ailleurs).
    """

    def __init__(self,
                 chain: Optional[KinematicChain] = None,
                 calib_path: Optional[Path] = None,
                 rotation_weight: float = 0.1,
                 max_iter: int = 100,
                 tol_residual: float = 1e-3,
                 lambda_damping: float = 1e-3,
                 fd_step: float = 1e-4,
                 n_random_restarts: int = 6,
                 random_seed: int = 0):
        """
        Args:
            chain           : chaine cinematique. Defaut : URDF SO-101.
            calib_path      : chemin calibration_follower.json (pour les
                              plages articulaires). Si None, contraintes
                              molles +/- pi.
            rotation_weight : alpha de ponderation rotation vs translation.
            max_iter        : nb max d'iterations Gauss-Newton.
            tol_residual    : seuil de convergence sur || r ||.
            lambda_damping  : facteur Levenberg-Marquardt (stabilite numerique).
            fd_step         : pas pour le Jacobien numerique (rad).
            n_random_restarts : nb d'essais avec q_init aleatoires (en plus du
                              q_init fourni). Permet d'echapper aux minima
                              locaux. 0 = pas de restart.
            random_seed     : graine RNG (reproductibilite).
        """
        self.chain = chain or KinematicChain()
        self.rotation_weight = float(rotation_weight)
        self.max_iter = int(max_iter)
        self.tol_residual = float(tol_residual)
        self.lambda_damping = float(lambda_damping)
        self.fd_step = float(fd_step)
        self.n_random_restarts = int(n_random_restarts)
        self._rng = np.random.default_rng(random_seed)
        self._restart_seed = int(random_seed)
        self.joints = list(self.chain.actuated)
        self.n_dof = len(self.joints)

        # Plages articulaires depuis calibration moteur
        self.joint_limits: dict[str, tuple[float, float]] = {}
        calib_path = calib_path or (REPO / "configs" / "calibration_follower.json")
        if calib_path.exists():
            calib = load_motor_calibration(calib_path)
            # Conversion plage raw -> radians.
            # LeRobot motors_bus.py:858 : angle_deg = (raw - mid) * 360 / 4095
            # donc 1 count = 360/4095 deg = 2*pi/4095 rad
            #
            # MARGE DE SECURITE : on retire 3 deg = 0.052 rad de chaque cote.
            # Eviter d'envoyer aux butees physiques pour 2 raisons :
            #   1. Le motor_controller a un seuil interne strict, et un
            #      angle pile sur la butee + arrondi flottant peut declencher
            #      un depassement de quelques counts.
            #   2. Les servos s'usent plus vite si on les pousse aux butees.
            SAFETY_MARGIN_RAD = np.radians(3.0)
            for j in self.joints:
                if j in calib:
                    c = calib[j]
                    span_rad = (c["range_max"] - c["range_min"]) * 2.0 * np.pi / 4095.0
                    half = span_rad / 2.0
                    half_safe = max(0.0, half - SAFETY_MARGIN_RAD)
                    self.joint_limits[j] = (-half_safe, +half_safe)

        # Defaut si pas de calibration : +/- pi
        for j in self.joints:
            self.joint_limits.setdefault(j, (-np.pi, +np.pi))

    # ----- API principale -------------------------------------------------

    def solve(self,
              T_target: np.ndarray,
              q_init: Optional[dict[str, float] | np.ndarray] = None,
              ) -> IKResult:
        """Resout l'IK pour la pose cible avec random restarts.

        Strategie : essaie d'abord avec q_init (ou zero), puis
        `n_random_restarts` essais avec configurations aleatoires dans les
        plages articulaires. Retourne la MEILLEURE solution trouvee.

        Args:
            T_target : matrice 4x4 SE(3) de la pose pince desiree (repere base).
            q_init   : configuration initiale. Dict {joint: rad}, ndarray (5,),
                       ou None (-> zero).

        Returns:
            IKResult avec angles solution + diagnostic.
        """
        if T_target.shape != (4, 4):
            raise ValueError(f"T_target doit etre 4x4, recu {T_target.shape}")

        # RNG RE-SEEDE A CHAQUE APPEL : les restarts aleatoires sont identiques
        # d'un run a l'autre pour une meme pose -> mouvement REPRODUCTIBLE (fini
        # le "meme situation, comportement different").
        rng = np.random.default_rng(self._restart_seed)
        starts = []
        if q_init is not None:
            starts.append(self._to_vector(q_init))
        else:
            starts.append(np.zeros(self.n_dof))
        # "smart init" : devine selon la position cible
        starts.append(self._smart_init(T_target))
        # Restarts aleatoires (deterministes grace au re-seed)
        for _ in range(self.n_random_restarts):
            q_rand = np.array([rng.uniform(lo, hi)
                               for (lo, hi) in self.joint_limits.values()])
            starts.append(q_rand)

        # On resout TOUS les departs et on collecte les solutions convergees
        # (plus de "premiere convergee" : c'etait elle qui pouvait etre
        # enroulee selon le hasard).
        converged: list[IKResult] = []
        best_any: Optional[IKResult] = None
        for q0 in starts:
            result = self._solve_once(T_target, q0)
            if best_any is None or result.residual_norm < best_any.residual_norm:
                best_any = result
            if result.converged:
                converged.append(result)

        if converged:
            # Parmi les convergees, on garde la PLUS PROCHE de q_init
            # (continuite articulaire -> mouvement lisse, pas de saut ni de
            # poignet retourne). Si q_init absent : reference = smart_init.
            ref = (self._to_vector(q_init) if q_init is not None
                   else self._smart_init(T_target))
            return min(converged, key=lambda r: float(np.linalg.norm(
                self._to_vector(r.joint_angles_rad) - ref)))

        return best_any  # rien n'a converge -> meilleure approche trouvee

    def _smart_init(self, T_target: np.ndarray) -> np.ndarray:
        """Heuristique : devine un q_init plausible selon la position cible.

        Pour le SO-101 :
          - shoulder_pan : pointe vers la cible (angle atan2(y, x))
          - shoulder_lift : 0 si cible au niveau bras, negatif si cible plus haute
          - elbow_flex : positif pour replier vers cible proche
          - wrist_flex : depend de l'orientation desiree
          - wrist_roll : 0 par defaut

        Ne pretend pas etre exact ; juste un meilleur point de depart que zero
        pour les poses top-down typiques.
        """
        x, y, z = T_target[:3, 3]
        # shoulder_pan : oriente le bras vers la cible
        pan = np.arctan2(y, x)
        # Pour une pose top-down : pince vers le bas, donc l'effecteur doit
        # etre au-dessus de la cible et la pince repliee de 90deg vers le bas
        # On suppose ~0.3m de portee maximale
        dist = np.sqrt(x * x + y * y)
        if dist < 0.30:
            # Cible proche : bras replie
            lift = -0.4   # epaule un peu vers le haut
            elbow = +0.7  # coude replie
            wflex = -0.6  # poignet vers le bas pour pince verticale
        else:
            # Cible lointaine : bras plus tendu
            lift = -0.2
            elbow = +0.4
            wflex = -0.4
        # wrist_roll : aligner sur le yaw demande par la pose cible. Le
        # shoulder_pan ET le wrist_roll tournent autour de la verticale, donc
        # wrist_roll ~= yaw_cible - pan. La pince etant SYMETRIQUE, yaw et
        # yaw+-180deg sont la MEME prise : on prend le representant le plus
        # proche du neutre (dans [-pi/2, pi/2]). Sans ca, l'IK partait de
        # wrist_roll=0 et convergeait parfois vers la solution RETOURNEE
        # (+-180deg) -> "tete a l'envers" intermittente (cf observations robot).
        yaw_target = float(np.arctan2(T_target[1, 0], T_target[0, 0]))
        wrist_roll = yaw_target - pan
        while wrist_roll > np.pi / 2:
            wrist_roll -= np.pi
        while wrist_roll < -np.pi / 2:
            wrist_roll += np.pi
        q_smart = np.array([pan, lift, elbow, wflex, wrist_roll])
        return self._clip_to_limits(q_smart)

    def _solve_once(self, T_target: np.ndarray, q_init: np.ndarray) -> IKResult:
        """Une seule passe Gauss-Newton/LM (sans restart)."""
        q = q_init.copy()
        # Damping local (ne modifie pas l'attribut partage entre restarts)
        damping = self.lambda_damping

        r = self._residual(q, T_target)
        prev_norm = np.linalg.norm(r)

        if prev_norm < self.tol_residual:
            return self._make_result(q, r, 0, converged=True,
                                      message="Pose initiale deja dans la tolerance")

        for it in range(self.max_iter):
            J = self._jacobian(q, T_target)
            JtJ = J.T @ J + damping * np.eye(self.n_dof)
            try:
                delta_q = -np.linalg.solve(JtJ, J.T @ r)
            except np.linalg.LinAlgError:
                return self._make_result(q, r, it, converged=False,
                                          message="Jacobien singulier")
            step_norm = np.linalg.norm(delta_q)
            if step_norm > 0.5:
                delta_q = delta_q * (0.5 / step_norm)
            q_new = self._clip_to_limits(q + delta_q)
            r_new = self._residual(q_new, T_target)
            r_norm_new = np.linalg.norm(r_new)
            if r_norm_new < prev_norm:
                q = q_new; r = r_new; prev_norm = r_norm_new
                damping = max(damping * 0.7, 1e-6)
                if r_norm_new < self.tol_residual:
                    return self._make_result(q, r, it + 1, converged=True,
                                              message=f"Converge en {it+1} iter")
            else:
                damping = min(damping * 2.0, 1.0)

        return self._make_result(q, r, self.max_iter, converged=False,
                                  message=f"Non-converge (residu {prev_norm:.4f})")

    def solve_grasp_pose(self, grasp_pose,
                         q_init: Optional[dict[str, float]] = None,
                         persist_choice: bool = True,
                         lock_orientation: bool = False,
                         ) -> tuple[IKResult, IKResult, IKResult]:
        """Resout l'IK pour les 3 poses d'un GraspPose (approach/grasp/retract).

        Strategie : chaque solve utilise la solution precedente comme point de
        depart (continuite articulaire, evite les sauts entre approach et grasp).

        PERSISTANCE DU CHOIX D'ORIENTATION (persist_choice=True) : si la
        variante retournee de 180deg est choisie, les matrices de grasp_pose
        sont REECRITES avec cette orientation. CONTRAT : apres l'appel,
        FK(solution) == grasp_pose.T_base_gripper_*.

        LOCK D'ORIENTATION (lock_orientation=True) : on NE re-explore PAS la
        symetrie 180deg ; on resout les matrices TELLES QUELLES. A utiliser a
        TOUS les re-solves (apres refinement cam_2, au retry) une fois
        l'orientation engagee : sinon le cout (penalite de non-convergence
        ASYMETRIQUE entre A et B en top-down sous-actionne) pouvait faire
        BASCULER le choix d'un re-solve a l'autre -> demi-tour 180deg du poignet
        (essai 10 du 2026-06-12 : retry -153deg -> +18deg). Le 1er solve garde
        lock_orientation=False (il choisit l'orientation), les suivants True.

        Returns:
            (result_approach, result_grasp, result_retract)
        """
        from src.planning.grasp import GraspPose  # import local pour eviter cycle
        if not isinstance(grasp_pose, GraspPose):
            raise TypeError(f"Attendu GraspPose, recu {type(grasp_pose).__name__}")

        # LOCK : orientation deja engagee -> resoudre les matrices telles quelles
        # (les corrections cam_2 ne touchent que la translation, donc l'orientation
        # persistee reste valide). Stabilite garantie entre re-solves.
        if lock_orientation:
            ra = self.solve(grasp_pose.T_base_gripper_approach, q_init=q_init)
            rg = self.solve(grasp_pose.T_base_gripper_grasp,
                            q_init=ra.joint_angles_rad or None)
            rr = self.solve(grasp_pose.T_base_gripper_retract,
                            q_init=rg.joint_angles_rad or None)
            return ra, rg, rr

        # === Symetrie 180deg de la pince ===
        # Une prise a l'orientation R est IDENTIQUE a R tournee de 180deg autour
        # de l'axe d'approche (la pince ferme sur la meme ligne). On resout les
        # DEUX orientations et on garde la config la plus NATURELLE (wrist_roll
        # proche du neutre). Sans ca, l'IK gardait la 1ere solution convergee
        # depuis q_init, parfois "enroulee" selon l'orientation de l'objet ->
        # chemins interminables + tete a l'envers (diagnostic Maxence).
        Rz180 = np.diag([-1.0, -1.0, 1.0])

        # La pince est ASYMETRIQUE (doigt fixe / doigt mobile) : retourner
        # l'orientation de 180deg met le doigt fixe DE L'AUTRE COTE de l'objet.
        # L'offset lateral A2 (qui plaque l'objet contre le doigt fixe) est
        # deja integre dans la translation -> la variante retournee doit le
        # MIROITER (t' = t - 2*offset) pour rester plaquee contre le doigt fixe.
        off = (grasp_pose.meta or {}).get("offset_base_xy_mm")
        d_off = np.zeros(3)
        if off is not None:
            d_off = np.array([float(off[0]) / 1000.0, float(off[1]) / 1000.0, 0.0])

        def _flip(T):
            T2 = np.array(T, dtype=np.float64, copy=True)
            T2[:3, :3] = T2[:3, :3] @ Rz180
            T2[:3, 3] = T2[:3, 3] - 2.0 * d_off
            return T2

        def _solve_trio(T_app, T_grp, T_ret):
            ra = self.solve(T_app, q_init=q_init)
            rg = self.solve(T_grp, q_init=ra.joint_angles_rad or None)
            rr = self.solve(T_ret, q_init=rg.joint_angles_rad or None)
            return (ra, rg, rr)

        T_app_b = _flip(grasp_pose.T_base_gripper_approach)
        T_grp_b = _flip(grasp_pose.T_base_gripper_grasp)
        T_ret_b = _flip(grasp_pose.T_base_gripper_retract)

        trio_a = _solve_trio(grasp_pose.T_base_gripper_approach,
                             grasp_pose.T_base_gripper_grasp,
                             grasp_pose.T_base_gripper_retract)
        trio_b = _solve_trio(T_app_b, T_grp_b, T_ret_b)

        # Cout = poignet naturel (|wrist_roll|) + CONTINUITE avec q_init.
        # La continuite empeche de basculer entre A et B d'un re-solve a
        # l'autre (apres correction cam_2) -> sinon le bras fait un "tour" a
        # chaque bascule (les "3 tours" observes). Au 1er calcul (q_init
        # lointain), c'est le poignet qui tranche -> orientation naturelle ;
        # aux re-solves (q_init = pose courante), la continuite fait RESTER sur
        # l'orientation deja prise.
        q0vec = self._to_vector(q_init) if q_init is not None else None

        wr_hi = self.joint_limits.get("wrist_roll", (-np.pi, np.pi))[1]

        def _cost(trio):
            rg = trio[1]
            if not rg.joint_angles_rad:
                return 1e9
            wr = abs(rg.joint_angles_rad.get("wrist_roll", 0.0))
            pen = sum(0.0 if r.converged else 5.0 for r in trio)
            cont = (float(np.linalg.norm(
                self._to_vector(rg.joint_angles_rad) - q0vec))
                if q0vec is not None else 0.0)
            # Penalite forte si le poignet frole sa butee (une orientation collee
            # a +/-164deg risque de clipper a un re-solve apres correction).
            near_limit = 3.0 if wr > wr_hi - np.radians(12) else 0.0
            # CONTINUITE DOMINANTE : les 2 orientations (symetrie 180deg de la
            # pince) saisissent l'objet de facon IDENTIQUE, donc on prend celle
            # qui BOUGE LE MOINS le poignet depuis la pose courante (anti rotation
            # ~90deg inutile sur les objets couches, signale par Maxence). |wrist|
            # ne reste qu'un faible tiebreaker.
            return pen + near_limit + 1.5 * cont + 0.25 * wr

        flipped = _cost(trio_b) < _cost(trio_a)
        chosen = trio_b if flipped else trio_a

        if flipped and persist_choice:
            grasp_pose.T_base_gripper_approach = T_app_b
            grasp_pose.T_base_gripper_grasp = T_grp_b
            grasp_pose.T_base_gripper_retract = T_ret_b
            if grasp_pose.meta is not None:
                if off is not None:
                    grasp_pose.meta["offset_base_xy_mm"] = (-float(off[0]),
                                                            -float(off[1]))
                grasp_pose.meta["yaw_rad"] = float(np.arctan2(
                    T_grp_b[1, 0], T_grp_b[0, 0]))
                grasp_pose.meta["flipped_180"] = not grasp_pose.meta.get(
                    "flipped_180", False)

        return chosen[0], chosen[1], chosen[2]

    def solve_grasp_pose_free_yaw(self, grasp_pose,
                                  q_init: Optional[dict[str, float]] = None,
                                  ) -> tuple[IKResult, IKResult, IKResult]:
        """Resout un GraspPose a YAW LIBRE (objet DEBOUT, empreinte ronde).

        Pour un objet rond, toute orientation de prise grippe de la meme facon.
        On BALAYE le yaw et on retient celui dont la solution MINIMISE le
        mouvement articulaire depuis q_init (continuite) -> le poignet reste pres
        de la pose de depart au lieu de tourner ~90deg pour rien (cylindres
        debout, essais 1,13,14 du 2026-06-12). On vise le CENTRE de l'objet
        (offset lateral A2 sans objet pour un rond) et on PERSISTE l'orientation
        choisie dans grasp_pose -> les re-solves ulterieurs (lock_orientation)
        la conservent.

        Returns: (approach, grasp, retract).
        """
        from src.planning.grasp import GraspPose, _rotation_top_down, _se3
        if not isinstance(grasp_pose, GraspPose):
            raise TypeError(f"Attendu GraspPose, recu {type(grasp_pose).__name__}")

        # Centre objet + hauteurs depuis la pose courante (on enleve l'offset
        # lateral : inutile/non oriente pour un objet rond).
        T_g = grasp_pose.T_base_gripper_grasp
        z_grasp = float(T_g[2, 3])
        meta = grasp_pose.meta or {}
        cxy = meta.get("object_center_xy_m")
        if cxy is not None:
            cx, cy = float(cxy[0]), float(cxy[1])
        else:
            off = meta.get("offset_base_xy_mm", (0.0, 0.0))
            cx = float(T_g[0, 3]) - float(off[0]) / 1000.0
            cy = float(T_g[1, 3]) - float(off[1]) / 1000.0
        h_app = float(grasp_pose.T_base_gripper_approach[2, 3]) - z_grasp
        h_ret = float(grasp_pose.T_base_gripper_retract[2, 3]) - z_grasp
        q0 = self._to_vector(q_init) if q_init is not None else None

        best = None  # (cost, yaw, trio, mats)
        for yaw_deg in range(-180, 180, 15):
            yaw = float(np.radians(yaw_deg))
            R = _rotation_top_down(yaw)
            T_app = _se3(R, [cx, cy, z_grasp + h_app])
            T_grp = _se3(R, [cx, cy, z_grasp])
            T_ret = _se3(R, [cx, cy, z_grasp + h_ret])
            ra = self.solve(T_app, q_init=q_init)
            rg = self.solve(T_grp, q_init=ra.joint_angles_rad or None)
            if not rg.joint_angles_rad or rg.translation_err_mm > 12.0:
                continue
            rr = self.solve(T_ret, q_init=rg.joint_angles_rad or None)
            # Cout = mouvement articulaire (continuite avec la pose courante)
            # -> minimise la rotation du poignet ET du reste du bras.
            if q0 is not None:
                cost = float(np.linalg.norm(self._to_vector(rg.joint_angles_rad) - q0))
            else:
                cost = abs(rg.joint_angles_rad.get("wrist_roll", 0.0))
            if best is None or cost < best[0]:
                best = (cost, yaw, (ra, rg, rr), (T_app, T_grp, T_ret))

        if best is None:
            # Aucun yaw n'atteint la position : repli sur le solveur standard.
            return self.solve_grasp_pose(grasp_pose, q_init=q_init)

        _, yaw, trio, mats = best
        # PERSISTE l'orientation choisie (offset lateral neutralise pour un rond).
        grasp_pose.T_base_gripper_approach, grasp_pose.T_base_gripper_grasp, \
            grasp_pose.T_base_gripper_retract = mats
        if grasp_pose.meta is not None:
            grasp_pose.meta["yaw_rad"] = yaw
            grasp_pose.meta["offset_base_xy_mm"] = (0.0, 0.0)
            grasp_pose.meta["yaw_committed_deg"] = float(np.degrees(yaw))
        return trio

    def solve_topdown_free_yaw(self, position_xyz, q_init=None):
        """Resout une pose pince-vers-le-bas a une POSITION donnee, YAW LIBRE.

        Cherche l'orientation (rotation autour de la verticale) qui donne la
        config la plus NATURELLE (wrist_roll proche du neutre) parmi celles qui
        atteignent la position. Pour la DEPOSE : seule la position compte (la
        pince ouvre, l'objet tombe), donc inutile de forcer un yaw qui retourne
        le poignet (contorsion / tete a l'envers a la boite lointaine).

        Returns: (IKResult, yaw_rad_choisi).
        """
        from src.planning.grasp import _rotation_top_down, _se3
        best = None  # (cost_wrist, result, yaw_rad)
        for yaw_deg in range(-180, 180, 15):
            yaw = float(np.radians(yaw_deg))
            r = self.solve(_se3(_rotation_top_down(yaw), position_xyz), q_init=q_init)
            if r.translation_err_mm < 15.0:  # position atteinte
                cost = abs(r.joint_angles_rad.get("wrist_roll", 0.0))
                if best is None or cost < best[0]:
                    best = (cost, r, yaw)
        if best is not None:
            return best[1], best[2]
        # Aucune orientation n'atteint la position : fallback yaw=0
        return self.solve(_se3(_rotation_top_down(0.0), position_xyz),
                          q_init=q_init), 0.0

    # ----- helpers internes -----------------------------------------------

    def _to_vector(self, q) -> np.ndarray:
        if isinstance(q, dict):
            return np.array([q.get(j, 0.0) for j in self.joints], dtype=np.float64)
        return np.asarray(q, dtype=np.float64).flatten()

    def _to_dict(self, q_vec: np.ndarray) -> dict[str, float]:
        return {j: float(q_vec[i]) for i, j in enumerate(self.joints)}

    def _residual(self, q_vec: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """Residu 6D : (dt en m) puis (alpha * dr en rad)."""
        T_cur = self.chain.fk(self._to_dict(q_vec))
        # Translation
        dt = T_cur[:3, 3] - T_target[:3, 3]
        # Rotation : err = R_cur.T @ R_target ; on convertit en Rodrigues
        R_err = T_cur[:3, :3].T @ T_target[:3, :3]
        rvec, _ = cv2.Rodrigues(R_err)
        dr = rvec.flatten()
        return np.concatenate([dt, self.rotation_weight * dr])

    def _jacobian(self, q_vec: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """Jacobien numerique : J[:, k] = dr/dq_k par differences finies centrees."""
        J = np.zeros((6, self.n_dof))
        for k in range(self.n_dof):
            q_plus = q_vec.copy(); q_plus[k] += self.fd_step
            q_minus = q_vec.copy(); q_minus[k] -= self.fd_step
            r_plus = self._residual(q_plus, T_target)
            r_minus = self._residual(q_minus, T_target)
            J[:, k] = (r_plus - r_minus) / (2 * self.fd_step)
        return J

    def _clip_to_limits(self, q_vec: np.ndarray) -> np.ndarray:
        clipped = q_vec.copy()
        for k, j in enumerate(self.joints):
            lo, hi = self.joint_limits[j]
            clipped[k] = min(hi, max(lo, q_vec[k]))
        return clipped

    def _make_result(self, q_vec: np.ndarray, r: np.ndarray, n_iter: int,
                     converged: bool, message: str = "") -> IKResult:
        # Decompose le residu en composantes lisibles
        dt = r[:3]
        dr = r[3:] / self.rotation_weight  # retire la ponderation
        t_err_mm = float(np.linalg.norm(dt) * 1000.0)
        r_err_deg = float(np.degrees(np.linalg.norm(dr)))
        return IKResult(
            joint_angles_rad=self._to_dict(q_vec),
            converged=converged,
            residual_norm=float(np.linalg.norm(r)),
            translation_err_mm=t_err_mm,
            rotation_err_deg=r_err_deg,
            n_iterations=n_iter,
            message=message,
        )


# ============================================================
# Self-tests (lance avec : python -m src.control.ik)
# ============================================================
if __name__ == "__main__":
    print("Tests ik.py")
    print()

    solver = IKSolver()
    print(f"  {solver.n_dof} DDL : {solver.joints}")
    print(f"  Plages articulaires (rad) :")
    for j, (lo, hi) in solver.joint_limits.items():
        print(f"    {j:<15} [{lo:+6.3f}, {hi:+6.3f}]")
    print()

    # ========================================================
    # Test 1 : FK -> IK roundtrip (cas le plus simple)
    # On choisit une config arbitraire q*, on calcule FK(q*) -> T,
    # on lance IK(T), on doit retomber sur q* (ou une config equivalente).
    # ========================================================
    rng = np.random.default_rng(42)
    n_ok = 0; n_total = 5
    max_t_err = 0.0; max_r_err = 0.0
    for trial in range(n_total):
        q_true_vec = np.array([
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
        ])
        q_true_dict = {j: float(q_true_vec[i]) for i, j in enumerate(ARM_JOINTS)}
        T_target = solver.chain.fk(q_true_dict)
        # IK avec q_init = zero (deliberement different de q_true)
        result = solver.solve(T_target, q_init=None)
        if result.converged and result.translation_err_mm < 1.0 and result.rotation_err_deg < 1.0:
            n_ok += 1
        max_t_err = max(max_t_err, result.translation_err_mm)
        max_r_err = max(max_r_err, result.rotation_err_deg)
    print(f"  [{n_ok}/{n_total}] FK->IK roundtrip : "
          f"erreur max trans {max_t_err:.3f} mm, rot {max_r_err:.3f} deg")
    assert n_ok >= n_total - 1, f"Trop d'echecs : {n_ok}/{n_total}"
    print()

    # ========================================================
    # Test 2 : pose tres simple (config zero) -> doit converger en 1-2 iter
    # ========================================================
    T_zero = solver.chain.fk({j: 0.0 for j in ARM_JOINTS})
    result = solver.solve(T_zero, q_init=None)
    assert result.converged, f"Devrait converger : {result.message}"
    # NB : depuis la selection par CONTINUITE (fc51b09), le resultat retourne
    # peut venir d'un restart (plus d'iterations que le depart zero). Le
    # critere pertinent est la convergence + la precision, pas le nb d'iter.
    assert result.translation_err_mm < 2.0 and result.rotation_err_deg < 2.0, \
        f"Pose zero imprecise : {result.translation_err_mm:.2f} mm / {result.rotation_err_deg:.2f} deg"
    print(f"  [OK] Pose zero : {result.n_iterations} iter, "
          f"err {result.translation_err_mm:.3f} mm / {result.rotation_err_deg:.3f} deg")
    print()

    # ========================================================
    # Test 3 : pose top-down a une position ATTEIGNABLE
    # La config zero donne l'effecteur a (39.1, 0, 22.7) cm pince horizontale.
    # Une pose top-down (pince verticale) au-dessus de la table demande de
    # "replier" le bras, ce qui reduit la portee. On choisit une cible
    # proche pour rester dans le workspace : (20cm devant, 10cm de haut).
    # ========================================================
    from src.planning.grasp import _rotation_top_down, _se3
    T_topdown = _se3(_rotation_top_down(0.0), [0.20, 0.0, 0.10])
    result = solver.solve(T_topdown, q_init=None)
    print(f"  Pose top-down (20cm devant, pince vers bas, z=10cm) : "
          f"{result.n_iterations} iter, "
          f"err {result.translation_err_mm:.2f} mm / {result.rotation_err_deg:.2f} deg")
    if result.converged:
        q_str = ", ".join(f"{j}={np.degrees(result.joint_angles_rad[j]):+.1f}deg"
                          for j in ARM_JOINTS)
        print(f"       Angles : {q_str}")
        print(f"  [OK] Pose top-down resolue")
    else:
        # Le SO-101 est sous-actionne (5 DDL pour SE(3)) ; certaines orientations
        # ne sont pas exactement atteignables. L'IK retourne la meilleure
        # approximation.
        print(f"  [INFO] Pose top-down non-converge exactement (sous-actuation 5/6 DDL) "
              f"-- residu position {result.translation_err_mm:.1f} mm")
    print()

    # ========================================================
    # Test 4 : pose impossible (hors workspace) -> non-converge mais ne plante pas
    # ========================================================
    T_far = _se3(_rotation_top_down(0.0), [2.0, 0.0, 0.0])  # 2m devant : impossible
    result = solver.solve(T_far, q_init=None)
    print(f"  Pose impossible (2m) : converged={result.converged}, "
          f"err {result.translation_err_mm:.0f} mm")
    assert not result.converged, "Devrait NE PAS converger pour pose hors workspace"
    print(f"  [OK] Pose hors workspace gerée gracieusement")
    print()

    # ========================================================
    # Test 5 : solve_grasp_pose (3 poses successives, continuite articulaire)
    # ========================================================
    from src.planning.grasp import TopDownGrasp
    from src.perception.scene import ObjectInstance
    obj = ObjectInstance(label="x", position_base_m=np.array([0.25, 0.0, 0.03]))
    grasp = TopDownGrasp().plan(obj)
    assert grasp is not None
    r_app, r_grp, r_ret = solver.solve_grasp_pose(grasp)
    print(f"  Grasp pose (3 poses) :")
    for name, r in [("approach", r_app), ("grasp", r_grp), ("retract", r_ret)]:
        print(f"    {name:<10} converged={r.converged}, "
              f"err {r.translation_err_mm:.1f} mm / {r.rotation_err_deg:.2f} deg")
    print()

    # ========================================================
    # Test 6 : CONTRAT de persistance du choix d'orientation.
    # Apres solve_grasp_pose, FK(solution grasp) doit correspondre a la
    # matrice ECRITE dans grasp_pose (meme si la variante 180deg a ete
    # choisie). C'est ce contrat qui garantit que mini-descente / re-IK
    # ulterieurs visent la meme orientation physique (anti demi-tours).
    # ========================================================
    for yaw_test_deg in (0.0, 75.0, -82.0):
        obj_y = ObjectInstance(label="x",
                               position_base_m=np.array([0.22, 0.08, 0.02]))
        gp_y = TopDownGrasp(grasp_lateral_offset_mm=8.0).plan(obj_y)
        from src.planning.grasp import _rotation_top_down as _rtd, _se3 as _s3
        yaw_t = np.radians(yaw_test_deg)
        for attr in ("T_base_gripper_approach", "T_base_gripper_grasp",
                     "T_base_gripper_retract"):
            T = getattr(gp_y, attr)
            T2 = _s3(_rtd(yaw_t), T[:3, 3])
            setattr(gp_y, attr, T2)
        ra_y, rg_y, rr_y = solver.solve_grasp_pose(
            gp_y, q_init={j: 0.0 for j in ARM_JOINTS})
        T_fk = solver.chain.fk(rg_y.joint_angles_rad)
        T_persisted = gp_y.T_base_gripper_grasp
        d_trans_mm = float(np.linalg.norm(T_fk[:3, 3] - T_persisted[:3, 3]) * 1000)
        # Angle entre les deux orientations (Rodrigues sur R_err)
        R_err6 = T_fk[:3, :3].T @ T_persisted[:3, :3]
        ang_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R_err6)[0])))
        flip_txt = " (retournee 180)" if gp_y.meta.get("flipped_180") else ""
        print(f"  Test 6 yaw={yaw_test_deg:+.0f}deg{flip_txt} : "
              f"FK vs matrice persistee = {d_trans_mm:.1f} mm / {ang_deg:.1f} deg")
        assert d_trans_mm < 25.0 and ang_deg < 20.0, \
            f"contrat persistance viole : {d_trans_mm:.1f} mm / {ang_deg:.1f} deg"
    print(f"  [OK] solve_grasp_pose : FK(solution) == matrices persistees")
    print()

    print("Tous les tests passent.")
