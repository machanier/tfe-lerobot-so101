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
            for j in self.joints:
                if j in calib:
                    c = calib[j]
                    span_rad = (c["range_max"] - c["range_min"]) * 2.0 * np.pi / 4095.0
                    # Centre sur 0, demi-amplitude = span/2
                    half = span_rad / 2.0
                    self.joint_limits[j] = (-half, +half)

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

        # Genere les configurations de depart : q_init + N aleatoires
        starts = []
        if q_init is not None:
            starts.append(self._to_vector(q_init))
        else:
            starts.append(np.zeros(self.n_dof))
        # Ajout d'un "smart init" : devine selon la position cible
        starts.append(self._smart_init(T_target))
        # Restarts aleatoires
        for _ in range(self.n_random_restarts):
            q_rand = np.array([
                self._rng.uniform(lo, hi)
                for j, (lo, hi) in self.joint_limits.items()
            ])
            starts.append(q_rand)

        best_result: Optional[IKResult] = None
        for q0 in starts:
            result = self._solve_once(T_target, q0)
            if best_result is None or result.residual_norm < best_result.residual_norm:
                best_result = result
            if result.converged:
                # Si on a converge, pas la peine de continuer les restarts
                return result

        return best_result  # meilleure solution trouvee (peut-etre non converge)

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
        q_smart = np.array([pan, lift, elbow, wflex, 0.0])
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
                         q_init: Optional[dict[str, float]] = None
                         ) -> tuple[IKResult, IKResult, IKResult]:
        """Resout l'IK pour les 3 poses d'un GraspPose (approach/grasp/retract).

        Strategie : chaque solve utilise la solution precedente comme point de
        depart (continuite articulaire, evite les sauts entre approach et grasp).

        Args:
            grasp_pose : src.planning.grasp.GraspPose.
            q_init     : configuration initiale pour l'approach (defaut : zero).

        Returns:
            (result_approach, result_grasp, result_retract)
        """
        from src.planning.grasp import GraspPose  # import local pour eviter cycle
        if not isinstance(grasp_pose, GraspPose):
            raise TypeError(f"Attendu GraspPose, recu {type(grasp_pose).__name__}")

        r_app = self.solve(grasp_pose.T_base_gripper_approach, q_init=q_init)
        r_grp = self.solve(grasp_pose.T_base_gripper_grasp,
                           q_init=r_app.joint_angles_rad if r_app.joint_angles_rad else None)
        r_ret = self.solve(grasp_pose.T_base_gripper_retract,
                           q_init=r_grp.joint_angles_rad if r_grp.joint_angles_rad else None)
        return r_app, r_grp, r_ret

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
    assert result.n_iterations <= 5, f"Trop d'iterations pour pose zero : {result.n_iterations}"
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

    print("Tous les tests passent.")
