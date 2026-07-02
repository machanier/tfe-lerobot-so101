"""Generation de trajectoires articulaires lisses pour le SO-101.

Objectif : passer de la configuration articulaire courante q_0 a une
configuration cible q_f en limitant les a-coups (vitesse et acceleration
nulles aux extremites, profil lisse entre les deux).

Deux methodes sont disponibles :

  linear   : interpolation lineaire q(t) = (1-s) q_0 + s q_f, s dans [0, 1].
             Simple mais a vitesse constante, d'ou un saut a t=0 et t=T.
             Acceptable pour le SO-101 (mouvements lents, reducteur eleve).

  quintic  : polynome de degre 5 verifiant :
               q(0) = q_0,  q(T) = q_f
               v(0) = v(T) = 0   (vitesse nulle aux extremites)
               a(0) = a(T) = 0   (acceleration nulle, donc pas de jerk)
             Formulation standard en robotique, a coefficients fermes
             (cf Sciavicco & Siciliano).

Pour le pipeline pick-and-place, plusieurs trajectoires sont enchainees :
  q_courant -> q_approach -> q_grasp -> q_retract -> q_drop_above -> q_drop_release

Chaque sous-trajectoire est echantillonnee a une cadence dt (par defaut 30 Hz,
coherente avec le rythme des cameras).

Reference :
  Sciavicco & Siciliano 2000, "Modelling and Control of Robot Manipulators",
  chapitre 4 "Trajectory Planning".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class JointTrajectory:
    """Sequence temporelle de configurations articulaires.

    Attributs :
        joint_names : ordre des joints (par exemple ARM_JOINTS).
        timestamps  : (N,) instants en secondes depuis le debut.
        positions   : (N, len(joints)) angles en radians par instant.
        velocities  : (N, len(joints)) vitesses (rad/s), optionnel.
        gripper_pct : (N,) ouverture de la pince (0=ferme, 100=ouvert), optionnel.
                      Si fourni, il est envoye en meme temps que les positions.
        meta        : metadonnees (strategie, source, etc.).
    """

    joint_names: list[str]
    timestamps: np.ndarray
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    gripper_pct: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamps)
        if self.positions.shape != (n, len(self.joint_names)):
            raise ValueError(
                f"positions doit avoir la forme ({n}, {len(self.joint_names)}), "
                f"recu {self.positions.shape}"
            )
        if self.velocities is not None and self.velocities.shape != self.positions.shape:
            raise ValueError("velocities doit avoir la meme forme que positions")
        if self.gripper_pct is not None and self.gripper_pct.shape != (n,):
            raise ValueError(f"gripper_pct doit avoir la forme ({n},)")

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def position_at(self, i: int) -> dict[str, float]:
        """Renvoie {joint: angle_rad} a l'instant i."""
        return {n: float(self.positions[i, k]) for k, n in enumerate(self.joint_names)}

    def __len__(self):
        return len(self.timestamps)


# ============================================================
# Generateurs de trajectoire
# ============================================================


def linear_trajectory(q_start: dict[str, float],
                      q_end: dict[str, float],
                      duration_s: float,
                      dt_s: float = 1.0 / 30.0,
                      joint_names: Optional[list[str]] = None,
                      gripper_start: Optional[float] = None,
                      gripper_end: Optional[float] = None) -> JointTrajectory:
    """Interpolation lineaire entre q_start et q_end."""
    if joint_names is None:
        joint_names = list(q_start.keys())
    n = max(2, int(np.ceil(duration_s / dt_s)) + 1)
    ts = np.linspace(0.0, duration_s, n)
    q0 = np.array([q_start[j] for j in joint_names], dtype=np.float64)
    q1 = np.array([q_end[j] for j in joint_names], dtype=np.float64)
    positions = np.outer(1.0 - ts / duration_s, q0) + np.outer(ts / duration_s, q1)
    # Vitesse constante egale a (q1 - q0) / duration
    velocities = np.tile((q1 - q0) / duration_s, (n, 1))
    # Convention : vitesse nulle aux extremites (le profil lineaire presente
    # en realite un saut a t=0 et t=T)
    velocities[0] = 0.0
    velocities[-1] = 0.0
    # Pince interpolee de la meme maniere
    gripper = None
    if gripper_start is not None and gripper_end is not None:
        gripper = (1.0 - ts / duration_s) * gripper_start + (ts / duration_s) * gripper_end
    return JointTrajectory(
        joint_names=joint_names,
        timestamps=ts,
        positions=positions,
        velocities=velocities,
        gripper_pct=gripper,
        meta={"profile": "linear", "duration_s": duration_s},
    )


def quintic_trajectory(q_start: dict[str, float],
                       q_end: dict[str, float],
                       duration_s: float,
                       dt_s: float = 1.0 / 30.0,
                       joint_names: Optional[list[str]] = None,
                       gripper_start: Optional[float] = None,
                       gripper_end: Optional[float] = None) -> JointTrajectory:
    """Polynome quintique : vitesse et acceleration nulles aux extremites.

    Formule (Sciavicco & Siciliano 2000, eq. 4.4) :
      s(t) = 10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5
      q(t) = q_0 + s(t) * (q_f - q_0)

    Verifie : s(0)=0, s(T)=1, s'(0)=s'(T)=0, s''(0)=s''(T)=0.
    """
    if joint_names is None:
        joint_names = list(q_start.keys())
    n = max(2, int(np.ceil(duration_s / dt_s)) + 1)
    ts = np.linspace(0.0, duration_s, n)
    tau = ts / duration_s  # temps normalise dans [0, 1]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    # Derivee de s, utilisee pour les vitesses
    s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration_s

    q0 = np.array([q_start[j] for j in joint_names], dtype=np.float64)
    q1 = np.array([q_end[j] for j in joint_names], dtype=np.float64)
    dq = q1 - q0
    positions = q0[None, :] + s[:, None] * dq[None, :]
    velocities = s_dot[:, None] * dq[None, :]

    gripper = None
    if gripper_start is not None and gripper_end is not None:
        gripper = gripper_start + s * (gripper_end - gripper_start)

    return JointTrajectory(
        joint_names=joint_names,
        timestamps=ts,
        positions=positions,
        velocities=velocities,
        gripper_pct=gripper,
        meta={"profile": "quintic", "duration_s": duration_s},
    )


def chain_trajectories(trajectories: Iterable[JointTrajectory]) -> JointTrajectory:
    """Concatene plusieurs trajectoires bout a bout (les timestamps sont decales).

    Si certains segments possedent velocities ou gripper_pct et d'autres non,
    les segments manquants sont completes (velocities a zero, ouverture de pince
    maintenue a la derniere valeur connue). Sans ce completage, np.concatenate
    produirait des formes incoherentes et __post_init__ leverait une exception.
    """
    trajs = list(trajectories)
    if not trajs:
        raise ValueError("Aucune trajectoire a chainer.")
    joint_names = trajs[0].joint_names
    for t in trajs[1:]:
        if t.joint_names != joint_names:
            raise ValueError("Toutes les trajectoires doivent avoir le meme ordre des joints.")

    has_velocities = any(t.velocities is not None for t in trajs)
    has_gripper = any(t.gripper_pct is not None for t in trajs)

    ts_list = []; pos_list = []; vel_list = []; grip_list = []
    offset = 0.0
    last_gripper = 100.0  # ouverture par defaut si elle n'est pas connue
    for t in trajs:
        ts_list.append(t.timestamps + offset)
        pos_list.append(t.positions)
        if has_velocities:
            if t.velocities is not None:
                vel_list.append(t.velocities)
            else:
                # Segment sans vitesses : complete par des zeros
                vel_list.append(np.zeros_like(t.positions))
        if has_gripper:
            if t.gripper_pct is not None:
                grip_list.append(t.gripper_pct)
                last_gripper = float(t.gripper_pct[-1])
            else:
                # Segment sans pince : maintient la derniere ouverture connue
                grip_list.append(np.full(len(t.timestamps), last_gripper))
        offset += t.duration_s

    return JointTrajectory(
        joint_names=joint_names,
        timestamps=np.concatenate(ts_list),
        positions=np.concatenate(pos_list, axis=0),
        velocities=np.concatenate(vel_list, axis=0) if has_velocities else None,
        gripper_pct=np.concatenate(grip_list) if has_gripper else None,
        meta={"profile": "chained", "n_segments": len(trajs)},
    )


def estimate_duration_safe(q_start: dict[str, float], q_end: dict[str, float],
                            max_velocity_rad_s: float = 0.5) -> float:
    """Estime une duree raisonnable pour qu'aucun joint ne depasse max_velocity.

    Approche conservative : duration = deplacement_max / vitesse_max, avec une
    marge x1.5 pour absorber le profil quintique (vitesse de pic superieure a
    la vitesse moyenne).

    Args:
        max_velocity_rad_s : 0.5 rad/s (environ 30 deg/s), valeur prudente
            pour le SO-101.
    """
    joints = set(q_start.keys()) | set(q_end.keys())
    max_disp = 0.0
    for j in joints:
        d = abs(q_start.get(j, 0.0) - q_end.get(j, 0.0))
        if d > max_disp:
            max_disp = d
    return max(0.5, 1.5 * max_disp / max_velocity_rad_s)


# ============================================================
# Tests internes (lancer avec : python -m src.control.trajectory)
# ============================================================
if __name__ == "__main__":
    print("Tests trajectory.py")
    print()

    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    q0 = {j: 0.0 for j in joints}
    q1 = {j: 0.5 for j in joints}

    # 1. linear
    t_lin = linear_trajectory(q0, q1, duration_s=2.0, dt_s=0.1, joint_names=joints)
    assert len(t_lin) == 21  # 2.0 / 0.1 + 1
    assert np.allclose(t_lin.positions[0], 0.0)
    assert np.allclose(t_lin.positions[-1], 0.5)
    # mi-parcours attendu autour de 0.25
    mid = len(t_lin) // 2
    assert np.allclose(t_lin.positions[mid], 0.25, atol=0.05)
    print(f"  [OK] linear_trajectory : N={len(t_lin)}, duree={t_lin.duration_s}s, "
          f"mi-parcours={t_lin.positions[mid][0]:.3f}")

    # 2. quintic : vitesse 0 aux extremites
    t_q = quintic_trajectory(q0, q1, duration_s=2.0, dt_s=0.1, joint_names=joints)
    assert np.allclose(t_q.positions[0], 0.0)
    assert np.allclose(t_q.positions[-1], 0.5)
    assert np.allclose(t_q.velocities[0], 0.0, atol=1e-8), \
        f"vitesse t=0 attendue 0, recu {t_q.velocities[0]}"
    assert np.allclose(t_q.velocities[-1], 0.0, atol=1e-8)
    # Vitesse maximale au milieu (environ 1.5 * (q1-q0)/T = 1.5 * 0.5/2 = 0.375)
    v_max = np.max(np.abs(t_q.velocities))
    assert 0.30 < v_max < 0.50, f"v_max quintic attendu ~0.375, recu {v_max}"
    print(f"  [OK] quintic_trajectory : v_max={v_max:.3f} (~1.5x linear)")

    # 3. chain_trajectories
    t_chain = chain_trajectories([t_lin, t_q])
    assert len(t_chain) == len(t_lin) + len(t_q)
    assert np.isclose(t_chain.duration_s, t_lin.duration_s + t_q.duration_s)
    print(f"  [OK] chain_trajectories : {len(t_chain)} points, "
          f"duree {t_chain.duration_s}s")

    # 4. estimate_duration_safe
    dur = estimate_duration_safe(q0, q1, max_velocity_rad_s=0.5)
    # 0.5 rad / 0.5 rad/s = 1s, puis x1.5 = 1.5s
    assert 1.4 < dur < 1.6, f"duration attendue ~1.5s, recu {dur}"
    print(f"  [OK] estimate_duration_safe : {dur:.2f}s pour 0.5 rad a 0.5 rad/s")

    # 5. position_at
    p = t_q.position_at(0)
    assert p == {j: 0.0 for j in joints}
    print(f"  [OK] position_at")

    # 6. gripper interpolation
    t_g = quintic_trajectory(q0, q1, 2.0, 0.1, joints,
                              gripper_start=100.0, gripper_end=0.0)
    assert t_g.gripper_pct is not None
    assert np.isclose(t_g.gripper_pct[0], 100.0)
    assert np.isclose(t_g.gripper_pct[-1], 0.0)
    print(f"  [OK] gripper interpolation : {t_g.gripper_pct[0]:.0f} -> {t_g.gripper_pct[-1]:.0f}")

    print()
    print("Tous les tests passent.")
