"""Momentum-based short-horizon state predictor and collision forecaster for the Franka cell.

The solver fuses two sources of state every control step:

* Arm proprioception: joint positions and velocities, the joint-space mass matrix,
  Coriolis and gravity terms, and the drive gains. These give the generalized
  momentum ``p = M(q) qd`` and let the solver roll the arm forward under the
  policy's commanded joint targets (action-conditioned) or under its current
  momentum alone (ballistic).
* Object history: the last ``K`` world positions of every tracked rigid object.
  A least-squares fit over that window gives each object's velocity, acceleration
  and linear momentum, which are then extrapolated with a physics prior (resting
  objects keep their momentum, free-flying objects follow gravity).

The predicted arm is represented by the cuMotion Franka collision spheres placed
with a vectorized forward-kinematics pass, so a whole horizon is evaluated in one
batch. Predicted spheres are tested against static boxes and the predicted object
boxes; the earliest penetration becomes an alert with a time-to-collision.

This module only depends on numpy so it can be tested outside Isaac Sim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


HALF_PI = math.pi / 2

# (child link, parent link, joint origin, joint type, axis) from the cuMotion Franka URDF.
_FRANKA_CHAIN = [
    ("panda_link1", "panda_link0", _origin((0, 0, 0.333)), "revolute", 0),
    ("panda_link2", "panda_link1", _origin((0, 0, 0), (-HALF_PI, 0, 0)), "revolute", 1),
    ("panda_link3", "panda_link2", _origin((0, -0.316, 0), (HALF_PI, 0, 0)), "revolute", 2),
    ("panda_link4", "panda_link3", _origin((0.0825, 0, 0), (HALF_PI, 0, 0)), "revolute", 3),
    ("panda_link5", "panda_link4", _origin((-0.0825, 0.384, 0), (-HALF_PI, 0, 0)), "revolute", 4),
    ("panda_link6", "panda_link5", _origin((0, 0, 0), (HALF_PI, 0, 0)), "revolute", 5),
    ("panda_link7", "panda_link6", _origin((0.088, 0, 0), (HALF_PI, 0, 0)), "revolute", 6),
    ("panda_hand", "panda_link7", _origin((0, 0, 0.107), (0, 0, -math.pi / 4)), "fixed", None),
    ("panda_leftfinger", "panda_hand", _origin((0, 0, 0.0584)), "prismatic+y", 7),
    ("panda_rightfinger", "panda_hand", _origin((0, 0, 0.0584)), "prismatic-y", 8),
    ("panda_leftfingertip", "panda_leftfinger", _origin((0, 0, 0.045), (0, 0, math.pi)), "fixed", None),
    ("panda_rightfingertip", "panda_rightfinger", _origin((0, 0, 0.045), (0, 0, math.pi)), "fixed", None),
]

# World-collision spheres from robot_configurations/franka/robot.xrdf: link -> [(center, radius)].
_FRANKA_SPHERES = {
    "panda_link1": [((0, -0.08, 0), 0.06), ((0, -0.03, 0), 0.06), ((0, 0, -0.12), 0.06), ((0, 0, -0.17), 0.06)],
    "panda_link2": [((0, 0, 0.03), 0.06), ((0, 0, 0.08), 0.06), ((0, -0.12, 0), 0.06), ((0, -0.17, 0), 0.06)],
    "panda_link3": [((0, 0, -0.06), 0.05), ((0, 0, -0.1), 0.06), ((0.08, 0.06, 0), 0.055), ((0.08, 0.02, 0), 0.055)],
    "panda_link4": [((0, 0, 0.02), 0.055), ((0, 0, 0.06), 0.055), ((-0.08, 0.095, 0), 0.06), ((-0.08, 0.06, 0), 0.055)],
    "panda_link5": [
        ((0, 0.055, 0), 0.06), ((0, 0.075, 0), 0.06), ((0, 0, -0.22), 0.06), ((0, 0.05, -0.18), 0.05),
        ((0.01, 0.08, -0.14), 0.025), ((0.01, 0.085, -0.11), 0.025), ((0.01, 0.09, -0.08), 0.025), ((0.01, 0.095, -0.05), 0.025),
        ((-0.01, 0.08, -0.14), 0.025), ((-0.01, 0.085, -0.11), 0.025), ((-0.01, 0.09, -0.08), 0.025), ((-0.01, 0.095, -0.05), 0.025),
    ],
    "panda_link6": [((0, 0, 0), 0.06), ((0.08, 0.03, 0), 0.06), ((0.08, -0.01, 0), 0.06)],
    "panda_link7": [((0, 0, 0.07), 0.05), ((0.02, 0.04, 0.08), 0.025), ((0.04, 0.02, 0.08), 0.025), ((0.04, 0.06, 0.085), 0.02), ((0.06, 0.04, 0.085), 0.02)],
    "panda_hand": [((0, y, z), r) for z, r in ((0.01, 0.028), (0.03, 0.026), (0.05, 0.024)) for y in (-0.075, -0.045, -0.015, 0.015, 0.045, 0.075)],
    "panda_leftfingertip": [((0, 0.0075, 0), 0.0108)],
    "panda_rightfingertip": [((0, -0.0075, 0), 0.0108)],
}

FRANKA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
# cspace acceleration limits from robot.xrdf (arm joints); the fingers get a generous bound.
FRANKA_ACCELERATION_LIMITS = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 20.0, 20.0)
GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_leftfingertip", "panda_rightfingertip")


class FrankaKinematics:
    """Batched forward kinematics and collision-sphere placement for the Franka arm."""

    def __init__(self) -> None:
        self.link_names = ["panda_link0"] + [c[0] for c in _FRANKA_CHAIN]
        self._index = {name: i for i, name in enumerate(self.link_names)}
        centers, radii, owners = [], [], []
        for link, spheres in _FRANKA_SPHERES.items():
            for center, radius in spheres:
                centers.append(center)
                radii.append(radius)
                owners.append(self._index[link])
        self.sphere_local = np.asarray(centers, dtype=np.float64)
        self.sphere_radius = np.asarray(radii, dtype=np.float64)
        self.sphere_link = np.asarray(owners, dtype=np.int64)
        self.sphere_link_name = [self.link_names[i] for i in owners]
        self.gripper_sphere = np.isin(self.sphere_link_name, GRIPPER_LINKS)

    def link_index(self, name: str) -> int:
        return self._index[name]

    def forward(self, q: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Link world transforms for a batch of joint vectors.

        Args:
            q: Joint positions (shape ``(B, 9)``) ordered as :data:`FRANKA_JOINTS`.
            base: World transform of ``panda_link0`` (shape ``(4, 4)``).

        Returns:
            Link transforms (shape ``(B, L, 4, 4)``) ordered as :attr:`link_names`.
        """
        q = np.atleast_2d(q)
        batch = q.shape[0]
        out = np.empty((batch, len(self.link_names), 4, 4))
        out[:, 0] = base
        for child, parent, origin, kind, j in _FRANKA_CHAIN:
            T = out[:, self._index[parent]] @ origin
            if kind == "revolute":
                c, s = np.cos(q[:, j]), np.sin(q[:, j])
                R = np.zeros((batch, 4, 4))
                R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
                R[:, 2, 2] = R[:, 3, 3] = 1.0
                T = T @ R
            elif kind.startswith("prismatic"):
                sign = 1.0 if kind.endswith("+y") else -1.0
                T = T.copy()
                T[:, :3, 3] += T[:, :3, 1] * (sign * q[:, j])[:, None]
            out[:, self._index[child]] = T
        return out

    def spheres(self, link_transforms: np.ndarray) -> np.ndarray:
        """World centers of the collision spheres (shape ``(B, S, 3)``)."""
        T = link_transforms[:, self.sphere_link]
        return np.einsum("bsij,sj->bsi", T[..., :3, :3], self.sphere_local) + T[..., :3, 3]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """Axis-aligned box obstacle."""

    name: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]


def sphere_box_distance(centers: np.ndarray, radii: np.ndarray, box_center: np.ndarray, box_half: np.ndarray) -> np.ndarray:
    """Signed sphere-surface to box distance, negative when penetrating (broadcasts over leading dims)."""
    d = np.abs(centers - box_center) - box_half
    outside = np.sqrt(np.sum(np.maximum(d, 0.0) ** 2, axis=-1))
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside - radii


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from WXYZ quaternions (shape ``(N, 4)`` -> ``(N, 3, 3)``)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


# ---------------------------------------------------------------------------
# Object momentum
# ---------------------------------------------------------------------------


class ObjectMomentumTracker:
    """Estimates velocity, acceleration and momentum of tracked objects from a window of past frames.

    Objects are oriented boxes; ``half_extents`` are in each object's own frame.
    """

    REST = 0
    FLIGHT = 1
    DRIVEN = 2

    def __init__(self, names: list[str], masses: np.ndarray, half_extents: np.ndarray, window: int = 8, floor_z: float = 0.0):
        self.names = list(names)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.half_extents = np.asarray(half_extents, dtype=np.float64)
        self.window = window
        self.floor_z = floor_z
        self._times: list[float] = []
        self._positions: list[np.ndarray] = []
        self.rotations = np.tile(np.eye(3), (len(names), 1, 1))
        self.velocity = np.zeros((len(names), 3))
        self.acceleration = np.zeros((len(names), 3))
        self.mode = np.full(len(names), self.REST)

    def forget(self, index: int) -> None:
        """Drop an object's history (e.g. after a teleport) so stale frames do not imply a huge velocity."""
        for p in self._positions:
            p[index] = np.nan

    def update(self, t: float, positions: np.ndarray, orientations: np.ndarray | None = None) -> None:
        self._times.append(t)
        self._positions.append(np.asarray(positions, dtype=np.float64).copy())
        if orientations is not None:
            self.rotations = quat_wxyz_to_matrix(np.asarray(orientations, dtype=np.float64))
        if len(self._times) > self.window:
            self._times.pop(0)
            self._positions.pop(0)
        self._fit()

    @property
    def positions(self) -> np.ndarray:
        return self._positions[-1]

    @property
    def momentum(self) -> np.ndarray:
        return self.masses[:, None] * self.velocity

    def _fit(self) -> None:
        n = len(self._times)
        if n < 3:
            self.velocity[:] = 0.0
            self.acceleration[:] = 0.0
            return
        tau = np.asarray(self._times) - self._times[-1]
        P = np.stack(self._positions)  # (K, N, 3)
        A = np.stack([np.ones_like(tau), tau, 0.5 * tau * tau], axis=1)
        finite = np.isfinite(P).all(axis=2)  # (K, N)
        full = finite.all(axis=0)
        v = np.zeros((P.shape[1], 3))
        a = np.zeros((P.shape[1], 3))
        if full.any():
            coeffs = np.linalg.lstsq(A, P[:, full].reshape(n, -1), rcond=None)[0].reshape(3, -1, 3)
            v[full], a[full] = coeffs[1], coeffs[2]
        # Objects with a partial (recently reset) history are fit on their finite frames only.
        for i in np.flatnonzero(~full & finite[-1]):
            rows = finite[:, i]
            if rows.sum() >= 3:
                c = np.linalg.lstsq(A[rows], P[rows, i], rcond=None)[0]
                v[i], a[i] = c[1], c[2]
            elif rows.sum() == 2:
                k0, k1 = np.flatnonzero(rows)
                v[i] = (P[k1, i] - P[k0, i]) / (tau[k1] - tau[k0])
        self.velocity, self.acceleration = v, a
        free_fall = np.abs(a[:, 2] + GRAVITY) < 0.25 * GRAVITY
        resting = (np.abs(v[:, 2]) < 0.03) & (np.abs(a[:, 2]) < 0.25 * GRAVITY)
        self.mode = np.where(free_fall, self.FLIGHT, np.where(resting, self.REST, self.DRIVEN))

    def predict(self, horizon: np.ndarray) -> np.ndarray:
        """Predicted positions at future offsets ``horizon`` seconds (shape ``(H, N, 3)``).

        Resting objects conserve horizontal momentum (belt transport), free-flying objects
        follow the gravity parabola until they reach the floor, and driven objects (being
        pushed or carried) extrapolate their fitted velocity.
        """
        p0 = self.positions
        v = self.velocity.copy()
        a = np.zeros_like(v)
        rest = self.mode == self.REST
        flight = self.mode == self.FLIGHT
        v[rest, 2] = 0.0
        a[flight] = (0.0, 0.0, -GRAVITY)
        t = horizon[:, None, None]
        pred = p0 + v * t + 0.5 * a * t * t
        floor = self.floor_z + self.half_extents[:, 2]
        pred[..., 2] = np.maximum(pred[..., 2], np.where(flight, floor, -np.inf))
        return np.nan_to_num(pred, nan=1e6)


# ---------------------------------------------------------------------------
# Arm momentum
# ---------------------------------------------------------------------------


@dataclass
class Proprioception:
    """Arm measurements for one control step (joint order :data:`FRANKA_JOINTS`)."""

    q: np.ndarray
    qd: np.ndarray
    mass_matrix: np.ndarray
    coriolis: np.ndarray
    gravity: np.ndarray
    base: np.ndarray


@dataclass
class DriveModel:
    """PhysX joint-drive parameters needed to predict the arm's response to a command."""

    stiffness: np.ndarray
    damping: np.ndarray
    max_effort: np.ndarray
    acceleration_drive: np.ndarray  # per joint: True if the drive output is an acceleration (inertia-scaled)
    gravity_enabled: bool


def extrapolate_target(q_target: np.ndarray, qd_target: np.ndarray, qdd_target: np.ndarray, horizon: np.ndarray,
                       accelerate_for: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Extend a streamed joint target over the horizon (shapes ``(H, J)``).

    A braking target (acceleration opposing velocity) decelerates to rest and stays there;
    an accelerating target keeps its acceleration for at most ``accelerate_for`` seconds,
    then cruises. Speed-ups are not extrapolated by default: the policy's first step out of
    rest looks like an enormous acceleration that it never sustains.
    """
    t = horizon[:, None]
    braking = qd_target * qdd_target < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(braking, -qd_target / qdd_target, accelerate_for)
    te = np.minimum(t, t_stop)
    q = q_target + qd_target * te + 0.5 * qdd_target * te * te
    qd = qd_target + qdd_target * te
    cruising = ~braking & (t > t_stop)
    q = q + np.where(cruising, qd * (t - t_stop), 0.0)
    qd = np.where(braking & (t >= t_stop), 0.0, qd)
    return q, qd


class ArmMomentumModel:
    """Rolls the arm's joint state forward from its generalized momentum."""

    def __init__(self, drive: DriveModel, dt: float):
        self.drive = drive
        self.dt = dt

    @staticmethod
    def generalized_momentum(proprio: Proprioception) -> np.ndarray:
        return proprio.mass_matrix @ proprio.qd

    @staticmethod
    def kinetic_energy(proprio: Proprioception) -> float:
        return 0.5 * float(proprio.qd @ proprio.mass_matrix @ proprio.qd)

    def ballistic(self, proprio: Proprioception, qdd: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Momentum-only rollout: keep the current joint acceleration (no knowledge of the next command)."""
        t = self.dt * np.arange(1, steps + 1)[:, None]
        q = proprio.q + proprio.qd * t + 0.5 * qdd * t * t
        return q, proprio.qd + qdd * t

    def commanded(self, proprio: Proprioception, q_targets: np.ndarray, qd_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Action-conditioned rollout through the joint drives (implicit PD, mass matrix frozen over the horizon).

        Args:
            proprio: Current arm measurements.
            q_targets: Drive position targets for each future step (shape ``(H, J)``).
            qd_targets: Drive velocity targets for each future step (shape ``(H, J)``).
        """
        M = proprio.mass_matrix
        Kp, Kd = self.drive.stiffness, self.drive.damping
        dt = self.dt
        accel = self.drive.acceleration_drive
        # PhysX acceleration drives scale gains by the joint's own inertia; force drives use them directly.
        Kp_eff = np.where(accel, Kp * np.diag(M), Kp)
        Kd_eff = np.where(accel, Kd * np.diag(M), Kd)
        bias = proprio.coriolis + (proprio.gravity if self.drive.gravity_enabled else 0.0)
        implicit = dt * Kd_eff + dt * dt * Kp_eff
        system_inv = np.linalg.inv(M + np.diag(implicit))
        limit = self.drive.max_effort
        q, qd = proprio.q.copy(), proprio.qd.copy()
        steps = q_targets.shape[0]
        qs, qds = np.empty((steps, q.size)), np.empty((steps, q.size))
        for k in range(steps):
            tau_explicit = Kp_eff * (q_targets[k] - q - dt * qd) + Kd_eff * (qd_targets[k] - qd)
            qdd = system_inv @ (tau_explicit - bias)
            tau = tau_explicit - implicit * qdd
            saturated = np.abs(tau) > limit
            if saturated.any():
                # Saturated joints apply their effort limit; only the rest keep the implicit drive terms.
                rhs = np.where(saturated, np.clip(tau, -limit, limit), tau_explicit) - bias
                qdd = np.linalg.solve(M + np.diag(np.where(saturated, 0.0, implicit)), rhs)
            qd = qd + dt * qdd
            q = q + dt * qd
            qs[k], qds[k] = q, qd
        return qs, qds


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """Earliest predicted contact within the horizon."""

    t: float
    time_to_collision: float
    model: str
    link: str
    obstacle: str
    distance: float
    link_speed: float
    arm_kinetic_energy: float
    point: tuple[float, float, float] = (math.nan, math.nan, math.nan)  # predicted contact location (world)


@dataclass
class Prediction:
    t: float
    q_commanded: np.ndarray | None
    q_ballistic: np.ndarray
    object_positions: np.ndarray
    hand_commanded: np.ndarray | None
    hand_ballistic: np.ndarray
    min_distance: float
    alert: Alert | None
    confirmed: bool  # a contact has been predicted for ``confirm_steps`` consecutive steps
    momentum: np.ndarray
    kinetic_energy: float
    momentum_residual: float
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ContactRules:
    """Contacts that are part of normal operation and must not raise alerts."""

    ignore_objects: set[int] = field(default_factory=set)  # objects the whole arm may touch (the grasp target)
    carried: int | None = None  # object rigidly attached to the hand
    carried_offset: np.ndarray | None = None  # carried object center in the hand frame
    carried_ignore_boxes: set[str] = field(default_factory=set)  # surfaces the carried object may touch (placing)


class MomentumSolver:
    """Predicts arm and object states ``horizon`` steps ahead and forecasts collisions.

    The action-conditioned (``commanded``) rollout is checked over the full horizon. The
    momentum-only (``ballistic``) rollout is still computed over the full horizon for
    reporting, but only its first ``ballistic_alert_steps`` raise alerts because a
    constant-acceleration guess degrades quickly once the policy changes its command.
    An alert is ``confirmed`` once a contact has been predicted on ``confirm_steps``
    consecutive steps, which filters single-step transients when the policy switches phase.
    """

    def __init__(
        self,
        kinematics: FrankaKinematics,
        arm: ArmMomentumModel,
        objects: ObjectMomentumTracker,
        boxes: list[Box],
        horizon_steps: int,
        dt: float,
        margin: float = 0.0,
        ballistic_alert_steps: int = 10,
        confirm_steps: int = 3,
        uncertainty_scale: float = 0.0,
        commanded_alert_steps: int | None = None,
    ):
        self.kin = kinematics
        self.arm = arm
        self.objects = objects
        self.boxes = boxes
        self.box_center = np.array([b.center for b in boxes], dtype=np.float64)
        self.box_half = np.array([b.half for b in boxes], dtype=np.float64)
        self.horizon_steps = horizon_steps
        self.ballistic_alert_steps = min(ballistic_alert_steps, horizon_steps)
        self.commanded_alert_steps = min(commanded_alert_steps or horizon_steps, horizon_steps)
        self.dt = dt
        self.margin = margin
        self.confirm_steps = confirm_steps
        self.uncertainty_scale = uncertainty_scale
        # Online running median of each model's k-step hand prediction error (m); scaled, it becomes
        # the penetration a predicted contact at step k must exceed to raise an alert.
        self.median_error = {"commanded": np.full(horizon_steps, 1e-3), "ballistic": np.full(horizon_steps, 1e-3)}
        self._hand_predictions: dict[str, list[np.ndarray]] = {"commanded": [], "ballistic": []}
        self.hand_index = kinematics.link_index("panda_hand")
        self.acceleration_limits = np.asarray(FRANKA_ACCELERATION_LIMITS)
        self._qd_history: list[tuple[float, np.ndarray]] = []
        self._target_history: list[tuple[float, np.ndarray]] = []
        self._last_commanded_qd: np.ndarray | None = None
        self._streak = 0

    def joint_acceleration(self, t: float, qd: np.ndarray) -> np.ndarray:
        self._qd_history.append((t, qd.copy()))
        if len(self._qd_history) > 4:
            self._qd_history.pop(0)
        if len(self._qd_history) < 2:
            return np.zeros_like(qd)
        (t0, v0), (t1, v1) = self._qd_history[0], self._qd_history[-1]
        return (v1 - v0) / max(t1 - t0, 1e-6)

    MEDIAN_RATE = 0.02  # multiplicative step of the running-median estimate per sample

    def _calibrate(self, hand_now: np.ndarray) -> None:
        """Score past hand predictions against the hand's current position."""
        for model, history in self._hand_predictions.items():
            n = len(history)
            if n == 0:
                continue
            # history[-j] was made j steps ago; its entry j-1 predicted the present.
            preds = np.stack([history[-j][j - 1] for j in range(1, n + 1)])
            err = np.linalg.norm(preds - hand_now, axis=1)
            med = self.median_error[model]
            med[:n] *= np.exp(self.MEDIAN_RATE * np.sign(err - med[:n]))

    def tolerance(self, model: str) -> np.ndarray:
        """Penetration (m) a predicted contact at each step must exceed to raise an alert."""
        return self.uncertainty_scale * self.median_error[model]

    def target_acceleration(self, t: float, qd_target: np.ndarray) -> np.ndarray:
        """Smoothed acceleration of the streamed velocity target, clamped to the joint acceleration limits."""
        history = self._target_history
        if history and t - history[-1][0] > 2.5 * self.dt:
            history.clear()
        history.append((t, qd_target.copy()))
        if len(history) > 4:
            history.pop(0)
        if len(history) < 2:
            return np.zeros_like(qd_target)
        (t0, v0), (t1, v1) = history[0], history[-1]
        qdd = (v1 - v0) / max(t1 - t0, 1e-6)
        return np.clip(qdd, -self.acceleration_limits, self.acceleration_limits)

    def solve(
        self,
        t: float,
        proprio: Proprioception,
        object_positions: np.ndarray,
        q_target: np.ndarray | None,
        qd_target: np.ndarray | None,
        rules: ContactRules,
        object_orientations: np.ndarray | None = None,
    ) -> Prediction:
        timings = {}
        t0 = time.perf_counter()
        H = self.horizon_steps
        horizon = self.dt * np.arange(1, H + 1)

        # Momentum residual: measured momentum minus what the last commanded rollout expected.
        residual = 0.0
        if self._last_commanded_qd is not None:
            residual = float(np.linalg.norm(proprio.mass_matrix[:7, :7] @ (proprio.qd[:7] - self._last_commanded_qd[:7])))
        momentum = ArmMomentumModel.generalized_momentum(proprio)
        energy = ArmMomentumModel.kinetic_energy(proprio)

        qdd = self.joint_acceleration(t, proprio.qd)
        q_ball, _ = self.arm.ballistic(proprio, qdd, H)
        q_cmd = None
        self._last_commanded_qd = None
        if q_target is not None:
            qd_target = qd_target if qd_target is not None else np.zeros_like(q_target)
            q_traj, qd_traj = extrapolate_target(q_target, qd_target, self.target_acceleration(t, qd_target), self.dt * np.arange(H))
            q_cmd, qd_cmd = self.arm.commanded(proprio, q_traj, qd_traj)
            self._last_commanded_qd = qd_cmd[0]
        t1 = time.perf_counter()
        timings["arm_rollout"] = (t1 - t0) * 1e3

        self.objects.update(t, object_positions, object_orientations)
        obj_pred = self.objects.predict(horizon)  # (H, N, 3)
        t2 = time.perf_counter()
        timings["object_rollout"] = (t2 - t1) * 1e3

        rollouts = ([("commanded", q_cmd, self.commanded_alert_steps)] if q_cmd is not None else []) + [("ballistic", q_ball, self.ballistic_alert_steps)]
        batch = np.concatenate([q for _, q, _ in rollouts], axis=0)
        links = self.kin.forward(batch, proprio.base).reshape(len(rollouts), H, -1, 4, 4)
        spheres = self.kin.spheres(links.reshape(-1, *links.shape[2:])).reshape(len(rollouts), H, -1, 3)
        hand = links[:, :, self.hand_index, :3, 3]
        hand_now = self.kin.forward(proprio.q[None], proprio.base)[0, self.hand_index, :3, 3]
        self._calibrate(hand_now)
        for r, (model, _, _) in enumerate(rollouts):
            history = self._hand_predictions[model]
            history.append(hand[r])
            if len(history) > H:
                history.pop(0)
        t3 = time.perf_counter()
        timings["kinematics"] = (t3 - t2) * 1e3

        alert, min_dist = self._check(t, rollouts, spheres, links, obj_pred, rules, energy)
        self._streak = self._streak + 1 if alert is not None else 0
        t4 = time.perf_counter()
        timings["collision"] = (t4 - t3) * 1e3
        timings["total"] = (t4 - t0) * 1e3

        return Prediction(
            t=t,
            q_commanded=q_cmd,
            q_ballistic=q_ball,
            object_positions=obj_pred,
            hand_commanded=hand[0] if q_cmd is not None else None,
            hand_ballistic=hand[-1],
            min_distance=min_dist,
            alert=alert,
            confirmed=self._streak >= self.confirm_steps,
            momentum=momentum,
            kinetic_energy=energy,
            momentum_residual=residual,
            timings_ms=timings,
        )

    def _check(self, t, rollouts, spheres, links, obj_pred, rules: ContactRules, energy: float):
        """One batched signed-distance pass: (rollout, step, sphere, obstacle)."""
        R, H = spheres.shape[:2]
        S = spheres.shape[2]
        B, N = len(self.boxes), obj_pred.shape[1]
        radii = self.kin.sphere_radius
        sphere_names = list(self.kin.sphere_link_name)

        # Obstacles: static boxes (identity rotation) followed by the predicted object boxes.
        centers = np.concatenate([np.broadcast_to(self.box_center, (H, B, 3)), obj_pred], axis=1)  # (H, O, 3)
        rot = np.concatenate([np.tile(np.eye(3), (B, 1, 1)), self.objects.rotations], axis=0)  # (O, 3, 3)
        half = np.concatenate([self.box_half, self.objects.half_extents], axis=0)  # (O, 3)
        names = [b.name for b in self.boxes] + self.objects.names
        ignore = np.zeros((S, B + N), dtype=bool)
        for i in rules.ignore_objects | ({rules.carried} if rules.carried is not None else set()):
            if 0 <= i < N:
                ignore[:, B + i] = True

        if rules.carried is not None and rules.carried_offset is not None:
            hand_T = links[:, :, self.hand_index]  # (R, H, 4, 4)
            carried = np.einsum("rhij,j->rhi", hand_T[..., :3, :3], rules.carried_offset) + hand_T[..., :3, 3]
            spheres = np.concatenate([spheres, carried[:, :, None, :]], axis=2)
            # The smallest half extent keeps the vertical clearance exact for upright parts.
            radii = np.append(radii, float(np.min(self.objects.half_extents[rules.carried])))
            sphere_names.append("carried:" + self.objects.names[rules.carried])
            row = np.zeros((1, B + N), dtype=bool)
            row[0, B + rules.carried] = True
            for j, b in enumerate(self.boxes):
                row[0, j] = b.name in rules.carried_ignore_boxes
            for i in rules.ignore_objects:
                row[0, B + i] = True
            ignore = np.vstack([ignore, row])

        for r, (_, _, steps) in enumerate(rollouts):
            spheres[r, steps:] = np.nan  # steps beyond a model's alert horizon never hit

        # Broad phase: keep obstacles whose world AABB overlaps the swept AABB of all spheres.
        valid = np.isfinite(spheres[..., 0])
        lo = np.min(np.where(valid[..., None], spheres - radii[:, None], np.inf), axis=(0, 1, 2))
        hi = np.max(np.where(valid[..., None], spheres + radii[:, None], -np.inf), axis=(0, 1, 2))
        extent = np.einsum("oij,oj->oi", np.abs(rot), half)  # (O, 3) world half extents
        o_lo = np.min(centers, axis=0) - extent - self.margin
        o_hi = np.max(centers, axis=0) + extent + self.margin
        keep = np.flatnonzero(np.all((o_lo <= hi) & (o_hi >= lo), axis=1) & ~ignore.all(axis=0))
        if keep.size == 0:
            return None, math.inf

        # Narrow phase per axis on contiguous (K, R, H, S') arrays in each obstacle's frame.
        c = centers[:, keep].transpose(1, 0, 2)[:, None, :, None, :]  # (K, 1, H, 1, 3)
        Rk, hk = rot[keep], half[keep]
        dx, dy, dz = (spheres[None, ..., i] - c[..., i] for i in range(3))
        q = []
        for i in range(3):
            local = dx * Rk[:, 0, i, None, None, None] + dy * Rk[:, 1, i, None, None, None] + dz * Rk[:, 2, i, None, None, None]
            q.append(np.abs(local) - hk[:, i, None, None, None])
        outside = np.sqrt(np.maximum(q[0], 0.0) ** 2 + np.maximum(q[1], 0.0) ** 2 + np.maximum(q[2], 0.0) ** 2)
        inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)
        d = (outside + inside - radii).transpose(1, 2, 3, 0)  # (R, H, S', K)
        d = np.where(np.isnan(d) | ignore[None, None][..., keep], np.inf, d)
        names = [names[o] for o in keep]
        min_dist = float(np.min(d))

        best: Alert | None = None
        tol = np.stack([self.tolerance(model) for model, _, _ in rollouts])  # (R, H)
        hit = d < (self.margin - tol)[:, :, None, None]
        for r, (model, _, _) in enumerate(rollouts):
            steps_hit = hit[r].any(axis=(1, 2))
            if not steps_hit.any():
                continue
            k = int(np.argmax(steps_hit))
            tc = (k + 1) * self.dt
            if best is not None and tc >= best.time_to_collision:
                continue
            s, o = np.unravel_index(int(np.argmin(d[r, k])), d[r, k].shape)
            prev = spheres[r, k - 1, s] if k > 0 else spheres[r, 0, s]
            best = Alert(t=t, time_to_collision=tc, model=model, link=sphere_names[s], obstacle=names[o],
                         distance=float(d[r, k, s, o]), link_speed=float(np.linalg.norm(spheres[r, k, s] - prev) / self.dt),
                         arm_kinetic_energy=energy, point=tuple(float(v) for v in spheres[r, k, s]))
        return best, min_dist


