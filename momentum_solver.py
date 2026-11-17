"""Momentum-based short-horizon state predictor and collision forecaster for the Franka cell.

Overview
========

Every control step (1/60 s), the pick-and-place policy produces a new joint target for
the arm. Before that target is sent to the simulator, this module asks: *if we execute
this command, and the objects around the arm keep moving the way they have been, will
anything hit anything in the next half second?*

To answer, the solver fuses two sources of state:

* **Arm proprioception.** These are quantities the robot can measure about itself:
  joint positions ``q``, joint velocities ``qd``, the joint-space mass (inertia) matrix
  ``M(q)``, the Coriolis/centrifugal torques, the gravity torques, and the joint-drive
  gains. From these it computes the arm's *generalized momentum* ``p = M(q) qd`` and
  rolls the arm forward in time in two ways:

  - ``commanded`` (action-conditioned): simulate how the joint drives (PD controllers)
    will pull the arm toward the policy's commanded targets. This is the main model.
  - ``ballistic`` (momentum-only): assume the arm keeps its current joint acceleration.
    This needs no knowledge of the command, so it still works if the command is unknown,
    but it degrades quickly as soon as the policy changes what it is doing.

* **Object history.** These are the last ``K`` world positions of every tracked rigid
  object (parts in the tote, parts on the belt, fault objects). A least-squares fit over
  that window gives each object's velocity, acceleration and linear momentum
  ``m * v``. The solver then extrapolates each object with a simple physics prior:

  - resting objects (e.g. riding the conveyor) keep their horizontal momentum;
  - free-flying objects follow a gravity parabola;
  - anything else ("driven", e.g. being pushed) keeps its current velocity.

Collision checking
==================

The arm's shape is approximated by 56 spheres (taken from the cuMotion robot
description). For every predicted future joint configuration, a forward-kinematics (FK)
pass places those spheres in the world. A whole horizon (30 steps × 2 models) is done in
one vectorized batch. Every predicted sphere is then tested against:

* static boxes (floor, pedestal, tote stand, tote walls, conveyor), and
* the predicted oriented boxes of every tracked object.

The earliest predicted penetration becomes an :class:`Alert` with a *time to
collision*. An alert counts as *confirmed* once contacts have been predicted on several
consecutive control steps. A single-step blip is usually a transient, not a real
collision course.

Why "momentum"?
===============

Two reasons:

1. Prediction. Both the arm and the objects carry momentum, and over half a second
   momentum is the dominant thing that determines where they end up.
2. Detection. Comparing the arm's *measured* momentum with the momentum the model
   *predicted* one step earlier gives a "momentum residual". It jumps when an unmodelled
   external force (a collision) acts on the arm. This is the same idea as the classic
   *momentum observer* used for collision detection on torque-controlled robots
   (De Luca et al., 2006). Here it is only reported, not used to raise alerts.

Notation used throughout
========================

* ``B``: batch size (number of joint configurations processed at once)
* ``J``: number of joints (9: seven arm joints and two finger joints)
* ``H``: horizon length in control steps (30 by default, i.e. 0.5 s at 60 Hz)
* ``L``: number of links in the kinematic chain
* ``S``: number of collision spheres on the arm
* ``N``: number of tracked objects
* ``O``: number of obstacles (static boxes plus tracked objects)
* ``R``: number of rollouts (1 or 2: commanded and/or ballistic)
* ``K``: number of obstacles that survive the broad phase (in :meth:`MomentumSolver._check`)

All positions are in metres, times in seconds, and angles in radians. Rotation matrices
map local (body) coordinates to world coordinates.

This module only depends on numpy, so it can be unit-tested outside Isaac Sim
(see ``test_momentum_solver.py``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81  # m/s^2, acting along world -Z

# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
#
# Forward kinematics (FK) answers: "given the joint angles, where is every link in the
# world?" A serial robot is a chain of rigid links connected by joints. Each joint has a
# fixed *origin* transform, which places the joint frame relative to the parent link,
# followed by a *motion* transform that depends on the joint variable:
#
#     T_child = T_parent @ T_origin @ T_motion(q_j)
#
# * revolute joint:  T_motion is a rotation by q_j about the joint's local Z axis
# * prismatic joint: T_motion is a translation by q_j along an axis (here the hand's Y)
# * fixed joint:     T_motion is the identity
#
# All transforms are 4x4 homogeneous matrices [[R, t], [0, 1]]: R is a 3x3 rotation and
# t a translation. Multiplying them chains frames together.


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from URDF roll-pitch-yaw angles.

    URDF uses fixed-axis XYZ rotations: first roll about X, then pitch about Y, then
    yaw about Z, all about the *fixed* parent axes. This equals
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, written out below in closed form.
    """
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(xyz, rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 homogeneous transform from a URDF ``<origin xyz=... rpy=...>`` tag."""
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


HALF_PI = math.pi / 2

# The Franka Panda kinematic chain, transcribed from the cuMotion Franka URDF.
# Each entry is (child link, parent link, joint origin, joint type, index into q).
#
# The 7 arm joints are revolute about their local Z axis; the alternating ±90° roll
# angles in the origins are what make consecutive joint axes perpendicular. panda_hand is
# rigidly attached to link7, rotated -45° about Z. The two fingers are prismatic and slide
# in opposite directions along the hand's Y axis, so a positive finger value opens the
# gripper on both sides. The fingertip frames are fixed offsets at the end of each finger.
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

# Collision geometry: each link is approximated by a few spheres, given as
# (center in the link's local frame, radius). Values are copied from the cuMotion robot
# description robot_configurations/franka/robot.xrdf, which motion planners use for the
# same purpose.
#
# Why spheres? The distance from a sphere to anything is just the distance from its center
# minus its radius. That makes collision checks cheap and easy to vectorize, at the cost of
# a slightly bumpy, conservative approximation of the true link mesh.
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
    # The hand is a 3 (heights) x 6 (across the palm) grid of small spheres.
    "panda_hand": [((0, y, z), r) for z, r in ((0.01, 0.028), (0.03, 0.026), (0.05, 0.024)) for y in (-0.075, -0.045, -0.015, 0.015, 0.045, 0.075)],
    "panda_leftfingertip": [((0, 0.0075, 0), 0.0108)],
    "panda_rightfingertip": [((0, -0.0075, 0), 0.0108)],
}

# Joint order used for every q / qd / torque vector in this module. It matches the order
# Isaac Sim's articulation reports for the Franka, so arrays can be passed straight through.
FRANKA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
# Joint acceleration limits (rad/s^2) from robot.xrdf for the arm joints; the fingers get a
# generous bound. Used to clamp the noisy finite-difference estimate of the target's
# acceleration in MomentumSolver.target_acceleration.
FRANKA_ACCELERATION_LIMITS = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 20.0, 20.0)
# Links that make up the gripper. Kept for callers that want to treat gripper contacts
# differently (e.g. contacts with the grasp target are expected).
GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_leftfingertip", "panda_rightfingertip")


class FrankaKinematics:
    """Batched forward kinematics and collision-sphere placement for the Franka arm.

    "Batched" means :meth:`forward` takes many joint configurations at once (shape
    ``(B, 9)``) and returns all of their link poses in one call, using numpy array
    operations instead of a Python loop over configurations. The solver uses this to
    evaluate every step of every predicted trajectory in one pass.

    The FK was checked against Isaac Sim's own link poses to within 10 µm
    (``probe_franka_dynamics.py``).
    """

    def __init__(self) -> None:
        # Link 0 (the base) comes first, then every child in chain order. Because the chain
        # list is ordered parent-before-child, a single forward pass over it always finds
        # the parent's transform already computed.
        self.link_names = ["panda_link0"] + [c[0] for c in _FRANKA_CHAIN]
        self._index = {name: i for i, name in enumerate(self.link_names)}
        # Flatten the per-link sphere lists into parallel arrays:
        #   sphere_local[s]  center of sphere s in its link's frame   (S, 3)
        #   sphere_radius[s] radius                                   (S,)
        #   sphere_link[s]   index of the owning link in link_names   (S,)
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
            base: World transform of ``panda_link0`` (shape ``(4, 4)``). It is where the
                robot is mounted (on top of the pedestal in this cell).

        Returns:
            Link transforms (shape ``(B, L, 4, 4)``) ordered as :attr:`link_names`.
            ``out[b, l, :3, 3]`` is the world position of link ``l`` for configuration
            ``b``; ``out[b, l, :3, :3]`` is its orientation.
        """
        q = np.atleast_2d(q)
        batch = q.shape[0]
        out = np.empty((batch, len(self.link_names), 4, 4))
        out[:, 0] = base
        for child, parent, origin, kind, j in _FRANKA_CHAIN:
            # Parent pose times the fixed joint origin. `@` on (B, 4, 4) arrays multiplies
            # each batch element separately.
            T = out[:, self._index[parent]] @ origin
            if kind == "revolute":
                # Rotation by q_j about local Z:
                #   [[cos -sin 0 0],
                #    [sin  cos 0 0],
                #    [ 0    0  1 0],
                #    [ 0    0  0 1]]
                # built for all B configurations at once.
                c, s = np.cos(q[:, j]), np.sin(q[:, j])
                R = np.zeros((batch, 4, 4))
                R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
                R[:, 2, 2] = R[:, 3, 3] = 1.0
                T = T @ R
            elif kind.startswith("prismatic"):
                # Slide along the joint frame's Y axis, which is column 1 of the rotation
                # block, i.e. that axis written in world coordinates. The left finger moves
                # along +Y and the right finger along -Y.
                sign = 1.0 if kind.endswith("+y") else -1.0
                T = T.copy()
                T[:, :3, 3] += T[:, :3, 1] * (sign * q[:, j])[:, None]
            # "fixed" joints use the origin transform alone.
            out[:, self._index[child]] = T
        return out

    def spheres(self, link_transforms: np.ndarray) -> np.ndarray:
        """World centers of the collision spheres (shape ``(B, S, 3)``).

        Each sphere center is fixed in its link's frame, so its world position is
        ``R_link @ c_local + t_link``. ``T`` gathers the owning link's transform for every
        sphere, shape ``(B, S, 4, 4)``. The einsum then applies the rotation to every local
        center in one call: ``out[b, s, i] = sum_j R[b, s, i, j] * c[s, j]``.
        """
        T = link_transforms[:, self.sphere_link]
        return np.einsum("bsij,sj->bsi", T[..., :3, :3], self.sphere_local) + T[..., :3, 3]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """Axis-aligned box obstacle (the static parts of the cell: floor, pedestal, tote, conveyor).

    ``half`` holds the half extents: the box spans ``center ± half`` along each world axis.
    """

    name: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]


def sphere_box_distance(centers: np.ndarray, radii: np.ndarray, box_center: np.ndarray, box_half: np.ndarray) -> np.ndarray:
    """Signed sphere-surface to box distance, negative when penetrating (broadcasts over leading dims).

    This is the standard signed distance function (SDF) of an axis-aligned box, minus the
    sphere radius. Work in the box's frame, using symmetry to fold the point into the
    positive octant:

        d = |p - c| - h           (per axis; d_i > 0 means outside along axis i)

    * If any ``d_i > 0``, the point is outside. The distance to the box surface is the
      length of the positive part of ``d``. One positive component means the closest
      feature is a face, two means an edge, three means a corner.
    * If all ``d_i <= 0``, the point is inside. The (negative) distance is
      ``max_i d_i``, the distance to the nearest face.

    The two cases combine into ``|max(d, 0)| + min(max_i d_i, 0)``. Subtracting the radius
    turns "distance from the center" into "distance from the sphere's surface".

    :meth:`MomentumSolver._check` uses the same formula inline, generalized to rotated
    boxes and fused across all spheres and obstacles at once. This standalone version is
    kept for tests and readability.
    """
    d = np.abs(centers - box_center) - box_half
    outside = np.sqrt(np.sum(np.maximum(d, 0.0) ** 2, axis=-1))
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside - radii


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from WXYZ quaternions (shape ``(N, 4)`` -> ``(N, 3, 3)``).

    Isaac Sim reports orientations as unit quaternions ``(w, x, y, z)``. This is the
    standard closed-form conversion. The quaternion is renormalized first, because
    float32 values from the simulator drift slightly off unit length and that would
    scale the resulting matrix.
    """
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

    How it works
    ------------
    The tracker only sees object *positions*. It never reads the simulator's velocities,
    the same as a camera-based tracker would. It keeps the last ``window`` frames and
    fits, per object and per axis, a constant-acceleration model by least squares:

        p(tau) = p0 + v * tau + 0.5 * a * tau^2

    Here ``tau`` is time relative to the newest frame, so ``tau <= 0`` for history and
    ``tau = 0`` is "now". The fitted ``v`` and ``a`` are the velocity and acceleration at
    the present moment. A least-squares fit over several frames is much less noisy than
    differencing two frames, and fitting the ``a`` term stops a curving trajectory (e.g. a
    thrown ball) from biasing the velocity estimate.

    Each object is then classified into a motion mode, which picks the physics prior used
    for prediction:

    * ``REST``: no vertical motion. The object sits on a surface; it may slide or be
      carried horizontally, e.g. on the conveyor belt.
    * ``FLIGHT``: vertical acceleration close to -g, i.e. the object is in free fall
      (a thrown or dropped object).
    * ``DRIVEN``: anything else (being pushed, carried in the gripper, bouncing).
    """

    REST = 0
    FLIGHT = 1
    DRIVEN = 2

    def __init__(self, names: list[str], masses: np.ndarray, half_extents: np.ndarray, window: int = 8, floor_z: float = 0.0):
        self.names = list(names)
        self.masses = np.asarray(masses, dtype=np.float64)  # (N,) kg, only used to report momentum
        self.half_extents = np.asarray(half_extents, dtype=np.float64)  # (N, 3) in each object's frame
        self.window = window  # frames of history used for the fit (8 frames = 0.13 s at 60 Hz)
        self.floor_z = floor_z  # height below which falling objects cannot go
        # Ring buffer of the most recent frames: times (K,) and positions (K, N, 3).
        self._times: list[float] = []
        self._positions: list[np.ndarray] = []
        self.rotations = np.tile(np.eye(3), (len(names), 1, 1))  # (N, 3, 3) latest orientation
        self.velocity = np.zeros((len(names), 3))
        self.acceleration = np.zeros((len(names), 3))
        self.mode = np.full(len(names), self.REST)

    def forget(self, index: int) -> None:
        """Drop an object's history (e.g. after a teleport) so stale frames do not imply a huge velocity.

        The history is overwritten with NaN instead of being deleted, so that all objects
        keep sharing one time axis. :meth:`_fit` then ignores the NaN frames for this
        object until enough fresh frames arrive.
        """
        for p in self._positions:
            p[index] = np.nan

    def update(self, t: float, positions: np.ndarray, orientations: np.ndarray | None = None) -> None:
        """Push a new frame (sim time ``t``, positions ``(N, 3)``, quaternions ``(N, 4)``) and refit."""
        self._times.append(t)
        self._positions.append(np.asarray(positions, dtype=np.float64).copy())
        if orientations is not None:
            # Only the latest orientation is kept. Rotation is not extrapolated: over 0.5 s
            # the parts barely rotate, and the collision check uses this orientation for
            # every future step.
            self.rotations = quat_wxyz_to_matrix(np.asarray(orientations, dtype=np.float64))
        if len(self._times) > self.window:
            self._times.pop(0)
            self._positions.pop(0)
        self._fit()

    @property
    def positions(self) -> np.ndarray:
        """Most recent measured positions ``(N, 3)``."""
        return self._positions[-1]

    @property
    def momentum(self) -> np.ndarray:
        """Linear momentum ``m * v`` of every object ``(N, 3)``, in kg·m/s."""
        return self.masses[:, None] * self.velocity

    def _fit(self) -> None:
        """Least-squares fit of ``p0 + v tau + 0.5 a tau^2`` over the history window.

        In matrix form, with one row per frame ``k``:

            [1, tau_k, 0.5 tau_k^2] @ [p0, v, a]^T = p_k

        i.e. ``A @ X = P`` with ``A`` of shape ``(K, 3)`` and ``P`` of shape ``(K, N*3)``.
        Every column of ``P`` (one coordinate of one object) shares the same ``A``. That
        means a single ``lstsq`` call solves all objects with a full history at once.
        """
        n = len(self._times)
        if n < 3:
            # Three unknowns (p0, v, a) need at least three frames.
            self.velocity[:] = 0.0
            self.acceleration[:] = 0.0
            return
        tau = np.asarray(self._times) - self._times[-1]
        P = np.stack(self._positions)  # (K, N, 3)
        A = np.stack([np.ones_like(tau), tau, 0.5 * tau * tau], axis=1)
        finite = np.isfinite(P).all(axis=2)  # (K, N): which frames are valid for each object
        full = finite.all(axis=0)  # (N,): objects whose whole window is valid
        v = np.zeros((P.shape[1], 3))
        a = np.zeros((P.shape[1], 3))
        if full.any():
            # Solve for every fully-observed object in one call; the result has shape
            # (3 coefficients, N_full, 3 axes), and row 1 is v, row 2 is a.
            coeffs = np.linalg.lstsq(A, P[:, full].reshape(n, -1), rcond=None)[0].reshape(3, -1, 3)
            v[full], a[full] = coeffs[1], coeffs[2]
        # Objects with a partial (recently reset) history are fit on their finite frames only.
        for i in np.flatnonzero(~full & finite[-1]):
            rows = finite[:, i]
            if rows.sum() >= 3:
                c = np.linalg.lstsq(A[rows], P[rows, i], rcond=None)[0]
                v[i], a[i] = c[1], c[2]
            elif rows.sum() == 2:
                # Two frames can only give a velocity (finite difference).
                k0, k1 = np.flatnonzero(rows)
                v[i] = (P[k1, i] - P[k0, i]) / (tau[k1] - tau[k0])
        self.velocity, self.acceleration = v, a
        # Mode classification, with thresholds chosen to tolerate fit noise:
        #   free fall: vertical acceleration within 25% of -g;
        #   resting:   vertical speed under 3 cm/s and no strong vertical acceleration.
        free_fall = np.abs(a[:, 2] + GRAVITY) < 0.25 * GRAVITY
        resting = (np.abs(v[:, 2]) < 0.03) & (np.abs(a[:, 2]) < 0.25 * GRAVITY)
        self.mode = np.where(free_fall, self.FLIGHT, np.where(resting, self.REST, self.DRIVEN))

    def predict(self, horizon: np.ndarray) -> np.ndarray:
        """Predicted positions at future offsets ``horizon`` seconds (shape ``(H, N, 3)``).

        Resting objects conserve horizontal momentum (belt transport), free-flying objects
        follow the gravity parabola until they reach the floor, and driven objects (being
        pushed or carried) extrapolate their fitted velocity.

        The fitted acceleration is deliberately *not* extrapolated for resting or driven
        objects. A fitted acceleration over 0.13 s is noisy, and squaring it over a 0.5 s
        horizon amplifies the noise a lot. The only acceleration that is trusted is gravity,
        which is known exactly.
        """
        p0 = self.positions
        v = self.velocity.copy()
        a = np.zeros_like(v)
        rest = self.mode == self.REST
        flight = self.mode == self.FLIGHT
        v[rest, 2] = 0.0  # resting objects stay on their support surface
        a[flight] = (0.0, 0.0, -GRAVITY)
        t = horizon[:, None, None]  # (H, 1, 1), broadcast against (N, 3)
        pred = p0 + v * t + 0.5 * a * t * t
        # Falling objects stop at the floor (their center sits half_extent_z above it).
        floor = self.floor_z + self.half_extents[:, 2]
        pred[..., 2] = np.maximum(pred[..., 2], np.where(flight, floor, -np.inf))
        # Objects with no valid history (NaN) are moved far away so they never collide.
        return np.nan_to_num(pred, nan=1e6)


# ---------------------------------------------------------------------------
# Arm momentum
# ---------------------------------------------------------------------------
#
# The arm is a multi-body system. Its joint-space equation of motion (the "manipulator
# equation") is
#
#     M(q) qdd + c(q, qd) + g(q) = tau
#
# where:
#   M(q)      (J, J) mass matrix: the joint-space inertia, symmetric positive definite
#   c(q, qd)  (J,)   Coriolis and centrifugal torques (velocity-dependent coupling)
#   g(q)      (J,)   gravity torques
#   tau       (J,)   torques applied by the joint drives (plus any external contact torques)
#
# Isaac Sim gives M, c and g directly through its articulation API, so the solver never
# has to derive them. The generalized momentum is p = M(q) qd, and the kinetic energy is
# 0.5 qd^T M qd.


@dataclass
class Proprioception:
    """Arm measurements for one control step (joint order :data:`FRANKA_JOINTS`).

    All of these come from Isaac Sim's articulation API each step:

    * ``q`` (9,): joint positions, rad (arm) or m (fingers)
    * ``qd`` (9,): joint velocities
    * ``mass_matrix`` (9, 9): ``M(q)``
    * ``coriolis`` (9,): ``c(q, qd)``, the torques needed to cancel Coriolis and
      centrifugal effects
    * ``gravity`` (9,): ``g(q)``, the torques needed to hold the arm still against gravity
    * ``base`` (4, 4): world transform of the robot base
    """

    q: np.ndarray
    qd: np.ndarray
    mass_matrix: np.ndarray
    coriolis: np.ndarray
    gravity: np.ndarray
    base: np.ndarray


@dataclass
class DriveModel:
    """PhysX joint-drive parameters needed to predict the arm's response to a command.

    Each joint is driven by a PD controller inside the physics engine:

        tau = Kp * (q_target - q) + Kd * (qd_target - qd),  clipped to ±max_effort

    For the Franka in this cell the drives are *force* drives with Kp ≈ 22918 and
    Kd ≈ 4584 on the arm joints, and effort limits of 87 N·m (joints 1-4) and 12 N·m
    (joints 5-7). These gains are very stiff, which matters for how the drive has to be
    simulated (see :meth:`ArmMomentumModel.commanded`).
    """

    stiffness: np.ndarray  # Kp per joint
    damping: np.ndarray  # Kd per joint
    max_effort: np.ndarray  # torque (or force) limit per joint
    acceleration_drive: np.ndarray  # per joint: True if the drive output is an acceleration (inertia-scaled)
    gravity_enabled: bool


def extrapolate_target(q_target: np.ndarray, qd_target: np.ndarray, qdd_target: np.ndarray, horizon: np.ndarray,
                       accelerate_for: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Extend a streamed joint target over the horizon (shapes ``(H, J)``).

    The problem
    -----------
    The policy (RMPflow) is *reactive*. Each step it emits only the *next* target
    ``(q_target, qd_target)``, not a full future trajectory. To roll the drives forward
    for H steps, the solver needs a guess of the targets the policy will send at each of
    those steps. This function makes that guess from the current target, its velocity,
    and its (estimated) acceleration.

    The rules, per joint
    --------------------
    * Braking (acceleration opposes velocity): the target is decelerating. It is assumed
      to keep decelerating until it stops, at ``t_stop = -qd / qdd``, and then stay at
      rest. This matters near the end of a motion, e.g. when descending onto the belt:
      the policy is slowing down, and assuming constant velocity would predict the hand
      overshooting into the surface.
    * Otherwise (speeding up or steady): keep the acceleration for at most
      ``accelerate_for`` seconds (0 by default), then cruise at constant velocity.

    The braking-only rule comes from an observation. On the first step of a new phase the
    policy goes from rest to moving, which looks like a huge acceleration that it never
    sustains. Extrapolating that acceleration produced false "overshoot" alarms at every
    phase start.

    Args:
        q_target, qd_target, qdd_target: Current target position, velocity and
            acceleration, each shape ``(J,)``.
        horizon: Future time offsets in seconds, shape ``(H,)``.
        accelerate_for: How long a speed-up is extrapolated, in seconds.

    Returns:
        ``(q, qd)`` target trajectories, each shape ``(H, J)``.
    """
    t = horizon[:, None]  # (H, 1) broadcast against (J,)
    braking = qd_target * qdd_target < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(braking, -qd_target / qdd_target, accelerate_for)
    # Constant-acceleration segment, clipped at t_stop.
    te = np.minimum(t, t_stop)
    q = q_target + qd_target * te + 0.5 * qdd_target * te * te
    qd = qd_target + qdd_target * te
    # After t_stop, non-braking joints cruise at the velocity they reached...
    cruising = ~braking & (t > t_stop)
    q = q + np.where(cruising, qd * (t - t_stop), 0.0)
    # ...and braking joints are stopped (their position is already frozen at the stop point).
    qd = np.where(braking & (t >= t_stop), 0.0, qd)
    return q, qd


class ArmMomentumModel:
    """Rolls the arm's joint state forward from its generalized momentum."""

    def __init__(self, drive: DriveModel, dt: float):
        self.drive = drive
        self.dt = dt  # integration step, equal to the physics step (1/60 s)

    @staticmethod
    def generalized_momentum(proprio: Proprioception) -> np.ndarray:
        """``p = M(q) qd``. It is the joint-space analogue of ``m * v``."""
        return proprio.mass_matrix @ proprio.qd

    @staticmethod
    def kinetic_energy(proprio: Proprioception) -> float:
        """``0.5 qd^T M(q) qd`` in joules. Reported with each alert as a severity cue."""
        return 0.5 * float(proprio.qd @ proprio.mass_matrix @ proprio.qd)

    def ballistic(self, proprio: Proprioception, qdd: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Momentum-only rollout: keep the current joint acceleration (no knowledge of the next command).

        Constant-acceleration kinematics in joint space:

            q(t)  = q + qd t + 0.5 qdd t^2
            qd(t) = qd + qdd t

        for ``t = dt, 2 dt, ..., steps * dt``. This is cheap and needs no model of the
        controller. It is accurate for a few steps (the arm cannot change its momentum
        instantly), but drifts once the policy changes its command. That is why the solver
        only trusts its first ~10 steps for alerts.
        """
        t = self.dt * np.arange(1, steps + 1)[:, None]
        q = proprio.q + proprio.qd * t + 0.5 * qdd * t * t
        return q, proprio.qd + qdd * t

    def commanded(self, proprio: Proprioception, q_targets: np.ndarray, qd_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Action-conditioned rollout through the joint drives (implicit PD, mass matrix frozen over the horizon).

        This reproduces, approximately, what PhysX will do with the targets: apply PD
        torques, solve the equation of motion for the acceleration, and integrate.

        Why "implicit"?
        ---------------
        A naive (explicit) step computes the torque from the *current* state:

            tau = Kp (q* - q) + Kd (qd* - qd);   qdd = M^-1 (tau - bias)

        With Kp ≈ 23000 and dt = 1/60 s this is numerically unstable. The drive behaves
        like a stiff spring, and an explicit integrator blows up once ``dt`` exceeds about
        ``2 / sqrt(Kp / M)``. That limit is a few milliseconds here. PhysX avoids this by
        evaluating the PD law at the *end* of the step (an implicit drive):

            tau = Kp (q* - q_next) + Kd (qd* - qd_next)
            qd_next = qd + dt qdd
            q_next  = q + dt qd_next = q + dt qd + dt^2 qdd

        Substituting the last two lines into the first:

            tau = Kp (q* - q - dt qd) + Kd (qd* - qd) - (dt Kd + dt^2 Kp) qdd
                = tau_explicit - implicit * qdd

        Inserting that into ``M qdd = tau - bias`` and moving the ``qdd`` terms to the left:

            (M + diag(implicit)) qdd = tau_explicit - bias

        This is one linear solve per step, and it is unconditionally stable. Because ``M``
        is frozen over the horizon, ``(M + diag(implicit))^-1`` is computed once and reused.

        Effort saturation
        -----------------
        Real drives cannot exceed ``max_effort``. After solving, the torque ``tau`` actually
        applied is recovered. For any joint where it exceeds the limit, the joint applies
        exactly ``±max_effort``, so its torque no longer depends on ``qdd``. The system is
        re-solved with the implicit term removed for those joints. This matters for fast
        motions (e.g. the corrupted command in the ``bad_action`` fault), where the wrist
        joints saturate.

        Simplifications
        ---------------
        * ``M``, Coriolis and gravity are frozen at their current values for the whole
          horizon. They actually change with ``q``, but slowly over 0.5 s.
        * Contacts are not simulated: the rollout predicts motion *as if nothing is hit*,
          and the collision check then asks whether that motion intersects anything.

        Args:
            proprio: Current arm measurements.
            q_targets: Drive position targets for each future step (shape ``(H, J)``).
            qd_targets: Drive velocity targets for each future step (shape ``(H, J)``).

        Returns:
            Predicted ``(q, qd)`` after each step, each of shape ``(H, J)``.
        """
        M = proprio.mass_matrix
        Kp, Kd = self.drive.stiffness, self.drive.damping
        dt = self.dt
        accel = self.drive.acceleration_drive
        # PhysX acceleration drives scale gains by the joint's own inertia; force drives use them directly.
        Kp_eff = np.where(accel, Kp * np.diag(M), Kp)
        Kd_eff = np.where(accel, Kd * np.diag(M), Kd)
        # "Bias" torques: everything on the left-hand side of the manipulator equation that
        # is not M qdd. Gravity only counts if gravity is enabled on the robot's links.
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
            tau = tau_explicit - implicit * qdd  # torque the drive actually applies
            saturated = np.abs(tau) > limit
            if saturated.any():
                # Saturated joints apply their effort limit; only the rest keep the implicit drive terms.
                rhs = np.where(saturated, np.clip(tau, -limit, limit), tau_explicit) - bias
                qdd = np.linalg.solve(M + np.diag(np.where(saturated, 0.0, implicit)), rhs)
            # Semi-implicit (symplectic) Euler: update velocity first, then position with
            # the new velocity. This matches the substitution used in the derivation above.
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

    t: float  # sim time at which the prediction was made
    time_to_collision: float  # seconds from t until the first predicted penetration
    model: str  # "commanded" or "ballistic": which arm rollout produced it
    link: str  # arm link (or "carried:<object>") predicted to make contact
    obstacle: str  # static box name or tracked object name it will hit
    distance: float  # signed distance at that step (negative = penetration depth, m)
    link_speed: float  # speed of the contacting sphere at that step (m/s), a severity cue
    arm_kinetic_energy: float  # current arm kinetic energy (J), a severity cue
    point: tuple[float, float, float] = (math.nan, math.nan, math.nan)  # predicted contact location (world)


@dataclass
class Prediction:
    """Everything the solver computed in one control step.

    Shapes: ``q_*`` are ``(H, J)``, ``object_positions`` is ``(H, N, 3)`` and ``hand_*``
    are ``(H, 3)``. Row ``k`` is the prediction for ``(k + 1) * dt`` seconds ahead.
    """

    t: float
    q_commanded: np.ndarray | None  # None if no command was given this step
    q_ballistic: np.ndarray
    object_positions: np.ndarray
    hand_commanded: np.ndarray | None  # predicted panda_hand position path
    hand_ballistic: np.ndarray
    min_distance: float  # smallest signed clearance over every sphere, obstacle and step (m)
    alert: Alert | None
    confirmed: bool  # a contact has been predicted for ``confirm_steps`` consecutive steps
    momentum: np.ndarray  # generalized momentum M qd (J,)
    kinetic_energy: float
    momentum_residual: float  # see MomentumSolver.solve
    timings_ms: dict[str, float] = field(default_factory=dict)  # per-stage wall time


@dataclass
class ContactRules:
    """Contacts that are part of normal operation and must not raise alerts.

    A pick-and-place arm is *supposed* to touch some things: the gripper closes on the
    part it is picking, and a carried part is set down on the belt. The caller (the
    monitor in ``franka_factory_pick_place.py``) updates these rules each phase, so only
    *unexpected* contacts raise alerts.

    The carried part is not one of the arm's links, but it moves with the hand. It is
    therefore added to the arm as an extra sphere attached at ``carried_offset`` in the
    hand frame. That way a carried part hitting something (like the intruder) is also
    forecast.
    """

    ignore_objects: set[int] = field(default_factory=set)  # objects the whole arm may touch (the grasp target)
    carried: int | None = None  # object rigidly attached to the hand
    carried_offset: np.ndarray | None = None  # carried object center in the hand frame
    carried_ignore_boxes: set[str] = field(default_factory=set)  # surfaces the carried object may touch (placing)
    # Radius of the carried-part sphere. If None, the carried object's smallest half extent is
    # used. Set it when the carried part is not a tracked object (``carried`` < 0), e.g. when the
    # part's size comes from the finger opening.
    carried_radius: float | None = None


class MomentumSolver:
    """Predicts arm and object states ``horizon`` steps ahead and forecasts collisions.

    The action-conditioned (``commanded``) rollout is checked over the full horizon. The
    momentum-only (``ballistic``) rollout is still computed over the full horizon for
    reporting, but only its first ``ballistic_alert_steps`` raise alerts because a
    constant-acceleration guess degrades quickly once the policy changes its command.
    An alert is ``confirmed`` once a contact has been predicted on ``confirm_steps``
    consecutive steps, which filters single-step transients when the policy switches phase.

    One call to :meth:`solve` per control step does the following:

    1. **Arm rollout.** Compute the momentum residual, roll the arm forward ballistically,
       and, if a command is given, extend it over the horizon and roll it through the
       drives.
    2. **Object rollout.** Push the new object positions into the tracker and extrapolate
       every object over the horizon.
    3. **Kinematics.** Run batched FK for every predicted configuration, place the
       collision spheres, and score earlier hand predictions against where the hand
       actually is now (online calibration).
    4. **Collision check.** Run a broad phase, then a fused signed-distance narrow phase
       over (rollout, step, sphere, obstacle), and pick the earliest hit.

    In this cell a call takes about 2.5 ms in total, versus about 4.5 ms for the policy
    step.

    Args:
        kinematics: FK and collision-sphere model.
        arm: Arm dynamics and drive model.
        objects: Object tracker; the solver feeds it new positions each step.
        boxes: Static obstacles.
        horizon_steps: How many control steps ahead to predict (H).
        dt: Control step in seconds.
        margin: Extra clearance in metres. A predicted distance below ``margin`` counts as a
            hit, so a positive margin raises alerts earlier and more often.
        ballistic_alert_steps: Only the first this-many ballistic steps can raise alerts.
        confirm_steps: Consecutive steps with a predicted hit before ``confirmed``.
        uncertainty_scale: If > 0, a predicted hit at step k must penetrate deeper than
            ``uncertainty_scale * median_error[k]``. This suppresses alerts that are
            within the model's typical prediction error. Off (0) by default.
        commanded_alert_steps: Optionally limit how many commanded steps can raise alerts
            (defaults to the full horizon).
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
        self.set_boxes(boxes)
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
        # The last H predicted hand paths per model (each (H, 3)), used by _calibrate.
        self._hand_predictions: dict[str, list[np.ndarray]] = {"commanded": [], "ballistic": []}
        self.hand_index = kinematics.link_index("panda_hand")
        self.acceleration_limits = np.asarray(FRANKA_ACCELERATION_LIMITS)
        self._qd_history: list[tuple[float, np.ndarray]] = []  # recent measured joint velocities
        self._target_history: list[tuple[float, np.ndarray]] = []  # recent commanded velocity targets
        self._last_commanded_qd: np.ndarray | None = None  # one-step-ahead velocity predicted last call
        self._streak = 0  # consecutive steps with a predicted hit

    def set_boxes(self, boxes: list[Box]) -> None:
        """Replace the static obstacles, e.g. with a freshly perceived map."""
        self.boxes = list(boxes)
        self.box_center = np.array([b.center for b in self.boxes], dtype=np.float64).reshape(-1, 3)
        self.box_half = np.array([b.half for b in self.boxes], dtype=np.float64).reshape(-1, 3)

    def joint_acceleration(self, t: float, qd: np.ndarray) -> np.ndarray:
        """Measured joint acceleration, by finite difference over the last 4 velocity samples.

        PhysX reports joint velocities but not accelerations. Differencing over 3 steps
        (rather than 1) averages out step-to-step jitter. The result feeds the ballistic
        rollout.
        """
        self._qd_history.append((t, qd.copy()))
        if len(self._qd_history) > 4:
            self._qd_history.pop(0)
        if len(self._qd_history) < 2:
            return np.zeros_like(qd)
        (t0, v0), (t1, v1) = self._qd_history[0], self._qd_history[-1]
        return (v1 - v0) / max(t1 - t0, 1e-6)

    MEDIAN_RATE = 0.02  # multiplicative step of the running-median estimate per sample

    def _calibrate(self, hand_now: np.ndarray) -> None:
        """Score past hand predictions against the hand's current position.

        Every step, each model predicted a hand path for the next H steps. Now that time
        has passed, the solver can check them. The prediction made ``j`` steps ago had,
        in its row ``j - 1``, a guess for *now*. Comparing that to the measured hand
        position gives the model's ``j``-step-ahead error, for every ``j`` at once.

        The median error for each ``j`` is tracked with a streaming median estimator.
        Nudge the estimate up by 2% if the new error is larger and down by 2% if smaller:

            m <- m * exp(+-rate)

        At equilibrium, half the samples are above and half below, which is the definition
        of the median. Unlike a mean, it ignores rare huge errors (e.g. at a collision),
        and it needs O(1) memory.
        """
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
        """Smoothed acceleration of the streamed velocity target, clamped to the joint acceleration limits.

        This is the same finite-difference idea as :meth:`joint_acceleration`, applied to the
        *commanded* velocity. Its output feeds :func:`extrapolate_target`, which uses it to
        detect braking.

        * If no target arrived for more than 2.5 steps (e.g. between pick-and-place
          cycles), the history is stale and is cleared. Otherwise the jump across the gap
          would look like an enormous acceleration.
        * The estimate is clamped to the robot's acceleration limits, because a single
          noisy step should not produce a physically impossible extrapolation.
        """
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
        object_time: float | None = None,
    ) -> Prediction:
        """Run one prediction step. Call once per control step, *before* applying the command.

        Args:
            t: Current sim time (s).
            proprio: Arm measurements this step.
            object_positions: Tracked object world positions ``(N, 3)``.
            q_target, qd_target: The joint command about to be sent ``(J,)``. Pass
                ``None`` to run the ballistic model only.
            rules: Which contacts are expected right now.
            object_orientations: Tracked object quaternions ``(N, 4)``, WXYZ.
            object_time: Time at which ``object_positions`` were observed, if earlier than ``t``
                (e.g. camera latency). Object predictions are then extrapolated over the
                extra delay so they still line up with the arm's horizon.

        Returns:
            A :class:`Prediction`. ``prediction.confirmed`` is the signal a guard should act on.
        """
        timings = {}
        t0 = time.perf_counter()
        H = self.horizon_steps
        horizon = self.dt * np.arange(1, H + 1)  # (H,) seconds ahead: dt, 2dt, ..., H dt

        # ---- 1. Arm rollout -------------------------------------------------
        # Momentum residual: measured momentum minus what the last commanded rollout expected.
        #
        # Last step, the commanded rollout predicted the velocity qd_pred we would have now.
        # Any difference between that and the measured qd was caused by forces the model
        # does not know about, mostly contacts. Multiplying by M turns the velocity error
        # into a momentum error, i.e. roughly the impulse (integrated external torque over
        # one step) that the environment applied to the arm. Only the 7 arm joints are
        # used; the fingers are dominated by grasp forces.
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
            # The target trajectory starts at offset 0, not dt: the target sent now is
            # what the drive tracks during the first step, the extrapolated target for dt
            # during the second step, and so on.
            q_traj, qd_traj = extrapolate_target(q_target, qd_target, self.target_acceleration(t, qd_target), self.dt * np.arange(H))
            q_cmd, qd_cmd = self.arm.commanded(proprio, q_traj, qd_traj)
            self._last_commanded_qd = qd_cmd[0]  # kept for next step's momentum residual
        t1 = time.perf_counter()
        timings["arm_rollout"] = (t1 - t0) * 1e3

        # ---- 2. Object rollout ----------------------------------------------
        t_obs = t if object_time is None else object_time
        self.objects.update(t_obs, object_positions, object_orientations)
        obj_pred = self.objects.predict(horizon + (t - t_obs))  # (H, N, 3)
        t2 = time.perf_counter()
        timings["object_rollout"] = (t2 - t1) * 1e3

        # ---- 3. Kinematics --------------------------------------------------
        # Each rollout is (name, joint trajectory (H, J), how many steps may raise alerts).
        # Both trajectories are stacked into one (R*H, J) batch so FK runs once.
        rollouts = ([("commanded", q_cmd, self.commanded_alert_steps)] if q_cmd is not None else []) + [("ballistic", q_ball, self.ballistic_alert_steps)]
        batch = np.concatenate([q for _, q, _ in rollouts], axis=0)
        links = self.kin.forward(batch, proprio.base).reshape(len(rollouts), H, -1, 4, 4)  # (R, H, L, 4, 4)
        spheres = self.kin.spheres(links.reshape(-1, *links.shape[2:])).reshape(len(rollouts), H, -1, 3)  # (R, H, S, 3)
        hand = links[:, :, self.hand_index, :3, 3]  # (R, H, 3) predicted hand paths
        hand_now = self.kin.forward(proprio.q[None], proprio.base)[0, self.hand_index, :3, 3]
        self._calibrate(hand_now)
        for r, (model, _, _) in enumerate(rollouts):
            history = self._hand_predictions[model]
            history.append(hand[r])
            if len(history) > H:
                history.pop(0)
        t3 = time.perf_counter()
        timings["kinematics"] = (t3 - t2) * 1e3

        # ---- 4. Collision check ---------------------------------------------
        alert, min_dist = self._check(t, rollouts, spheres, links, obj_pred, rules, energy)
        # Confirmation: count consecutive steps with any predicted hit. One step without a
        # hit resets the count.
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
        """One batched signed-distance pass: (rollout, step, sphere, obstacle).

        Conceptually this evaluates ``distance(sphere s at step k of rollout r, obstacle o)``
        for every combination. With 2 rollouts, 30 steps, 56 arm spheres plus the carried part,
        and 8 static boxes plus 7 objects, that is roughly 2 × 30 × 57 × 15 ≈ 51k distances per call.
        Two tricks keep it under half a millisecond:

        * **Broad phase.** Compute one axis-aligned bounding box (AABB) around *every*
          predicted sphere position, and one around each obstacle's whole predicted path.
          Obstacles whose box does not overlap the arm's box cannot be hit, so they are
          dropped before the expensive part. Usually only a handful survive.
        * **Fused narrow phase.** The box SDF from :func:`sphere_box_distance` is
          evaluated for all survivors at once on contiguous per-axis arrays. Each sphere
          center is first transformed into each obstacle's local frame, so rotated
          objects are handled exactly.

        Returns:
            ``(alert, min_distance)``. ``alert`` is the earliest predicted penetration
            across rollouts, or ``None``. ``min_distance`` is the smallest clearance found
            (``inf`` if nothing is nearby).
        """
        R, H = spheres.shape[:2]
        S = spheres.shape[2]
        B, N = len(self.boxes), obj_pred.shape[1]
        radii = self.kin.sphere_radius
        sphere_names = list(self.kin.sphere_link_name)

        # Obstacles: static boxes (identity rotation) followed by the predicted object boxes.
        # Index o < B is a static box; index B + i is tracked object i.
        centers = np.concatenate([np.broadcast_to(self.box_center, (H, B, 3)), obj_pred], axis=1)  # (H, O, 3)
        rot = np.concatenate([np.tile(np.eye(3), (B, 1, 1)), self.objects.rotations], axis=0)  # (O, 3, 3)
        half = np.concatenate([self.box_half, self.objects.half_extents], axis=0)  # (O, 3)
        names = [b.name for b in self.boxes] + self.objects.names
        # ignore[s, o] = True means a contact between sphere s and obstacle o is expected
        # and must not raise an alert. The grasp target and the carried part are ignored
        # for the arm's own spheres.
        ignore = np.zeros((S, B + N), dtype=bool)
        for i in rules.ignore_objects | ({rules.carried} if rules.carried is not None else set()):
            if 0 <= i < N:
                ignore[:, B + i] = True

        if rules.carried is not None and rules.carried_offset is not None:
            # Attach the carried part to the arm as one extra sphere. Its center is the
            # part's center, transformed by each predicted hand pose.
            hand_T = links[:, :, self.hand_index]  # (R, H, 4, 4)
            carried = np.einsum("rhij,j->rhi", hand_T[..., :3, :3], rules.carried_offset) + hand_T[..., :3, 3]
            spheres = np.concatenate([spheres, carried[:, :, None, :]], axis=2)
            tracked = 0 <= rules.carried < N
            if rules.carried_radius is not None:
                radius = rules.carried_radius
            else:
                # The smallest half extent keeps the vertical clearance exact for upright parts.
                radius = float(np.min(self.objects.half_extents[rules.carried]))
            radii = np.append(radii, radius)
            sphere_names.append("carried:" + (self.objects.names[rules.carried] if tracked else "part"))
            # The carried part may touch itself, the placing surface, and ignored objects.
            row = np.zeros((1, B + N), dtype=bool)
            if tracked:
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
        # World-axis half extents of a rotated box: each world axis picks up |R| times the
        # local half extents. This is the tightest AABB around an oriented box.
        extent = np.einsum("oij,oj->oi", np.abs(rot), half)  # (O, 3) world half extents
        o_lo = np.min(centers, axis=0) - extent - self.margin  # over the whole horizon
        o_hi = np.max(centers, axis=0) + extent + self.margin
        # Two AABBs overlap iff their intervals overlap on all three axes. Obstacles that
        # every sphere ignores are dropped too.
        keep = np.flatnonzero(np.all((o_lo <= hi) & (o_hi >= lo), axis=1) & ~ignore.all(axis=0))
        if keep.size == 0:
            return None, math.inf

        # Narrow phase per axis on contiguous (K, R, H, S') arrays in each obstacle's frame.
        #
        # For each kept obstacle k: d = sphere_center - obstacle_center (world frame), then
        # local = R_k^T d, i.e. local_i = sum_j d_j R_k[j, i]. Working per axis (dx, dy, dz)
        # instead of one big einsum keeps memory access contiguous. Together with the broad
        # phase, this took the collision check from ~6.6 ms to ~0.44 ms.
        # S' is the number of spheres including the carried one, if any.
        c = centers[:, keep].transpose(1, 0, 2)[:, None, :, None, :]  # (K, 1, H, 1, 3)
        Rk, hk = rot[keep], half[keep]
        dx, dy, dz = (spheres[None, ..., i] - c[..., i] for i in range(3))  # each (K, R, H, S')
        q = []
        for i in range(3):
            local = dx * Rk[:, 0, i, None, None, None] + dy * Rk[:, 1, i, None, None, None] + dz * Rk[:, 2, i, None, None, None]
            q.append(np.abs(local) - hk[:, i, None, None, None])
        # Box SDF, as in sphere_box_distance.
        outside = np.sqrt(np.maximum(q[0], 0.0) ** 2 + np.maximum(q[1], 0.0) ** 2 + np.maximum(q[2], 0.0) ** 2)
        inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)
        d = (outside + inside - radii).transpose(1, 2, 3, 0)  # (R, H, S', K)
        # Masked-out steps (NaN) and expected contacts become +inf: "never a hit".
        d = np.where(np.isnan(d) | ignore[None, None][..., keep], np.inf, d)
        names = [names[o] for o in keep]
        min_dist = float(np.min(d))

        # Find the earliest step at which any sphere penetrates any obstacle, per rollout,
        # and keep the earliest across rollouts.
        best: Alert | None = None
        tol = np.stack([self.tolerance(model) for model, _, _ in rollouts])  # (R, H)
        hit = d < (self.margin - tol)[:, :, None, None]
        for r, (model, _, _) in enumerate(rollouts):
            steps_hit = hit[r].any(axis=(1, 2))  # (H,) any hit at step k?
            if not steps_hit.any():
                continue
            k = int(np.argmax(steps_hit))  # argmax of a boolean array = first True
            tc = (k + 1) * self.dt  # row k is (k + 1) steps ahead
            if best is not None and tc >= best.time_to_collision:
                continue
            # At that step, report the deepest sphere/obstacle pair.
            s, o = np.unravel_index(int(np.argmin(d[r, k])), d[r, k].shape)
            # Sphere speed by finite difference along the predicted path.
            prev = spheres[r, k - 1, s] if k > 0 else spheres[r, 0, s]
            best = Alert(t=t, time_to_collision=tc, model=model, link=sphere_names[s], obstacle=names[o],
                         distance=float(d[r, k, s, o]), link_speed=float(np.linalg.norm(spheres[r, k, s] - prev) / self.dt),
                         arm_kinetic_energy=energy, point=tuple(float(v) for v in spheres[r, k, s]))
        return best, min_dist
