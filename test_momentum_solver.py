"""Unit tests for momentum_solver (numpy only): ``python3 -m pytest test_momentum_solver.py`` or ``python3 test_momentum_solver.py``."""

import numpy as np
from momentum_solver import (
    GRAVITY,
    ArmMomentumModel,
    Box,
    ContactRules,
    DriveModel,
    FrankaKinematics,
    MomentumSolver,
    ObjectMomentumTracker,
    Proprioception,
    extrapolate_target,
    sphere_box_distance,
)

DT = 1.0 / 60.0


def test_fk_zero_pose():
    kin = FrankaKinematics()
    T = kin.forward(np.zeros((1, 9)), np.eye(4))[0]
    hand = T[kin.link_index("panda_hand")]
    # Panda flange at q=0 sits at (0.088, 0, 0.926); the hand is the flange yawed by -45 deg.
    assert np.allclose(hand[:3, 3], [0.088, 0.0, 0.926], atol=1e-9)
    assert np.allclose(hand[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)


def test_fk_batch_matches_single():
    kin = FrankaKinematics()
    rng = np.random.default_rng(0)
    q = rng.uniform(-1, 1, size=(5, 9))
    base = np.eye(4)
    base[:3, 3] = (0.1, 0.2, 0.75)
    batch = kin.forward(q, base)
    for i in range(5):
        assert np.allclose(batch[i], kin.forward(q[i : i + 1], base)[0])
    assert kin.spheres(batch).shape == (5, len(kin.sphere_radius), 3)


def test_sphere_box_distance():
    c = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.05]])
    d = sphere_box_distance(c, np.array([0.1, 0.1]), np.zeros(3), np.array([0.5, 0.5, 0.1]))
    assert np.allclose(d, [0.8, -0.15])


def _tracker_fit(traj):
    tr = ObjectMomentumTracker(["a"], np.array([1.0]), np.array([[0.02, 0.02, 0.02]]), window=8)
    for k in range(8):
        tr.update(k * DT, traj(k * DT)[None])
    return tr


def test_belt_object_keeps_momentum():
    tr = _tracker_fit(lambda t: np.array([0.15 * t, 0.7, 0.8]))
    assert tr.mode[0] == tr.REST
    pred = tr.predict(np.array([0.5]))
    assert np.allclose(pred[0, 0], [0.15 * (7 * DT + 0.5), 0.7, 0.8], atol=1e-6)
    assert np.allclose(tr.momentum[0], [0.15, 0, 0], atol=1e-6)


def test_projectile_follows_gravity():
    v0 = np.array([1.0, -0.5, 2.0])
    p0 = np.array([0.0, 0.0, 1.0])
    traj = lambda t: p0 + v0 * t + 0.5 * np.array([0, 0, -GRAVITY]) * t * t  # noqa: E731
    tr = _tracker_fit(traj)
    assert tr.mode[0] == tr.FLIGHT
    t_end = 7 * DT
    pred = tr.predict(np.array([0.2]))
    assert np.allclose(pred[0, 0], traj(t_end + 0.2), atol=1e-6)


def _drive():
    n = 9
    return DriveModel(
        stiffness=np.full(n, 400.0), damping=np.full(n, 40.0), max_effort=np.full(n, 1e6),
        acceleration_drive=np.zeros(n, dtype=bool), gravity_enabled=False,
    )


def test_commanded_rollout_converges():
    arm = ArmMomentumModel(_drive(), DT)
    proprio = Proprioception(q=np.zeros(9), qd=np.zeros(9), mass_matrix=np.eye(9), coriolis=np.zeros(9), gravity=np.zeros(9), base=np.eye(4))
    target = np.full(9, 0.3)
    q, _ = arm.commanded(proprio, np.tile(target, (240, 1)), np.zeros((240, 9)))
    assert np.allclose(q[-1], target, atol=1e-3)
    assert np.all(np.diff(q[:10, 0]) > 0)


def test_braking_target_stops():
    horizon = DT * np.arange(30)
    q, qd = extrapolate_target(np.zeros(1), np.array([1.0]), np.array([-4.0]), horizon)
    # Decelerating from 1 rad/s at 4 rad/s^2 stops after 0.25 s, 0.125 rad from the start.
    assert np.isclose(q[-1, 0], 0.125) and qd[-1, 0] == 0.0
    assert np.all(np.diff(q[:, 0]) >= -1e-12)


def test_oriented_object_box():
    tr = ObjectMomentumTracker(["part"], np.array([0.1]), np.array([[0.1, 0.01, 0.01]]))
    yaw = np.pi / 2
    tr.update(0.0, np.zeros((1, 3)), np.array([[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]]))
    # A long box yawed 90 deg extends along world Y.
    assert np.allclose(tr.rotations[0] @ [1.0, 0, 0], [0, 1.0, 0], atol=1e-9)


def test_solver_flags_box_in_path():
    kin = FrankaKinematics()
    arm = ArmMomentumModel(_drive(), DT)
    objects = ObjectMomentumTracker(["part"], np.array([0.1]), np.array([[0.02, 0.02, 0.02]]))
    q0 = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.8, 0.04, 0.04])
    hand0 = kin.forward(q0[None], np.eye(4))[0, kin.link_index("panda_hand"), :3, 3]
    wall = Box("wall", (hand0[0], hand0[1], hand0[2] - 0.2), (0.08, 0.08, 0.02))
    solver = MomentumSolver(kin, arm, objects, [wall], horizon_steps=60, dt=DT)
    proprio = Proprioception(q=q0, qd=np.zeros(9), mass_matrix=np.eye(9), coriolis=np.zeros(9), gravity=np.zeros(9), base=np.eye(4))
    far = np.array([[5.0, 5.0, 0.0]])
    calm = solver.solve(0.0, proprio, far, q0, np.zeros(9), ContactRules())
    assert calm.alert is None
    # Pitching the shoulder forward drives the hand down into the wall.
    dive = solver.solve(DT, proprio, far, q0 + np.array([0, 0.6, 0, 0, 0, 0, 0, 0, 0]), np.zeros(9), ContactRules())
    assert dive.alert is not None and dive.alert.obstacle == "wall" and dive.alert.model == "commanded"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
