"""Unit tests for scene_perception (numpy only; run with ``python3 test_scene_perception.py``)."""

import math

import numpy as np

from momentum_solver import FrankaKinematics, ObjectMomentumTracker
from scene_perception import (
    ClusterTracker,
    ContactEstimator,
    DepthFrame,
    RobotSelfFilter,
    VoxelMap,
    backproject,
    project,
)

K = np.array([[200.0, 0, 80], [0, 200.0, 60], [0, 0, 1]])
READY = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.04, 0.04])


def looking_down(height: float) -> np.ndarray:
    """Camera at (0, 0, height) looking straight down (camera -Z = world -Z), image up = world +Y."""
    T = np.eye(4)
    T[:3, 3] = (0.0, 0.0, height)
    return T


def plane_depth(height: float, plane_z: float, shape=(120, 160)) -> np.ndarray:
    return np.full(shape, height - plane_z, dtype=np.float32)


def test_backproject_project_roundtrip():
    T = looking_down(1.0)
    pts = backproject(plane_depth(1.0, 0.2), K, T, stride=4)
    assert np.allclose(pts[:, 2], 0.2)
    u, v, z = project(pts, K, T)
    assert np.allclose(z, 0.8)
    assert u.min() >= 0 and u.max() < 160 and v.min() >= 0 and v.max() < 120
    print("ok test_backproject_project_roundtrip")


def test_map_builds_and_clears_box():
    """A slab seen for longer than STATIC_AFTER becomes a static box; once seen through, it clears."""
    vmap = VoxelMap(lo=(-0.5, -0.5, -0.03), hi=(0.5, 0.5, 1.0), voxel=0.03)
    T = looking_down(1.0)
    frame = DepthFrame(plane_depth(1.0, 0.3), K, T)
    t = 0.0
    for _ in range(int(2.0 * 60)):
        t += 1 / 60
        pts = backproject(frame.depth, K, T)
        vmap.update(t, [frame], pts, np.zeros(0, dtype=np.int64))
    boxes = vmap.static_boxes()
    assert boxes, "no static boxes"
    tops = [b.center[2] + b.half[2] for b in boxes]
    assert abs(max(tops) - 0.31) < 0.01, max(tops)
    # The slab disappears: the camera now sees the floor at z=0 through where it was.
    frame = DepthFrame(plane_depth(1.0, 0.0), K, T)
    for _ in range(15):
        t += 1 / 60
        vmap.update(t, [frame], backproject(frame.depth, K, T), np.zeros(0, dtype=np.int64))
    # Voxels at the edge of the view whose centers project outside the image are never seen through.
    central = (np.abs(vmap.centers[:, 0]) < 0.15) & (np.abs(vmap.centers[:, 1]) < 0.1)
    occupied_z = vmap.centers[(vmap.L > 0) & central][:, 2]
    assert np.all(occupied_z < 0.05), occupied_z.max()
    print("ok test_map_builds_and_clears_box")


def test_foreground_cluster_tracks_moving_object():
    vmap = VoxelMap(lo=(-0.5, -0.5, -0.03), hi=(0.5, 0.5, 1.0), voxel=0.03)
    slots = ClusterTracker(4)
    tracker = ObjectMomentumTracker([f"s{i}" for i in range(4)], np.ones(4), np.full((4, 3), 0.03))
    T = looking_down(1.0)
    t = 0.0
    # Warm-up: until the floor has been seen for STATIC_AFTER it is foreground too, and an object
    # resting on it would merge with it into one cluster.
    floor = DepthFrame(plane_depth(1.0, 0.0), K, T)
    while t < VoxelMap.STATIC_AFTER + 0.1:
        t += 1 / 60
        vmap.update(t, [floor], backproject(floor.depth, K, T), np.zeros(0, dtype=np.int64))
    t0 = t
    xs = []
    for k in range(40):
        t += 1 / 60
        depth = plane_depth(1.0, 0.0)
        x = -0.2 + 0.5 * (t - t0)  # 0.5 m/s along +x
        # A 6 cm cube top at z=0.06 drawn into the depth image.
        u0 = int(K[0, 0] * (x - 0.03) / 0.94 + K[0, 2])
        u1 = int(K[0, 0] * (x + 0.03) / 0.94 + K[0, 2])
        v0, v1 = int(-K[1, 1] * 0.03 / 0.94 + K[1, 2]), int(K[1, 1] * 0.03 / 0.94 + K[1, 2])
        depth[v0:v1, u0:u1] = 0.94
        frame = DepthFrame(depth, K, T)
        vmap.update(t, [frame], backproject(depth, K, T), np.zeros(0, dtype=np.int64))
        pos = slots.update(t, vmap.foreground_clusters())
        for i in slots.reset_slots:
            tracker.forget(i)
        tracker.update(t, pos)
        if k > 10:
            s = np.flatnonzero(np.isfinite(pos[:, 0]))
            assert s.size >= 1
            xs.append(pos[s[0], 0] - x)
    assert np.median(np.abs(xs)) < 0.03, np.median(np.abs(xs))
    s = np.flatnonzero(np.isfinite(slots.position[:, 0]))[0]
    assert abs(tracker.velocity[s, 0] - 0.5) < 0.1, tracker.velocity[s]
    print("ok test_foreground_cluster_tracks_moving_object")


def test_self_filter_removes_arm_points():
    kin = FrankaKinematics()
    filt = RobotSelfFilter(kin)
    links = kin.forward(READY[None], np.eye(4))[0]
    centers, radii = filt.spheres(links)
    on_arm = centers + np.array([0.0, 0.0, 0.01])
    away = np.array([[1.5, 1.5, 0.2], [-1.0, 0.5, 0.1]])
    keep = filt.keep(np.vstack([on_arm, away]), centers, radii)
    assert not keep[: len(on_arm)].any() and keep[len(on_arm):].all()
    print("ok test_self_filter_removes_arm_points")


def test_contact_localization_recovers_point():
    """A force pushing on link5 from the side is localized on link5, pointing the right way."""
    kin = FrankaKinematics()
    est = ContactEstimator(kin, 1 / 60)
    base = np.eye(4)
    q = READY
    # Pick a candidate point on a link5 sphere and push inward along its outward direction.
    links = kin.forward(q[None], base)[0]
    s = kin.sphere_link_name.index("panda_link5")
    c = kin.spheres(links[None])[0, s]
    out = np.array([0.0, 1.0, 0.0])
    out = links[kin.link_index("panda_link5"), :3, :3] @ out
    p = c + kin.sphere_radius[s] * out
    f = -20.0 * out
    eps = 1e-5
    J = np.zeros((3, 7))
    for j in range(7):
        dq = q.copy()
        dq[j] += eps
        lk = kin.forward(dq[None], base)[0]
        T = lk[kin.link_index("panda_link5")]
        p_local = links[kin.link_index("panda_link5")][:3, :3].T @ (p - links[kin.link_index("panda_link5")][:3, 3])
        J[:, j] = (T[:3, :3] @ p_local + T[:3, 3] - p) / eps
    tau = J.T @ f
    loc = est.localize(q, base, tau)
    assert loc is not None
    assert loc["residual"] < 0.1, loc
    assert np.linalg.norm(loc["point"] - p) < 0.08, (loc["point"], p, loc["link"])
    assert np.dot(loc["normal"], out) > 0.5
    print("ok test_contact_localization_recovers_point")


def test_external_torque_zero_in_free_motion():
    kin = FrankaKinematics()
    est = ContactEstimator(kin, 1 / 60, noise_floor=np.full(7, 0.1))
    M = np.diag(np.full(9, 0.5))
    c = np.zeros(9)
    g = np.full(9, 1.0)
    qd = np.zeros(9)
    for k in range(20):
        qd_new = qd + (1 / 60) * 0.3
        tau = M @ np.full(9, 0.3) + c + g  # joint torque that produces exactly that acceleration
        est.update(k / 60, READY, qd_new, M, c, g, tau)
        qd = qd_new
    assert np.all(np.abs(est.tau_ext) < 1e-6) and not est.in_contact()
    print("ok test_external_torque_zero_in_free_motion")


if __name__ == "__main__":
    test_backproject_project_roundtrip()
    test_map_builds_and_clears_box()
    test_foreground_cluster_tracks_moving_object()
    test_self_filter_removes_arm_points()
    test_contact_localization_recovers_point()
    test_external_torque_zero_in_free_motion()
