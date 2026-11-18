"""Tests for curobo_scene (needs a CUDA GPU, PyTorch and cuRobo): ``python3 test_curobo_scene.py``."""

import numpy as np

from curobo_scene import CuroboPerceivedScene, cluster_points, strided_intrinsics
from momentum_solver import FrankaKinematics
from scene_perception import DepthFrame, backproject

DT = 1.0 / 60.0
K = np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]])
# USD camera 1 m above a floor at z = 0.3, looking straight down (-Z).
CAMERA = np.eye(4)
CAMERA[:3, 3] = (0.5, 0.2, 1.3)


def test_strided_intrinsics_match_backproject():
    depth = np.full((60, 80), 2.0, dtype=np.float32)
    pts, mask = backproject(depth, K, np.eye(4), stride=2, drop_edges=False, return_mask=True)
    Ks = strided_intrinsics(K, 2)
    v, u = np.nonzero(mask)
    # cuRobo's ray for subsampled pixel (v, u): ((u - cx) / fx, (v - cy) / fy, 1) in OpenCV axes.
    x = (u - Ks[0, 2]) / Ks[0, 0] * 2.0
    y = (v - Ks[1, 2]) / Ks[1, 1] * 2.0
    assert np.allclose(pts[:, 0], x) and np.allclose(pts[:, 1], -y)


def test_cluster_points_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal((0.0, 0.0, 0.5), 0.01, size=(50, 3))
    b = rng.normal((0.5, 0.0, 0.5), 0.01, size=(50, 3))
    clusters = cluster_points(np.vstack([a, b]))
    assert len(clusters) == 2
    centers = sorted(c["center"][0] for c in clusters)
    assert abs(centers[0]) < 0.02 and abs(centers[1] - 0.5) < 0.02


def _step(scene, t, depth):
    zeros = np.zeros(9)
    base = np.eye(4)
    base[:3, 3] = (10.0, 10.0, 0.0)  # the robot is far away, out of view
    scene.update(t, [DepthFrame(depth, K, CAMERA)], zeros, zeros, np.eye(9), zeros, zeros, zeros, base)


def test_static_floor_moving_box_and_settling():
    scene = CuroboPerceivedScene(FrankaKinematics(), DT, [(60, 80)], warmup_frames=5, static_after=0.5)
    floor = np.full((60, 80), 1.0, dtype=np.float32)
    t = 0.0
    for _ in range(15):
        _step(scene, t, floor)
        t += DT
    probe = np.array([[0.5, 0.2, 0.5]])
    assert abs(scene.field.distance(probe)[0] - 0.2) < 0.03
    # A 10 cm box, 20 cm tall, slides across the floor; its top is at z = 0.5.
    for k in range(20):
        depth = floor.copy()
        c0 = 20 + k
        depth[25:31, c0:c0 + 6] = 0.8
        _step(scene, t, depth)
        t += DT
    assert len(scene.foreground_points) > 0
    assert np.all(np.abs(scene.foreground_points[:, 2] - 0.5) < 0.03)
    assert scene.field.distance(probe)[0] > 0.1  # the moving box is not fused into the static map
    # The box stops. After static_after seconds it becomes static scene.
    for _ in range(60):
        _step(scene, t, depth)
        t += DT
    box_top = np.array([[0.5 + (c0 + 3 - 40) / 60.0 * 0.8, 0.2 - (28 - 30) / 60.0 * 0.8, 0.5]])
    assert len(scene.foreground_points) == 0
    assert abs(scene.field.distance(box_top)[0]) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
