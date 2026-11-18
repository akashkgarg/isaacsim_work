"""Perceived scene backed by cuRobo's GPU TSDF/ESDF mapper.

This is a drop-in alternative to :class:`scene_perception.PerceivedScene`. The robot self
filter, depth-latency handling, touch localization and object-slot tracking are the same; what
changes is how the static scene is built and handed to the solver:

* **Static scene.** Self-filtered depth from both cameras is fused into cuRobo's block-sparse
  TSDF (1 cm voxels) on the GPU, and an ESDF (Euclidean signed distance field, 2 cm voxels) is
  recomputed from it every step. The solver queries the ESDF at its predicted sphere centers
  (:class:`EsdfField`) instead of checking a list of boxes, so curved and cluttered geometry is
  kept as measured rather than approximated by greedy boxes.
* **Clearing.** Blocks inside a camera frustum lose weight every frame (``frustum_decay``) and
  are kept alive only by fresh observations. Surfaces that went away (a picked part, robot
  points that leaked through the filter) fade within about a second and a half.
* **Change detection.** Before fusing, the current map is rendered from each camera pose. A
  depth pixel clearly in front of the rendered surface means the camera now sees something
  where it previously saw through to that surface: something moved in. Those pixels are kept
  out of the TSDF, so moving objects do not burn into the static map, and are clustered into
  foreground objects for the slot tracker and the momentum solver, as before. The margin grows
  at grazing view angles, where a small surface offset is a large depth difference.
* **Open background.** Where a camera sees nothing within range (the far end of the hall),
  the map has no surface to render, so the test above cannot work. For a camera that does not
  move (the cell camera), each pixel also remembers how deep it has seen, with no return
  counting as infinitely deep; a return in front of that is foreground too. This is what
  catches a part thrown through open air.
* **Settling.** Foreground that stays in the same 3 cm cell for ``static_after`` seconds is
  fused after all. Something that stopped moving becomes part of the scene, and a surface the
  map lost (decayed while occluded, or never resolved, like a thin rail) cannot stay
  foreground forever just because foreground is never fused.
* **Touch.** Localized contacts become small cuboids in the mapper's static channel, which
  never decays and is merged into the ESDF.

cuRobo and PyTorch are only imported by this module, so the numpy pipeline keeps working
without them.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from curobo.perception import Mapper, MapperCfg
from curobo.scene import Cuboid, SceneData
from curobo.types import CameraObservation, DeviceCfg, Pose

from momentum_solver import FrankaKinematics, StaticField
from scene_perception import DepthFrame, PerceivedScene, backproject

# USD cameras look down -Z with +Y up; cuRobo uses the OpenCV convention (+Z forward, +Y down).
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# Pose for an unused camera slot: far outside the map, so its frustum decays nothing.
_PARKED = np.array([[1.0, 0.0, 0.0, 1e3], [0.0, 1.0, 0.0, 1e3], [0.0, 0.0, 1.0, 1e3], [0.0, 0.0, 0.0, 1.0]])
_CUBE_OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


def strided_intrinsics(K: np.ndarray, stride: int) -> np.ndarray:
    """Intrinsics of ``depth[::stride, ::stride]``, matching :func:`scene_perception.backproject`.

    ``backproject`` puts subsampled pixel ``j`` at full-resolution column ``j * stride + 0.5``;
    cuRobo casts rays through ``(j - cx) / fx``. Both agree when ``fx' = fx / stride`` and
    ``cx' = (cx - 0.5) / stride`` (same for y).
    """
    Ks = K.astype(np.float64).copy()
    Ks[0, 0] /= stride
    Ks[1, 1] /= stride
    Ks[0, 2] = (K[0, 2] - 0.5) / stride
    Ks[1, 2] = (K[1, 2] - 0.5) / stride
    return Ks


class EsdfField(StaticField):
    """Trilinear lookups into the mapper's latest ESDF (on the GPU)."""

    name = "map"

    def __init__(self, device: str):
        self.device = device
        self.grid: torch.Tensor | None = None  # (1, 1, nx, ny, nz)

    def set(self, voxel_grid) -> None:
        E = voxel_grid.feature_tensor
        nx, ny, nz = E.shape
        vs = float(voxel_grid.voxel_size)
        self.grid = E.float()[None, None]
        self.center = torch.tensor(voxel_grid.pose[:3], device=self.device, dtype=torch.float32)
        self.half = torch.tensor([(nx - 1) * vs / 2, (ny - 1) * vs / 2, (nz - 1) * vs / 2], device=self.device)

    def distance(self, points: np.ndarray) -> np.ndarray:
        if self.grid is None or len(points) == 0:
            return np.full(len(points), np.inf)
        p = torch.as_tensor(points, device=self.device, dtype=torch.float32)
        n = (p - self.center) / self.half
        # The grid is (x, y, z) from slowest to fastest; grid_sample wants coordinates as (z, y, x).
        s = F.grid_sample(self.grid, n[:, [2, 1, 0]].view(1, 1, 1, -1, 3), align_corners=True, padding_mode="border")
        return s.view(-1).double().cpu().numpy()


def cluster_points(points: np.ndarray, voxel: float = 0.03, min_voxels: int = 2, max_size: float = 0.3) -> list[dict]:
    """Group foreground points into connected clusters of ``voxel``-sized cells (26-connectivity).

    Same output as :meth:`scene_perception.VoxelMap.foreground_clusters`: per cluster, the point
    centroid, half extents covering the per-cell point means (extended downward, since a camera
    mostly sees the top), and the cell count. Clusters wider than ``max_size`` are dropped.
    """
    if len(points) == 0:
        return []
    ijk = np.floor(points / voxel).astype(np.int64)
    cells, inverse, counts = np.unique(ijk, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    mean = np.stack([np.bincount(inverse, weights=points[:, a]) for a in range(3)], 1) / counts[:, None]
    lo_cell = cells.min(0)
    grid = np.zeros(tuple(cells.max(0) - lo_cell + 1), dtype=bool)
    local = cells - lo_cell
    grid[tuple(local.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    cell_label = labels[tuple(local.T)]
    clusters = []
    for lab in range(1, n + 1):
        group = np.flatnonzero(cell_label == lab)
        if group.size < min_voxels:
            continue
        m = mean[group]
        lo, hi = m.min(0) - voxel / 2, m.max(0) + voxel / 2
        if np.any(hi - lo > max_size):
            continue
        depth = min(hi[0] - lo[0], hi[1] - lo[1])
        lo[2] = min(lo[2], hi[2] - depth)
        w = counts[group].astype(np.float64)
        center = (m * w[:, None]).sum(0) / w.sum()
        half = 0.5 * (hi - lo) + np.abs(center - 0.5 * (lo + hi))
        clusters.append({"center": center, "half": half, "voxels": int(group.size)})
    return clusters


class CuroboPerceivedScene(PerceivedScene):
    """:class:`PerceivedScene` with the voxel map and boxes replaced by a cuRobo TSDF/ESDF.

    After :meth:`update`, :attr:`field` holds the static distance field for
    :attr:`momentum_solver.MomentumSolver.static_field`; :attr:`boxes` stays empty.

    Args:
        camera_shapes: Full-resolution ``(height, width)`` of every camera, in a fixed order.
            Frames passed to :meth:`update` are matched to cameras by image shape.
        stride: Depth subsampling before fusion (2 gives 120×160 and 120×212 images here).
        voxel: TSDF voxel size; ``esdf_voxel`` the ESDF's.
        robot_margin: Foreground within this distance (m) beyond the self-filter spheres is taken
            to be robot or carried-part pixels that slipped past the filter, and dropped.
        foreground_tol: How far (m) in front of the rendered surface, measured along its normal,
            a pixel must be to count as something that moved in. Covers wrist-pose error and
            conveyor-roller flicker.
        surface_tol: Foreground points closer than this (m) to a mapped surface are that surface
            seen from another angle (roller tops over the gaps the cell camera sees into, a
            misregistered wrist frame), not change. Must stay below the height of a part.
        frustum_decay: Per-frame weight decay of in-view blocks.
        warmup_frames: Frames fused before change detection starts (the map is empty before).
        static_after: Seconds foreground may stay in one cell before it is fused as static.
    """

    def __init__(self, kin: FrankaKinematics, dt: float, camera_shapes: list[tuple[int, int]], n_slots: int = 8,
                 stride: int = 2, voxel: float = 0.01, esdf_voxel: float = 0.02, foreground_tol: float = 0.03,
                 frustum_decay: float = 0.97, minimum_weight: float = 2.0, warmup_frames: int = 10,
                 static_after: float = 1.5, robot_margin: float = 0.03, surface_tol: float = 0.015,
                 contact_threshold: float = 20.0,
                 noise_floor: np.ndarray | None = None,
                 device: str = "cuda:0", lo=(-0.6, -1.0, -0.03), hi=(1.7, 1.4, 1.6), max_touches: int = 64):
        super().__init__(kin, dt, n_slots=n_slots, contact_threshold=contact_threshold, noise_floor=noise_floor)
        self.map = None
        self.device = device
        self.stride = stride
        self.foreground_tol = foreground_tol
        self.robot_margin = robot_margin
        self.surface_tol = surface_tol
        self.warmup_frames = warmup_frames
        self.camera_shapes = [tuple(s) for s in camera_shapes]
        self.H = max(-(-h // stride) for h, _ in self.camera_shapes)
        self.W = max(-(-w // stride) for _, w in self.camera_shapes)
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.mapper = Mapper(MapperCfg(
            extent_meters_xyz=tuple(float(v) for v in hi - lo),
            grid_center=torch.tensor((lo + hi) / 2, dtype=torch.float32),
            voxel_size=voxel, esdf_voxel_size=esdf_voxel, truncation_distance=4 * voxel,
            depth_minimum_distance=0.1, depth_maximum_distance=4.0, minimum_tsdf_weight=minimum_weight,
            frustum_decay_factor=frustum_decay, enable_static=True,
            num_cameras=len(self.camera_shapes), image_height=self.H, image_width=self.W, device=device))
        self.field = EsdfField(device)
        self._rgb = torch.zeros((len(self.camera_shapes), self.H, self.W, 3), dtype=torch.uint8, device=device)
        self._frames = 0
        self._touch_scene = SceneData.create_cache(num_envs=1, device_cfg=DeviceCfg(device=torch.device(device)),
                                                   cuboid_cache=max_touches)
        self._n_touch_cubes = 0
        self.max_touches = max_touches
        self.foreground_points = np.zeros((0, 3))
        # Settling grid: since when each 3 cm cell has held foreground continuously (NaN = not now).
        self.static_after = static_after
        self.cell = 0.03
        self.cell_lo = lo
        self.cell_shape = tuple(int(np.ceil(v)) for v in (hi - lo) / self.cell)
        self.fg_since = np.full(int(np.prod(self.cell_shape)), np.nan)
        v, u = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        self._uv = (u, v)
        # Per camera slot: last pose, and the deepest depth each pixel has seen from it (0 = never
        # observed, inf = no return). Only used while the camera keeps the same pose.
        self.max_range = 4.0
        self._last_pose = [None] * len(self.camera_shapes)
        self._seen_depth = np.zeros((len(self.camera_shapes), self.H * self.W))

    def update(self, t: float, frames: list[DepthFrame], q, qd, M, c, g, tau_joint, base: np.ndarray,
               carried: tuple[np.ndarray, float] | None = None, exclude: list[tuple[np.ndarray, float]] = (),
               expected_touch: bool = False, t_sensor: float | None = None, q_sensor: np.ndarray | None = None,
               carried_sensor: tuple[np.ndarray, float] | None = None,
               q_uncertain: np.ndarray | None = None) -> None:
        """Integrate one step. Same arguments as :meth:`PerceivedScene.update`.

        ``exclude`` is not needed here: expected contacts with the static scene are expressed
        as :attr:`momentum_solver.ContactRules.static_ignore` regions instead.
        """
        t_sensor = t if t_sensor is None else t_sensor
        t0 = time.perf_counter()
        links, (s_centers, s_radii), _ = self.filter_spheres(q, qd, base, carried, q_sensor, carried_sensor, q_uncertain)

        # Self-filtered, subsampled depth per camera slot; unused slots stay empty and parked.
        n_cam = len(self.camera_shapes)
        depth = np.zeros((n_cam, self.H, self.W), dtype=np.float32)
        poses = np.repeat(_PARKED[None], n_cam, axis=0)
        Ks = np.repeat(np.eye(3)[None], n_cam, axis=0)
        kept: list[tuple[int, np.ndarray, np.ndarray]] = []  # (slot, flat pixel index, world points)
        fixed = np.zeros(n_cam, dtype=bool)  # camera slots whose pose did not change since last frame
        for fr in frames:
            slot = self.camera_shapes.index(tuple(fr.depth.shape))
            pts, mask = backproject(fr.depth, fr.K, fr.cam_to_world, stride=self.stride, max_range=self.max_range,
                                    return_mask=True)
            keep = self.filter.keep(pts, s_centers, s_radii)
            h, w = mask.shape
            raw = fr.depth[::self.stride, ::self.stride].reshape(-1)
            pix = np.flatnonzero(mask)[keep]
            img = np.zeros(h * w, dtype=np.float32)
            img[pix] = raw[pix]
            depth[slot, :h, :w] = img.reshape(h, w)
            poses[slot] = fr.cam_to_world @ USD_TO_CV
            Ks[slot] = strided_intrinsics(fr.K, self.stride)
            pad = lambda p: (p // w) * self.W + p % w  # noqa: E731  pixel index into the padded (H, W) image
            kept.append((slot, pad(pix), pts[keep]))
            last = self._last_pose[slot]
            fixed[slot] = last is not None and np.allclose(last, fr.cam_to_world, atol=1e-6)
            if not fixed[slot]:
                self._seen_depth[slot] = 0.0
            self._last_pose[slot] = fr.cam_to_world.copy()
            self._seen_depth[slot, pad(np.flatnonzero(~np.isfinite(raw) | (raw >= self.max_range)))] = np.inf
        t1 = time.perf_counter()

        touch = self.touch(t, q, qd, M, c, g, tau_joint, base, links, carried, expected_touch)
        if touch is not None and self._n_touch_cubes < self.max_touches:
            self._touch_scene.add_obstacle(Cuboid(name=f"touch_{self._n_touch_cubes}",
                                                  pose=[*map(float, touch[0]), 1.0, 0.0, 0.0, 0.0], dims=[0.03] * 3), env_idx=0)
            self._n_touch_cubes += 1
            self.mapper.update_static_obstacles(self._touch_scene)
        t2 = time.perf_counter()

        d = torch.from_numpy(depth).to(self.device)
        pose = Pose.from_matrix(torch.tensor(poses, dtype=torch.float32, device=self.device))
        K = torch.tensor(Ks, dtype=torch.float32, device=self.device)
        fg = np.zeros((n_cam, self.H * self.W), dtype=bool)
        observed = depth.reshape(n_cam, -1) > 0
        if self._frames >= self.warmup_frames:
            rendered, normals, valid = self.mapper.render(K, pose, (self.H, self.W))
            cos = torch.abs((normals * self._world_rays(Ks, poses)).sum(-1)).clamp(0.25, 1.0)
            fg = ((d > 0) & valid.bool() & (d < rendered - self.foreground_tol / cos)).view(n_cam, -1).cpu().numpy()
            fg |= fixed[:, None] & observed & np.isinf(self._seen_depth)
        candidates = fg.copy()
        (fg_slot, fg_pix, fg_pts), (rb_slot, rb_pix) = self._unsettled(t_sensor, fg, kept, (s_centers, s_radii))
        fg[:] = False
        fg[fg_slot, fg_pix] = True
        # Settled foreground is scene now: its pixels stop counting as open background.
        flat = depth.reshape(n_cam, -1)
        settled = candidates & ~fg
        settled[rb_slot, rb_pix] = False
        self._seen_depth[settled] = flat[settled]
        grow = observed & ~candidates
        self._seen_depth[grow] = np.maximum(self._seen_depth[grow], flat[grow])
        # Pixels kept out of the TSDF: foreground, and what is left of the robot.
        excluded = fg.copy()
        excluded[rb_slot, rb_pix] = True
        mask = torch.from_numpy(excluded.reshape(n_cam, self.H, self.W)).to(self.device)
        self.mapper.integrate(CameraObservation(depth_image=torch.where(mask, 0.0, d), rgb_image=self._rgb,
                                                pose=pose, intrinsics=K))
        self._frames += 1
        self.field.set(self.mapper.compute_esdf())
        t3 = time.perf_counter()

        self.foreground_points = fg_pts
        self.object_positions = self.slots.update(t_sensor, cluster_points(self.foreground_points))
        self.object_time = t_sensor
        self.object_half = self.slots.half.copy()
        self.boxes_changed = False
        t4 = time.perf_counter()
        self.timings_ms = {"backproject": (t1 - t0) * 1e3, "touch": (t2 - t1) * 1e3, "map": (t3 - t2) * 1e3,
                           "clusters": (t4 - t3) * 1e3, "boxes": 0.0, "total": (t4 - t0) * 1e3}

    def _world_rays(self, Ks: np.ndarray, poses: np.ndarray) -> torch.Tensor:
        """Unit world-frame view ray of every pixel, ``(n_cam, H, W, 3)``."""
        u, v = self._uv
        rays = np.stack([(u[None] - Ks[:, 0, 2, None, None]) / Ks[:, 0, 0, None, None],
                         (v[None] - Ks[:, 1, 2, None, None]) / Ks[:, 1, 1, None, None],
                         np.ones((len(Ks), *u.shape))], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        world = np.einsum("nij,nhwj->nhwi", poses[:, :3, :3], rays)
        return torch.from_numpy(world.astype(np.float32)).to(self.device)

    def _unsettled(self, t: float, fg: np.ndarray, kept: list[tuple[int, np.ndarray, np.ndarray]],
                   robot: tuple[np.ndarray, np.ndarray]):
        """Split foreground candidates into robot leftovers, settled scene and real foreground.

        Candidates within ``robot_margin`` of the robot's filter spheres ``robot`` are robot
        pixels the filter missed. The rest is foreground until its 3 cm cell has held
        foreground for ``static_after`` seconds, after which it is fused as static scene.

        Returns ``((slot, pixel, point) of foreground, (slot, pixel) of robot leftovers)``.
        """
        slots, pixels, points = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)], [np.zeros((0, 3))]
        for slot, pix, p in kept:
            sel = fg[slot][pix]
            slots.append(np.full(int(sel.sum()), slot))
            pixels.append(pix[sel])
            points.append(p[sel])
        slots, pixels, points = np.concatenate(slots), np.concatenate(pixels), np.vstack(points)
        on_robot = ~self.filter.keep(points, robot[0], robot[1] + self.robot_margin)
        robot_left = (slots[on_robot], pixels[on_robot])
        slots, pixels, points = slots[~on_robot], pixels[~on_robot], points[~on_robot]
        off = np.abs(self.field.distance(points)) >= self.surface_tol
        slots, pixels, points = slots[off], pixels[off], points[off]
        ijk = np.floor((points - self.cell_lo) / self.cell).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.cell_shape)), axis=1)
        flat = np.full(len(points), -1)
        flat[inside] = np.ravel_multi_index(ijk[inside].T, self.cell_shape)
        hit = np.zeros(self.fg_since.size, dtype=bool)
        hit[flat[inside]] = True
        self.fg_since[hit & np.isnan(self.fg_since)] = t
        self.fg_since[~hit] = np.nan
        settled = np.zeros(len(points), dtype=bool)
        settled[inside] = t - self.fg_since[flat[inside]] >= self.static_after
        stay = inside & ~settled
        return (slots[stay], pixels[stay], points[stay]), robot_left
