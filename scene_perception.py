"""Build the momentum solver's scene model on the fly from the robot's own sensors.

The world-knowledge version of the monitor hands the solver the exact boxes of the cell
(``cell_obstacles``) and the simulator's poses for every object. This module replaces both
with estimates built only from sensors a real Franka cell could have:

* **Depth images.** One RGB-D camera is mounted on the hand, and one is fixed on the cell.
  The fixed camera's pose is known from calibration. The wrist camera's pose comes from
  forward kinematics (joint angles + the known mount), so it is proprioceptive too.
* **Joint state and torques.** Joint positions/velocities, the dynamics model (mass
  matrix, Coriolis and gravity terms) and the measured joint torques, like the Franka's
  joint torque sensors.

Pipeline, once per control step
===============================

1. **Back-project** every depth image into a world-frame point cloud.
2. **Self-filter.** Drop points that lie on the robot (its collision spheres, slightly
   inflated) or on the part it is carrying. The robot sees itself constantly; without
   this step it would map its own arm as an obstacle.
3. **Update a voxel occupancy map** (3 cm voxels) with log-odds evidence:

   * a voxel that contains a depth point gains evidence (``+L_HIT``);
   * a voxel that a camera ray passed *through* (the measured depth at its pixel is
     further away) loses evidence (``-L_MISS``). This is how geometry that moves away
     is cleared;
   * voxels inside the arm's own collision spheres are known free: the arm is there, so
     nothing else can be ("swept-volume carving", pure proprioception).

4. **Split static from moving.** A voxel that has stayed occupied for ``STATIC_AFTER``
   seconds is *static* scene: tote, conveyor, pedestal, parts resting in the tote. Voxels
   hit in the current frame that are not static are *foreground*: things that just
   appeared or are moving (parts riding the belt, an intruder, a thrown ball).
5. **Static geometry → boxes.** Static voxels are merged greedily into axis-aligned boxes.
   Box extents come from the mean measured surface position inside each voxel, not the
   voxel bounds, so surfaces are not inflated by up to a voxel.
6. **Foreground → tracked objects.** Foreground voxels are grouped into connected clusters.
   Each cluster's centroid and extent are associated with a fixed set of tracking slots
   (nearest neighbour with a gate), which feed the existing
   :class:`~momentum_solver.ObjectMomentumTracker`. The momentum fit and prediction are
   then exactly as before.
7. **Touch.** :class:`ContactEstimator` turns joint torques into an external-torque
   estimate. When the arm is pushed, it finds the point on the arm surface that best
   explains the torque. That point is added to the map as strong occupied evidence, which
   covers surfaces the cameras cannot see (occluded by the arm, or inside the tote).

Ground truth is never read here. The monitor only uses it afterwards, to score the map
and the forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from momentum_solver import Box, FrankaKinematics

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def depth_edges(depth: np.ndarray, jump: float = 0.05, rel: float = 0.04) -> np.ndarray:
    """Pixels at depth discontinuities: a 4-neighbour differs by more than max(jump, rel * depth).

    Along an object's silhouette, the rendered and the sensed depth both put some pixels between
    the object and the background, i.e. in free space. Moving with the arm, they look exactly
    like something fast entering free space, so they are dropped.
    """
    d = np.where(np.isfinite(depth), depth, 1e3).astype(np.float64)
    diff = np.zeros_like(d)
    diff[1:, :] = np.maximum(diff[1:, :], np.abs(d[1:, :] - d[:-1, :]))
    diff[:-1, :] = np.maximum(diff[:-1, :], np.abs(d[1:, :] - d[:-1, :]))
    diff[:, 1:] = np.maximum(diff[:, 1:], np.abs(d[:, 1:] - d[:, :-1]))
    diff[:, :-1] = np.maximum(diff[:, :-1], np.abs(d[:, 1:] - d[:, :-1]))
    return diff > np.maximum(jump, rel * d)


def backproject(depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray, stride: int = 2,
                max_range: float = 4.0, drop_edges: bool = True) -> np.ndarray:
    """World points ``(M, 3)`` from a depth image (distance to the image plane, m).

    USD cameras look down their local -Z axis with +Y up, while image rows grow downward.
    So a pixel ``(u, v)`` at depth ``z`` is the camera-frame point
    ``((u - cx) z / fx, -(v - cy) z / fy, -z)``. ``stride`` subsamples pixels; every second
    pixel is plenty for 3 cm voxels. Pixels with no return (inf), beyond ``max_range``, or
    on a depth discontinuity (:func:`depth_edges`) are dropped.
    """
    d = depth[::stride, ::stride].astype(np.float64)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride + 0.5
    v = v * stride + 0.5
    ok = np.isfinite(d) & (d > 0.0) & (d < max_range)
    if drop_edges:
        ok &= ~depth_edges(depth)[::stride, ::stride]
    z = d[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = -(v[ok] - K[1, 2]) * z / K[1, 1]
    p_cam = np.stack([x, y, -z], axis=1)
    return p_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def project(points: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel columns, rows (int) and depth along the view axis for world points ``(M, 3)``."""
    world_to_cam = np.linalg.inv(cam_to_world)
    p = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = -p[:, 2]
    zs = np.where(z > 1e-6, z, np.inf)
    u = np.floor(K[0, 0] * p[:, 0] / zs + K[0, 2]).astype(np.int64)
    v = np.floor(-K[1, 1] * p[:, 1] / zs + K[1, 2]).astype(np.int64)
    return u, v, z


@dataclass
class DepthFrame:
    """One depth image with everything needed to interpret it."""

    depth: np.ndarray  # (h, w) metres, inf = no return
    K: np.ndarray  # (3, 3) intrinsics
    cam_to_world: np.ndarray  # (4, 4)


# ---------------------------------------------------------------------------
# Self filter
# ---------------------------------------------------------------------------

# Extra filter spheres on the finger bodies (the collision model only covers the fingertips),
# as (link, local center, radius).
_FINGER_FILTER = [
    ("panda_leftfinger", (0.0, 0.01, 0.02), 0.02), ("panda_leftfinger", (0.0, 0.01, 0.04), 0.02),
    ("panda_rightfinger", (0.0, -0.01, 0.02), 0.02), ("panda_rightfinger", (0.0, -0.01, 0.04), 0.02),
    # The fixed base has no collision spheres (it cannot hit anything), but the cameras see it.
    ("panda_link0", (0.0, 0.0, 0.06), 0.11), ("panda_link0", (-0.09, 0.0, 0.06), 0.09),
    ("panda_link0", (0.0, 0.0, 0.16), 0.09),
    # Gaps between the collision spheres at the shoulder, found from surviving depth points.
    ("panda_link1", (0.0, -0.05, -0.07), 0.07), ("panda_link2", (0.0, -0.06, 0.05), 0.07),
]


class RobotSelfFilter:
    """Spheres covering the robot (and the carried part) at the current joint state."""

    def __init__(self, kin: FrankaKinematics, inflate: float = 0.025):
        self.kin = kin
        self.inflate = inflate
        self.extra_link = np.array([kin.link_index(n) for n, _, _ in _FINGER_FILTER])
        self.extra_local = np.array([c for _, c, _ in _FINGER_FILTER], dtype=np.float64)
        self.extra_radius = np.array([r for _, _, r in _FINGER_FILTER])

    def spheres(self, links: np.ndarray, carried: tuple[np.ndarray, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Centers ``(S, 3)`` and radii ``(S,)`` of the robot at link transforms ``links`` ``(L, 4, 4)``."""
        centers = self.kin.spheres(links[None])[0]
        T = links[self.extra_link]
        extra = np.einsum("sij,sj->si", T[:, :3, :3], self.extra_local) + T[:, :3, 3]
        centers = np.vstack([centers, extra])
        radii = np.concatenate([self.kin.sphere_radius, self.extra_radius])
        if carried is not None:
            centers = np.vstack([centers, carried[0][None]])
            radii = np.append(radii, carried[1])
        return centers, radii

    def keep(self, points: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Boolean mask of points that are NOT on the robot."""
        r = radii + self.inflate
        keep = np.ones(len(points), dtype=bool)
        # Cheap pre-cull: only test points inside the robot's bounding box, then inside the
        # bounding box of each run of consecutive spheres (consecutive spheres belong to the
        # same link, so each run is compact).
        lo, hi = (centers - r[:, None]).min(0), (centers + r[:, None]).max(0)
        near = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        if near.size == 0:
            return keep
        p = points[near]
        hit = np.zeros(len(p), dtype=bool)
        for s in range(0, len(centers), 8):
            c, rr = centers[s:s + 8], r[s:s + 8]
            clo, chi = (c - rr[:, None]).min(0), (c + rr[:, None]).max(0)
            sel = np.flatnonzero(~hit & np.all((p >= clo) & (p <= chi), axis=1))
            if sel.size:
                d2 = ((p[sel, None, :] - c[None]) ** 2).sum(-1)
                hit[sel] = np.any(d2 < (rr * rr)[None], axis=1)
        keep[near] = ~hit
        return keep


class LatencyEstimator:
    """Estimates the depth latency (in control steps) from the robot's own body in the depth.

    The renderer delivers depth a few steps late, and how late depends on load (4 steps in a
    lightly loaded run, 1 with perception running in the loop). The fixed cell camera almost
    always sees the arm, so for each candidate lag ``L`` count the depth points that fall inside
    the robot's spheres posed ``L`` steps ago: the true lag captures the most points whenever
    the arm moves. One frame's margin is tiny (the arm moves ~1 cm per step against thousands
    of robot points), so the raw count deficits relative to the best lag are summed over time,
    with slow forgetting so the estimate follows load changes. The estimate is the lag with the
    smallest accumulated deficit; it switches only when the new lag leads the current one by
    ``min_evidence`` points, which keeps it from flickering between neighbouring lags.
    """

    def __init__(self, robot_filter: RobotSelfFilter, max_lag: int = 8, forget: float = 0.999,
                 initial: int = 4, min_evidence: float = 50.0):
        self.filter = robot_filter
        self.max_lag = max_lag
        self.forget = forget
        self.min_evidence = min_evidence
        self.scores = np.zeros(max_lag + 1)
        self.lag = initial
        # Depth fused at a wrong lag leaves robot points in the static map for good, so until
        # one lag clearly leads callers filter the robot at every candidate lag.
        self.locked = False

    def update(self, points: np.ndarray, links_history: np.ndarray) -> int:
        """``points``: cell-camera world points; ``links_history[k]``: link transforms ``k`` steps ago."""
        n = min(len(links_history), self.max_lag + 1)
        if n < 2 or len(points) == 0:
            return self.lag
        counts = np.zeros(self.max_lag + 1)
        for k in range(n):
            c, r = self.filter.spheres(links_history[k])
            counts[k] = np.count_nonzero(~self.filter.keep(points, c, r))
        self.scores[:n] = self.forget * self.scores[:n] + counts[:n] - counts[:n].max()
        best = int(np.argmax(self.scores[:n]))
        current = self.scores[self.lag] if self.lag < n else -np.inf
        if self.scores[best] - current > self.min_evidence:
            self.lag = best
        if n == self.max_lag + 1 and self.scores[best] - np.sort(self.scores)[-2] > self.min_evidence:
            self.locked = True
        return self.lag


# ---------------------------------------------------------------------------
# Voxel map
# ---------------------------------------------------------------------------


_NEIGHBORS_26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


class VoxelMap:
    """Log-odds occupancy grid with static/foreground separation.

    Each voxel stores:

    * ``L``: log-odds of being occupied. Positive means occupied. Clamped to
      ``[L_MIN, L_MAX]``, so evidence never becomes so certain that it cannot change.
    * ``since``: the sim time the voxel last became occupied (NaN while free). Used to tell
      static scene from things that are moving through.
    * ``mean``: running mean of the depth points that landed in the voxel. Static boxes are
      sized from it, so surfaces sit where they were measured instead of at voxel bounds.
    """

    L_HIT = 0.85
    L_MISS = 0.4
    L_TOUCH = 3.0  # a localized touch contact is near-certain evidence
    L_MIN = -2.0
    L_MAX = 3.5
    STATIC_AFTER = 1.5  # seconds a voxel must stay occupied to count as static scene
    DYNAMIC_BELOW = -1.2  # log-odds below which a voxel counts as observed free
    SURFACE_TOL = 0.02  # points this close to an adjacent static surface are that surface
    FLICKER_RATE = 0.1  # recent hit rate above which a free voxel is a flickering surface

    def __init__(self, lo=(-0.6, -1.0, -0.03), hi=(1.7, 1.4, 1.6), voxel: float = 0.03):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.voxel = voxel
        self.shape = tuple(int(math.ceil(v)) for v in (np.asarray(hi) - self.lo) / voxel)
        n = int(np.prod(self.shape))
        self.L = np.zeros(n, dtype=np.float32)
        self.since = np.full(n, np.nan)
        self.mean = np.zeros((n, 3))
        self.count = np.zeros(n, dtype=np.float32)
        # Voxel centers, flattened in C order to match the flat arrays above.
        idx = np.indices(self.shape).reshape(3, -1).T
        self.centers = self.lo + (idx + 0.5) * voxel
        self.hit_now = np.zeros(n, dtype=bool)
        self.dynamic = np.zeros(n, dtype=bool)
        self.hit_raw = np.zeros(n, dtype=bool)  # voxels hit this step (outside the arm)
        self.frame_count = np.zeros(n)  # number of this step's points per voxel (valid where hit_raw)
        self.hit_rate = np.zeros(n, dtype=np.float32)  # running fraction of observations that were hits
        self.frame_mean = np.zeros((n, 3))  # mean of this step's points per voxel (valid where hit_now)
        self.t = 0.0

    # -- indexing -----------------------------------------------------------

    def flat_index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat voxel index of each point, and a mask of points inside the grid."""
        ijk = np.floor((points - self.lo) / self.voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=1)
        flat = np.ravel_multi_index(ijk[inside].T, self.shape) if inside.any() else np.zeros(0, dtype=np.int64)
        return flat, inside

    def voxels_in_spheres(self, centers: np.ndarray, radii: np.ndarray, margin: float = 0.0):
        """Flat indices of voxels whose centers lie inside any of the spheres.

        With ``margin > 0``, returns ``(inside, inside_with_margin)``: the voxels inside the
        spheres, and those inside the spheres grown by ``margin``.
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        empty = np.zeros(0, dtype=np.int64)
        if len(centers) == 0:
            return (empty, empty) if margin > 0 else empty
        # A cubic stencil of voxel offsets placed at every sphere; spheres are grouped by the
        # stencil size they need so small spheres do not pay for the largest one.
        reach = np.ceil((radii + margin) / self.voxel).astype(int) + 1
        inner, outer = [], []
        for rch in np.unique(reach):
            g = reach == rch
            o = np.arange(-rch, rch + 1)
            stencil = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
            c, r = centers[g], radii[g]
            ijk = np.floor((c - self.lo) / self.voxel).astype(np.int64)[:, None, :] + stencil[None]  # (S, K, 3)
            d2 = ((self.lo + (ijk + 0.5) * self.voxel - c[:, None, :]) ** 2).sum(-1)
            valid = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=2)
            m_out = valid & (d2 <= ((r + margin) ** 2)[:, None])
            outer.append(np.ravel_multi_index(ijk[m_out].T, self.shape))
            if margin > 0:
                m_in = m_out & (d2 <= (r * r)[:, None])
                inner.append(np.ravel_multi_index(ijk[m_in].T, self.shape))
        if margin > 0:
            return np.unique(np.concatenate(inner)), np.unique(np.concatenate(outer))
        return np.unique(np.concatenate(outer))

    # -- update -------------------------------------------------------------

    def update(self, t: float, frames: list[DepthFrame], points: np.ndarray, free: np.ndarray,
               touch: np.ndarray | None = None) -> None:
        """Integrate one time step.

        Args:
            t: Sim time.
            frames: Raw depth frames (used for the see-through test).
            points: Self-filtered world points from all frames ``(M, 3)``.
            free: Flat indices known to be free (inside the robot right now).
            touch: World points of localized touch contacts ``(C, 3)`` (optional).
        """
        self.t = t
        flat, inside = self.flat_index(points)
        hit = np.zeros(self.L.size, dtype=bool)
        hit[flat] = True
        # Running mean of measured surface position per voxel. Each frame counts as one
        # observation (its own point mean); capping the count lets old evidence fade.
        n = self.L.size
        upd = np.flatnonzero(hit)
        w = np.bincount(flat, minlength=n)[upd].astype(np.float64)
        p = points[inside]
        acc = np.stack([np.bincount(flat, weights=p[:, a], minlength=n)[upd] for a in range(3)], 1)
        fm = acc / w[:, None]
        self.frame_mean[upd] = fm
        self.frame_count[upd] = w
        # Change detection: a return inside space the cameras have seen through means something
        # has moved in. It stays "dynamic" for as long as it keeps being hit. Returns from
        # never-observed space are new scene surfaces instead (the wrist camera looking somewhere
        # new), and only become static boxes after STATIC_AFTER.
        # A voxel that is free now but was hit often lately is a flickering surface (conveyor
        # rollers and the gaps between them, seen alternately as the wrist moves), not change.
        consistently_free = (self.L < self.DYNAMIC_BELOW) & (self.hit_rate < self.FLICKER_RATE)
        cand = hit & (consistently_free | self.dynamic)
        self.dynamic = cand & ~self._on_static_surface(np.flatnonzero(cand))
        self.count[upd] = np.minimum(self.count[upd] + 1.0, 50.0)
        alpha = (1.0 / self.count[upd])[:, None]
        self.mean[upd] = (1 - alpha) * self.mean[upd] + alpha * fm

        # See-through test: only voxels that currently hold some evidence can lose it.
        miss = np.zeros(self.L.size, dtype=bool)
        cand = np.flatnonzero((self.L > self.L_MIN) & ~hit)
        if cand.size:
            for fr in frames:
                h, wd = fr.depth.shape
                u, v, z = project(self.centers[cand], fr.K, fr.cam_to_world)
                ok = (z > 0.1) & (u >= 0) & (u < wd) & (v >= 0) & (v < h)
                meas = np.full(cand.size, -np.inf)
                meas[ok] = fr.depth[v[ok], u[ok]]
                meas = np.where(np.isinf(meas) & (meas > 0), 1e9, meas)  # no return: the ray went through
                miss[cand[ok & (meas > z + self.voxel)]] = True

        self.L[hit] += self.L_HIT
        self.L[miss] -= self.L_MISS
        seen = hit | miss
        self.hit_rate[seen] = 0.9 * self.hit_rate[seen] + 0.1 * hit[seen]
        if touch is not None and len(touch):
            tf, tin = self.flat_index(touch)
            self.L[tf] += self.L_TOUCH
            hit[tf] = True
            self.mean[tf] = touch[tin]
        # Space inside the arm cannot be occupied, but carving is not a camera observation of
        # free space: with the margin it also clips surfaces the arm passes close to, and those
        # must not then look like something that moved in. So carve to "unknown", not "free".
        self.L[free] = np.minimum(self.L[free], 0.0)
        self.dynamic[free] = False
        hit[free] = False
        self.hit_raw = hit
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)

        occ = self.L > 0
        newly = occ & np.isnan(self.since)
        self.since[newly] = t
        self.since[~occ] = np.nan
        self.mean[newly & (self.count == 0)] = self.centers[newly & (self.count == 0)]
        self.hit_now = hit & occ

    def _on_static_surface(self, idx: np.ndarray) -> np.ndarray:
        """Mask (over all voxels) of ``idx`` whose points this step lie on a neighbouring static surface.

        A surface lying on a voxel boundary spills into the free voxel next to it under
        centimetre-level wrist-pose error, which would look like change. Such a voxel's points
        sit within SURFACE_TOL of the measured points of an adjacent static voxel, whereas the
        top of a part resting on that surface is several centimetres away.
        """
        out = np.zeros(self.L.size, dtype=bool)
        if idx.size == 0:
            return out
        static = self.static_mask()
        ijk = np.stack(np.unravel_index(idx, self.shape), 1)
        nbr = ijk[:, None, :] + _NEIGHBORS_26[None]
        valid = np.all((nbr >= 0) & (nbr < np.asarray(self.shape)), axis=2)
        nflat = np.ravel_multi_index(np.clip(nbr, 0, np.asarray(self.shape) - 1).transpose(2, 0, 1), self.shape)
        valid &= static[nflat]
        dist = np.linalg.norm(self.frame_mean[idx][:, None] - self.mean[nflat], axis=2)
        out[idx] = np.any(valid & (dist < self.SURFACE_TOL), axis=1)
        return out

    # -- queries --------------------------------------------------------------

    def static_mask(self) -> np.ndarray:
        return (self.L > 0) & (self.t - self.since >= self.STATIC_AFTER)

    def foreground_clusters(self, min_voxels: int = 2, max_size: float = 0.3) -> list[dict]:
        """Connected groups of voxels hit this step that moved into observed free space.

        Clusters wider than ``max_size`` along any axis are dropped: they are scene surfaces that
        were mistaken for change (e.g. a whole tote floor), not graspable-size objects.
        """
        cube = np.ones((3, 3, 3), dtype=bool)
        static = self.static_mask()
        seed_idx = np.flatnonzero(self.dynamic & ~static)
        clusters = []
        if seed_idx.size == 0:
            return clusters
        # Work in the bounding region of the seeds (plus the one-voxel growth) only.
        ijk = np.stack(np.unravel_index(seed_idx, self.shape), 1)
        lo = np.maximum(ijk.min(0) - 1, 0)
        hi = np.minimum(ijk.max(0) + 2, self.shape)
        sub = tuple(slice(a, b) for a, b in zip(lo, hi))
        seeds = (self.dynamic & ~static).reshape(self.shape)[sub]
        fg = ndimage.binary_dilation(seeds, structure=cube) & (self.hit_raw & ~static).reshape(self.shape)[sub]
        labels_sub, n = ndimage.label(fg, structure=cube)
        if n == 0:
            return clusters
        labels = np.zeros(self.shape, dtype=labels_sub.dtype)
        labels[sub] = labels_sub
        flat_labels = labels.reshape(-1)
        idx = np.flatnonzero(flat_labels)
        order = np.argsort(flat_labels[idx])
        idx = idx[order]
        splits = np.flatnonzero(np.diff(flat_labels[idx])) + 1
        for group in np.split(idx, splits):
            if group.size < min_voxels:
                continue
            m = self.frame_mean[group]
            lo, hi = m.min(0) - self.voxel / 2, m.max(0) + self.voxel / 2
            if np.any(hi - lo > max_size):
                continue
            # A camera sees only the near faces, often just the top: give the box at least the
            # height of its smaller footprint side, hanging down from the observed top.
            depth = min(hi[0] - lo[0], hi[1] - lo[1])
            lo[2] = min(lo[2], hi[2] - depth)
            # Report the point centroid (it moves smoothly, unlike voxel bounds, which jump as
            # columns enter and leave) and grow the half extents so the box still covers lo..hi.
            w = self.frame_count[group]
            center = (m * w[:, None]).sum(0) / w.sum()
            half = 0.5 * (hi - lo) + np.abs(center - 0.5 * (lo + hi))
            clusters.append({"center": center, "half": half, "voxels": group.size})
        return clusters

    def static_boxes(self, exclude: np.ndarray | None = None, pad: float = 0.01) -> list[Box]:
        """Greedy merge of static voxels into axis-aligned boxes.

        Scan voxels in index order. From each still-unclaimed static voxel, grow a box as far
        as possible along +x, then +y (whole rows must be static), then +z (whole slabs), and
        claim its voxels. This is the classic "greedy meshing" of voxel worlds. It gives few
        boxes for big flat surfaces (floor, belt top, tote walls) and many for clutter.

        Args:
            exclude: Flat indices to leave out (e.g. around the part about to be grasped).
            pad: Half-thickness added around the measured surface points.
        """
        grid = self.static_mask()
        if exclude is not None and len(exclude):
            grid[exclude] = False
        grid = grid.reshape(self.shape)
        mean = self.mean.reshape(*self.shape, 3)
        boxes: list[Box] = []
        nx, ny, nz = self.shape
        for i, j, k in np.argwhere(grid):
            if not grid[i, j, k]:
                continue
            i1 = i
            while i1 + 1 < nx and grid[i1 + 1, j, k]:
                i1 += 1
            j1 = j
            while j1 + 1 < ny and grid[i:i1 + 1, j1 + 1, k].all():
                j1 += 1
            k1 = k
            while k1 + 1 < nz and grid[i:i1 + 1, j:j1 + 1, k1 + 1].all():
                k1 += 1
            grid[i:i1 + 1, j:j1 + 1, k:k1 + 1] = False
            m = mean[i:i1 + 1, j:j1 + 1, k:k1 + 1].reshape(-1, 3)
            # Along axes where the box spans several voxels, the voxel bounds are the honest
            # extent; along a one-voxel axis, the measured surface is tighter.
            vlo = self.lo + np.array([i, j, k]) * self.voxel
            vhi = self.lo + (np.array([i1, j1, k1]) + 1) * self.voxel
            span = np.array([i1 - i, j1 - j, k1 - k])
            lo = np.where(span == 0, m.min(0) - pad, vlo)
            hi = np.where(span == 0, m.max(0) + pad, vhi)
            boxes.append(Box(f"map_{len(boxes)}", tuple(0.5 * (lo + hi)), tuple(0.5 * (hi - lo))))
        return boxes


# ---------------------------------------------------------------------------
# Object slots
# ---------------------------------------------------------------------------


class ClusterTracker:
    """Associates foreground clusters with a fixed number of slots across time.

    :class:`~momentum_solver.ObjectMomentumTracker` expects a fixed object list, so perceived
    objects live in ``n_slots`` slots. Each step, clusters are matched to slots greedily by
    distance to where the slot's object was predicted to be (last position + velocity × dt),
    within ``gate`` metres. Unmatched clusters take free slots; slots unmatched for
    ``drop_after`` seconds are freed. A slot's position is NaN while it is empty, which the
    momentum tracker treats as "no history".

    A new slot is only published after it has been matched in ``confirm`` consecutive steps.
    Depth flicker (e.g. conveyor rollers and the gaps between them, alternately seen as the
    wrist camera moves) makes clusters that last a step or two; real objects persist.
    """

    def __init__(self, n_slots: int = 8, gate: float = 0.15, drop_after: float = 0.2, min_voxels: int = 2,
                 confirm: int = 3):
        self.n = n_slots
        self.gate = gate
        self.drop_after = drop_after
        self.min_voxels = min_voxels
        self.confirm = confirm
        self.streak = np.zeros(n_slots, dtype=int)
        self.position = np.full((n_slots, 3), np.nan)
        self.velocity = np.zeros((n_slots, 3))
        self.half = np.full((n_slots, 3), 0.02)
        self.last_seen = np.full(n_slots, -np.inf)
        self.t = None
        self.reset_slots: list[int] = []  # slots (re)assigned this step: history must be forgotten

    def update(self, t: float, clusters: list[dict]) -> np.ndarray:
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        self.reset_slots = []
        active = np.isfinite(self.position[:, 0])
        predicted = self.position + self.velocity * dt
        clusters = [c for c in clusters if c["voxels"] >= self.min_voxels]
        used = np.zeros(self.n, dtype=bool)
        new_pos = np.full((self.n, 3), np.nan)
        # Largest clusters first: they are the most reliable to keep identity.
        for c in sorted(clusters, key=lambda c: -c["voxels"]):
            d = np.where(active & ~used, np.linalg.norm(predicted - c["center"], axis=1), np.inf)
            s = int(np.argmin(d))
            if not np.isfinite(d[s]) or d[s] > self.gate:
                free = np.flatnonzero(~active & ~used)
                if free.size == 0:
                    continue
                s = int(free[0])
                self.reset_slots.append(s)
                self.velocity[s] = 0.0
                self.streak[s] = 0
            used[s] = True
            self.streak[s] += 1
            if dt > 0 and np.isfinite(self.position[s, 0]) and s not in self.reset_slots:
                self.velocity[s] = 0.5 * self.velocity[s] + 0.5 * (c["center"] - self.position[s]) / dt
            new_pos[s] = c["center"]
            self.half[s] = np.maximum(c["half"], 0.015)
            self.last_seen[s] = t
        # Keep briefly-unseen objects coasting on their velocity, then free the slot.
        coast = ~used & active & (t - self.last_seen <= self.drop_after)
        new_pos[coast] = predicted[coast]
        self.velocity[~used & ~coast] = 0.0
        self.streak[~used & ~coast] = 0
        self.position = new_pos
        published = new_pos.copy()
        published[self.streak < self.confirm] = np.nan
        return published


# ---------------------------------------------------------------------------
# Touch
# ---------------------------------------------------------------------------


# Unit directions used to place candidate contact points on each collision sphere's surface:
# the 6 face, 12 edge and 8 corner directions of a cube.
_DIRECTIONS = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)], dtype=np.float64) - 1.0
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True)


class ContactEstimator:
    """External joint torque from measured torques, and localization of the contact on the arm.

    External torque
    ---------------
    The manipulator equation with an external force is

        M qdd + c + g = tau_joint + tau_ext,      tau_ext = J_c^T f_ext

    so ``tau_ext = M qdd + c + g - tau_joint``. The joint torque comes from the joint
    torque sensors (in Isaac Sim, ``get_dof_projected_joint_forces``; ``get_dof_efforts`` is
    zero for implicit PD drives), and ``qdd`` from differencing measured velocities. The
    estimate is noisy, so it is low-pass filtered. A contact is declared when any arm joint
    exceeds ``threshold`` times that joint's running noise level.

    On the captured runs, free motion stays below a score of about 15, while hard contacts
    score 20-400. Light grazes (the hand brushing the tote rim) stay at noise level: torque
    sensing only catches contacts that actually push the arm.

    Localization
    ------------
    A single contact force ``f`` (3-vector) at point ``p`` on the arm produces
    ``tau_ext = J_p(q)^T f``, where ``J_p`` is the 3×7 Jacobian of the point. For every
    candidate point on the arm surface (26 points on each collision sphere), solve the small
    least-squares problem ``min_f |J_p^T f - tau_ext|`` and keep the candidate with the
    smallest residual whose force pushes *into* the arm (contacts can only push). This is a
    one-shot version of the "contact particle filter" (Manuelli & Tedrake, 2016).

    Jacobians are computed by finite differences of the batched forward kinematics, with 8
    FK evaluations in one batch.
    """

    # Mean absolute filtered residual per arm joint in free motion, measured on the captured runs
    # (N·m). The running estimate starts here and never drops below NOISE_FLOOR.
    NOISE_INIT = (0.25, 0.9, 0.6, 0.45, 0.08, 0.08, 0.08)
    NOISE_FLOOR = (0.2, 0.2, 0.2, 0.2, 0.06, 0.06, 0.06)

    def __init__(self, kin: FrankaKinematics, dt: float, threshold: float = 20.0, alpha: float = 0.3,
                 noise_floor: np.ndarray | None = None):
        self.kin = kin
        self.dt = dt
        self.threshold = threshold
        self.alpha = alpha
        self.noise_floor = np.asarray(noise_floor if noise_floor is not None else self.NOISE_FLOOR, dtype=np.float64)
        self.noise = np.maximum(np.asarray(self.NOISE_INIT, dtype=np.float64), self.noise_floor)
        self._prev_qd: np.ndarray | None = None
        self._prev_t: float | None = None
        self.tau_ext = np.zeros(7)

    def update(self, t: float, q, qd, M, c, g, tau_joint, learn: bool = True) -> np.ndarray:
        """Filtered external torque on the 7 arm joints."""
        if self._prev_qd is None or self._prev_t is None or t - self._prev_t <= 0:
            self._prev_qd, self._prev_t = qd.copy(), t
            return self.tau_ext
        qdd = (qd - self._prev_qd) / (t - self._prev_t)
        raw = (M @ qdd + c + g - tau_joint)[:7]
        self._prev_qd, self._prev_t = qd.copy(), t
        self.tau_ext = (1 - self.alpha) * self.tau_ext + self.alpha * raw
        if learn and not self.in_contact():
            # Slow running estimate of each joint's free-motion noise (mean absolute value).
            self.noise = np.maximum(0.995 * self.noise + 0.005 * np.abs(self.tau_ext), self.noise_floor)
        return self.tau_ext

    def score(self) -> float:
        """How far the external torque is above noise (max over joints, in noise units)."""
        return float(np.max(np.abs(self.tau_ext) / self.noise))

    def in_contact(self) -> bool:
        return self.score() > self.threshold

    def localize(self, q: np.ndarray, base: np.ndarray, tau_ext: np.ndarray | None = None,
                 extra: tuple[np.ndarray, float] | None = None, eps: float = 1e-4) -> dict | None:
        """Best single-contact explanation of ``tau_ext``.

        Args:
            q: Joint positions (9,).
            base: Robot base transform.
            tau_ext: External torque on the 7 arm joints (defaults to the filtered estimate).
            extra: Optional (center in the panda_hand frame, radius) of a carried part, which
                becomes extra candidate points attached to the hand.

        Returns:
            ``{"point", "normal", "force", "residual", "link"}`` or None. ``normal`` is the
            arm's outward surface normal at the contact, which points toward the obstacle.
        """
        tau = self.tau_ext if tau_ext is None else tau_ext
        batch = np.repeat(q[None], 8, axis=0)
        batch[1:, :7] += eps * np.eye(7)
        links = self.kin.forward(batch, base)  # (8, L, 4, 4)
        T = links[:, self.kin.sphere_link]  # (8, S, 4, 4)
        R = T[..., :3, :3]
        centers = np.einsum("bsij,sj->bsi", R, self.kin.sphere_local) + T[..., :3, 3]
        radius = self.kin.sphere_radius
        names = list(self.kin.sphere_link_name)
        if extra is not None:
            H = links[:, self.kin.link_index("panda_hand")]
            ec = np.einsum("bij,j->bi", H[:, :3, :3], extra[0]) + H[:, :3, 3]
            centers = np.concatenate([centers, ec[:, None]], axis=1)
            R = np.concatenate([R, H[:, None, :3, :3]], axis=1)
            radius = np.append(radius, extra[1])
            names.append("carried")
        # Candidate points: sphere center + radius * (direction fixed in the link frame).
        dirs = np.einsum("bsij,dj->bsdi", R, _DIRECTIONS)  # (8, S, D, 3)
        pts = centers[:, :, None, :] + radius[None, :, None, None] * dirs
        J = (pts[1:] - pts[0]) / eps  # (7, S, D, 3): d point / d q_j
        J = J.transpose(1, 2, 3, 0).reshape(-1, 3, 7)  # (C, 3, 7)
        A = J @ J.transpose(0, 2, 1) + 1e-9 * np.eye(3)  # (C, 3, 3)
        f = np.linalg.solve(A, (J @ tau)[..., None])[..., 0]  # (C, 3)
        resid = np.linalg.norm(np.einsum("cij,ci->cj", J, f) - tau, axis=1)
        normal = dirs[0].reshape(-1, 3)
        fn = np.linalg.norm(f, axis=1) + 1e-12
        pushes_in = np.einsum("ci,ci->c", f, normal) < -0.5 * fn  # within a ~60 deg friction cone
        if not pushes_in.any():
            return None
        resid = np.where(pushes_in, resid, np.inf)
        best = int(np.argmin(resid))
        s = best // len(_DIRECTIONS)
        return {
            "point": pts[0].reshape(-1, 3)[best],
            "normal": normal[best],
            "force": f[best],
            "residual": float(resid[best] / (np.linalg.norm(tau) + 1e-9)),
            "link": names[s],
        }


# ---------------------------------------------------------------------------
# Task knowledge: what the gripper holds and which contacts are expected
# ---------------------------------------------------------------------------

# Midpoint between the fingertip pads in the panda_hand frame (finger joint origin + fingertip offset).
FINGERTIP_MID = np.array([0.0, 0.0, 0.0584 + 0.045])
PICK_PHASES = ("APPROACH_PICK", "DESCEND_PICK", "GRASP", "LIFT")
HOLDING_PHASES = ("LIFT", "APPROACH_PLACE", "DESCEND_PLACE")
PLACE_PHASES = ("DESCEND_PLACE", "RELEASE", "RETREAT")
EXPECTED_TOUCH = ("DESCEND_PICK", "GRASP", "LIFT", "DESCEND_PLACE", "RELEASE")  # contact is part of the task
WRIST_MIN_RANGE = 0.17  # closer wrist-camera returns are the robot's own hand and fingers


def carried_part(q: np.ndarray, hand: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """The grasped part, from proprioception alone.

    The part sits between the fingertip pads, so its center is the pad midpoint, and the
    finger opening (sum of both finger joints) is its width. Returns (world center, radius,
    center in the hand frame).
    """
    radius = max(0.5 * float(q[7] + q[8]), 0.012)
    center = hand[:3, :3] @ FINGERTIP_MID + hand[:3, 3]
    return center, radius, FINGERTIP_MID.copy()


def task_rules(phase: str, boxes: list[Box], slot_positions: np.ndarray, pick: np.ndarray | None,
               place: np.ndarray | None, carried_offset: np.ndarray | None, carried_radius: float | None = None):
    """Expected contacts for the current pick-and-place phase, expressed on perceived geometry.

    The world-knowledge monitor knows which object is the grasp target and which box is
    "the conveyor". Here, the only task knowledge is the policy's own commanded pick and
    place poses (``pick``, ``place``: world points between the fingertips), so:

    * while picking, perceived objects near the pick point are the target (may be touched),
      and static voxels around it are left out of the boxes;
    * while lifting, the carried part may brush boxes around the pick location (the tote);
    * while placing, the carried part may touch the box it is placed on, i.e. boxes under
      the place point, and the just-released part may be touched by the gripper.

    Returns (ContactRules, exclude regions for :meth:`VoxelMap.static_boxes`).
    """
    from momentum_solver import ContactRules

    rules = ContactRules()
    exclude: list[tuple[np.ndarray, float]] = []
    valid = np.isfinite(slot_positions[:, 0])

    def slots_near(p, r):
        return {int(i) for i in np.flatnonzero(valid & (np.linalg.norm(slot_positions - p, axis=1) < r))}

    if phase in PICK_PHASES and pick is not None:
        rules.ignore_objects = slots_near(pick, 0.10)
        exclude.append((pick, 0.06))
    if phase in PLACE_PHASES and place is not None:
        rules.ignore_objects |= slots_near(place, 0.10)
    if phase in HOLDING_PHASES and carried_offset is not None:
        rules.carried_offset = carried_offset
        rules.carried = -1  # the carried part is not one of the tracked slots
        rules.carried_radius = carried_radius
        if phase == "LIFT" and pick is not None:
            rules.carried_ignore_boxes = {b.name for b in boxes
                                          if np.hypot(*np.maximum(np.abs(np.asarray(b.center[:2]) - pick[:2]) - b.half[:2], 0.0)) < 0.12}
        elif phase == "DESCEND_PLACE" and place is not None:
            rules.carried_ignore_boxes = {b.name for b in boxes
                                          if np.all(np.abs(np.asarray(b.center[:2]) - place[:2]) <= np.asarray(b.half[:2]) + 0.05)
                                          and b.center[2] + b.half[2] <= place[2] + 0.02}
    return rules, exclude


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------


class PerceivedScene:
    """Owns the map, self filter, cluster tracker and contact estimator; produces solver inputs.

    Call :meth:`update` once per control step with the depth frames and proprioception, then
    read :attr:`boxes` (static obstacles) and :attr:`object_positions` / :attr:`object_half`
    (tracked foreground objects).
    """

    CARVE_MARGIN = 0.02

    def __init__(self, kin: FrankaKinematics, dt: float, n_slots: int = 8, voxel: float = 0.03,
                 box_every: float = 0.25, contact_threshold: float = 20.0, noise_floor: np.ndarray | None = None):
        self.kin = kin
        self.dt = dt
        self.map = VoxelMap(voxel=voxel)
        self.filter = RobotSelfFilter(kin)
        self.slots = ClusterTracker(n_slots)
        self.contacts = ContactEstimator(kin, dt, threshold=contact_threshold, noise_floor=noise_floor)
        self.box_every = box_every
        self.boxes: list[Box] = []
        self.boxes_changed = False
        self._last_boxes = -np.inf
        self.object_positions = np.full((n_slots, 3), np.nan)
        self.object_half = np.full((n_slots, 3), 0.02)
        self.object_time = 0.0
        self.touches: list[dict] = []  # localized contacts (for logging)
        self.timings_ms: dict[str, float] = {}

    def update(self, t: float, frames: list[DepthFrame], q, qd, M, c, g, tau_joint, base: np.ndarray,
               carried: tuple[np.ndarray, float] | None = None, exclude: list[tuple[np.ndarray, float]] = (),
               expected_touch: bool = False, t_sensor: float | None = None, q_sensor: np.ndarray | None = None,
               carried_sensor: tuple[np.ndarray, float] | None = None,
               q_uncertain: np.ndarray | None = None) -> None:
        """Integrate one step.

        Args:
            t: Sim time.
            frames: Depth frames from both cameras. Their wrist pose must be the pose at
                ``t_sensor``.
            q, qd, M, c, g, tau_joint: Proprioception and measured joint torques at ``t``.
            base: Robot base transform.
            carried: (world center, radius) of the part in the gripper, filtered like the robot.
            exclude: (world center, radius) regions left out of the static boxes, e.g. the part
                about to be grasped, which the gripper is meant to touch.
            expected_touch: True while contact is part of the task (grasping, placing). Touches
                are then not added to the map.
            t_sensor, q_sensor, carried_sensor: Time, joint state and carried part when the depth
                frames were captured. Depth arrives a few steps late (camera latency), so the
                robot must be filtered out at the pose it had *then*. Default to the current ones.
            q_uncertain: Joint states (one per row) the robot may have had when the frames were
                captured, while the latency is not yet known. The robot is filtered at all of
                them, which keeps it out of the map at the cost of hiding a little scene.
        """
        import time

        t_sensor = t if t_sensor is None else t_sensor
        q_sensor = q if q_sensor is None else q_sensor
        carried_sensor = carried if q_sensor is q else carried_sensor
        t0 = time.perf_counter()
        links = self.kin.forward(q[None], base)[0]
        # Filter with the arm swept over ±1 step around the sensor pose: at 2 m/s the hand moves
        # 3 cm per step, so a one-step error in the depth timestamp would leak arm points.
        sweep = q_sensor[None] + np.array([-1.0, 0.0, 1.0])[:, None] * self.dt * qd[None]
        sweep_links = self.kin.forward(sweep, base)
        hand_i = self.kin.link_index("panda_hand")
        s_centers, s_radii, sensor_spheres = [], [], None
        for k in range(len(sweep)):
            car = None
            if carried_sensor is not None:
                Hs, Hk = sweep_links[1, hand_i], sweep_links[k, hand_i]
                local = Hs[:3, :3].T @ (carried_sensor[0] - Hs[:3, 3])
                car = (Hk[:3, :3] @ local + Hk[:3, 3], carried_sensor[1])
            c_k, r_k = self.filter.spheres(sweep_links[k], car)
            if k == 1:
                sensor_spheres = (c_k, r_k)
            s_centers.append(c_k)
            s_radii.append(r_k)
        if q_uncertain is not None:
            for links_k in self.kin.forward(np.asarray(q_uncertain, dtype=np.float64), base):
                c_k, r_k = self.filter.spheres(links_k)
                s_centers.append(c_k)
                s_radii.append(r_k)
        s_centers, s_radii = np.vstack(s_centers), np.concatenate(s_radii)
        pts = [backproject(fr.depth, fr.K, fr.cam_to_world) for fr in frames]
        pts = np.vstack(pts) if pts else np.zeros((0, 3))
        pts = pts[self.filter.keep(pts, s_centers, s_radii)]
        centers, radii = self.filter.spheres(links, carried)
        t1 = time.perf_counter()

        tau_ext = self.contacts.update(t, q, qd, M, c, g, tau_joint)
        touch = None
        if self.contacts.in_contact() and not expected_touch:
            hand = links[self.kin.link_index("panda_hand")]
            extra = None
            if carried is not None:
                extra = (hand[:3, :3].T @ (carried[0] - hand[:3, 3]), carried[1])
            loc = self.contacts.localize(q, base, tau_ext, extra)
            if loc is not None and loc["residual"] < 0.35:
                p = loc["point"] + loc["normal"] * 0.5 * self.map.voxel
                touch = p[None]
                self.touches.append({"t": t, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in loc.items()}})
        t2 = time.perf_counter()

        # Carve with a margin: the spheres under-approximate the arm's surface in places, and a
        # voxel whose center is just outside a sphere can still hold arm points. Static scene is
        # only carved where the arm strictly is, so the gripper approaching a surface (to grasp
        # or place) does not reset it.
        all_c, all_r = np.vstack([centers, sensor_spheres[0]]), np.concatenate([radii, sensor_spheres[1]])
        strict, free = self.map.voxels_in_spheres(all_c, all_r, margin=self.CARVE_MARGIN)
        static = self.map.static_mask()
        free = np.union1d(free[~static[free]], strict)
        self.map.update(t_sensor, frames, pts, free, touch)
        t3 = time.perf_counter()

        self.object_positions = self.slots.update(t_sensor, self.map.foreground_clusters())
        self.object_time = t_sensor
        self.object_half = self.slots.half.copy()
        t4 = time.perf_counter()

        self.boxes_changed = False
        if t - self._last_boxes >= self.box_every:
            ex = [self.map.voxels_in_spheres(c[None], np.array([r])) for c, r in exclude]
            self.boxes = self.map.static_boxes(np.concatenate(ex) if ex else None)
            self._last_boxes = t
            self.boxes_changed = True
        t5 = time.perf_counter()
        self.timings_ms = {"backproject": (t1 - t0) * 1e3, "touch": (t2 - t1) * 1e3, "map": (t3 - t2) * 1e3,
                           "clusters": (t4 - t3) * 1e3, "boxes": (t5 - t4) * 1e3, "total": (t5 - t0) * 1e3}
