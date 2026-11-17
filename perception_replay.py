"""Replay a raw sensor capture through the perceived-scene pipeline and the momentum solver, offline.

A capture (``franka_factory_pick_place.py --perception --capture X.npz``) holds, per control
step: both depth images, proprioception, joint torques, the drive targets, and ground truth.
This script rebuilds the scene from the sensors only, runs the solver against it, and scores:

* the map against the ground-truth cell boxes (how much of the real surface is covered, and
  how much perceived geometry is spurious);
* perceived object tracks against ground-truth object positions;
* collision forecasts against the real PhysX contacts. The same replay with the
  ground-truth scene is the baseline.

Usage::

    python perception_replay.py runs/perception/intruder_raw.npz --out runs/perception/intruder_perceived.npz
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from momentum_eval import alert_episodes, contact_events, score, timing_stats
from momentum_solver import (
    ArmMomentumModel,
    Box,
    ContactRules,
    DriveModel,
    FrankaKinematics,
    MomentumSolver,
    ObjectMomentumTracker,
    Proprioception,
    sphere_box_distance,
)
from scene_perception import (
    EXPECTED_TOUCH,
    FINGERTIP_MID,
    HOLDING_PHASES,
    WRIST_MIN_RANGE,
    DepthFrame,
    LatencyEstimator,
    PerceivedScene,
    RobotSelfFilter,
    backproject,
    carried_part,
    task_rules,
)

PHASES = ["", "APPROACH_PICK", "DESCEND_PICK", "GRASP", "LIFT", "APPROACH_PLACE", "DESCEND_PLACE", "RELEASE", "RETREAT", "DONE", "FAILED"]


def load(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def commanded_poses(d: dict, kin: FrankaKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Per-step pick and place points (fingertip midpoint of the commanded grasp/place pose).

    Online these come straight from the policy's goal. The capture does not store the goal,
    so the replay uses where the fingertips actually were when GRASP and RELEASE started.
    That is the commanded pose to within the controller's tracking error.
    """
    n = len(d["t"])
    pick, place = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    hand = kin.forward(d["q"].astype(np.float64), d["base"])[:, kin.link_index("panda_hand")]
    tips = np.einsum("nij,j->ni", hand[:, :3, :3], FINGERTIP_MID) + hand[:, :3, 3]
    phase = d["phase"]
    for c in np.unique(d["cycle"]):
        idx = np.flatnonzero(d["cycle"] == c)
        g = idx[phase[idx] == PHASES.index("GRASP")]
        r = idx[phase[idx] == PHASES.index("RELEASE")]
        if g.size:
            pick[idx] = tips[g[0]]
        if r.size:
            place[idx] = tips[r[0]]
    return pick, place


def drive_model(d: dict) -> DriveModel:
    return DriveModel(stiffness=d["kp"].astype(np.float64), damping=d["kd"].astype(np.float64),
                      max_effort=d["max_effort"].astype(np.float64), acceleration_drive=np.zeros(9, dtype=bool),
                      gravity_enabled=True)


def gt_distance(points: np.ndarray, d: dict, step: int) -> np.ndarray:
    """Unsigned distance from points to the ground-truth scene (cell boxes and objects) at a step."""
    C, H = d["gt_box_centers"], d["gt_box_halves"]
    dist = [np.abs(sphere_box_distance(points, 0.0, C[i], H[i])) for i in range(len(C))]
    for j, p in enumerate(d["gt_positions"][step]):
        dist.append(np.abs(sphere_box_distance(points, 0.0, p, d["object_half_extents"][j] + 0.01)))
    return np.min(np.stack(dist, 1), axis=1)


def map_metrics(boxes: list[Box], d: dict, step: int, workspace=((-0.3, -0.8, 0.0), (1.1, 0.9, 1.4))) -> dict:
    """Coverage and precision of the perceived boxes against ground truth.

    * precision: fraction of perceived box volume (sampled on a 2 cm grid) that lies within
      3 cm of real geometry;
    * coverage: fraction of real top/side surface points inside the workspace that lie within
      3 cm of a perceived box. The floor is skipped because the arm cannot reach it.
    """
    if not boxes:
        return {"boxes": 0, "precision": None, "coverage": 0.0}
    pts = []
    for b in boxes:
        c, h = np.asarray(b.center), np.asarray(b.half)
        axes = [np.arange(c[i] - h[i], c[i] + h[i] + 1e-9, 0.02) if h[i] > 0.01 else np.array([c[i]]) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pts.append(g)
    pts = np.vstack(pts)
    precision = float(np.mean(gt_distance(pts, d, step) < 0.03))
    # Ground-truth surface samples: the top face of every cell box (except the floor), on a 3 cm grid.
    surf = []
    lo, hi = np.asarray(workspace[0]), np.asarray(workspace[1])
    for name, c, h in zip(d["gt_box_names"], d["gt_box_centers"], d["gt_box_halves"]):
        if name == "floor":
            continue
        xs = np.arange(c[0] - h[0], c[0] + h[0] + 1e-9, 0.03)
        ys = np.arange(c[1] - h[1], c[1] + h[1] + 1e-9, 0.03)
        g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        surf.append(np.column_stack([g, np.full(len(g), c[2] + h[2])]))
    surf = np.vstack(surf)
    surf = surf[np.all((surf >= lo) & (surf <= hi), axis=1)]
    bc = np.array([b.center for b in boxes])
    bh = np.array([b.half for b in boxes])
    near = np.zeros(len(surf), dtype=bool)
    for c, h in zip(bc, bh):
        near |= sphere_box_distance(surf, 0.0, c, h) < 0.03
    return {"boxes": len(boxes), "precision": precision, "coverage": float(near.mean())}


def run(d: dict, mode: str = "perceived", lag: int | None = None, horizon: int = 30, verbose: bool = True,
        noise_floor: np.ndarray | None = None) -> dict:
    kin = FrankaKinematics()
    dt = float(d["dt"])
    base = d["base"]
    hand_i = kin.link_index("panda_hand")
    n = len(d["t"])
    pick, place = commanded_poses(d, kin)
    q_all = d["q"].astype(np.float64)
    links_all = kin.forward(q_all, base)
    names = list(d["object_names"])

    n_slots = 8
    if mode == "perceived":
        scene = PerceivedScene(kin, dt, n_slots=n_slots, noise_floor=noise_floor)
        tracker = ObjectMomentumTracker([f"obj_{i}" for i in range(n_slots)], np.full(n_slots, 0.1),
                                        np.full((n_slots, 3), 0.02), window=8)
        boxes0: list[Box] = []
    else:
        scene = None
        tracker = ObjectMomentumTracker(names, np.full(len(names), 0.1), d["object_half_extents"].astype(np.float64), window=8)
        boxes0 = [Box(str(nm), tuple(c), tuple(h)) for nm, c, h in zip(d["gt_box_names"], d["gt_box_centers"], d["gt_box_halves"])]
    solver = MomentumSolver(kin, ArmMomentumModel(drive_model(d), dt), tracker, boxes0, horizon, dt)

    alerts, trace = [], {k: [] for k in ("t", "ttc", "confirmed", "alert_point", "alert_label", "hand_path", "slots",
                                          "slot_half", "touch", "tau_score", "perception_ms", "solve_ms", "n_boxes")}
    box_snapshots, map_scores = [], []
    latency = LatencyEstimator(RobotSelfFilter(kin, inflate=0.01)) if lag is None else None
    lock_t = None if lag is None else 0.0
    lags: list[int] = [] if mode == "perceived" else [lag or 0] * (n - 1)
    for i in range(n - 1):
        t = float(d["t"][i])
        phase = PHASES[int(d["phase"][i])] if int(d["phase"][i]) < len(PHASES) else ""
        q, qd = q_all[i], d["qd"][i].astype(np.float64)
        M, c, g = (d[k][i].astype(np.float64) for k in ("mass_matrix", "coriolis", "gravity"))
        hand = links_all[i, hand_i]
        carried_center, carried_radius, carried_offset = carried_part(q, hand)
        holding = phase in HOLDING_PHASES

        t0 = time.perf_counter()
        if scene is not None:
            if latency is not None:
                cell_pts = backproject(d["depth_cell"][i].astype(np.float32), d["K_cell"], d["cell_pose"], stride=4)
                lag_i = latency.update(cell_pts, links_all[np.maximum(i - np.arange(latency.max_lag + 1), 0)])
            else:
                lag_i = lag
            lags.append(lag_i)
            if latency is not None and latency.locked and lock_t is None:
                lock_t = t
            j = max(i - lag_i, 0)  # the step whose state the (late) depth frames show
            wrist_pose = links_all[j, hand_i] @ d["wrist_mount"]
            phase_j = PHASES[int(d["phase"][j])] if int(d["phase"][j]) < len(PHASES) else ""
            cj, rj, _ = carried_part(q_all[j], links_all[j, hand_i])
            carried_j = (cj, rj + 0.01) if phase_j in HOLDING_PHASES else None
            wd = d["depth_wrist"][i].astype(np.float32)
            wd = np.where(wd < WRIST_MIN_RANGE, np.inf, wd)
            frames = [DepthFrame(wd, d["K_wrist"], wrist_pose), DepthFrame(d["depth_cell"][i].astype(np.float32), d["K_cell"], d["cell_pose"])]
            q_uncertain = None
            if latency is not None and not latency.locked:
                frames = frames[1:]  # the wrist pose depends on the lag; the cell camera's does not
                q_uncertain = q_all[np.maximum(i - np.arange(latency.max_lag + 1), 0)]
            rules_pre, exclude = task_rules(phase, scene.boxes, scene.object_positions, pick[i] if np.isfinite(pick[i, 0]) else None,
                                            place[i] if np.isfinite(place[i, 0]) else None, carried_offset, carried_radius)
            scene.update(t, frames, q, qd, M, c, g, d["tau_projected"][i].astype(np.float64), base,
                         carried=(carried_center, carried_radius + 0.01) if holding else None, exclude=exclude,
                         expected_touch=phase in EXPECTED_TOUCH, t_sensor=float(d["t"][j]), q_sensor=q_all[j],
                         carried_sensor=carried_j, q_uncertain=q_uncertain)
            if scene.boxes_changed:
                solver.set_boxes(scene.boxes)
            for s in scene.slots.reset_slots:
                tracker.forget(s)
            tracker.half_extents[:] = scene.object_half
            positions = scene.object_positions
            rules, _ = task_rules(phase, scene.boxes, positions, pick[i] if np.isfinite(pick[i, 0]) else None,
                                  place[i] if np.isfinite(place[i, 0]) else None, carried_offset, carried_radius)
            orient = None
        else:
            positions = d["gt_positions"][i].astype(np.float64)
            orient = d["gt_orientations"][i].astype(np.float64)
            target = int(d["target"][i])
            rules = ContactRules(ignore_objects={target} if target >= 0 else set())
            if holding:
                rules.carried, rules.carried_offset = -1, carried_offset
                rules.carried_radius = carried_radius
                if phase == "LIFT":
                    rules.carried_ignore_boxes = {b.name for b in boxes0 if b.name.startswith("tote")}
                elif phase == "DESCEND_PLACE":
                    rules.carried_ignore_boxes = {"conveyor"}
        t1 = time.perf_counter()

        proprio = Proprioception(q=q, qd=qd, mass_matrix=M, coriolis=c, gravity=g, base=base)
        q_cmd, qd_cmd = d["q_target"][i + 1].astype(np.float64), d["qd_target"][i + 1].astype(np.float64)
        pred = solver.solve(t, proprio, positions, q_cmd, qd_cmd, rules, orient,
                            object_time=scene.object_time if scene is not None else None)
        t2 = time.perf_counter()

        if pred.alert is not None:
            alerts.append({**pred.alert.__dict__, "confirmed": pred.confirmed, "phase": phase})
        tr = trace
        tr["t"].append(t)
        tr["ttc"].append(pred.alert.time_to_collision if pred.alert else np.nan)
        tr["confirmed"].append(pred.confirmed)
        tr["alert_point"].append(pred.alert.point if pred.alert else (np.nan,) * 3)
        tr["alert_label"].append(f"{pred.alert.link} -> {pred.alert.obstacle} ({pred.alert.model})" if pred.alert else "")
        tr["hand_path"].append(pred.hand_commanded if pred.hand_commanded is not None else pred.hand_ballistic)
        tr["slots"].append(positions if scene is not None else np.full((n_slots, 3), np.nan))
        tr["slot_half"].append(scene.object_half if scene is not None else np.full((n_slots, 3), np.nan))
        tr["touch"].append(scene.touches[-1]["point"] if scene is not None and scene.touches and scene.touches[-1]["t"] == t else (np.nan,) * 3)
        tr["tau_score"].append(scene.contacts.score() if scene is not None else np.nan)
        tr["perception_ms"].append((t1 - t0) * 1e3)
        tr["solve_ms"].append((t2 - t1) * 1e3)
        tr["n_boxes"].append(len(solver.boxes))
        if scene is not None and scene.boxes_changed:
            box_snapshots.append((t, np.array([[*b.center, *b.half] for b in scene.boxes]).reshape(-1, 6)))
        if scene is not None and i % 120 == 0 and i > 0:
            m = map_metrics(scene.boxes, d, i)
            m["t"] = t
            map_scores.append(m)
            if verbose:
                print(f"[map] t={t:5.1f}s boxes={m['boxes']} precision={m['precision']} coverage={m['coverage']:.2f} "
                      f"perception {np.mean(tr['perception_ms'][-120:]):.1f} ms solve {np.mean(tr['solve_ms'][-120:]):.2f} ms")

    contacts = [{"t": float(t), "a": str(a), "b": str(b)} for t, a, b in zip(d["contact_t"], d["contact_link"], d["contact_other"])]
    contacts = [c for c in contacts if c["t"] <= d["t"][-1]]
    events = contact_events(contacts)
    episodes = alert_episodes([a for a in alerts if a["confirmed"]], dt)
    result = {
        "mode": mode,
        "forecast": score(episodes, events, horizon * dt),
        "map": map_scores,
        "timing": {"perception": timing_stats(trace["perception_ms"]), "solve": timing_stats(trace["solve_ms"])},
    }
    if scene is not None:
        result["touches"] = scene.touches
        result["tracking"] = tracking_error(d, np.asarray(trace["slots"]), lag=lags)
        result["lag_steps"] = {"final": int(lags[-1]), "histogram": np.bincount(lags).tolist(),
                               "locked_at_s": lock_t}
    result["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    result["box_snapshots"] = box_snapshots
    return result


def tracking_error(d: dict, slots: np.ndarray, moving: float = 0.05, lag=0) -> dict:
    """For each moving ground-truth object, distance to the nearest perceived slot per step.

    Steps where the object is in the gripper are skipped: the carried part is filtered out of
    the depth like the robot, and the solver models it from the finger width instead.
    Slots at step i describe the scene at the depth timestamp, ``lag`` steps earlier (a
    constant or one value per step), so they are compared with ground truth from then (the
    solver extrapolates them to the present).
    """
    n = len(slots)
    lag = np.broadcast_to(np.asarray(lag), (n,)) if np.ndim(lag) == 0 else np.asarray(lag)[:n]
    gt = d["gt_positions"][np.maximum(np.arange(n) - lag, 0)]
    v = np.gradient(gt, d["t"][:n], axis=0)
    phase = np.array([PHASES[int(p)] if int(p) < len(PHASES) else "" for p in d["phase"][:n]])
    holding = np.isin(phase, HOLDING_PHASES)
    out = {}
    for j, name in enumerate(d["object_names"]):
        carried = holding & (d["target"][:n] == j)
        sel = (np.linalg.norm(v[:, j], axis=1) > moving) & (gt[:, j, 2] > 0.3) & (np.abs(gt[:, j]).max(1) < 2.0) & ~carried
        if sel.sum() < 5:
            continue
        dist = np.nanmin(np.linalg.norm(slots[sel] - gt[sel, j][:, None], axis=2), axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        out[str(name)] = {"steps_moving": int(sel.sum()), "detected_frac": float(np.mean(dist < 0.08)),
                          "median_error_m": float(np.median(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else None}
    return out


def summarize(res: dict) -> str:
    f = res["forecast"]
    ev = [(round(e["t"], 2), e["pairs"][0][0] + "->" + e["pairs"][0][1].rsplit("/", 1)[-1], e["forecast"],
           None if e["lead_time"] is None else round(e["lead_time"], 2)) for e in f["events"]]
    return (f"{res['mode']}: forecast {f['num_forecast']}/{f['num_events']} events, false alarms {f['num_false_alarm_episodes']}, "
            f"mean lead {f['mean_lead_time']}\n  events {ev}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", default=None, help="Write the perceived-mode trace (.npz) for rendering")
    ap.add_argument("--report", default=None)
    ap.add_argument("--lag", default="auto", help="Depth latency in control steps, or 'auto' to estimate it from the depth")
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()
    d = load(a.capture)
    lag = None if a.lag == "auto" else int(a.lag)
    res = run(d, "perceived", lag=lag)
    print(summarize(res))
    print("tracking", json.dumps(res["tracking"]))
    print("lag", json.dumps(res.get("lag_steps")))
    print("touches", len(res["touches"]), [(round(x["t"], 2), x["link"], round(x["residual"], 2)) for x in res["touches"][:12]])
    report = {"perceived": {k: v for k, v in res.items() if k not in ("trace", "box_snapshots")}}
    if not a.no_baseline:
        base = run(d, "world", lag=lag, verbose=False)
        print(summarize(base))
        report["world"] = {k: v for k, v in base.items() if k not in ("trace", "box_snapshots")}
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(report, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if a.out:
        snaps = res["box_snapshots"]
        np.savez_compressed(a.out, **{f"trace_{k}": v for k, v in res["trace"].items()},
                            box_t=np.array([s[0] for s in snaps]),
                            box_count=np.array([len(s[1]) for s in snaps]),
                            box_data=np.vstack([s[1] for s in snaps]) if snaps else np.zeros((0, 6)))
