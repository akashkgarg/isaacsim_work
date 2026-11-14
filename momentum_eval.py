"""Scoring for the momentum solver: timing statistics, prediction error, and collision-forecast quality."""

from __future__ import annotations

import numpy as np


def timing_stats(samples_ms: list[float]) -> dict:
    if not samples_ms:
        return {}
    a = np.asarray(samples_ms)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "n": int(a.size),
    }


def group_episodes(times: list[float], gap: float) -> list[tuple[int, int]]:
    """Index ranges ``[start, end]`` of runs whose consecutive timestamps are at most ``gap`` apart."""
    episodes = []
    start = None
    for i, t in enumerate(times):
        if start is None:
            start = i
        elif t - times[i - 1] > gap:
            episodes.append((start, i - 1))
            start = i
    if start is not None:
        episodes.append((start, len(times) - 1))
    return episodes


def alert_episodes(alerts: list[dict], dt: float) -> list[dict]:
    alerts = sorted(alerts, key=lambda a: a["t"])
    out = []
    for s, e in group_episodes([a["t"] for a in alerts], gap=2.5 * dt):
        chunk = alerts[s : e + 1]
        first = chunk[0]
        out.append(
            {
                "start": first["t"],
                "end": chunk[-1]["t"],
                "steps": len(chunk),
                "first_time_to_collision": first["time_to_collision"],
                "predicted_contact_time": first["t"] + first["time_to_collision"],
                "min_time_to_collision": min(a["time_to_collision"] for a in chunk),
                "models": sorted({a["model"] for a in chunk}),
                "links": sorted({a["link"] for a in chunk}),
                "obstacles": sorted({a["obstacle"] for a in chunk}),
                "max_link_speed": max(a["link_speed"] for a in chunk),
            }
        )
    return out


def contact_events(contacts: list[dict], gap: float = 0.5) -> list[dict]:
    contacts = sorted(contacts, key=lambda c: c["t"])
    out = []
    for s, e in group_episodes([c["t"] for c in contacts], gap=gap):
        chunk = contacts[s : e + 1]
        out.append({"t": chunk[0]["t"], "end": chunk[-1]["t"], "pairs": sorted({(c["a"], c["b"]) for c in chunk})})
    return out


def score(episodes: list[dict], events: list[dict], horizon: float, slack: float = 0.25) -> dict:
    """Match alert episodes to real unexpected contacts.

    An event counts as forecast when an episode starts before the contact and is still
    active within one horizon of it; the lead time is contact time minus episode start.
    Episodes not followed by a contact within ``horizon + slack`` are false alarms.
    """
    matched = set()
    results = []
    for ev in events:
        best = None
        for i, ep in enumerate(episodes):
            if ep["start"] <= ev["t"] + 1e-9 and ep["end"] >= ev["t"] - horizon - slack:
                if best is None or ep["start"] < episodes[best]["start"]:
                    best = i
        if best is not None:
            matched.add(best)
            ep = episodes[best]
            results.append({**ev, "forecast": True, "lead_time": ev["t"] - ep["start"], "episode": ep})
        else:
            results.append({**ev, "forecast": False, "lead_time": None, "episode": None})
    false_alarms = [ep for i, ep in enumerate(episodes) if i not in matched]
    leads = [r["lead_time"] for r in results if r["forecast"]]
    return {
        "events": results,
        "num_events": len(events),
        "num_forecast": sum(r["forecast"] for r in results),
        "num_alert_episodes": len(episodes),
        "num_false_alarm_episodes": len(false_alarms),
        "false_alarms": false_alarms,
        "mean_lead_time": float(np.mean(leads)) if leads else None,
        "min_lead_time": float(np.min(leads)) if leads else None,
    }


def prediction_error(times: np.ndarray, predicted: dict[int, np.ndarray], actual: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean/p95 Euclidean error of k-step predictions against what actually happened k steps later.

    Args:
        times: Step timestamps (shape ``(T,)``).
        predicted: Horizon step k -> predictions made at each step for step ``i + k`` (shape ``(T, ..., 3)``).
        actual: Actual positions at each step (shape ``(T, ..., 3)``).
        mask: Optional per-step/per-entity boolean selection (shape ``(T, ...)``).
    """
    out = {}
    for k, pred in predicted.items():
        if len(times) <= k:
            continue
        err = np.linalg.norm(pred[:-k] - actual[k:], axis=-1)
        if mask is not None:
            err = err[mask[:-k]]
        err = err[np.isfinite(err)]
        if err.size:
            out[str(k)] = {"mean_m": float(err.mean()), "p95_m": float(np.percentile(err, 95)), "max_m": float(err.max())}
    return out
