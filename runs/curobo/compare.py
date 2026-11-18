"""Side-by-side table of the voxel and cuRobo perceived-scene replays.

Usage: compare.py [cuRobo result dir ...]  (default: this directory). Voxel and world rows come
from ``<capture>_voxel.json`` in this directory; each cuRobo dir holds ``<capture>_curobo.json``.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURES = ("intruder", "projectile", "bad_action", "nominal")


def row(capture, name, rep):
    f = rep["forecast"]
    m = [x for x in rep.get("map", []) if x.get("precision") is not None]
    st = rep["timing"].get("perception_stages_mean_ms", {})
    tracks = rep.get("tracking", {})
    return (capture, name, f"{f['num_forecast']}/{f['num_events']}", f["num_false_alarm_episodes"],
            None if f["mean_lead_time"] is None else round(f["mean_lead_time"], 2),
            round(float(np.mean([x["precision"] for x in m])), 2) if m else "",
            round(float(np.mean([x["coverage"] for x in m])), 2) if m else "",
            round(st.get("total", 0.0), 1) if st else "", round(st.get("map", 0.0), 1) if st else "",
            " ".join(f"{k.rsplit('/', 1)[-1]}:{v['detected_frac']:.2f}" for k, v in tracks.items()))


dirs = sys.argv[1:] or [HERE]
rows, totals = [], {}
for s in CAPTURES:
    voxel = json.load(open(os.path.join(HERE, f"{s}_voxel.json")))
    reps = [("world", voxel["world"]), ("voxel", voxel["perceived"])]
    for d in dirs:
        p = os.path.join(d, f"{s}_curobo.json")
        if os.path.exists(p):
            reps.append(("curobo" if d == HERE else "curobo:" + os.path.basename(os.path.normpath(d)), json.load(open(p))["perceived"]))
    for name, rep in reps:
        rows.append(row(s, name, rep))
        f = rep["forecast"]
        t = totals.setdefault(name, [0, 0, 0])
        t[0] += f["num_forecast"]
        t[1] += f["num_events"]
        t[2] += f["num_false_alarm_episodes"]
print("| capture | scene | forecast | false alarms | mean lead (s) | map precision | map coverage | perception ms | map ms | moving-object detection |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
print()
for name, (fc, ev, fa) in totals.items():
    print(f"total {name}: forecast {fc}/{ev}, false alarms {fa}")
