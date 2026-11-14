"""Overlay momentum-solver predictions on frames recorded with ``--record`` and encode an mp4.

Usage::

    python render_overlay.py runs/render/intruder.npz runs/render/intruder_frames runs/render/intruder.mp4 \
        --title "Intruder fault"

Draws, per frame: the solver's predicted hand path over its horizon (green clear, orange forecast contact,
red confirmed), predicted object paths for moving objects, the predicted contact point, the realised hand
trail, fault/contact/guard banners, a timing HUD, and a clearance/TTC strip with a time cursor.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess

import cv2
import numpy as np

FFMPEG = "/home/builder/.local/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2"
HORIZONTAL_APERTURE = 20.955  # USD camera default (mm)
PHASES = ["", "APPROACH_PICK", "DESCEND_PICK", "GRASP", "LIFT", "APPROACH_PLACE", "DESCEND_PLACE", "RELEASE", "RETREAT"]

# BGR
GREEN, ORANGE, RED, CYAN, WHITE, GREY, YELLOW = (80, 220, 80), (0, 160, 255), (40, 40, 240), (255, 220, 0), (255, 255, 255), (170, 170, 170), (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Camera:
    def __init__(self, eye, target, focal, resolution):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, (0.0, 0.0, 1.0))
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.w, self.h = (int(v) for v in resolution)
        self.fx = float(focal) / HORIZONTAL_APERTURE * self.w

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """World points (..., 3) -> pixel coordinates (..., 2) and a visibility mask."""
        d = np.asarray(points, float) - self.eye
        z = d @ self.f
        ok = (z > 0.05) & np.all(np.isfinite(d), axis=-1)
        zs = np.where(ok, z, 1.0)
        px = np.stack([self.w / 2 + self.fx * (d @ self.r) / zs, self.h / 2 - self.fx * (d @ self.u) / zs], axis=-1)
        return px, ok


def text(img, s, org, scale=0.6, color=WHITE, thick=1, bg=True):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg:
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def banner(img, s, color, y=150):
    scale, thick = 1.0, 2
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    text(img, s, ((img.shape[1] - tw) // 2, y), scale, color, thick)


def polyline(img, px, ok, color, thick=2):
    pts = px[ok].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


class Strip:
    """Clearance / TTC timeline drawn along the bottom edge."""

    def __init__(self, d, t0, t1, w, h=72, max_clear=0.25):
        self.t0, self.t1, self.w, self.h, self.max_clear = t0, t1, w, h, max_clear
        t = d["t"]
        sel = (t >= t0) & (t <= t1)
        self.t, self.clear = t[sel], np.clip(np.nan_to_num(d["min_distance"][sel], posinf=max_clear), -0.02, max_clear)
        self.alert = np.isfinite(d["ttc"][sel])
        self.confirmed = d["confirmed"][sel]
        self.guarded = d["guarded"][sel]
        self.contacts = [c for c in d["contact_t"] if t0 <= c <= t1]
        self.faults = [f for f in d["fault_t"] if t0 <= f <= t1]

    def x(self, t):
        return int(40 + (t - self.t0) / max(self.t1 - self.t0, 1e-6) * (self.w - 80))

    def draw(self, img, now):
        y0 = img.shape[0] - self.h - 10
        overlay = img.copy()
        cv2.rectangle(overlay, (30, y0 - 22), (self.w - 30, y0 + self.h + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.putText(img, "min predicted clearance (0-25 cm)   orange: forecast contact   red: confirmed   blue: guard hold"
                    "   | contact   ^ fault", (40, y0 - 6), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        for t, a, c, g in zip(self.t, self.alert, self.confirmed, self.guarded):
            if a or g:
                col = (255, 120, 40) if g else (RED if c else ORANGE)
                cv2.line(img, (self.x(t), y0 + 2), (self.x(t), y0 + 14), col, 1)
        ys = (y0 + self.h - (self.clear / self.max_clear) * (self.h - 20)).astype(np.int32)
        pts = np.stack([[self.x(t) for t in self.t], ys], axis=-1).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, GREEN, 1, cv2.LINE_AA)
        for c in self.contacts:
            cv2.line(img, (self.x(c), y0 + 16), (self.x(c), y0 + self.h), RED, 2)
        for f in self.faults:
            x = self.x(f)
            cv2.fillPoly(img, [np.array([[x, y0 + self.h - 12], [x - 7, y0 + self.h], [x + 7, y0 + self.h]], np.int32)], YELLOW)
        cv2.line(img, (self.x(now), y0), (self.x(now), y0 + self.h), WHITE, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slow", type=float, default=1.0, help="Playback slow-down factor")
    args = ap.parse_args()

    d = dict(np.load(args.trace, allow_pickle=False))
    cam = Camera(d["camera_eye"], d["camera_target"], d["camera_focal"], d["camera_resolution"])
    files = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))
    frame_t = d["frame_times"][: len(files)]
    t = d["t"]
    strip = Strip(d, float(frame_t[0]), float(frame_t[-1]), cam.w)
    out_dir = args.frames.rstrip("/") + "_overlay"
    os.makedirs(out_dir, exist_ok=True)
    names = d["object_names"]
    contact_t, contact_label, fault_t = d["contact_t"], d["contact_label"], d["fault_t"]
    pred_keys = sorted(int(k) for k in d["eval_steps"])

    for n, (path, tf) in enumerate(zip(files, frame_t)):
        img = cv2.imread(path)
        i = int(np.clip(np.searchsorted(t, tf), 0, len(t) - 1))
        confirmed, alerting, guarded = bool(d["confirmed"][i]), bool(np.isfinite(d["ttc"][i])), bool(d["guarded"][i])

        # realised hand trail over the last second
        trail = d["hand"][max(0, i - 60): i + 1]
        px, ok = cam.project(trail)
        polyline(img, px, ok, GREY, 1)

        # predicted object paths (only objects that are actually moving)
        for j in range(d["objects"].shape[1]):
            if d["object_speed"][i, j] < 0.05:
                continue
            path3 = np.concatenate([d["objects"][i, j][None]] + [d[f"objects_pred_{k}"][i, j][None] for k in pred_keys])
            px, ok = cam.project(path3)
            polyline(img, px, ok, CYAN, 2)
            for p, o in zip(px[1:], ok[1:]):
                if o:
                    cv2.circle(img, tuple(int(v) for v in p), 3, CYAN, -1, cv2.LINE_AA)
            if ok[0]:
                text(img, f"{names[j]} {d['object_speed'][i, j]:.2f} m/s", px[0] + (10, -10), 0.45, CYAN)

        # predicted hand path over the solver horizon
        color = RED if confirmed else (ORANGE if alerting else GREEN)
        hp = d["hand_path"][i]
        px, ok = cam.project(np.concatenate([d["hand"][i][None], hp]))
        polyline(img, px, ok, color, 3)
        if ok[-1]:
            cv2.circle(img, tuple(int(v) for v in px[-1]), 5, color, -1, cv2.LINE_AA)

        # predicted contact location
        if alerting and np.all(np.isfinite(d["alert_point"][i])):
            (p,), (o,) = cam.project(d["alert_point"][i][None])
            if o:
                c = tuple(int(v) for v in p)
                cv2.circle(img, c, 16, color, 2, cv2.LINE_AA)
                cv2.drawMarker(img, c, color, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
                text(img, f"{d['alert_label'][i]}  in {d['ttc'][i]:.2f}s", (c[0] + 22, c[1] + 6), 0.5, color)

        # banners
        recent_contacts = [lbl for ct, lbl in zip(contact_t, contact_label) if 0 <= tf - ct < 0.5]
        if guarded:
            banner(img, "GUARD: action forecast to collide -> holding arm", (255, 160, 60), 150)
        elif confirmed:
            banner(img, f"COLLISION FORECAST in {d['ttc'][i]:.2f}s", RED, 150)
        if any(0 <= tf - ft < 1.0 for ft in fault_t):
            banner(img, "FAULT INJECTED", YELLOW, 195)
        if recent_contacts:
            banner(img, "CONTACT: " + recent_contacts[-1], RED, 240)
            cv2.rectangle(img, (2, 2), (cam.w - 3, cam.h - 3), RED, 4)

        # HUD
        sl = slice(max(0, i - 30), i + 1)
        hud = [
            args.title,
            f"t = {t[i]:6.2f} s   cycle {d['cycle'][i]}   {PHASES[int(d['phase'][i])] if int(d['phase'][i]) < len(PHASES) else d['phase'][i]}",
            f"min predicted clearance {d['min_distance'][i] * 100:6.1f} cm" if np.isfinite(d["min_distance"][i]) else "min predicted clearance   --",
            f"solver {np.mean(d['solve_ms'][sl]):.2f} ms   policy {np.mean(d['policy_ms'][sl]):.2f} ms   "
            f"horizon {hp.shape[0]} steps ({hp.shape[0] / 60:.2f} s)",
        ]
        for row, s in enumerate(hud):
            if s:
                text(img, s, (14, 28 + 26 * row), 0.6 if row else 0.75, WHITE, 2 if row == 0 else 1)
        strip.draw(img, tf)
        cv2.imwrite(os.path.join(out_dir, f"frame_{n:05d}.png"), img)

    fps = args.fps / args.slow
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", os.path.join(out_dir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", "30", args.out], check=True)
    print(f"wrote {args.out} ({len(files)} frames, {len(files) / fps:.1f} s)")


if __name__ == "__main__":
    main()
