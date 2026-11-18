# Momentum solver runs

This file explains each rendered run in `runs/render/`. It covers what was injected, what the momentum solver predicted, what actually happened, and how the run ended.

All runs use the same cell and settings: seed 3, six parts, a 0.5 s (30-step) prediction horizon, an 8-frame object history, and 3-step alert confirmation. Faults fire in pick-and-place cycle 2.

The rendered runs stop early with `--max-time`, so the last part listed in each render report is cut off mid-cycle. Delivery counts below come from the full-length runs in `runs/*.json`.

| Video | Scenario | Solver warning before first contact | Outcome (full run) |
|---|---|---|---|
| [`nominal.mp4`](runs/render/nominal.mp4) | Normal operation, cycle 4 | 0.31 s (hand touches tote) | 6/6 delivered |
| [`intruder.mp4`](runs/render/intruder.mp4) | Box placed on the belt in the place path | 0.48 s | 6/6 delivered, part knocked around |
| [`projectile.mp4`](runs/render/projectile.mp4) | Ball thrown at the hand | 0.40 s | 6/6 delivered |
| [`bad_action.mp4`](runs/render/bad_action.mp4) | Command corrupted to drive the hand 25 cm into the belt | 0.15 s | 5/6 delivered, part knocked off |
| [`bad_action_guard.mp4`](runs/render/bad_action_guard.mp4) | Same corruption, guard enabled | n/a (no contact) | 6/6 delivered, no contact |
| [`bad_action_vs_guard.mp4`](runs/render/bad_action_vs_guard.mp4) | The two bad-action runs side by side | | |

"Warning" means the time from the first *confirmed* forecast to the first real PhysX contact. It can never exceed the 0.5 s horizon.

## Reading the overlay

- **Hand path:** the hand's predicted path over the next 0.5 s, from the commanded-motion model. It is green when clear, orange when a contact is forecast but not yet confirmed (fewer than 3 consecutive steps), and red once confirmed.
- **Cyan paths:** the predicted motion of any object moving faster than 5 cm/s, labelled with its speed.
- **Crosshair:** where the solver expects contact, labelled `link -> obstacle (model)` and the time until impact. The model is `commanded` (follows the controller's targets through the joint drives) or `ballistic` (keeps the current joint accelerations).
- **Banners:**
  - `COLLISION FORECAST`: a confirmed alert.
  - `GUARD`: the guard is overriding the command.
  - `FAULT INJECTED`: the first second after a fault.
  - `CONTACT`: a real unexpected contact reported by PhysX in the last 0.5 s, shown with a red border.
- **HUD:**
  - **Clearance:** the smallest predicted clearance over the horizon. Negative values mean predicted penetration.
  - **Timing:** a rolling mean of solver time and policy time per step.
- **Bottom strip:** predicted clearance over the clip (0–25 cm). Ticks mark forecasts (orange or red) and guard holds (blue). Red bars mark real contacts, and the yellow triangle marks the fault.

Timing in the rendered runs: the solver takes about 2.2 ms per step and the policy about 5.2 ms. The sim step looks like about 240 ms only because the RTX renderer runs every step. Without rendering, a sim step is 4.0 ms; see `runs/nominal.json`.

## Nominal (`nominal.mp4`, 27–34 s)

**Setup.** No fault. This is cycle 4: the arm finishes placing `Part_4` on the belt at 27.7 s, then goes back to the tote for `Part_3`.

**Timeline.**
- **30.07 s:** the forecast is confirmed: `panda_hand -> tote_wall_+y`, contact expected in about 0.45 s. The grasp pose for `Part_3` is close to the tote's +y wall.
- **30.38 s and 30.42 s:** the hand touches the tote. The solver warned 0.31 s ahead.
- **30.53 s:** a second confirmed forecast, `panda_hand -> Part_2`: the hand passes close to a neighbouring part in the tote.
- **31.32 s:** the hand touches the tote again on the way out. This is the same wall area, not `Part_2`, so the forecast named the wrong obstacle.

**Outcome.** These are light grazes. The policy is unaffected and all 6 parts are delivered in the full run.

**Notes.**
- The momentum residual stays low, peaking at 0.5 against a p99 of 1.4. Grazing contacts barely show up in proprioception, so the forecast is the only signal for them.
- Across the full nominal run there are 7 real contacts. 6 were forecast, with a mean warning of 0.45 s, and there were 13 false-alarm episodes. Most false alarms are near-misses during descent into the tote or onto the belt, where the true clearance is a few millimetres.

## Intruder (`intruder.mp4`, 10–20 s)

**Setup.** When cycle 2 enters the move-to-place phase (13.95 s), a 12 × 12 × 20 cm, 2 kg box is put on the belt 0.30 m upstream of the drop point. It rides the belt at 0.15 m/s and reaches the drop point about 2 s later, while the arm is bringing `Part_1` in.

**Timeline.**
- **13.95 s:** the box appears (pink, on the rollers). The object tracker restarts its history for it and estimates its velocity from the next few frames.
- **14.93 s:** the first alert, `Part_1 -> Intruder`, with contact expected in 0.50 s (the full horizon). It is confirmed at 14.97 s. The carried part's predicted path runs into the box.
- **15.45 s:** the right finger and `Part_1` hit the box, 0.48 s after confirmation. The predicted contact time was right to within one step.
- **15.5–16.3 s:** more contacts as the arm keeps pushing to the place pose: the left finger hits the box at 16.25 s. `Part_1` is knocked to about 2 m/s and lands on the belt at 16.28 s. The box itself is pushed to about 0.5 m/s.
- **16.6–17.3 s:** short forecasts against the conveyor as the arm finishes the descent. These are near-misses with no contact.

**Outcome.** No guard runs for object faults, so the arm doesn't react. `Part_1` still ends up on the belt and the full run delivers 6/6. The momentum residual peaks at only 0.82, so proprioception alone would not have flagged this collision.

**Why the alert waits until 14.93 s.** The box starts well clear of the arm. The solver can only flag a contact that falls within its 0.5 s horizon, so the alert starts once the box and arm paths are about 0.5 s from crossing.

## Projectile (`projectile.mp4`, 10–20 s)

**Setup.** 0.5 s into cycle 2's move-to-place phase (14.47 s), a 0.3 kg ball (3.5 cm radius) is launched from 1 m in front of the arm and 10 cm above. It is aimed at where the hand will be 0.5 s later, extrapolated from the hand's current velocity. The launch velocity is (−2.0, 0, +2.25) m/s, and the ball reaches about 4.2 m/s.

**Timeline.**
- **14.47 s:** launch. The ball starts outside the frame on the right, and only its predicted cyan arc is visible at first.
- **14.50 s:** the first alert, `panda_hand -> Projectile`, with contact in about 0.43 s. It is confirmed at 14.53 s, just 4 steps after launch. The tracker's quadratic fit picks up the ballistic arc within a few frames.
- **14.93 s:** the ball hits `panda_link7`, then the hand at 14.95 s and the left finger at 14.97 s. The warning was 0.40 s.
- The ball bounces off, and the arm carries on with `Part_1`.

**Outcome.** 6/6 delivered. As with the intruder, the residual barely moves (peak 0.92), so only the forecast caught this.

## Bad action without guard (`bad_action.mp4`, 11–21 s)

**Setup.** 0.1 s into cycle 2's lowering-to-place phase, the policy's command is corrupted. Using the hand Jacobian at that moment, the script computes the joint offset that would move the hand straight down 25 cm, about (−0.26, −0.72, +0.34, 0.02, 0.02, 0.14, 0) rad. It adds that offset to the policy's joint targets every step for the next 1.1 s. The policy itself is unaware. The joint drives are very stiff (kp ≈ 23k), so the arm snaps toward the bad target, with links reaching about 2.8 m/s.

**Timeline.**
- **17.20–17.23 s:** the first alert, then confirmation, `panda_leftfingertip -> conveyor (commanded)`. The solver sees the corrupted target on the step it is applied, so the alert starts at the moment of injection (the fault log stamps the fault at 17.25 s). The predicted path turns red and points about 14 cm into the belt.
- **17.38 s:** `Part_1` hits the belt, 0.15 s after confirmation. Both fingers follow at 17.40–17.47 s, and the hand at 17.77 s.
- **18.30 s:** the wrist (`panda_link5`) touches the belt while the arm recovers after the corruption window closes.
- The momentum residual peaks at 4.8 at 17.40 s, against a p99 of 2.9 for this run. This is the one fault where proprioception also reacts, but only after impact.

**Outcome.** `Part_1` is knocked around on the belt, and in the full run one part ends up off the belt (5/6 delivered).

**Why the warning is short.** The corrupted command didn't exist before it was issued, and the stiff arm reaches the belt in about 9 steps. A 0.15 s warning is therefore close to the physical limit for this fault. It is enough for a guard to act, as the next run shows.

## Bad action with guard (`bad_action_guard.mp4`, 11–21 s)

**Setup.** The same corruption, at the same point in cycle 2 (17.38 s in this run), with `--guard`. Whenever a *confirmed* forecast comes from the `commanded` model, the guard replaces that step's command with "hold current joint positions, zero velocity". When the forecast clears, the policy's command goes through again.

**Timeline.**
- **17.33–17.37 s:** the forecast for the corrupted command is raised and then confirmed, and the guard engages.
- **17.37–18.47 s:** the guard holds for 67 steps, about 1.1 s, which matches the corruption window. The hand stays within about 1 cm of where it was at injection, roughly 27 cm above the belt surface (measured at the hand frame). The −13.8 cm in the HUD is how far the *rejected* command would have driven the gripper into the belt.
- **About 18.5 s:** the corruption ends, the command no longer predicts a collision, and the guard releases. A one-step ballistic alert appears as the arm starts moving again. The policy then finishes the place normally.

**Outcome.** No unexpected contacts. All 6 parts are delivered, and the full run takes 79.1 s instead of 77.1 s.

**Notes.**
- The guard also fired 25 times before the fault, for short holds at about 3.0–3.7 s, 7.3 s and 12.7 s. These happen on approaches to the tote and belt where clearance is small. Some precede real light tote touches and some are false alarms. Each causes a few frames of hesitation.
- In `runs/bad_action_guard.json`, the fault-window alert is counted as a false alarm because no contact followed it. In fact, no contact followed because the guard stopped the motion.
- The guard only protects against the arm's *own* commanded motion. It is not used for the intruder or projectile faults, where holding still would not avoid the hit.

## Reproducing

```bash
cd /home/builder/isaacsim_work/franka_factory
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0
/home/builder/isaacsim/python.sh franka_factory_pick_place.py --headless --gpu 0 --momentum \
    --fault bad_action --guard --camera close \
    --record runs/render/bad_action_guard_frames --record-window 11 21 --max-time 21.5 \
    --trace runs/render/bad_action_guard.npz --report runs/render/bad_action_guard.json
/home/builder/isaacsim/python.sh render_overlay.py runs/render/bad_action_guard.npz \
    runs/render/bad_action_guard_frames runs/render/bad_action_guard.mp4 --title "Fault: corrupted action, guard enabled"
```

The windows used for the other runs:

| Run | `--fault` | `--record-window` | `--max-time` |
|---|---|---|---|
| intruder | `intruder` | `10 20` | `20.5` |
| projectile | `projectile` | `10 20` | `20.5` |
| bad_action | `bad_action` | `11 21` | `21.5` |
| nominal | none | `27 34` | `34.5` |

Add `--slow 2` to `render_overlay.py` for half-speed playback. Rendering five runs in parallel on one GPU takes about 7 minutes.

## Perceived scene (`runs/perceived/`)

With `--scene perceived` the solver gets no world knowledge. Everything it collides against is rebuilt on the fly from robot sensing:

- **Depth:** RGB-D from the wrist camera and the fixed cell camera.
- **Proprioception:** joint positions, velocities and commands, plus the dynamics model.
- **Torque sensing:** measured joint torques, used to detect and localise touches.
- **The policy's own pick and place goals:** used to decide which contacts are expected.

Ground truth is read only to score the result.

How the scene is built, in `scene_perception.py`:

1. **Back-project and filter the robot.** Each depth frame becomes a point cloud. Points on the robot are removed with a set of spheres posed at the joint state the frame was captured at, swept ±1 step. Flying pixels at depth edges are dropped.
2. **Estimate the depth latency.** The renderer delivers depth a few steps late, and how late depends on load. It is 1 step for a plain run, 2 steps while also recording video, and 4 steps when only capturing. `LatencyEstimator` measures it online from the robot's own body in the cell camera: for each candidate lag it counts depth points inside the robot spheres posed that many steps ago, and keeps a running sum of how far each lag falls short of the best one. It locks after about 2 s. Until then, only the cell camera is fused, and the robot is filtered at every candidate lag so it cannot leak into the map.
3. **Build a voxel map.** A 3 cm log-odds grid separates the static scene from foreground. A voxel counts as dynamic only if it was consistently free before, is not flickering (conveyor rollers), and is not just a static surface spilling across a voxel boundary.
4. **Extract boxes and objects.** Static voxels become axis-aligned boxes for the solver. Foreground voxels are clustered and tracked into 8 object slots, which feed the solver's object momentum tracker exactly as true object poses do in world mode. A slot is published after 3 consecutive matches.
5. **Fuse touches.** An unexpected torque spike is localised on the arm and added to the map as an obstacle.

Each step costs 41–48 ms of perception, mostly map update and back-projection, on top of a 2–3 ms solve.

Results from live runs, compared with world mode over the same length:

| Run | Perceived forecast | Perceived false alarms | Perceived mean lead | World forecast | World false alarms | World mean lead | Map precision | Depth lag (steps) |
|---|---|---|---|---|---|---|---|---|
| [intruder](runs/perceived/render/intruder.mp4) | 3/3 | 15 | 0.68 s | 3/3 | 5 | 0.72 s | 0.87 | 2 |
| [projectile](runs/perceived/render/projectile.mp4) | 2/2 | 8 | 0.43 s | 2/2 | 4 | 0.39 s | 0.89 | 1–3 |
| [bad_action](runs/perceived/render/bad_action.mp4) | 3/3 | 8 | 0.55 s | 3/3 | 3 | 0.53 s | 0.76 | 2–3 |
| [nominal](runs/perceived/render/nominal.mp4) | 4/4 | 18 | 0.51 s | 3/4 | 8 | 0.46 s | 0.83 | 2 |

How to read the table:

- **Map precision** is the fraction of perceived box volume that lies within 3 cm of a real surface. Coverage of the real surfaces is 0.93 in every run.
- **Forecasting:** every unexpected contact was forecast, with lead times matching world mode.
- **Fault objects:** the intruder box was detected on 95% of its moving steps and the thrown part on 81%, with median position errors of 5 cm.
- **The perceived runs above also record video.** That adds render load and raises the latency to 2 steps. The same intruder run without recording locked at 1 step and had 7 false alarms with map precision 0.89.
- **False alarms are the remaining gap:** about 2–3 times as many as world mode. Most last 1–3 frames and come from short-lived foreground "objects" 0.3–0.5 m from the hand, likely brief leaks from the arm body or belt. That cause has not been confirmed.

Overlay additions:

- **Blue wireframes:** perceived static boxes. The floor is not drawn.
- **Magenta boxes:** tracked object slots, with their predicted paths.
- **Stars:** torque-localised touches.
- **HUD line:** box count, object count and perception time.

Ground-truth object paths are not drawn in perceived mode.

Reproducing a perceived run, here the intruder:

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json OMP_NUM_THREADS=1
/home/builder/isaacsim/python.sh franka_factory_pick_place.py --headless --gpu 2 --scene perceived --fault intruder \
    --camera close --record runs/perceived/render/intruder_frames --record-window 10 20 --max-time 20.5 \
    --trace runs/perceived/intruder_trace.npz --report runs/perceived/intruder.json
/home/builder/isaacsim/python.sh render_overlay.py runs/perceived/intruder_trace.npz \
    runs/perceived/render/intruder_frames runs/perceived/render/intruder.mp4 --title "Perceived scene: intruder box"
```

For world-mode baselines, replace `--scene perceived` with `--momentum`.

`perception_replay.py` replays raw captures offline. Captures are made with `--perception --capture runs/perception/<name>_raw.npz`. Replays use the same estimator and accept `--lag auto` (the default) or a fixed lag. `test_scene_perception.py` holds the unit tests.

### cuRobo backend (`runs/curobo/`)

`--perception-backend curobo` (live) or `--backend curobo` (replay) replaces the voxel map and box extraction with cuRobo's TSDF fusion and ESDF (`curobo_scene.py`). The robot filter, latency estimator, object slots and touch localisation are unchanged. The solver queries the ESDF directly (`MomentumSolver.static_field`, obstacle name `map`), with the pick and place regions exempted through `ContactRules.static_ignore`/`carried_static_ignore` instead of being cut out of the boxes.

Foreground is a depth pixel at least 3 cm in front of the map rendered from the same camera, or, for the fixed camera, a return where that pixel had only ever seen open background. Foreground is dropped if it lies within 3 cm of the robot spheres or within 1.5 cm of a mapped surface; the second rule removes conveyor-roller flicker and wrist misregistration. Foreground is fused as static once its 3 cm cell has held foreground for 1.5 s.

Offline replays of the four raw captures (`bash runs/curobo/run_all.sh`, then `python.sh runs/curobo/compare.py`):

| Capture | World forecast / false alarms | Voxel forecast / false alarms | cuRobo forecast / false alarms | Fault-object detection, voxel vs cuRobo |
|---|---|---|---|---|
| intruder | 3/3, 4 | 3/3, 11 | 3/3, 3 | 0.95 vs 0.96 |
| projectile | 2/2, 4 | 2/2, 10 | 2/2, 3 | 0.80 vs 0.70 |
| bad_action | 3/3, 3 | 3/3, 8 | 3/3, 2 | |
| nominal | 3/4, 8 | 4/4, 13 | 4/4, 11 | |
| **total** | 11/12, 19 | 12/12, 42 | 12/12, 19 | |

Map precision is 0.79 for cuRobo and 0.80 for voxel, and coverage is 0.94 for both. Lead times match within 0.1 s. Perception takes about 20 ms per step, against about 30 ms for voxel. `runs/curobo/sweep.sh <tag> '<json kwargs>'` replays all captures with other `CuroboPerceivedScene` parameters. Faster frustum decay (0.93, 0.90) adds false alarms.

cuRobo v0.9 is installed with `pip install --target ~/isaacsim_work/deps/site "~/isaacsim_work/deps/curobo[cu13-torch]"`. Replays run under `PYTHONPATH=~/isaacsim_work/deps/site /home/builder/isaacsim/kit/python/bin/python3`. The live script appends `--curobo-site` to `sys.path`. A first live attempt hit the Vulkan device-lost crash during startup, so live numbers are still to come.

Leave `CUDA_VISIBLE_DEVICES` unset and pick the GPU with `--gpu`. A Vulkan device-lost error ("GPU crash is detected") makes the node health check fail, and Karpenter replaces the node about 10 minutes later. When that happens, stop launching runs and push your work.
