"""Franka bin-to-conveyor pick-and-place in a warehouse/factory work cell.

The Franka sits on a pedestal between a tote of parts and two chained roller
conveyors. The conveyors move items with a PhysX surface velocity, so dropped
parts are carried away physically. Grasps are real finger contacts (no fixed
joints or pose writes); arm motion comes from cuMotion RMPflow through the
maintained ``PickPlaceController`` state machine.

With ``--momentum`` a momentum solver (``momentum_solver.py``) runs every control step
next to the policy: it reads the arm's proprioception (joint state, mass matrix,
Coriolis/gravity terms, drive gains) and the recent frames of every part, predicts arm
and object states ``--horizon-steps`` ahead, and forecasts collisions. PhysX contact
reports provide the ground truth the forecasts are scored against, and wall-clock
timings of the simulator, the policy and the solver are reported side by side.
``--fault`` injects an erroneous situation and ``--guard`` holds the arm whenever the
policy's next action is forecast to collide.

Run (from the Isaac Sim root)::

    ./python.sh franka_factory_pick_place.py --headless --test
    ./python.sh franka_factory_pick_place.py --save-usd franka_factory.usd
    ./python.sh franka_factory_pick_place.py --headless --momentum --report runs/nominal.json
    ./python.sh franka_factory_pick_place.py --headless --momentum --fault bad_action --guard
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num-items", type=int, default=6)
parser.add_argument("--belt-speed", type=float, default=0.15, help="Conveyor surface speed in m/s")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--no-dressing", action="store_true", help="Skip decorative factory props")
parser.add_argument("--save-usd", default=None, help="Export the composed stage before simulation starts")
parser.add_argument("--record", default=None, help="Directory for rendered RGB frames (needs a working renderer)")
parser.add_argument("--record-every", type=int, default=2)
parser.add_argument("--record-window", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help="Only save frames between these sim times (s)")
parser.add_argument("--max-time", type=float, default=None, help="Stop the run at this sim time (s)")
parser.add_argument("--camera", choices=("wide", "close"), default="wide", help="Recording viewpoint")
parser.add_argument("--preview", action="store_true", help="Render stills from a few viewpoints into --record and exit")
parser.add_argument("--report", default=None, help="Write a JSON run report to this path")
parser.add_argument("--gpu", type=int, default=0, help="Single GPU used for rendering")
parser.add_argument("--test", action="store_true", help="Fail with a non-zero exit code if any gate fails")
parser.add_argument("--momentum", action="store_true", help="Run the momentum solver alongside the policy and score its forecasts")
parser.add_argument("--horizon-steps", type=int, default=30, help="Momentum solver prediction horizon in physics steps")
parser.add_argument("--history", type=int, default=8, help="Object frames used to estimate momentum")
parser.add_argument("--margin", type=float, default=0.0, help="Predicted clearance (m) below which an alert is raised")
parser.add_argument("--confirm-steps", type=int, default=3, help="Consecutive predicted-contact steps before an alert is confirmed")
parser.add_argument("--arm-alert-steps", type=int, default=None,
                    help="Steps of the action-conditioned arm rollout that may raise alerts (default: full horizon)")
parser.add_argument("--uncertainty-scale", type=float, default=0.0,
                    help="Multiplier on the solver's self-measured k-step error used as penetration tolerance (0 disables)")
parser.add_argument("--fault", choices=("none", "intruder", "bad_action", "projectile"), default="none",
                    help="Inject an erroneous situation: an unplanned box riding the conveyor into the drop zone, "
                         "a corrupted policy action, or a part thrown at the gripper")
parser.add_argument("--fault-cycle", type=int, default=2, help="Pick-and-place cycle (1-based) in which the fault fires")
parser.add_argument("--guard", action="store_true", help="Hold the arm instead of executing an action forecast to collide")
parser.add_argument("--trace", default=None, help="Write per-step momentum solver traces (.npz)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "active_gpu": args.gpu,
        "physics_gpu": args.gpu,
        "multi_gpu": False,
        "max_gpu_count": 1,
        # Headless runs without recording never need pixels; skipping the viewport keeps RTX idle.
        "disable_viewport_updates": args.headless and not args.record,
        **({"renderer": "MinimalRendering"} if args.headless and not args.record else {}),
    }
)

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.client
import warp as wp
from isaacsim.core.experimental.objects import Cube, Cylinder
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot
from isaacsim.robot_motion.examples.manipulation import (
    GripperCommand,
    JointGripperController,
    ManipulationScenario,
    PickPlaceController,
    PickPlacePhase,
)
from isaacsim.storage.native import get_assets_root_path
from momentum_eval import alert_episodes, contact_events, prediction_error, score, timing_stats
from momentum_solver import (
    FRANKA_JOINTS,
    ArmMomentumModel,
    Box,
    ContactRules,
    DriveModel,
    FrankaKinematics,
    MomentumSolver,
    ObjectMomentumTracker,
    Proprioception,
)
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

PHYSICS_DT = 1.0 / 60.0

# Work-cell layout (meters, Z-up). The robot faces +X; the tote is to its right
# (-Y) and the conveyor line to its left (+Y), running away along +X.
ROBOT_BASE = (0.0, 0.0, 0.75)
TOTE_CENTER_XY = (0.36, -0.38)
TOTE_STAND_TOP = 0.62
TOTE_SCALE = (1.8, 1.8, 0.85)  # small_KLT stretched into a parts tote; long axis ends up along X
KLT_HALF = (0.0989, 0.1483, 0.0732)  # small_KLT half extents in its own frame
CONVEYOR_Y = 0.70
CONVEYOR_X0 = -0.6
CONVEYOR_LENGTH = 2.0  # ConveyorBelt_A05 runs along local +X from 0 to 2 m
ROLLER_TOP = 0.7693
BELT_TOP = 0.772
BELT_HALF_WIDTH = 0.45
NUM_CONVEYORS = 2
DROP_XY = (0.38, 0.42)
DROP_CLEARANCE = 0.02  # release height of the item bottom above the belt

ITEM_CUBE = 0.045
ITEM_CYL_RADIUS = 0.022
ITEM_CYL_HEIGHT = 0.05
ITEM_MASS = 0.08

FINGER_REACH = 0.055  # open finger centers from the grasp axis
HAND_HALF_LENGTH = 0.105  # hand body half-length along the finger axis

TOTE_YAW = math.pi / 2.0
TOTE_INNER_HALF = (KLT_HALF[1] * TOTE_SCALE[1] - 0.012, KLT_HALF[0] * TOTE_SCALE[0] - 0.012)  # world X, Y


def tote_center_z() -> float:
    return TOTE_STAND_TOP + KLT_HALF[2] * TOTE_SCALE[2]


# ---------------------------------------------------------------------------
# USD authoring helpers
# ---------------------------------------------------------------------------


def _precision(prim: Usd.Prim, name: str, default: UsdGeom.XformOp.Precision) -> UsdGeom.XformOp.Precision:
    # Referenced assets may already author these ops; re-adding them must match the authored type.
    attr = prim.GetAttribute(name)
    if not attr or not attr.GetTypeName():
        return default
    type_name = str(attr.GetTypeName())
    if type_name.startswith("double"):
        return UsdGeom.XformOp.PrecisionDouble
    if type_name.startswith("half"):
        return UsdGeom.XformOp.PrecisionHalf
    return UsdGeom.XformOp.PrecisionFloat


def set_xform(prim: Usd.Prim, translate=(0, 0, 0), yaw_deg: float = 0.0, scale=(1, 1, 1)) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    p = UsdGeom.XformOp.PrecisionDouble
    f = UsdGeom.XformOp.PrecisionFloat
    for name in ("xformOp:orient", "xformOp:rotateXYZ", "xformOp:transform"):
        if prim.GetAttribute(name):
            prim.RemoveProperty(name)
    xf.AddTranslateOp(_precision(prim, "xformOp:translate", p)).Set(Gf.Vec3d(*translate))
    xf.AddRotateZOp(_precision(prim, "xformOp:rotateZ", f)).Set(float(yaw_deg))
    xf.AddScaleOp(_precision(prim, "xformOp:scale", f)).Set(Gf.Vec3d(*scale))


def add_reference(stage: Usd.Stage, path: str, url: str, translate=(0, 0, 0), yaw_deg=0.0, scale=(1, 1, 1)) -> Usd.Prim:
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(url)
    set_xform(prim, translate, yaw_deg, scale)
    return prim


def preview_material(stage: Usd.Stage, path: str, color, roughness=0.5, metallic=0.0) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def physics_material(stage: Usd.Stage, path: str, static: float, dynamic: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return material


def bind(prim: Usd.Prim, material: UsdShade.Material, purpose: str | None = None) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        binding.Bind(material)


def static_box(stage: Usd.Stage, path: str, center, size, material: UsdShade.Material) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), center, 0.0, size)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    bind(cube.GetPrim(), material)
    return cube.GetPrim()


def first_usd(url: str) -> str | None:
    """Return the first USD file in a folder (digital-twin props ship one asset per folder)."""
    result, entries = omni.client.list(url)
    if result != omni.client.Result.OK:
        return None
    names = sorted(e.relative_path for e in entries if e.relative_path.endswith((".usd", ".usda", ".usdc")))
    return f"{url}/{names[0]}" if names else None


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def build_factory(stage: Usd.Stage, assets_root: str) -> None:
    add_reference(stage, "/World/Factory", assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd")

    # The warehouse floor has collision, but keep an explicit invisible floor so
    # ManipulationScenario does not add its own visible grid plane.
    floor = UsdGeom.Plane.Define(stage, "/World/ground_plane")
    floor.CreateAxisAttr("Z")
    floor.CreateWidthAttr(60.0)
    floor.CreateLengthAttr(60.0)
    floor.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(250.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 0.95))
    for i, (x, y) in enumerate([(0.3, -0.6), (0.3, 1.0), (2.6, 0.7)]):
        light = UsdLux.RectLight.Define(stage, f"/World/CellLights/Rect_{i}")
        light.CreateWidthAttr(1.2)
        light.CreateHeightAttr(0.4)
        light.CreateIntensityAttr(6000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
        set_xform(light.GetPrim(), (x, y, 3.2))


def build_cell(stage: Usd.Stage, assets_root: str) -> None:
    steel = preview_material(stage, "/World/Looks/PaintedSteel", (0.18, 0.2, 0.22), roughness=0.45, metallic=0.6)
    safety = preview_material(stage, "/World/Looks/SafetyYellow", (0.95, 0.72, 0.05), roughness=0.5)

    static_box(stage, "/World/Cell/Pedestal", (ROBOT_BASE[0], ROBOT_BASE[1], ROBOT_BASE[2] / 2), (0.32, 0.32, ROBOT_BASE[2]), steel)
    static_box(stage, "/World/Cell/PedestalPlate", (ROBOT_BASE[0], ROBOT_BASE[1], 0.01), (0.5, 0.5, 0.02), safety)
    stand_size = (2 * KLT_HALF[1] * TOTE_SCALE[1] + 0.06, 2 * KLT_HALF[0] * TOTE_SCALE[0] + 0.06, TOTE_STAND_TOP)
    static_box(stage, "/World/Cell/ToteStand", (*TOTE_CENTER_XY, TOTE_STAND_TOP / 2), stand_size, steel)

    klt = assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd"
    tote = add_reference(stage, "/World/Cell/Tote", klt, (*TOTE_CENTER_XY, tote_center_z()), math.degrees(TOTE_YAW), TOTE_SCALE)
    UsdPhysics.RigidBodyAPI.Apply(tote).CreateKinematicEnabledAttr(True)
    stage.GetPrimAtPath("/World/Cell/Tote/DomeLight").SetActive(False)

    belt_material = physics_material(stage, "/World/PhysicsMaterials/Belt", 0.9, 0.8)
    conveyor_url = assets_root + "/Isaac/Props/Conveyors/ConveyorBelt_A05.usd"
    for i in range(NUM_CONVEYORS):
        x0 = CONVEYOR_X0 + i * CONVEYOR_LENGTH
        conveyor = add_reference(stage, f"/World/Cell/Conveyor_{i}", conveyor_url, (x0, CONVEYOR_Y, 0.0))
        # Rollers do not spin; an invisible kinematic slab with surface velocity
        # carries the parts instead, so take the rollers out of contact.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{conveyor.GetPath()}/Rollers")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        belt = UsdGeom.Cube.Define(stage, f"/World/Cell/Belt_{i}")
        belt.CreateSizeAttr(1.0)
        belt.CreateVisibilityAttr("invisible")
        set_xform(belt.GetPrim(), (x0 + CONVEYOR_LENGTH / 2, CONVEYOR_Y, BELT_TOP - 0.01), 0.0, (CONVEYOR_LENGTH, 2 * BELT_HALF_WIDTH, 0.02))
        UsdPhysics.CollisionAPI.Apply(belt.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(belt.GetPrim()).CreateKinematicEnabledAttr(True)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt.GetPrim())
        surface.CreateSurfaceVelocityEnabledAttr(True)
        surface.CreateSurfaceVelocityLocalSpaceAttr(False)
        surface.CreateSurfaceVelocityAttr(Gf.Vec3f(args.belt_speed, 0.0, 0.0))
        bind(belt.GetPrim(), belt_material, "physics")

    catch_scale = (2.6, 2.6, 2.0)
    catch_x = CONVEYOR_X0 + NUM_CONVEYORS * CONVEYOR_LENGTH + KLT_HALF[0] * catch_scale[0] + 0.03
    catch = add_reference(stage, "/World/Cell/CatchBin", klt, (catch_x, CONVEYOR_Y, KLT_HALF[2] * catch_scale[2]), 0.0, catch_scale)
    UsdPhysics.RigidBodyAPI.Apply(catch).CreateKinematicEnabledAttr(True)
    stage.GetPrimAtPath("/World/Cell/CatchBin/DomeLight").SetActive(False)


def build_dressing(stage: Usd.Stage, assets_root: str) -> None:
    """Non-colliding-with-cell props that make the warehouse read as a production floor."""
    dt = assets_root + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse"
    props = [
        ("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A", (-1.0, -1.9, 0.0), 0.0),
        ("Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_B", (0.6, -2.0, 0.0), 90.0),
        ("Shipping/Wood_Crate_on_Pallet/Standard_A", (4.9, -1.6, 0.0), 15.0),
        ("Shipping/Wood_Crate_on_Pallet/HeavyDuty_A", (6.4, 1.6, 0.0), 0.0),
        ("Equipment/Forklifts/Forklift_A", (7.2, -1.2, 0.0), 160.0),
        ("Equipment/Carts/BinCart_A", (-1.0, 1.8, 0.0), 90.0),
        ("Safety/Barriers/GuardRail_A", (1.6, -1.25, 0.0), 0.0),
        ("Safety/Fire_Protection/FireExtinguisherStand_A", (-2.2, -2.2, 0.0), 45.0),
    ]
    for i, (folder, pos, yaw) in enumerate(props):
        url = first_usd(f"{dt}/{folder}")
        if url is None:
            print(f"[dressing] skipped missing prop folder {folder}")
            continue
        # Digital-twin props are authored in centimeters; references do not convert units.
        layer = Sdf.Layer.FindOrOpen(url)
        mpu = layer.pseudoRoot.GetInfo("metersPerUnit") if layer and layer.pseudoRoot.HasInfo("metersPerUnit") else 1.0
        s = float(mpu)
        add_reference(stage, f"/World/Dressing/Prop_{i}_{folder.rsplit('/', 1)[-1]}", url, pos, yaw, (s, s, s))


def spawn_items(stage: Usd.Stage, rng: random.Random) -> list[dict]:
    colors = [(0.9, 0.35, 0.05), (0.1, 0.35, 0.8), (0.75, 0.75, 0.72), (0.15, 0.55, 0.25), (0.85, 0.1, 0.1), (0.95, 0.8, 0.1)]
    item_material = physics_material(stage, "/World/PhysicsMaterials/Part", 1.0, 0.9)
    slots = [(dx, dy) for dx in (-0.12, 0.0, 0.12) for dy in (-0.065, 0.065)]
    rng.shuffle(slots)
    items = []
    floor_z = TOTE_STAND_TOP + 0.02
    for i in range(args.num_items):
        dx, dy = slots[i % len(slots)]
        layer = i // len(slots)
        x = TOTE_CENTER_XY[0] + dx + rng.uniform(-0.01, 0.01)
        y = TOTE_CENTER_XY[1] + dy + rng.uniform(-0.01, 0.01)
        yaw = rng.uniform(-math.pi / 4, math.pi / 4)
        quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        path = f"/World/Items/Part_{i}"
        is_cube = i % 2 == 0
        if is_cube:
            z = floor_z + ITEM_CUBE / 2 + 0.01 + layer * 0.06
            Cube(path, positions=[(x, y, z)], orientations=[quat], sizes=ITEM_CUBE, colors=[colors[i % len(colors)]])
        else:
            z = floor_z + ITEM_CYL_HEIGHT / 2 + 0.01 + layer * 0.06
            Cylinder(path, positions=[(x, y, z)], orientations=[quat], radii=ITEM_CYL_RADIUS, heights=ITEM_CYL_HEIGHT, axes="Z", colors=[colors[i % len(colors)]])
        GeomPrim(path, apply_collision_apis=True)
        RigidPrim(path)
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateMassAttr(ITEM_MASS)
        bind(stage.GetPrimAtPath(path), item_material, "physics")
        items.append({"path": path, "kind": "cube" if is_cube else "cylinder", "half_height": (ITEM_CUBE if is_cube else ITEM_CYL_HEIGHT) / 2})
    return items


CAMERA_VIEWS = {
    "wide": ((2.3, -2.0, 2.0), (0.9, 0.15, 0.75)),
    "close": ((1.55, -1.25, 1.55), (0.3, 0.05, 0.85)),  # tote, arm and drop zone fill the frame
}
CAMERA_EYE, CAMERA_TARGET = CAMERA_VIEWS[args.camera]
CAMERA_FOCAL = 18.0
CAMERA_RESOLUTION = (1280, 720)


def aim_camera(stage: Usd.Stage, path: str, eye, target) -> None:
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())


def add_camera(stage: Usd.Stage) -> str:
    path = "/World/Cameras/CellCam"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(CAMERA_FOCAL)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    aim_camera(stage, path, CAMERA_EYE, CAMERA_TARGET)
    return path


# ---------------------------------------------------------------------------
# Grasp selection
# ---------------------------------------------------------------------------


def wrap_half_turn(angle: float) -> float:
    """Map an angle to [-pi/2, pi/2); parallel-jaw grasps are symmetric under a half turn."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def up_z(q) -> float:
    _, x, y, _ = q
    return 1 - 2 * (x * x + y * y)


def top_down_orientation(psi: float) -> tuple[float, float, float, float]:
    """Hand pointing down, rotated by psi about world Z; fingers close along (sin psi, -cos psi)."""
    return (0.0, math.cos(psi / 2), math.sin(psi / 2), 0.0)


def grasp_clearance(center_xy: np.ndarray, psi: float, others_xy: list[np.ndarray]) -> float:
    tote = np.asarray(TOTE_CENTER_XY)
    d = np.array([math.sin(psi), -math.cos(psi)])
    n = np.array([-d[1], d[0]])
    rel = center_xy - tote
    score = math.inf
    for s in (1.0, -1.0):
        tip = rel + s * FINGER_REACH * d
        for other in others_xy:
            score = min(score, float(np.linalg.norm(tip - (other - tote))) - 0.034)
        for t in (-0.03, 0.03):
            corner = rel + s * HAND_HALF_LENGTH * d + t * n
            score = min(score, TOTE_INNER_HALF[0] - abs(corner[0]), TOTE_INNER_HALF[1] - abs(corner[1]))
    return score


def choose_grasp(items: list[dict], positions: np.ndarray, orientations: np.ndarray, skip: set[int]):
    tote = np.asarray(TOTE_CENTER_XY)
    rim = TOTE_STAND_TOP + 2 * KLT_HALF[2] * TOTE_SCALE[2]
    in_tote = [
        i
        for i in range(len(items))
        if i not in skip
        and abs(positions[i, 0] - tote[0]) < TOTE_INNER_HALF[0]
        and abs(positions[i, 1] - tote[1]) < TOTE_INNER_HALF[1]
        and positions[i, 2] < rim
    ]
    if not in_tote:
        return None
    top = max(positions[i, 2] for i in in_tote)
    best = None
    for i in in_tote:
        if positions[i, 2] < top - 0.02 or up_z(orientations[i]) < 0.9:
            continue
        others = [positions[j, :2] for j in in_tote if j != i]
        if items[i]["kind"] == "cube":
            # Closing jaws square up a cube that is up to ~15 deg off-axis; prefer face-aligned grasps.
            base = yaw_of(orientations[i])
            candidates = [
                (wrap_half_turn(base + k * math.pi / 2 + math.radians(off)), 0.002 * abs(off) / 15)
                for k in (0, 1)
                for off in (0, -15, 15)
            ]
        else:
            candidates = [(math.radians(a), 0.0) for a in range(-90, 90, 15)]
        for psi, penalty in candidates:
            score = grasp_clearance(positions[i, :2], psi, others) - penalty
            if best is None or score > best[2]:
                best = (i, psi, score)
    if best is None or best[2] < 0.0:
        return None
    return best


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


def make_controller(scenario: ManipulationScenario) -> PickPlaceController:
    config = scenario.robot_config
    tool_frame = config.tool.controller_frame
    arm = RmpFlowController(
        cumotion_robot=load_cumotion_supported_robot(config.name),
        cumotion_world_interface=scenario.world_interface,
        robot_joint_space=scenario.joint_space,
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
    )
    gripper = config.gripper
    gripper_args = {
        "robot_joint_space": scenario.joint_space,
        "joint_names": gripper.joint_names,
        "open_positions": gripper.open_positions,
        "closed_positions": gripper.closed_positions,
        "position_tolerance": gripper.position_tolerance,
        "velocity_tolerance": gripper.velocity_tolerance,
        "minimum_close_fraction": gripper.minimum_close_fraction,
    }
    return PickPlaceController(
        arm_controller=arm,
        gripper_open_controller=JointGripperController(**gripper_args, command=GripperCommand.OPEN),
        gripper_close_controller=JointGripperController(**gripper_args, command=GripperCommand.CLOSE),
        robot_site_space=scenario.site_space,
        tool_frame=tool_frame,
        controller_to_grasp_position=config.tool.controller_to_grasp_position,
        controller_to_grasp_orientation=config.tool.controller_to_grasp_orientation,
        grasp_orientation=config.grasp_orientation,
        approach_height=0.15,
        lift_height=0.25,
        phase_timeouts={PickPlacePhase.APPROACH_PICK: 8.0, PickPlacePhase.APPROACH_PLACE: 9.0},
    )


def make_goal(pick, pick_quat, place, place_quat) -> mg.RobotState:
    names = [PickPlaceController.PICK_SITE, PickPlaceController.PLACE_SITE]
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=names,
            positions=(names, wp.array([pick, place], dtype=wp.float32, device="cpu")),
            orientations=(names, wp.array([pick_quat, place_quat], dtype=wp.float32, device="cpu")),
        )
    )


class Recorder:
    """Optional RGB frame capture; disabled automatically when no renderer is available."""

    def __init__(self, camera_path: str, directory: str | None):
        self.directory = directory
        self.annotator = None
        self.count = 0
        self.times: list[float] = []
        if not directory:
            return
        try:
            import omni.replicator.core as rep

            os.makedirs(directory, exist_ok=True)
            product = rep.create.render_product(camera_path, CAMERA_RESOLUTION)
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.annotator.attach([product])
        except Exception as exc:  # noqa: BLE001
            print(f"[record] disabled: {exc}")
            self.annotator = None

    def capture(self, t: float | None = None) -> None:
        if self.annotator is None:
            return
        if t is not None and args.record_window and not (args.record_window[0] <= t <= args.record_window[1]):
            return
        data = self.annotator.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return
        from PIL import Image

        Image.fromarray(np.asarray(data)[..., :3]).save(os.path.join(self.directory, f"frame_{self.count:05d}.png"))
        self.times.append(t if t is not None else math.nan)
        self.count += 1


# ---------------------------------------------------------------------------
# Momentum solver integration
# ---------------------------------------------------------------------------

ROBOT_PATH = "/World/Cell/Franka"
FAULT_ROOT = "/World/Faults"
EVAL_STEPS = (1, 5, 10, 15, 30, 60)
INTRUDER_SIZE = (0.12, 0.12, 0.20)
INTRUDER_MASS = 2.0
INTRUDER_LEAD = 2.0  # seconds between spawning upstream and reaching the drop point
PROJECTILE_RADIUS = 0.035
PROJECTILE_MASS = 0.3
PROJECTILE_FLIGHT = 0.5  # seconds from launch to the gripper's extrapolated position
BAD_ACTION_DROP = 0.25  # m: the corrupted policy commands the gripper this far below the placement pose
BAD_ACTION_WINDOW = (0.1, 1.2)  # seconds after DESCEND_PLACE starts
CARRY_PHASES = (PickPlacePhase.APPROACH_PLACE, PickPlacePhase.DESCEND_PLACE)
HOLDING_PHASES = (PickPlacePhase.LIFT, PickPlacePhase.APPROACH_PLACE, PickPlacePhase.DESCEND_PLACE)


def cell_obstacles() -> list[Box]:
    """Axis-aligned boxes for the fixed cell geometry the arm can hit."""
    cx, cy = TOTE_CENTER_XY
    ox, oy = KLT_HALF[1] * TOTE_SCALE[1], KLT_HALF[0] * TOTE_SCALE[0]  # tote outer half extents in world X, Y
    rim = TOTE_STAND_TOP + 2 * KLT_HALF[2] * TOTE_SCALE[2]
    wall_h = (rim - TOTE_STAND_TOP) / 2
    zc = TOTE_STAND_TOP + wall_h
    t = 0.006
    line_half = NUM_CONVEYORS * CONVEYOR_LENGTH / 2
    return [
        Box("floor", (0.0, 0.0, -0.5), (50.0, 50.0, 0.5)),
        Box("pedestal", (ROBOT_BASE[0], ROBOT_BASE[1], ROBOT_BASE[2] / 2), (0.16, 0.16, ROBOT_BASE[2] / 2)),
        Box("tote_stand", (cx, cy, TOTE_STAND_TOP / 2), (ox + 0.03, oy + 0.03, TOTE_STAND_TOP / 2)),
        Box("tote_wall_+x", (cx + ox - t, cy, zc), (t, oy, wall_h)),
        Box("tote_wall_-x", (cx - ox + t, cy, zc), (t, oy, wall_h)),
        Box("tote_wall_+y", (cx, cy + oy - t, zc), (ox, t, wall_h)),
        Box("tote_wall_-y", (cx, cy - oy + t, zc), (ox, t, wall_h)),
        Box("conveyor", (CONVEYOR_X0 + line_half, CONVEYOR_Y, BELT_TOP / 2), (line_half, BELT_HALF_WIDTH, BELT_TOP / 2)),
    ]


def item_half_extents(item: dict) -> tuple[float, float, float]:
    """Half extents in the part's own frame (cylinders use their bounding square prism)."""
    if item["kind"] == "cube":
        return (ITEM_CUBE / 2,) * 3
    return (ITEM_CYL_RADIUS, ITEM_CYL_RADIUS, ITEM_CYL_HEIGHT / 2)


def spawn_fault_objects(stage: Usd.Stage) -> list[dict]:
    """Pre-spawn the fault's dynamic object parked off-cell; it is teleported in when the fault fires."""
    specs = []
    if args.fault == "intruder":
        path = f"{FAULT_ROOT}/Intruder"
        Cube(path, positions=[(-3.5, 3.5, INTRUDER_SIZE[2] / 2 + 0.01)], sizes=1.0, scales=[INTRUDER_SIZE], colors=[(0.9, 0.1, 0.6)])
        specs.append({"path": path, "mass": INTRUDER_MASS, "half": tuple(s / 2 for s in INTRUDER_SIZE)})
    elif args.fault == "projectile":
        from isaacsim.core.experimental.objects import Sphere

        path = f"{FAULT_ROOT}/Projectile"
        Sphere(path, positions=[(-3.5, 4.0, PROJECTILE_RADIUS + 0.01)], radii=PROJECTILE_RADIUS, colors=[(0.9, 0.1, 0.6)])
        specs.append({"path": path, "mass": PROJECTILE_MASS, "half": (PROJECTILE_RADIUS,) * 3})
    for spec in specs:
        GeomPrim(spec["path"], apply_collision_apis=True)
        RigidPrim(spec["path"])
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(spec["path"])).CreateMassAttr(spec["mass"])
    return specs


def enable_contact_reports(stage: Usd.Stage, paths: list[str]) -> None:
    robot_bodies = [p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for path in robot_bodies + paths:
        api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path))
        api.CreateThresholdAttr().Set(0.0)


def quat_to_matrix(position, wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = position
    return T


class MomentumMonitor:
    """Runs the momentum solver every control step, injects faults, and records ground-truth contacts."""

    def __init__(self, scenario: ManipulationScenario, items: list[dict], item_view: RigidPrim, faults: list[dict]):
        art = scenario.articulation
        if list(art.dof_names) != FRANKA_JOINTS:
            raise RuntimeError(f"Unexpected Franka joint order {art.dof_names}")
        self.art = art
        self.items = items
        self.item_view = item_view
        self.faults = faults
        self.fault_view = RigidPrim([f["path"] for f in faults]) if faults else None
        self.object_paths = [i["path"] for i in items] + [f["path"] for f in faults]

        kp, kd = (g.numpy()[0].astype(np.float64) for g in art.get_dof_gains())
        max_effort = art.get_dof_max_efforts().numpy()[0].astype(np.float64)
        drive_types = art.get_dof_drive_types()[0]
        # panda_finger_joint2 is a mimic joint without its own drive; model it as a copy of finger 1.
        for arr in (kp, kd, max_effort):
            arr[8] = arr[7]
        drive = DriveModel(
            stiffness=kp,
            damping=kd,
            max_effort=max_effort,
            acceleration_drive=np.array([d == "acceleration" for d in drive_types[:8]] + [drive_types[7] == "acceleration"]),
            gravity_enabled=bool(art.get_link_enabled_gravities().numpy()[0].any()),
        )
        base_p, base_q = art.get_world_poses()
        self.base = quat_to_matrix(base_p.numpy()[0], base_q.numpy()[0])
        self.kin = FrankaKinematics()
        self.hand = self.kin.link_index("panda_hand")
        tracker = ObjectMomentumTracker(
            self.object_paths,
            masses=np.array([ITEM_MASS] * len(items) + [f["mass"] for f in faults]),
            half_extents=np.array([item_half_extents(i) for i in items] + [f["half"] for f in faults]),
            window=args.history,
        )
        self.solver = MomentumSolver(self.kin, ArmMomentumModel(drive, PHYSICS_DT), tracker, cell_obstacles(),
                                     args.horizon_steps, PHYSICS_DT, margin=args.margin, confirm_steps=args.confirm_steps,
                                     uncertainty_scale=args.uncertainty_scale, commanded_alert_steps=args.arm_alert_steps)
        self.tote_boxes = {b.name for b in self.solver.boxes if b.name.startswith("tote")}
        self.targets_q = art.get_dof_position_targets().numpy()[0].astype(np.float64)
        self.targets_qd = art.get_dof_velocity_targets().numpy()[0].astype(np.float64)

        self.now = 0.0
        self.cycle = 0
        self.target: int | None = None
        self.phase: PickPlacePhase | None = None
        self.phase_start = 0.0
        self.carried_offset: np.ndarray | None = None
        self.fault_fired = False
        self.fault_log: list[dict] = []
        self.prev_hand: np.ndarray | None = None
        self.contacts: list[dict] = []
        self.expected_contacts = 0
        self.alerts: list[dict] = []
        self.guard_steps = 0
        self.trace: dict[str, list] = {k: [] for k in (
            "t", "cycle", "phase", "carried", "min_distance", "ttc", "residual", "kinetic_energy",
            "sim_ms", "policy_ms", "sense_ms", "solve_ms", "arm_rollout_ms", "object_rollout_ms", "kinematics_ms",
            "collision_ms", "apply_ms", "hand", "objects", "object_speed", "guarded",
            "confirmed", "alert_point", "alert_label", "hand_path",
        )}
        self.hand_pred = {"commanded": {k: [] for k in EVAL_STEPS if k <= args.horizon_steps},
                          "ballistic": {k: [] for k in EVAL_STEPS if k <= args.horizon_steps}}
        self.object_pred = {k: [] for k in EVAL_STEPS if k <= args.horizon_steps}
        self._subscription = None

    # -- ground truth ---------------------------------------------------

    def subscribe_contacts(self) -> None:
        from omni.physics.core import ContactEventType, get_physics_simulation_interface

        found = ContactEventType.CONTACT_FOUND

        def on_contact(headers, _data, _anchors) -> None:
            for h in headers:
                if h.type == found:
                    self._classify(str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1)))

        self._subscription = get_physics_simulation_interface().subscribe_physics_contact_report_events(on_contact)

    def _classify(self, a: str, b: str) -> None:
        if self.target is None or self.phase is None:
            return
        robot_a, robot_b = a.startswith(ROBOT_PATH), b.startswith(ROBOT_PATH)
        target = self.object_paths[self.target]
        if robot_a and robot_b:
            return
        if robot_a or robot_b:
            link, other = (a, b) if robot_a else (b, a)
            if other == target:
                self.expected_contacts += 1
                return
            self.contacts.append({"t": self.now, "a": link.rsplit("/", 1)[-1], "b": other, "phase": self.phase.name, "cycle": self.cycle})
        elif target in (a, b) and self.phase in CARRY_PHASES:
            other = b if a == target else a
            self.contacts.append({"t": self.now, "a": "carried:" + target, "b": other, "phase": self.phase.name, "cycle": self.cycle})

    # -- per-step -------------------------------------------------------

    def start_cycle(self, target: int) -> None:
        self.cycle += 1
        self.target = target
        self.carried_offset = None

    def sense(self) -> tuple[Proprioception, np.ndarray, np.ndarray]:
        art = self.art
        proprio = Proprioception(
            q=art.get_dof_positions().numpy()[0].astype(np.float64),
            qd=art.get_dof_velocities().numpy()[0].astype(np.float64),
            mass_matrix=art.get_mass_matrices().numpy()[0].astype(np.float64),
            coriolis=art.get_dof_coriolis_and_centrifugal_compensation_forces().numpy()[0].astype(np.float64),
            gravity=art.get_dof_gravity_compensation_forces().numpy()[0].astype(np.float64),
            base=self.base,
        )
        positions, orientations = (a.numpy() for a in self.item_view.get_world_poses())
        if self.fault_view is not None:
            fp, fq = (a.numpy() for a in self.fault_view.get_world_poses())
            positions, orientations = np.vstack([positions, fp]), np.vstack([orientations, fq])
        return proprio, positions.astype(np.float64), orientations.astype(np.float64)

    def propose(self, desired) -> tuple[np.ndarray, np.ndarray]:
        q_t, qd_t = self.targets_q.copy(), self.targets_qd.copy()
        joints = desired.joints if desired is not None else None
        if joints is not None:
            if joints.positions is not None:
                q_t[joints.position_indices.numpy()] = joints.positions.numpy().reshape(-1)
            if joints.velocities is not None:
                qd_t[joints.velocity_indices.numpy()] = joints.velocities.numpy().reshape(-1)
        q_t[8], qd_t[8] = q_t[7], qd_t[7]
        return q_t, qd_t

    def rules(self) -> ContactRules:
        rules = ContactRules(ignore_objects={self.target})
        if self.phase in HOLDING_PHASES and self.carried_offset is not None:
            rules.carried = self.target
            rules.carried_offset = self.carried_offset
            if self.phase is PickPlacePhase.LIFT:
                rules.carried_ignore_boxes = set(self.tote_boxes)
            elif self.phase is PickPlacePhase.DESCEND_PLACE:
                rules.carried_ignore_boxes = {"conveyor"}
        return rules

    def _fault_active(self) -> bool:
        phase = PickPlacePhase.DESCEND_PLACE if args.fault == "bad_action" else PickPlacePhase.APPROACH_PLACE
        return args.fault != "none" and self.cycle == args.fault_cycle and self.phase is phase

    def corrupt_action(self, t: float, q: np.ndarray, q_t: np.ndarray) -> bool:
        if args.fault != "bad_action" or not self._fault_active():
            return False
        dt = t - self.phase_start
        if not (BAD_ACTION_WINDOW[0] <= dt <= BAD_ACTION_WINDOW[1]):
            return False
        if not self.fault_fired:
            # Joint offset that moves the hand straight down, from a finite-difference hand Jacobian.
            eps = 1e-4
            batch = np.repeat(q[None], 8, axis=0)
            batch[1:, :7] += eps * np.eye(7)
            hands = self.kin.forward(batch, self.base)[:, self.hand, :3, 3]
            J = (hands[1:] - hands[0]).T / eps  # (3, 7)
            self.bad_action_offset = np.linalg.pinv(J) @ np.array([0.0, 0.0, -BAD_ACTION_DROP])
            self.fault_fired = True
            self.fault_log.append({"t": t, "fault": "bad_action", "hand_drop_m": BAD_ACTION_DROP,
                                   "joint_offset_rad": self.bad_action_offset.round(4).tolist()})
        q_t[:7] += self.bad_action_offset
        return True

    def trigger_object_fault(self, t: float, hand: np.ndarray, hand_velocity: np.ndarray) -> None:
        if self.fault_fired or self.fault_view is None or not self._fault_active():
            return
        index = len(self.items)
        if args.fault == "intruder":
            x = DROP_XY[0] - args.belt_speed * INTRUDER_LEAD
            p0 = np.array([x, DROP_XY[1], BELT_TOP + INTRUDER_SIZE[2] / 2 + 0.002])
            v0 = np.array([args.belt_speed, 0.0, 0.0])
        elif args.fault == "projectile":
            if t - self.phase_start < 0.5:
                return
            aim = hand + hand_velocity * PROJECTILE_FLIGHT
            p0 = aim + np.array([1.0, 0.0, 0.1])
            g = np.array([0.0, 0.0, -9.81])
            v0 = (aim - p0 - 0.5 * g * PROJECTILE_FLIGHT**2) / PROJECTILE_FLIGHT
        else:
            return
        self.fault_view.set_world_poses(positions=p0[None].astype(np.float32), orientations=np.array([[1.0, 0, 0, 0]], np.float32))
        self.fault_view.set_velocities(v0[None].astype(np.float32), np.zeros((1, 3), np.float32))
        self.solver.objects.forget(index)
        self.fault_fired = True
        self.fault_log.append({"t": t, "fault": args.fault, "position": p0.tolist(), "velocity": v0.tolist()})

    def control_step(self, t: float, phase: PickPlacePhase, scenario: ManipulationScenario, desired, sim_ms: float, policy_ms: float) -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_start = t
        t0 = time.perf_counter()
        proprio, objects, orientations = self.sense()
        t1 = time.perf_counter()
        hand_T = self.kin.forward(proprio.q[None], self.base)[0, self.hand]
        if phase is PickPlacePhase.LIFT and self.carried_offset is None:
            self.carried_offset = hand_T[:3, :3].T @ (objects[self.target] - hand_T[:3, 3])
        q_t, qd_t = self.propose(desired)
        modified = self.corrupt_action(t, proprio.q, q_t)
        t2 = time.perf_counter()
        pred = self.solver.solve(t, proprio, objects, q_t, qd_t, self.rules(), orientations)
        t3 = time.perf_counter()

        guarded = False
        if args.guard and pred.confirmed and pred.alert.model == "commanded":
            q_t[:7], qd_t[:7] = proprio.q[:7], 0.0
            modified = guarded = True
            self.guard_steps += 1
        t4 = time.perf_counter()
        if modified:
            self.art.set_dof_position_targets(q_t[None].astype(np.float32))
            self.art.set_dof_velocity_targets(qd_t[None].astype(np.float32))
        else:
            scenario.apply_robot_state(desired)
        self.targets_q, self.targets_qd = q_t, qd_t
        apply_ms = (time.perf_counter() - t4) * 1e3

        hand = hand_T[:3, 3]
        hand_velocity = (hand - self.prev_hand) / PHYSICS_DT if self.prev_hand is not None else np.zeros(3)
        self.prev_hand = hand
        self.trigger_object_fault(t, hand, hand_velocity)

        if pred.alert is not None:
            self.alerts.append({**pred.alert.__dict__, "confirmed": pred.confirmed, "cycle": self.cycle, "phase": phase.name, "guarded": guarded})
        tr = self.trace
        tr["t"].append(t)
        tr["cycle"].append(self.cycle)
        tr["phase"].append(phase.value)
        tr["carried"].append(self.target if phase in HOLDING_PHASES else -1)
        tr["min_distance"].append(pred.min_distance)
        tr["ttc"].append(pred.alert.time_to_collision if pred.alert else np.nan)
        tr["residual"].append(pred.momentum_residual)
        tr["kinetic_energy"].append(pred.kinetic_energy)
        tr["sim_ms"].append(sim_ms)
        tr["policy_ms"].append(policy_ms)
        tr["sense_ms"].append((t1 - t0) * 1e3)
        tr["solve_ms"].append((t3 - t2) * 1e3)
        for key in ("arm_rollout", "object_rollout", "kinematics", "collision"):
            tr[f"{key}_ms"].append(pred.timings_ms[key])
        tr["apply_ms"].append(apply_ms)
        tr["hand"].append(hand)
        tr["objects"].append(objects)
        tr["object_speed"].append(np.linalg.norm(self.solver.objects.velocity, axis=1))
        tr["guarded"].append(guarded)
        tr["confirmed"].append(pred.confirmed)
        tr["hand_path"].append(pred.hand_commanded if pred.hand_commanded is not None else pred.hand_ballistic)
        tr["alert_point"].append(pred.alert.point if pred.alert else (np.nan,) * 3)
        tr["alert_label"].append(
            f"{pred.alert.link.rsplit('/', 1)[-1]} -> {pred.alert.obstacle.rsplit('/', 1)[-1]} ({pred.alert.model})" if pred.alert else "")
        for k in self.object_pred:
            self.object_pred[k].append(pred.object_positions[k - 1])
            self.hand_pred["ballistic"][k].append(pred.hand_ballistic[k - 1])
            self.hand_pred["commanded"][k].append(pred.hand_commanded[k - 1] if pred.hand_commanded is not None else np.full(3, np.nan))

    # -- report ---------------------------------------------------------

    def summarize(self) -> dict:
        tr = {k: np.asarray(v) for k, v in self.trace.items()}
        times = tr["t"]
        # Steps where the cycle changed are discontinuous for k-step comparisons; consecutive indices only.
        hand_err = {model: prediction_error(times, {k: np.asarray(v) for k, v in preds.items()}, tr["hand"])
                    for model, preds in self.hand_pred.items()}
        n_obj = tr["objects"].shape[1] if tr["objects"].size else 0
        moving = (tr["object_speed"] > 0.02) & (np.arange(n_obj)[None, :] != tr["carried"][:, None])
        obj_err = prediction_error(times, {k: np.asarray(v) for k, v in self.object_pred.items()}, tr["objects"], moving)
        episodes = alert_episodes([a for a in self.alerts if a["confirmed"]], PHYSICS_DT)
        raw_episodes = alert_episodes(self.alerts, PHYSICS_DT)
        events = contact_events(self.contacts)
        horizon = args.horizon_steps * PHYSICS_DT
        solver_ms = tr["sense_ms"] + tr["solve_ms"]
        policy_ms = tr["policy_ms"]
        timing = {
            "physics_dt_ms": PHYSICS_DT * 1e3,
            "horizon_s": horizon,
            "sim_step": timing_stats(tr["sim_ms"].tolist()),
            "policy": timing_stats(policy_ms.tolist()),
            "apply_action": timing_stats(tr["apply_ms"].tolist()),
            "solver_sense": timing_stats(tr["sense_ms"].tolist()),
            "solver_predict": timing_stats(tr["solve_ms"].tolist()),
            "solver_total": timing_stats(solver_ms.tolist()),
            "solver_breakdown_mean_ms": {k: float(np.mean(tr[f"{k}_ms"])) for k in ("arm_rollout", "object_rollout", "kinematics", "collision")},
            "solver_over_policy": float(np.mean(solver_ms) / max(np.mean(policy_ms), 1e-9)),
            "solver_over_control_step": float(np.mean(solver_ms) / max(np.mean(tr["sim_ms"] + policy_ms + tr["apply_ms"]), 1e-9)),
            "predicted_sim_seconds_per_wall_second": float(horizon / max(np.mean(solver_ms) * 1e-3, 1e-9)),
        }
        quiet = np.isnan(tr["ttc"])
        return {
            "median_hand_error": {model: {str(k): float(v[k - 1]) for k in self.object_pred}
                                    for model, v in self.solver.median_error.items()},
            "config": {"horizon_steps": args.horizon_steps, "history": args.history, "margin": args.margin,
                       "confirm_steps": args.confirm_steps, "uncertainty_scale": args.uncertainty_scale,
                       "arm_alert_steps": args.arm_alert_steps,
                       "fault": args.fault, "fault_cycle": args.fault_cycle, "guard": args.guard},
            "faults": self.fault_log,
            "timing": timing,
            "hand_prediction_error": hand_err,
            "object_prediction_error": obj_err,
            "momentum_residual": {
                "p99_quiet": float(np.percentile(tr["residual"][quiet], 99)) if quiet.any() else None,
                "max": float(tr["residual"].max()) if tr["residual"].size else None,
                "max_at_t": float(times[np.argmax(tr["residual"])]) if tr["residual"].size else None,
            },
            "expected_contacts": self.expected_contacts,
            "unexpected_contacts": self.contacts[:50],
            "forecast": score(episodes, events, horizon),
            "forecast_unconfirmed": {k: v for k, v in score(raw_episodes, events, horizon).items() if k not in ("events", "false_alarms")},
            "guard_steps": self.guard_steps,
        }

    def save_trace(self, path: str, frame_times: list[float]) -> None:
        data = {k: np.asarray(v) for k, v in self.trace.items()}
        for model, preds in self.hand_pred.items():
            for k, v in preds.items():
                data[f"hand_{model}_{k}"] = np.asarray(v)
        for k, v in self.object_pred.items():
            data[f"objects_pred_{k}"] = np.asarray(v)
        data["eval_steps"] = np.asarray(list(self.object_pred))
        data["frame_times"] = np.asarray(frame_times, dtype=np.float64)
        data["camera_eye"], data["camera_target"] = np.asarray(CAMERA_EYE), np.asarray(CAMERA_TARGET)
        data["camera_focal"], data["camera_resolution"] = np.asarray(CAMERA_FOCAL), np.asarray(CAMERA_RESOLUTION)
        data["contact_t"] = np.asarray([c["t"] for c in self.contacts])
        data["contact_label"] = np.asarray([f"{c['a'].rsplit('/', 1)[-1]} hit {c['b'].rsplit('/', 1)[-1]}" for c in self.contacts])
        data["object_names"] = np.asarray([p.rsplit("/", 1)[-1] for p in self.object_paths])
        data["fault_t"] = np.asarray([f["t"] for f in self.fault_log])
        np.savez_compressed(path, **data)


def print_momentum_summary(m: dict) -> None:
    t = m["timing"]
    row = lambda name, s: print(f"[timing] {name:<16} mean {s['mean_ms']:7.3f} ms  p95 {s['p95_ms']:7.3f} ms  max {s['max_ms']:7.3f} ms")  # noqa: E731
    print(f"[timing] physics dt {t['physics_dt_ms']:.2f} ms, prediction horizon {t['horizon_s']:.2f} s")
    for name in ("sim_step", "policy", "apply_action", "solver_sense", "solver_predict", "solver_total"):
        row(name, t[name])
    print(f"[timing] solver breakdown (mean ms) {json.dumps({k: round(v, 3) for k, v in t['solver_breakdown_mean_ms'].items()})}")
    print(f"[timing] solver/policy {t['solver_over_policy']:.2f}x, solver/full control step {t['solver_over_control_step']:.2f}x, "
          f"predicts {t['predicted_sim_seconds_per_wall_second']:.0f} s of future per wall-clock second")
    for model, err in m["hand_prediction_error"].items():
        print(f"[accuracy] hand {model:<9} " + "  ".join(f"k={k}: {v['mean_m'] * 1e3:6.1f} mm" for k, v in err.items()))
    print("[accuracy] moving objects  " + "  ".join(f"k={k}: {v['mean_m'] * 1e3:6.1f} mm" for k, v in m["object_prediction_error"].items()))
    u = m["forecast_unconfirmed"]
    print(f"[forecast] unconfirmed alerts: forecast {u['num_forecast']}/{u['num_events']}, episodes {u['num_alert_episodes']}, "
          f"false alarms {u['num_false_alarm_episodes']}")
    f = m["forecast"]
    print(f"[forecast] confirmed alerts: unexpected contact events {f['num_events']}, forecast {f['num_forecast']}, alert episodes "
          f"{f['num_alert_episodes']}, false alarms {f['num_false_alarm_episodes']}, mean lead {f['mean_lead_time']}")
    for ev in f["events"]:
        ep = ev["episode"]
        print(f"[forecast]   contact t={ev['t']:.3f}s pairs={ev['pairs'][:3]} forecast={ev['forecast']} lead={ev['lead_time']}"
              + (f" predicted_by={ep['models']} obstacles={ep['obstacles']} links={ep['links'][:4]}" if ep else ""))
    for ep in f["false_alarms"][:10]:
        print(f"[forecast]   false alarm t={ep['start']:.3f}-{ep['end']:.3f}s ttc={ep['first_time_to_collision']:.3f} "
              f"models={ep['models']} links={ep['links'][:4]} obstacles={ep['obstacles']}")
    print(f"[momentum] residual p99(quiet)={m['momentum_residual']['p99_quiet']} max={m['momentum_residual']['max']} "
          f"at t={m['momentum_residual']['max_at_t']} guard_steps={m['guard_steps']} faults={m['faults']}")


def main() -> int:
    rng = random.Random(args.seed)
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Unable to resolve the Isaac Sim assets root.")

    build_factory(stage, assets_root)
    build_cell(stage, assets_root)
    if not args.no_dressing:
        build_dressing(stage, assets_root)
    camera_path = add_camera(stage)

    scenario = ManipulationScenario("franka", robot_prim_path="/World/Cell/Franka", offset=ROBOT_BASE)
    scenario.setup_scene()
    items = spawn_items(stage, rng)
    item_view = RigidPrim([item["path"] for item in items])
    faults = spawn_fault_objects(stage) if args.momentum else []
    if args.momentum:
        enable_contact_reports(stage, [item["path"] for item in items] + [f["path"] for f in faults])

    scenario.initialize_world_binding(
        exclude_prim_paths=(
            # Fault objects are unknown to the planner by design.
            FAULT_ROOT,
            "/World/Factory",
            "/World/Dressing",
            "/World/ground_plane",
            "/World/Cell/Pedestal",
            "/World/Cell/PedestalPlate",
            # cuMotion rejects non-uniformly scaled ancestors; approach/lift heights clear the rim and
            # choose_grasp keeps the hand inside the walls during the vertical descent.
            "/World/Cell/Tote",
            "/World/Cell/CatchBin",
            *[f"/World/Cell/Conveyor_{i}/Rollers" for i in range(NUM_CONVEYORS)],
            *[f"/World/Cell/Belt_{i}" for i in range(NUM_CONVEYORS)],
            *[f"/World/Cell/Conveyor_{i}" for i in range(1, NUM_CONVEYORS)],
        )
    )
    controller = make_controller(scenario)

    if args.save_usd:
        stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
        print(f"[scene] exported {os.path.abspath(args.save_usd)}")

    recorder = Recorder(camera_path, args.record)
    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    monitor = MomentumMonitor(scenario, items, item_view, faults) if args.momentum else None
    if monitor is not None:
        monitor.subscribe_contacts()
    sim_time = 0.0
    steps = 0

    def step() -> bool:
        nonlocal sim_time, steps
        if monitor is not None:
            monitor.now = sim_time + PHYSICS_DT
        simulation_app.update()
        if not (app_utils.is_playing() and SimulationManager.is_simulating()):
            return False
        sim_time += PHYSICS_DT
        steps += 1
        if steps % args.record_every == 0:
            recorder.capture(sim_time)
        return True

    def poses() -> tuple[np.ndarray, np.ndarray]:
        p, q = item_view.get_world_poses()
        return p.numpy(), q.numpy()

    settle = 0
    while simulation_app.is_running() and settle < 90:
        settle += int(step())

    if args.preview:
        views = [
            (CAMERA_EYE, CAMERA_TARGET),
            ((0.9, -0.1, 2.6), (0.6, 0.05, 0.7)),
            ((-1.5, 0.2, 1.6), (0.6, 0.2, 0.75)),
            ((1.2, -1.6, 1.3), (0.3, 0.0, 0.8)),
        ]
        for eye, target in views:
            aim_camera(stage, camera_path, eye, target)
            for _ in range(30):
                simulation_app.update()
            recorder.capture()
        return 0

    report = {"belt_speed": args.belt_speed, "items": [], "failures": []}
    attempts: dict[int, int] = {}
    skip: set[int] = set()
    disabled: set[str] = set()
    place_quat = top_down_orientation(0.0)

    while simulation_app.is_running():
        positions, orientations = poses()
        choice = choose_grasp(items, positions, orientations, skip)
        if choice is None:
            break
        index, psi, clearance = choice
        item = items[index]
        attempts[index] = attempts.get(index, 0) + 1
        pick = tuple(float(v) for v in positions[index])
        place = (DROP_XY[0], DROP_XY[1], BELT_TOP + item["half_height"] + DROP_CLEARANCE)
        goal = make_goal(pick, top_down_orientation(psi), place, place_quat)
        record = {
            "path": item["path"],
            "kind": item["kind"],
            "attempt": attempts[index],
            "pick": pick,
            "grasp_yaw_deg": math.degrees(psi),
            "clearance": clearance,
            "start_time": sim_time,
        }
        if not controller.reset(scenario.read_robot_state(), goal, t=sim_time):
            raise RuntimeError(controller.failure_reason or "controller reset failed")
        if monitor is not None:
            monitor.start_cycle(index)
        max_carry_z = pick[2]
        while simulation_app.is_running() and not controller.is_done and not controller.failed:
            if args.max_time is not None and sim_time >= args.max_time:
                break
            t_step = time.perf_counter()
            if not step():
                continue
            t_policy = time.perf_counter()
            scenario.sync_world()
            previous = controller.phase
            desired = controller.forward(scenario.read_robot_state(), goal, sim_time)
            t_done = time.perf_counter()
            if desired is None:
                break
            if monitor is None:
                scenario.apply_robot_state(desired)
            else:
                monitor.control_step(sim_time, controller.phase, scenario, desired,
                                     sim_ms=(t_policy - t_step) * 1e3, policy_ms=(t_done - t_policy) * 1e3)
            if controller.phase is not previous and controller.phase is PickPlacePhase.DESCEND_PICK and item["path"] not in disabled:
                # Once grasped, a part never returns to the cell; keep it out of the planning world.
                scenario.set_planning_obstacles_enabled((item["path"],), False)
                disabled.add(item["path"])
            if controller.phase in (PickPlacePhase.LIFT, PickPlacePhase.APPROACH_PLACE, PickPlacePhase.DESCEND_PLACE):
                max_carry_z = max(max_carry_z, float(poses()[0][index, 2]))

        positions, _ = poses()
        record.update(
            end_time=sim_time,
            controller_done=controller.is_done,
            failure=controller.failure_reason,
            lifted=max_carry_z > pick[2] + 0.05,
            release_position=positions[index].tolist(),
            on_belt=bool(
                abs(positions[index, 1] - CONVEYOR_Y) < BELT_HALF_WIDTH
                and abs(positions[index, 2] - (BELT_TOP + item["half_height"])) < 0.02
            ),
        )
        report["items"].append(record)
        print(f"[cycle] {item['path']} done={controller.is_done} lifted={record['lifted']} on_belt={record['on_belt']} "
              f"t={sim_time:.1f}s failure={controller.failure_reason}")
        if not controller.is_done:
            report["failures"].append(record)
            if attempts[index] >= 2:
                skip.add(index)
        if args.max_time is not None and sim_time >= args.max_time:
            break

    # Let the line carry everything to the catch bin.
    catch_x0 = CONVEYOR_X0 + NUM_CONVEYORS * CONVEYOR_LENGTH
    tracking = {r["path"]: [] for r in report["items"] if r["on_belt"]}
    end_time = sim_time + (catch_x0 - DROP_XY[0]) / max(args.belt_speed, 1e-3) + 4.0
    if args.max_time is not None:
        end_time = min(end_time, args.max_time)
    while simulation_app.is_running() and sim_time < end_time:
        if not step():
            continue
        if steps % 30 == 0:
            positions, _ = poses()
            for i, item in enumerate(items):
                if item["path"] in tracking:
                    tracking[item["path"]].append((sim_time, *positions[i].tolist()))

    positions, _ = poses()
    delivered = []
    for i, item in enumerate(items):
        x, y, z = positions[i]
        if x > catch_x0 and abs(y - CONVEYOR_Y) < 0.6 and z < BELT_TOP:
            delivered.append(item["path"])
    speeds = {}
    for path, samples in tracking.items():
        on_line = [s for s in samples if s[1] < catch_x0 - 0.05 and abs(s[3] - BELT_TOP) < 0.06]
        if len(on_line) >= 2:
            speeds[path] = (on_line[-1][1] - on_line[0][1]) / (on_line[-1][0] - on_line[0][0])
    report.update(
        delivered=delivered,
        measured_belt_speeds=speeds,
        final_positions={item["path"]: positions[i].tolist() for i, item in enumerate(items)},
        sim_time=sim_time,
        frames=recorder.count,
    )
    successes = [r for r in report["items"] if r["controller_done"] and r["lifted"] and r["on_belt"]]
    report["summary"] = {
        "items": len(items),
        "placed_on_belt": len({r["path"] for r in successes}),
        "delivered_to_catch_bin": len(delivered),
        "failed_cycles": len(report["failures"]),
    }
    print("[summary]", json.dumps(report["summary"]), "speeds", {k: round(v, 3) for k, v in speeds.items()})
    if monitor is not None:
        report["momentum"] = monitor.summarize()
        print_momentum_summary(report["momentum"])
        if args.trace:
            monitor.save_trace(args.trace, recorder.times)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    if args.test:
        ok = report["summary"]["placed_on_belt"] == len(items) and len(delivered) == len(items)
        ok = ok and all(abs(v - args.belt_speed) < 0.3 * args.belt_speed for v in speeds.values())
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    if exit_code:
        raise SystemExit(exit_code)
