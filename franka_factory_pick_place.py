"""Franka bin-to-conveyor pick-and-place in a warehouse/factory work cell.

The Franka sits on a pedestal between a tote of parts and two chained roller
conveyors. The conveyors move items with a PhysX surface velocity, so dropped
parts are carried away physically. Grasps are real finger contacts (no fixed
joints or pose writes); arm motion comes from cuMotion RMPflow through the
maintained ``PickPlaceController`` state machine.

Run (from the Isaac Sim root)::

    ./python.sh franka_factory_pick_place.py --headless --test
    ./python.sh franka_factory_pick_place.py --save-usd franka_factory.usd
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num-items", type=int, default=6)
parser.add_argument("--belt-speed", type=float, default=0.15, help="Conveyor surface speed in m/s")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--no-dressing", action="store_true", help="Skip decorative factory props")
parser.add_argument("--save-usd", default=None, help="Export the composed stage before simulation starts")
parser.add_argument("--record", default=None, help="Directory for rendered RGB frames (needs a working renderer)")
parser.add_argument("--record-every", type=int, default=2)
parser.add_argument("--preview", action="store_true", help="Render stills from a few viewpoints into --record and exit")
parser.add_argument("--report", default=None, help="Write a JSON run report to this path")
parser.add_argument("--gpu", type=int, default=0, help="Single GPU used for rendering")
parser.add_argument("--test", action="store_true", help="Fail with a non-zero exit code if any gate fails")
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
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

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


CAMERA_EYE = (2.3, -2.0, 2.0)
CAMERA_TARGET = (0.9, 0.15, 0.75)


def aim_camera(stage: Usd.Stage, path: str, eye, target) -> None:
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())


def add_camera(stage: Usd.Stage) -> str:
    path = "/World/Cameras/CellCam"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(18.0)
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
        if not directory:
            return
        try:
            import omni.replicator.core as rep

            os.makedirs(directory, exist_ok=True)
            product = rep.create.render_product(camera_path, (1280, 720))
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.annotator.attach([product])
        except Exception as exc:  # noqa: BLE001
            print(f"[record] disabled: {exc}")
            self.annotator = None

    def capture(self) -> None:
        if self.annotator is None:
            return
        data = self.annotator.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return
        from PIL import Image

        Image.fromarray(np.asarray(data)[..., :3]).save(os.path.join(self.directory, f"frame_{self.count:05d}.png"))
        self.count += 1


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

    scenario.initialize_world_binding(
        exclude_prim_paths=(
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

    sim_time = 0.0
    steps = 0

    def step() -> bool:
        nonlocal sim_time, steps
        simulation_app.update()
        if not (app_utils.is_playing() and SimulationManager.is_simulating()):
            return False
        sim_time += PHYSICS_DT
        steps += 1
        if steps % args.record_every == 0:
            recorder.capture()
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
        max_carry_z = pick[2]
        while simulation_app.is_running() and not controller.is_done and not controller.failed:
            if not step():
                continue
            scenario.sync_world()
            previous = controller.phase
            desired = controller.forward(scenario.read_robot_state(), goal, sim_time)
            if desired is None:
                break
            scenario.apply_robot_state(desired)
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

    # Let the line carry everything to the catch bin.
    catch_x0 = CONVEYOR_X0 + NUM_CONVEYORS * CONVEYOR_LENGTH
    tracking = {r["path"]: [] for r in report["items"] if r["on_belt"]}
    end_time = sim_time + (catch_x0 - DROP_XY[0]) / max(args.belt_speed, 1e-3) + 4.0
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
