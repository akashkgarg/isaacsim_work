"""Print the Franka drive/dynamics configuration and check FrankaKinematics against simulated link poses."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False, "renderer": "MinimalRendering", "disable_viewport_updates": True})

import omni.kit.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.examples.manipulation import ManipulationScenario
from momentum_solver import FRANKA_JOINTS, FrankaKinematics

SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
scenario = ManipulationScenario("franka", robot_prim_path="/World/Franka", offset=(0.0, 0.0, 0.75))
art = scenario.setup_scene()
simulation_app.update()
app_utils.play()
for _ in range(10):
    simulation_app.update()

print("dof_names", art.dof_names)
print("link_names", art.link_names)
print("link_paths", art.link_paths[0])
print("drive_types", art.get_dof_drive_types())
kp, kd = art.get_dof_gains()
print("gains kp", kp.numpy(), "kd", kd.numpy())
print("max_efforts", art.get_dof_max_efforts().numpy())
print("gravity enabled", art.get_link_enabled_gravities().numpy())
print("mass matrix shape", art.get_mass_matrices().shape, "jacobian", art.get_jacobian_matrices().shape)
print("link masses", art.get_link_masses().numpy())

kin = FrankaKinematics()
order = [art.dof_names.index(n) for n in FRANKA_JOINTS]
rng = np.random.default_rng(0)
base_p, base_q = art.get_world_poses()
print("base", base_p.numpy(), base_q.numpy())
link_paths = art.link_paths[0]
names = [p.rsplit("/", 1)[-1] for p in link_paths]
common = [n for n in kin.link_names if n in names]
view = RigidPrim([link_paths[names.index(n)] for n in common])
for trial in range(3):
    q = art.get_dof_positions().numpy()[0].copy()
    q[:7] += rng.uniform(-0.4, 0.4, 7)
    art.set_dof_positions(q[None])
    art.set_dof_position_targets(q[None])
    for _ in range(3):
        simulation_app.update()
    q = art.get_dof_positions().numpy()[0]
    base = np.eye(4)
    base[:3, 3] = base_p.numpy()[0]
    T = kin.forward(q[order][None], base)[0]
    pos = view.get_world_poses()[0].numpy()
    err = {n: float(np.linalg.norm(T[kin.link_index(n), :3, 3] - pos[i])) for i, n in enumerate(common)}
    print("fk error (m)", {k: round(v, 5) for k, v in err.items()})

simulation_app.close()
