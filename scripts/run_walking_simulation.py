"""
run_walking_simulation.py — High-Performance Closed-Loop Bipedal Locomotion Runner.

Orchestrates the entire execution pipeline for the Unitree G1:
1. Environments and visualizer thread initialization.
2. Low-level PD and whole-body controller mapping setups.
3. Initial crouch footprint initialization.
4. Live execution of the 500 Hz control loop.

Zero-Allocation Compliance:
---------------------------
All velocity references and timeline variables are updated in-place via pre-allocated 
memory buffers. No heap mutations are permitted on the hot execution path.
"""

import sys
import os
import time
import math
import mujoco
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_parser import SimConfig
from src.env.mujoco_env import MujocoEnv
from src.estimators.state_estimator import StateEstimator
from src.controllers.joint_pd_controller import ControllerConfig, JointPDController
from src.controllers.locomotion.locomotion_controler import BipedLocomotionController
from src.utils.logger.terminal_logger import TerminalLogger as Logger

# ── Target Crouch Configuration Parameters (Sourced from verify_stance.py) ───
_HP = -0.50   # Hip Pitch (rad)
_KN =  1.00   # Knee Flexion (rad)
_AP = -0.50   # Ankle Pitch (rad)

Q_CROUCH: np.ndarray = np.array([
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,  # Left Leg
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,  # Right Leg
     0.0,  0.0,   0.0,                     # Waist
     0.30,  0.30,  0.0,  0.50,  0.0,  0.0,  0.0, # Left Arm
     0.30, -0.30,  0.0,  0.50,  0.0,  0.0,  0.0, # Right Arm
], dtype=np.float64)

KP: np.ndarray = np.array([
    400.0, 200.0, 150.0, 450.0, 250.0, 100.0,  # Left Leg
    400.0, 200.0, 150.0, 450.0, 250.0, 100.0,  # Right Leg
    200.0, 150.0, 200.0,                       # Waist
     80.0,  80.0,  50.0,  80.0,  40.0,  20.0, 20.0, # Left Arm
     80.0,  80.0,  50.0,  80.0,  40.0,  20.0, 20.0, # Right Arm
], dtype=np.float64)

KD: np.ndarray = np.array([
    15.0,  8.0,  6.0, 18.0, 20.0,  5.0,  # Left Leg
    15.0,  8.0,  6.0, 18.0, 20.0,  5.0,  # Right Leg
     8.0,  6.0,  8.0,                    # Waist
     3.0,  3.0,  2.0,  3.0,  1.5,  0.8,  0.8, # Left Arm
     3.0,  3.0,  2.0,  3.0,  1.5,  0.8,  0.8, # Right Arm
], dtype=np.float64)

_WARMUP_STEPS: int = 500
_VIEWER_SYNC_EVERY: int = 5
_FOOT_SPHERE_TARGET_Z: float = 0.006

def _locate_foot_spheres(model: mujoco.MjModel) -> list[int]:
    """Extracts contact geometry addresses matching the sole specifications."""
    return [i for i in range(model.ngeom) if abs(model.geom_size[i, 0] - 0.005) < 1e-6]

def _set_crouch_initial_state(env: MujocoEnv, foot_geom_idx: list[int]) -> None:
    """Enforces stable baseline kinematic foot positioning to anchor height initialization."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[7:36] = Q_CROUCH
    mujoco.mj_forward(env.model, env.data)

    min_sphere_z = min(env.data.geom_xpos[i, 2] for i in foot_geom_idx)
    env.data.qpos[2] += _FOOT_SPHERE_TARGET_Z - min_sphere_z
    mujoco.mj_forward(env.model, env.data)

def main() -> None:
    # 1. Parse Simulation Constants Layout
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/simulation.yaml"))
    config = SimConfig(config_path)

    Logger.info(f"Loading environment configuration from: {config.scene_xml_path}")
    env = MujocoEnv(config)
    env.init_viewer() # Spin up native visualization rendering

    foot_geom_idx = _locate_foot_spheres(env.model)
    
    # 2. Instantiate Locomotion Components
    estimator = StateEstimator(env.model, env.data)
    
    low_level_pd = JointPDController(
        model=env.model,
        data=env.data,
        cfg=ControllerConfig(kp=KP, kd=KD, log_interval=500),
    )
    
    locomotion_controller = BipedLocomotionController(
        model=env.model,
        data=env.data,
        state_estimator=estimator,
        low_level_pd=low_level_pd,
        step_duration=0.35,   # Parameterized stride timing configuration
        nominal_width=0.18    # Normalized operational tracking stance width
    )

    # Pre-allocate velocity command vectors to eliminate heap changes inside the loop
    v_des_live = np.zeros(2, dtype=np.float64)

    # 3. Establish Stable Geometric Footprint Before Loop Starts
    Logger.info("Aligning contact constraints...")
    _set_crouch_initial_state(env, foot_geom_idx)

    # 4. Execute Rigid Zero-Velocity Warmup Phase
    Logger.info(f"Executing {_WARMUP_STEPS} steps of stand-still initialization...")
    qdot_zero = np.zeros(env.model.nu, dtype=np.float64)
    for _ in range(_WARMUP_STEPS):
        env.data.ctrl[:] = low_level_pd.compute_torques(Q_CROUCH, qdot_zero)
        env.step()
        if env.viewer.is_running() and _ % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()
            
    # Reset master clock frame boundaries for accurate tracking initialization
    env.data.time = 0.0
    Logger.info("Locomotion loop engaged. Executing tracking test profile...")

    step = 0
    sim_duration = 15.0 # Max validation timeline
    
    # 5. Core Control Loop Execution Block
    while env.viewer.is_running() and env.data.time < sim_duration:
        step_wall_start = time.perf_counter()
        t_current = env.data.time

        # ── Velocity Profile State Machine (Zero Allocations) ────────────────
        # 0.0s -> 2.0s: Steady stand state to confirm operational convergence
        # 2.0s -> 10.0s: Linear forward command execution at 0.15 m/s
        # 10.0s -> End: De-accelerate baseline to check braking profile stability
        if t_current < 2.0:
            v_des_live[0] = 0.0
        elif t_current < 10.0:
            v_des_live[0] = 0.15
        else:
            v_des_live[0] = 0.0

        # ── Step Advanced Kinematics Core ───────────────────────────────────
        # Process scheduler flags, LIPM filtering, and map normal equations down to torques
        torques = locomotion_controller.advance_control(v_des=v_des_live)
        env.data.ctrl[:] = torques

        # ── Advance Physics Domain ──────────────────────────────────────────
        env.step()

        # ── Sync Graphics Pipelines ─────────────────────────────────────────
        if step % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()

        # ── Diagnostic Performance Tracking Telemetry ───────────────────────
        if step % 500 == 0:
            com = estimator.get_com()
            stable = estimator.is_com_stable()
            Logger.debug(
                f"t={t_current:.2f}s | "
                f"Cmd_v_x={v_des_live[0]:.2f}m/s | "
                f"CoM_XYZ=[{com[0]:+.3f}, {com[1]:+.3f}, {com[2]:.3f}]m | "
                f"Phase_L={locomotion_controller.scheduler.swing_phases[0]:.2f} | "
                f"Stable={('YES' if stable else 'NO')}"
            )

        # ── Precision Real-Time Pacing Lock ──────────────────────────────────
        elapsed_wall = time.perf_counter() - step_wall_start
        remaining = env.model.opt.timestep - elapsed_wall
        if remaining > 0:
            time.sleep(remaining)

        step += 1

    Logger.info("Execution complete. Closing context interfaces.")
    env.close_viewer()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("Locomotion loop killed via interrupt.")
    except Exception as exc:
        Logger.error(f"Fatal crash inside pipeline execution loop: {exc}")
        raise