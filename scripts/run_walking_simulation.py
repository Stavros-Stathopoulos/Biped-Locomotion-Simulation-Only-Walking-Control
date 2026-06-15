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
     0.0,  0.0,   0.3,                     # Waist
     0.30,  0.30,  0.0,  0.50,  0.0,  0.0,  0.0, # Left Arm
     0.30, -0.30,  0.0,  0.50,  0.0,  0.0,  0.0, # Right Arm
], dtype=np.float64)

# Gains identical to the verified verify_stance.py CTRL_CFG.
# The prior values (Kp_hip=400, Kp_knee=450) were 2-3× too high, amplifying
# any residual IK error into oscillating torques that destabilised the stance.
KP: np.ndarray = np.array([
    150.0, 100.0,  80.0, 200.0, 150.0,  60.0,  # Left Leg
    150.0, 100.0,  80.0, 200.0, 150.0,  60.0,  # Right Leg
    100.0,  80.0, 100.0,                        # Waist
     50.0,  50.0,  30.0,  50.0,  25.0,  12.0, 12.0,  # Left Arm
     50.0,  50.0,  30.0,  50.0,  25.0,  12.0, 12.0,  # Right Arm
], dtype=np.float64)

KD: np.ndarray = np.array([
     8.0,  5.0,  4.0, 10.0,  8.0,  3.0,  # Left Leg
     8.0,  5.0,  4.0, 10.0,  8.0,  3.0,  # Right Leg
     5.0,  4.0,  5.0,                    # Waist
     2.0,  2.0,  1.5,  2.0,  1.0,  0.5,  0.5,  # Left Arm
     2.0,  2.0,  1.5,  2.0,  1.0,  0.5,  0.5,  # Right Arm
], dtype=np.float64)

# Ankle balance correction during warmup (identical to verify_stance.py).
# Prevents the free pelvis from tipping backward before the IK loop engages.
_K_ANKLE_BALANCE: float = 1.5
_IDX_ANKLE_L: int = 4
_IDX_ANKLE_R: int = 10

_WARMUP_STEPS: int = 1000   # 2 s at 500 Hz — gives contact forces time to settle
_VIEWER_SYNC_EVERY: int = 5
_FOOT_SPHERE_TARGET_Z: float = 0.006

def _locate_foot_spheres(model: mujoco.MjModel) -> list[int]:
    """Extracts contact geometry addresses matching the sole specifications."""
    return [i for i in range(model.ngeom) if abs(model.geom_size[i, 0] - 0.005) < 1e-6]

def _set_crouch_initial_state(env: MujocoEnv, foot_geom_idx: list[int]) -> None:
    """Reset to crouched stance with correct floor contact height and CoM X centering."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[7:36] = Q_CROUCH
    mujoco.mj_forward(env.model, env.data)

    # Pelvis Z: lower until the lowest foot sphere just touches the floor.
    min_sphere_z = min(env.data.geom_xpos[i, 2] for i in foot_geom_idx)
    env.data.qpos[2] += _FOOT_SPHERE_TARGET_Z - min_sphere_z
    mujoco.mj_forward(env.model, env.data)

    # Pelvis X: translate so whole-body CoM is directly above the foot mid-point.
    # The crouched pose leaves the CoM ~35 mm behind the foot centroid; without
    # this correction the free pelvis immediately tips backward once gravity acts.
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com_x  = env.data.subtree_com[pelvis_id, 0]
    foot_x = float(np.mean([env.data.geom_xpos[i, 0] for i in foot_geom_idx]))
    env.data.qpos[0] += foot_x - com_x
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
        step_duration=0.40,
        # nominal_width is the Y offset added to left_hip_pitch_link to reach the
        # natural ankle position.  left_hip_pitch_link is at Y≈+0.064 m from pelvis;
        # the ankle is at Y≈+0.118 m, so the offset is ≈0.054 m → width ≈0.108 m.
        # The prior value (0.18) placed each swing foot ~38 mm too wide, causing
        # progressive lateral splay and a lateral fall.
        nominal_width=0.10,
    )

    # Pre-allocate velocity command vectors to eliminate heap changes inside the loop
    v_des_live = np.zeros(2, dtype=np.float64)
    q_ref_warmup = Q_CROUCH.copy()   # mutable copy for ankle balance correction

    # 3. Establish Stable Geometric Footprint Before Loop Starts
    Logger.info("Aligning contact constraints...")
    _set_crouch_initial_state(env, foot_geom_idx)

    # 4. Execute Warmup Phase with pelvis-pitch balance feedback.
    # Mirrors verify_stance.py: prevents the free pelvis from tipping before IK engages.
    Logger.info(f"Executing {_WARMUP_STEPS} steps of stand-still initialization...")
    qdot_zero = np.zeros(env.model.nu, dtype=np.float64)
    for _ in range(_WARMUP_STEPS):
        q_ref_warmup[:] = Q_CROUCH
        _qw, _qx, _qy, _qz = (env.data.qpos[3], env.data.qpos[4],
                                env.data.qpos[5], env.data.qpos[6])
        _t = max(-1.0, min(1.0, 2.0 * (_qw * _qy - _qz * _qx)))
        _pelvis_pitch = math.asin(_t)
        q_ref_warmup[_IDX_ANKLE_L] += _K_ANKLE_BALANCE * _pelvis_pitch
        q_ref_warmup[_IDX_ANKLE_R] += _K_ANKLE_BALANCE * _pelvis_pitch
        env.data.ctrl[:] = low_level_pd.compute_torques(q_ref_warmup, qdot_zero)
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