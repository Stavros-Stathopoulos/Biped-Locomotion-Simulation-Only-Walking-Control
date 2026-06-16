"""
Integration smoke-test for BipedLocomotionController.

Run from the project root:
    python -m src.controllers.locomotion

What this tests
---------------
* GaitScheduler — alternating contact states and swing-phase clock.
* FootstepPlanner — Raibert foot placement relative to the hips.
* SwingFootTrajectoryGenerator — cubic-Hermite + quartic hump trajectory.
* BipedLocomotionController.advance_control() — full orchestration loop.

The IK layer is still a placeholder (joints track current state), so the
robot will not physically walk.  The gait state printed to the terminal and
the foot target markers visible in the viewer confirm that the planning stack
is running correctly.
"""

import sys
import os
import math
import time

# Make the project root importable regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import mujoco
import numpy as np

from src.utils.config_parser import SimConfig
from src.env.mujoco_env import MujocoEnv
from src.controllers.joint_pd_controller import ControllerConfig, JointPDController
from src.estimators.state_estimator import StateEstimator
from src.utils.logger.terminal_logger import TerminalLogger as Logger
from src.controllers.locomotion.locomotion_controler import BipedLocomotionController
from src.controllers.fall_recovery_controller import FallRecoveryController


# ── Desired walking velocity ───────────────────────────────────────────────────
_V_DES = np.array([0.3, 0.0], dtype=np.float64)   # [v_x, v_y] m/s, forward walk

# ── Crouched reference pose (validated in verify_stance.py) ───────────────────
_HP, _KN, _AP = -0.50, 1.00, -0.50
Q_CROUCH = np.array([
    _HP,  0.0,  0.0, _KN,  _AP,  0.0,   # left leg
    _HP,  0.0,  0.0, _KN,  _AP,  0.0,   # right leg
     0.0,  0.0,  0.3,                    # waist
     0.30,  0.30, 0.0, 0.50, 0.0, 0.0, 0.0,  # left arm
     0.30, -0.30, 0.0, 0.50, 0.0, 0.0, 0.0,  # right arm
], dtype=np.float64)

# ── PD gains (same values as verify_stance.py) ────────────────────────────────
_CTRL_CFG = ControllerConfig(
    kp=np.array([
        150.0, 100.0,  80.0, 200.0, 150.0,  60.0,  # left leg
        150.0, 100.0,  80.0, 200.0, 150.0,  60.0,  # right leg
        100.0,  80.0, 100.0,                        # waist
         50.0,  50.0,  30.0,  50.0, 25.0, 12.0, 12.0,  # left arm
         50.0,  50.0,  30.0,  50.0, 25.0, 12.0, 12.0,  # right arm
    ], dtype=np.float64),
    kd=np.array([
          8.0,  5.0,  4.0, 10.0,  8.0,  3.0,  # left leg
          8.0,  5.0,  4.0, 10.0,  8.0,  3.0,  # right leg
          5.0,  4.0,  5.0,                     # waist
          2.0,  2.0,  1.5,  2.0, 1.0, 0.5, 0.5,  # left arm
          2.0,  2.0,  1.5,  2.0, 1.0, 0.5, 0.5,  # right arm
    ], dtype=np.float64),
    gravity_comp=True,
    nan_check=True,
    log_interval=500,
)

_FOOT_SPHERE_RADIUS   = 0.005
_FOOT_SPHERE_TARGET_Z = 0.006
_VIEWER_SYNC_EVERY    = 5
_LOG_EVERY            = 500     # terminal status every N steps (~1 Hz at 500 Hz)
_SIM_DURATION_S       = 60.0


def _locate_foot_spheres(model: mujoco.MjModel) -> list[int]:
    return [i for i in range(model.ngeom)
            if abs(model.geom_size[i, 0] - _FOOT_SPHERE_RADIUS) < 1e-6]


def _set_crouch_state(env: MujocoEnv, foot_geom_idx: list[int]) -> None:
    """Reset to the crouched stance with floor contact and CoM centered."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[7:36] = Q_CROUCH
    mujoco.mj_forward(env.model, env.data)

    # Pelvis Z: lower until the lowest foot sphere just touches the floor.
    min_z = min(env.data.geom_xpos[i, 2] for i in foot_geom_idx)
    env.data.qpos[2] += _FOOT_SPHERE_TARGET_Z - min_z
    mujoco.mj_forward(env.model, env.data)

    # Pelvis X: translate so CoM is over the foot mid-point.
    pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com_x  = env.data.subtree_com[pelvis_id, 0]
    foot_x = float(np.mean([env.data.geom_xpos[i, 0] for i in foot_geom_idx]))
    env.data.qpos[0] += foot_x - com_x
    mujoco.mj_forward(env.model, env.data)


def _log_gait_state(loco: BipedLocomotionController, step: int) -> None:
    sched = loco.scheduler
    side = "L-stance" if sched.current_step_index % 2 == 0 else "R-stance"
    Logger.debug(
        f"step={step:>6d}  t={loco.data.time:.2f}s  "
        f"{side}  stride_phase={sched.stride_phase:.2f}  "
        f"contacts=[L:{sched.contact_states[0]} R:{sched.contact_states[1]}]  "
        f"swing=[L:{sched.swing_phases[0]:.2f} R:{sched.swing_phases[1]:.2f}]  "
        f"tgt_L_z={loco.foot_target_pos_l[2]:.3f}m  "
        f"tgt_R_z={loco.foot_target_pos_r[2]:.3f}m"
    )


def main() -> None:
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../config/simulation.yaml")
    )
    config = SimConfig(config_path)

    Logger.info(f"Loading scene: {config.scene_xml_path}")
    env = MujocoEnv(config)
    env.init_viewer()

    foot_geom_idx = _locate_foot_spheres(env.model)
    Logger.debug(f"Found {len(foot_geom_idx)} foot sphere geoms.")

    pd_controller = JointPDController(env.model, env.data, _CTRL_CFG)
    estimator     = StateEstimator(env.model, env.data)
    loco          = BipedLocomotionController(
        model          = env.model,
        data           = env.data,
        state_estimator= estimator,
        low_level_pd   = pd_controller,
        step_duration  = 0.4,
        nominal_width  = 0.10,
    )
    recovery = FallRecoveryController(env.model, env.data)

    Logger.info("Setting crouched initial state ...")
    _set_crouch_state(env, foot_geom_idx)
    Logger.info(
        f"pelvis_z={env.data.qpos[2]:.4f} m  "
        f"pelvis_x={env.data.qpos[0]:.4f} m"
    )

    # ── Warmup: let contact forces and gravity comp settle ─────────────────────
    Logger.info("Warmup (500 steps) ...")
    for _ in range(500):
        torques = loco.advance_control(_V_DES)
        env.data.ctrl[:] = recovery.process(torques)
        env.step()
        if env.viewer.is_running() and _ % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()
    env.data.time = 0.0

    Logger.info(f"Running locomotion test for {_SIM_DURATION_S:.0f} s ...")
    Logger.info(f"Commanded velocity: vx={_V_DES[0]:.2f} m/s  vy={_V_DES[1]:.2f} m/s")

    step = 0
    last_sim_time = 0.0
    real_t0 = time.perf_counter()

    while env.viewer.is_running() and env.data.time < _SIM_DURATION_S:
        # Handle viewer pause
        is_paused = False
        if hasattr(env.viewer, 'run'):
            is_paused = not env.viewer.run
        if is_paused:
            env.sync_viewer()
            time.sleep(0.01)
            continue

        # Handle viewer reset
        if env.data.time < last_sim_time:
            Logger.info("Viewer reset detected — reinitialising ...")
            _set_crouch_state(env, foot_geom_idx)
            step = 0
            last_sim_time = 0.0
            continue

        wall_start = time.perf_counter()

        torques = loco.advance_control(_V_DES)
        env.data.ctrl[:] = recovery.process(torques)
        env.step()

        if step % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()

        if step % _LOG_EVERY == 0 and step > 0:
            _log_gait_state(loco, step)
            Logger.debug(
                f"  fall={recovery.fall_state.name} "
                f"pitch={math.degrees(recovery.pitch):+.1f}° "
                f"roll={math.degrees(recovery.roll):+.1f}° "
                f"CoM_z={recovery.com_height:.3f}m"
            )

        elapsed = time.perf_counter() - wall_start
        remaining = env.model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)

        last_sim_time = env.data.time
        step += 1

    real_elapsed = time.perf_counter() - real_t0
    Logger.info(
        f"Done. Simulated {env.data.time:.2f} s in {real_elapsed:.1f} s wall time "
        f"({env.data.time / max(real_elapsed, 1e-9):.2f}x real-time)."
    )

    time.sleep(2.0)
    env.close_viewer()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("Interrupted by user.")
    except Exception as exc:
        Logger.error(f"Fatal: {exc}")
        raise
    finally:
        Logger.info("Locomotion test concluded.")
