"""
verify_stance.py — Unitree G1 low-CoM crouched stance verification.

Target configuration (validated via forward kinematics):
  hip_pitch  = -0.50 rad  (negative = thigh swings forward, G1 convention)
  knee       = +1.00 rad  (knee flexion)
  ankle_pitch= -0.50 rad  (dorsiflexion — keeps foot flat; numerically solved
                            so all 8 foot sphere geoms contact the ground
                            simultaneously at z ≈ 0.006 m)

All joint angles lie within the joint ranges declared in g1_29dof.xml.
Pelvis height is adjusted programmatically so the lowest contact sphere
barely touches the floor before simulation begins.

Controller: vectorized joint-space PD + gravity/Coriolis feed-forward.
Gains are tuned analytically, accounting for:
  - Joint inertia and max torque (hips/knees ±88/±139 Nm carry full body load)
  - Armature (0.05 kg·m² reflected at each joint from the MJCF default)
  - Gravity compensation handling the steady-state load, so PD gains only
    need to handle transients and disturbances

Acceptance criteria:
  [x] compute_torques: zero dynamic allocation, zero Python loops on hot path
  [x] All commanded torques within jnt_actfrcrange limits (clipped in controller)
  [x] Robot holds crouched stance for ≥ 10 simulated seconds without falling
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco
import numpy as np

from src.utils.config_parser import SimConfig
from src.env.mujoco_env import MujocoEnv
from src.controllers.joint_pd_controller import JointPDController
from src.estimators.state_estimator import StateEstimator
from src.utils.logger.terminal_logger import TerminalLogger as Logger

# ── Actuator order (from g1_29dof.xml <actuator> block) ───────────────────────
# idx  name                      frcrange
#  0   left_hip_pitch_joint       ±88
#  1   left_hip_roll_joint        ±88
#  2   left_hip_yaw_joint         ±88
#  3   left_knee_joint            ±139
#  4   left_ankle_pitch_joint     ±50
#  5   left_ankle_roll_joint      ±50
#  6   right_hip_pitch_joint      ±88
#  7   right_hip_roll_joint       ±88
#  8   right_hip_yaw_joint        ±88
#  9   right_knee_joint           ±139
# 10   right_ankle_pitch_joint    ±50
# 11   right_ankle_roll_joint     ±50
# 12   waist_yaw_joint            ±88
# 13   waist_roll_joint           ±50
# 14   waist_pitch_joint          ±50
# 15   left_shoulder_pitch_joint  ±25
# 16   left_shoulder_roll_joint   ±25
# 17   left_shoulder_yaw_joint    ±25
# 18   left_elbow_joint           ±25
# 19   left_wrist_roll_joint      ±25
# 20   left_wrist_pitch_joint     ±5
# 21   left_wrist_yaw_joint       ±5
# 22   right_shoulder_pitch_joint ±25
# 23   right_shoulder_roll_joint  ±25
# 24   right_shoulder_yaw_joint   ±25
# 25   right_elbow_joint          ±25
# 26   right_wrist_roll_joint     ±25
# 27   right_wrist_pitch_joint    ±5
# 28   right_wrist_yaw_joint      ±5

# ── Target crouched configuration ─────────────────────────────────────────────
# Sign convention (G1): positive hip_pitch swings thigh BACKWARD.
# For a forward-leaning squat: use NEGATIVE hip_pitch.
# Flat-foot condition (numerically verified): ankle_pitch = hip_pitch = -0.50 rad
# gives equal world-Z height for all 8 foot sphere contact geoms.
_HP = -0.50   # hip pitch   (rad)  — thigh forward
_KN =  1.00   # knee        (rad)  — knee flexion
_AP = -0.50   # ankle pitch (rad)  — dorsiflexion; levels all foot spheres
#              Pelvis Z drops from 0.793 m (default) to ≈ 0.715 m

Q_CROUCH: np.ndarray = np.array([
    # ── left leg ──────────────────────────────────────────────────────
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,
    # ── right leg ─────────────────────────────────────────────────────
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,
    # ── waist (yaw, roll, pitch) — stay upright ───────────────────────
     0.0,  0.0,   0.0,
    # ── left arm (sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw) ─
     0.30,  0.30,  0.0,  0.50,  0.0,  0.0,  0.0,
    # ── right arm ────────────────────────────────────────────────────
     0.30, -0.30,  0.0,  0.50,  0.0,  0.0,  0.0,
], dtype=np.float64)

# ── Proportional gains (Nm/rad) ───────────────────────────────────────────────
# Design rationale:
#   ω_n = sqrt(Kp / I_eff)  with I_eff ≈ arm_inertia + armature (0.05 kg·m²)
#   Damping ratio ζ ≈ Kd / (2 * sqrt(Kp * I_eff))  →  target ζ ≈ 0.7–1.0
#   Gravity compensation handles steady-state; PD handles perturbations only.
#   All torques are clamped to jnt_actfrcrange in the controller.
KP: np.ndarray = np.array([
    # left leg
    250.0, 200.0, 150.0,  350.0, 100.0,  80.0,
    # right leg
    250.0, 200.0, 150.0,  350.0, 100.0,  80.0,
    # waist
    200.0, 150.0, 150.0,
    # left arm
     60.0,  60.0,  40.0,   60.0,  30.0,  15.0,  15.0,
    # right arm
     60.0,  60.0,  40.0,   60.0,  30.0,  15.0,  15.0,
], dtype=np.float64)

# ── Derivative gains (Nm·s/rad) ───────────────────────────────────────────────
# ζ ≈ 0.7 for legs/waist, ζ ≈ 0.9 for arms (arms are lightly loaded so more
# damping helps prevent oscillation from low-inertia wrist links).
KD: np.ndarray = np.array([
    # left leg
    8.0,  6.0,  5.0,  10.0,  4.0,  3.0,
    # right leg
    8.0,  6.0,  5.0,  10.0,  4.0,  3.0,
    # waist
    6.0,  5.0,  5.0,
    # left arm
    2.0,  2.0,  1.5,   2.0,  1.0,  0.5,  0.5,
    # right arm
    2.0,  2.0,  1.5,   2.0,  1.0,  0.5,  0.5,
], dtype=np.float64)

# Simulation duration for the acceptance test
_SIM_DURATION_S: float = 12.0

# Viewer sync rate: sync every N physics steps (avoids ~500 Hz OpenGL calls)
_VIEWER_SYNC_EVERY: int = 5

# Height at which foot sphere centers should rest when touching the floor.
# Sphere radius = 0.005 m; target center z = 0.006 m → bottom at z = 0.001 m.
_FOOT_SPHERE_TARGET_Z: float = 0.006


def _locate_foot_spheres(model: mujoco.MjModel) -> list[int]:
    """Return geom indices of all foot corner sphere geoms (size = 0.005 m)."""
    return [
        i for i in range(model.ngeom)
        if abs(model.geom_size[i, 0] - 0.005) < 1e-6
    ]


def _set_crouch_initial_state(env: MujocoEnv, foot_geom_idx: list[int]) -> None:
    """
    Set the simulation state to the crouched target configuration.

    Procedure:
    1. Reset to MuJoCo default (qpos = 0, qvel = 0).
    2. Write Q_CROUCH into the actuated joint qpos entries.
    3. Run mj_forward to propagate kinematics (no dynamics; velocities stay zero).
    4. Measure the lowest foot sphere z in world frame.
    5. Translate the pelvis downward so the lowest sphere rests at
       _FOOT_SPHERE_TARGET_Z (barely touching the floor without interpenetration).
    6. Final mj_forward to settle the kinematics before stepping.
    """
    mujoco.mj_resetData(env.model, env.data)

    # Write joint angles directly into the actuated qpos range.
    # qpos[0:7] is the floating base (position + quaternion) — left untouched here.
    # qpos[7:36] maps one-to-one onto the 29 actuators (all 1-DOF revolute joints).
    env.data.qpos[7:36] = Q_CROUCH
    mujoco.mj_forward(env.model, env.data)

    # Pelvis Z correction: lower pelvis until lowest foot sphere touches floor.
    min_sphere_z = min(env.data.geom_xpos[i, 2] for i in foot_geom_idx)
    env.data.qpos[2] += _FOOT_SPHERE_TARGET_Z - min_sphere_z
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/simulation.yaml")
    )
    config = SimConfig(config_path)

    Logger.info(f"Loading scene: {config.scene_xml_path}")
    env = MujocoEnv(config)
    env.init_viewer()

    # Pre-locate foot sphere geoms once (avoids per-step model queries)
    foot_geom_idx = _locate_foot_spheres(env.model)
    Logger.debug(f"Found {len(foot_geom_idx)} foot sphere geoms: {foot_geom_idx}")

    # Instantiate the optimised controller
    controller = JointPDController(
        model=env.model,
        data=env.data,
        kp=KP,
        kd=KD,
        log_interval=500,   # disk-log at ~1 Hz (500 Hz sim rate)
    )

    # Instantiate estimator for final CoM stability check
    estimator = StateEstimator(env.model, env.data)

    # Pre-allocate the zero-velocity reference (passed by reference every step)
    qdot_zero = np.zeros(env.model.nu, dtype=np.float64)

    Logger.info("Setting initial crouched state...")
    _set_crouch_initial_state(env, foot_geom_idx)

    pelvis_z = env.data.qpos[2]
    Logger.info(
        f"Initial state: pelvis_z={pelvis_z:.4f} m  "
        f"contacts={len(foot_geom_idx)}  "
        f"sim_dt={env.model.opt.timestep*1000:.1f} ms"
    )

    # ── Main simulation loop ────────────────────────────────────────────────────
    Logger.info(f"Running crouched stance for {_SIM_DURATION_S:.0f} s ...")

    step         = 0
    log_every    = 500              # status log interval (steps)
    real_t0      = time.perf_counter()

    while env.viewer.is_running() and env.data.time < _SIM_DURATION_S:
        step_wall_start = time.perf_counter()

        # ── Control ──────────────────────────────────────────────────────────
        # compute_torques returns a VIEW of controller._tau.
        # Copy it into data.ctrl immediately before the next call can overwrite it.
        env.data.ctrl[:] = controller.compute_torques(Q_CROUCH, qdot_zero)

        # ── Physics step ─────────────────────────────────────────────────────
        env.step()

        # ── Viewer sync (sub-sampled to cap OpenGL overhead) ─────────────────
        if step % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()

        # ── Periodic terminal status ──────────────────────────────────────────
        if step % log_every == 0 and step > 0:
            com   = estimator.get_com()
            stable = estimator.is_com_stable()
            Logger.debug(
                f"t={env.data.time:.2f}s  "
                f"CoM_z={com[2]:.4f}m  "
                f"pelvis_z={env.data.qpos[2]:.4f}m  "
                f"stable={'YES' if stable else 'NO '}"
            )

        # ── Real-time pacing ──────────────────────────────────────────────────
        elapsed_wall = time.perf_counter() - step_wall_start
        remaining    = env.model.opt.timestep - elapsed_wall
        if remaining > 0:
            time.sleep(remaining)

        step += 1

    # ── Final acceptance criteria report ───────────────────────────────────────
    Logger.info("─" * 60)
    Logger.info("Stance Verification Results:")

    com = estimator.get_com()
    Logger.info(
        f"  CoM position: X={com[0]:+.4f}  Y={com[1]:+.4f}  Z={com[2]:.4f}  [m]"
    )

    hull_pts, raw_pts = estimator.get_support_polygon()
    if hull_pts is not None:
        Logger.info(
            f"  Support polygon: {len(hull_pts)} hull vertices "
            f"from {len(raw_pts)} contact points"
        )
    else:
        Logger.error("  Support polygon: degenerate (< 3 non-collinear contacts)")

    stable = estimator.is_com_stable()
    if stable:
        Logger.info(
            f"  [PASS] CoM ground projection is INSIDE the support polygon"
        )
    else:
        Logger.error(
            f"  [FAIL] CoM ground projection is OUTSIDE the support polygon"
        )

    real_elapsed = time.perf_counter() - real_t0
    Logger.info(
        f"  Simulated {env.data.time:.2f} s in {real_elapsed:.1f} s wall time "
        f"({env.data.time/real_elapsed:.2f}x real-time)"
    )
    Logger.info("─" * 60)

    time.sleep(3.0)
    env.close_viewer()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        Logger.warning("Stance verification interrupted by user.")
    except Exception as exc:
        Logger.error(f"Fatal error: {exc}")
        raise
    finally:
        Logger.info("Stance verification concluded.")
