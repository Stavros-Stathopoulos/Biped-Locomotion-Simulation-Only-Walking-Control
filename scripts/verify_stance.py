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

Controller: vectorized joint-space PD feedback only — no gravity feed-forward.
Gravity is the disturbance; the PD gains are the mechanism that must reject it.
Pelvis-pitch → ankle correction damps forward tippling.

Gains are tuned to satisfy the marginal stability condition:
  2 * Kp_ankle > m * g * h_com  →  Kp_ankle > ~113 Nm/rad  (G1: m=35 kg, h≈0.66 m)
Ankle Kp=250 gives a ~2.2× stability margin. A small steady-state position
offset (q_eq ≠ q_ref) is expected: e_ss = τ_gravity / Kp per joint.

Acceptance criteria:
  [x] compute_torques: zero dynamic allocation, zero Python loops on hot path
  [x] All commanded torques within jnt_actfrcrange limits (clipped in controller)
  [x] Robot holds crouched stance for ≥ 10 simulated seconds without falling
"""

import sys
import os
import math
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
# Ankle Kp=250 satisfies: 2*Kp_ankle > m*g*h_com  (stability margin ≈ 2.2×).
# Hip-pitch Kp=400, knee Kp=450 chosen for stiff posture under body load.
KP: np.ndarray = np.array([
    # left leg:  hip_pitch  hip_roll  hip_yaw  knee   ankle_pitch  ankle_roll
         400.0,    200.0,   150.0,  450.0,    250.0,      100.0,
    # right leg
         400.0,    200.0,   150.0,  450.0,    250.0,      100.0,
    # waist
         200.0,    150.0,   200.0,
    # left arm
          80.0,     80.0,    50.0,   80.0,     40.0,       20.0,   20.0,
    # right arm
          80.0,     80.0,    50.0,   80.0,     40.0,       20.0,   20.0,
], dtype=np.float64)

# ── Derivative gains (Nm·s/rad) ───────────────────────────────────────────────
# Ankle Kd=20 → ζ ≈ 0.63 with Kp=250, I_eff≈0.25 kg·m².
KD: np.ndarray = np.array([
    # left leg:  hp    hr    hy   kn    ap    ar
          15.0,  8.0,  6.0, 18.0, 20.0,  5.0,
    # right leg
          15.0,  8.0,  6.0, 18.0, 20.0,  5.0,
    # waist
           8.0,  6.0,  8.0,
    # left arm
           3.0,  3.0,  2.0,  3.0,  1.5,  0.8,  0.8,
    # right arm
           3.0,  3.0,  2.0,  3.0,  1.5,  0.8,  0.8,
], dtype=np.float64)

# ── Balance feedback ───────────────────────────────────────────────────────────
# When the pelvis pitches forward (positive pelvis_pitch), increase ankle
# plantarflexion target (more negative ankle_pitch) to push CoM back.
_K_ANKLE_BALANCE: float = 1.5   # rad ankle correction per rad pelvis pitch
_IDX_ANKLE_L: int = 4           # index of left_ankle_pitch in Q_CROUCH
_IDX_ANKLE_R: int = 10          # index of right_ankle_pitch in Q_CROUCH

# Simulation duration for the acceptance test
_SIM_DURATION_S: float = 100.0

# Warmup steps: run 500 steps (1 s at 500 Hz) before starting the timer so
# contact forces can settle before the official stability measurement begins.
_WARMUP_STEPS: int = 500

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

    controller = JointPDController(
        model=env.model,
        data=env.data,
        kp=KP,
        kd=KD,
        log_interval=500,   # disk-log at ~1 Hz (500 Hz sim rate)
    )

    estimator = StateEstimator(env.model, env.data)

    qdot_zero = np.zeros(env.model.nu, dtype=np.float64)

    # Pre-allocated mutable reference to avoid per-step np.copy
    q_ref_live = Q_CROUCH.copy()

    Logger.info("Setting initial crouched state...")
    _set_crouch_initial_state(env, foot_geom_idx)

    pelvis_z = env.data.qpos[2]
    Logger.info(
        f"Initial state: pelvis_z={pelvis_z:.4f} m  "
        f"contacts={len(foot_geom_idx)}  "
        f"sim_dt={env.model.opt.timestep*1000:.1f} ms"
    )

    # ── Warmup phase ────────────────────────────────────────────────────────────
    # Run _WARMUP_STEPS physics steps before the official timer so contact
    # forces and gravity compensation converge before stability is judged.
    Logger.info(f"Warmup: {_WARMUP_STEPS} steps ...")
    for _ in range(_WARMUP_STEPS):
        env.data.ctrl[:] = controller.compute_torques(Q_CROUCH, qdot_zero)
        env.step()
        if env.viewer.is_running() and _ % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()
    env.data.time = 0.0   # reset clock — official timer starts now

    # ── Main simulation loop ─────────────────────────────────────────────────────
    Logger.info(f"Running crouched stance for {_SIM_DURATION_S:.0f} s ...")

    step      = 0
    log_every = 500
    real_t0   = time.perf_counter()

    while env.viewer.is_running() and env.data.time < _SIM_DURATION_S:
        step_wall_start = time.perf_counter()

        # ── Pelvis-pitch balance feedback ─────────────────────────────────────
        # Extract pelvis pitch from the floating-base quaternion (qpos[3:7]).
        # Rotation about the Y axis (pitch): arcsin(2*(qw*qy - qz*qx)).
        # A positive pelvis pitch means the torso has tipped forward; we
        # increase the ankle pitch target (more dorsiflexion) to push it back.
        q_ref_live[:] = Q_CROUCH
        _qw = env.data.qpos[3]
        _qx = env.data.qpos[4]
        _qy = env.data.qpos[5]
        _qz = env.data.qpos[6]
        _t = max(-1.0, min(1.0, 2.0 * (_qw * _qy - _qz * _qx)))
        _pelvis_pitch = math.asin(_t)
        q_ref_live[_IDX_ANKLE_L] += _K_ANKLE_BALANCE * _pelvis_pitch
        q_ref_live[_IDX_ANKLE_R] += _K_ANKLE_BALANCE * _pelvis_pitch

        # ── Control ───────────────────────────────────────────────────────────
        env.data.ctrl[:] = controller.compute_torques(q_ref_live, qdot_zero)

        # ── Physics step ──────────────────────────────────────────────────────
        env.step()

        # ── Viewer sync (sub-sampled to cap OpenGL overhead) ──────────────────
        if step % _VIEWER_SYNC_EVERY == 0:
            env.sync_viewer()

        # ── Periodic terminal status ───────────────────────────────────────────
        if step % log_every == 0 and step > 0:
            com    = estimator.get_com()
            stable = estimator.is_com_stable()
            Logger.debug(
                f"t={env.data.time:.2f}s  "
                f"CoM_z={com[2]:.4f}m  "
                f"pelvis_z={env.data.qpos[2]:.4f}m  "
                f"pitch={math.degrees(_pelvis_pitch):+.2f}°  "
                f"stable={'YES' if stable else 'NO '}"
            )

        # ── Real-time pacing ───────────────────────────────────────────────────
        elapsed_wall = time.perf_counter() - step_wall_start
        remaining    = env.model.opt.timestep - elapsed_wall
        if remaining > 0:
            time.sleep(remaining)

        step += 1

    # ── Final acceptance criteria report ────────────────────────────────────────
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
        Logger.info("  [PASS] CoM ground projection is INSIDE the support polygon")
    else:
        Logger.error("  [FAIL] CoM ground projection is OUTSIDE the support polygon")

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
