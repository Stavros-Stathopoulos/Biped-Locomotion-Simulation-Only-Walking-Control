"""
dcm_walking_controller.py — Adapter between DcmTrajectoryPlanner and UnitreeG1 IK.

This module does NOT modify any existing class. It composes:
  - DcmTrajectoryPlanner  (pure math, produces CoM + foot references)
  - UnitreeG1             (provides IK solver, PD replay, and visualisation)

Architecture
------------
DcmTrajectoryPlanner.plan()  →  DcmTrajectory  →  ik_adapter()  →  List[qpos]
                                                        ↑
                                              UnitreeG1.inverse_kinematics()
"""

import numpy as np
import mujoco

from src.controllers.unitree_g1 import UnitreeG1
from src.planners.dcm_trajectory_planner import (
    DcmTrajectory,
    DcmTrajectoryPlanner,
    Footstep,
    GaitTiming,
    LipmConfig,
    generate_footstep_sequence,
)
from src.utils import math_utils, mujoco_utils


# ══════════════════════════════════════════════════════════════════════════════
# IK adapter — maps DcmTrajectory → list of qpos via existing IK solver
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_to_joint_configs(
    robot: UnitreeG1,
    traj: DcmTrajectory,
    *,
    foot_weight: float = 40.0,
    com_weight: float = 20.0,
    ik_step: float = 0.001,
    ik_max_iters: int = 3000,
    subsample: int = 1,
) -> list[np.ndarray]:
    """
    Convert a DcmTrajectory into a list of full-body joint configurations.

    Uses the existing UnitreeG1.inverse_kinematics QP solver without
    modifying it. This is the Adapter Pattern: it transforms the planner's
    Cartesian-space output into the joint-space format expected by
    visualize_traj / position_control / simulate.

    Parameters
    ----------
    robot         : UnitreeG1 instance (provides IK, model, body IDs).
    traj          : DcmTrajectory from the planner.
    foot_weight   : IK task weight for each foot axis.
    com_weight    : IK task weight for each CoM axis.
    ik_step       : QP integration step size.
    ik_max_iters  : Max QP solver iterations per waypoint.
    subsample     : Process every Nth sample (1 = all, 2 = half, etc.).

    Returns
    -------
    List of (nq,) joint configuration arrays.
    """
    data_ik = mujoco.MjData(robot.model)
    robot._reset_to_walk_pose(data_ik)

    # Cache initial foot rotations (roll/pitch from walk pose)
    R_left_init = data_ik.xmat[robot.left_foot_id].copy().reshape(3, 3)
    R_right_init = data_ik.xmat[robot.right_foot_id].copy().reshape(3, 3)

    w_feet = [np.full(6, foot_weight), np.full(6, foot_weight)]
    w_com = np.full(3, com_weight)

    N = len(traj.time_s)
    indices = range(0, N, subsample)
    joint_traj: list[np.ndarray] = []

    total = len(list(range(0, N, subsample)))
    for count, k in enumerate(range(0, N, subsample)):
        if count % 50 == 0:
            print(f"    IK waypoint {count}/{total} ...")

        # Build SE(3) targets from planner output
        R_left = math_utils.Ryaw(traj.left_yaw_rad[k]) @ R_left_init
        R_right = math_utils.Ryaw(traj.right_yaw_rad[k]) @ R_right_init

        T_left = mujoco_utils.transformation(traj.left_foot_m[k], R_left)
        T_right = mujoco_utils.transformation(traj.right_foot_m[k], R_right)

        q_sol = robot.inverse_kinematics(
            data_ik, [T_left, T_right], w_feet,
            traj.com_m[k], w_com,
            step=ik_step, max_iters=ik_max_iters,
        )
        joint_traj.append(q_sol.copy())

    return joint_traj


# ══════════════════════════════════════════════════════════════════════════════
# Physics replay — correct timestep matching
# ══════════════════════════════════════════════════════════════════════════════

def replay_physics(
    robot: UnitreeG1,
    traj: list[np.ndarray],
    planning_dt_s: float,
) -> None:
    """
    Replay a joint trajectory with physics, correctly matching the
    planning timestep to MuJoCo's simulation timestep.

    The existing position_control() runs ONE mj_step per waypoint.
    If the MuJoCo timestep (dt_sim) differs from the planning timestep
    (dt_plan), the motion plays at the wrong speed.

    This function holds each PD target for ceil(dt_plan / dt_sim)
    physics substeps, so simulated time matches planned time exactly.

    Parameters
    ----------
    robot          : UnitreeG1 instance with PD controller.
    traj           : List of (nq,) joint configurations from IK.
    planning_dt_s  : Timestep between consecutive waypoints [s].
    """
    dt_sim = robot.model.opt.timestep   # MuJoCo simulation timestep [s]
    n_substeps = max(1, int(round(planning_dt_s / dt_sim)))

    print(f"    Physics replay: dt_sim={dt_sim:.4f}s, dt_plan={planning_dt_s:.4f}s, "
          f"substeps={n_substeps}")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as vis:
        vis.cam.lookat[:] = [0, 0, 0.7]
        vis.cam.distance  = 3.0
        vis.cam.azimuth   = 180
        vis.cam.elevation = -20

        while vis.is_running():
            for q_des in traj:
                for _ in range(n_substeps):
                    robot._apply_pd_control(q_des)
                    mujoco.mj_step(robot.model, robot.data)
                vis.sync()
            # Hold final pose
            for _ in range(500):
                robot._apply_pd_control(traj[-1])
                mujoco.mj_step(robot.model, robot.data)
                vis.sync()


# ══════════════════════════════════════════════════════════════════════════════
# High-level convenience: plan + adapt in one call
# ══════════════════════════════════════════════════════════════════════════════

def plan_dcm_walk(
    robot: UnitreeG1,
    n_steps: int = 5,
    travel_distance_m: float = 0.5,
    theta_rad: float = 0.0,
    arc_height_m: float = 0.05,
    timing: GaitTiming | None = None,
    subsample: int = 1,
) -> tuple[list[np.ndarray], float]:
    """
    End-to-end: footstep plan → DCM trajectory → IK joint configs.

    Parameters
    ----------
    robot              : UnitreeG1 robot instance.
    n_steps            : Number of full L+R step cycles.
    travel_distance_m  : Total forward distance [m].
    theta_rad          : Walking heading [rad].
    arc_height_m       : Swing foot lift height [m].
    timing             : GaitTiming config (uses defaults if None).
    subsample          : IK subsampling factor (1 = full density).

    Returns
    -------
    (joint_traj, planning_dt_s) — trajectory and its timestep.
    """
    if timing is None:
        timing = GaitTiming()

    # Read robot state to get initial foot positions and CoM
    data_tmp = mujoco.MjData(robot.model)
    robot._reset_to_walk_pose(data_tmp)

    left_pos = data_tmp.xpos[robot.left_foot_id].copy()
    right_pos = data_tmp.xpos[robot.right_foot_id].copy()
    com_xy = data_tmp.subtree_com[0, :2].copy()
    com_height = data_tmp.subtree_com[0, 2]

    # Configure LIPM with robot's actual CoM height
    lipm_cfg = LipmConfig(com_height_m=float(com_height))
    print(f"    LIPM: ω = {lipm_cfg.omega_rps:.2f} rad/s, "
          f"z_c = {lipm_cfg.com_height_m:.3f} m")

    # Generate footsteps
    init_l, init_r, swings = generate_footstep_sequence(
        left_pos, right_pos, n_steps, travel_distance_m, theta_rad,
    )
    print(f"    Planned {len(swings)} swing footsteps")

    # Plan DCM trajectory
    planner = DcmTrajectoryPlanner(lipm_cfg)
    dcm_traj = planner.plan(
        init_l, init_r, swings, timing, com_xy,
        arc_height_m=arc_height_m,
    )
    print(f"    DCM trajectory: {len(dcm_traj.time_s)} samples, "
          f"{dcm_traj.time_s[-1]:.2f} s")

    # Effective planning dt (accounting for subsampling)
    effective_dt = timing.dt_s * subsample

    # Adapt to joint-space via IK
    joint_traj = trajectory_to_joint_configs(
        robot, dcm_traj, subsample=subsample,
    )
    print(f"    Joint trajectory: {len(joint_traj)} waypoints")

    return joint_traj, effective_dt
