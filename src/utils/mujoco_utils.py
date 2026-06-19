"""
mujoco_utils.py — MuJoCo helper functions for the offline IK pipeline.

Ported from the reference Talos implementation (test/mujoco_utils.py).
Provides body-frame Jacobian extraction, SE(3) pose queries, position
control mapping, and joint-limit helpers used by the QP-based IK solver.
"""

import mujoco
import numpy as np


def apply_pos_control(model, data, qd):
    """
    Map a full qpos vector to actuator control signals.

    For each actuator, reads the target joint angle from ``qd`` at the
    corresponding qpos address and writes it into ``data.ctrl``.

    Note: this works directly only when actuators accept position targets
    (e.g. position servos).  For torque-mode motors (like the G1), use
    a PD wrapper on top of this mapping.
    """
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_qpos_addr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = qd[joint_qpos_addr]


def transformation(p, R):
    """Build a 4×4 homogeneous transformation from position and rotation."""
    T = np.eye(4)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = p.reshape(3)
    return T


def get_body_full_transformation(model, data, body_id):
    """Return the 4×4 world-frame SE(3) pose of a body."""
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def get_body_rot_jac_local_frame(model, data, body_id):
    """3×nv rotational Jacobian expressed in the body's local frame."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    R_bw = data.xmat[body_id].reshape(3, 3).T
    return R_bw @ jacr


def get_body_trans_jac_local_frame(model, data, body_id):
    """3×nv translational Jacobian expressed in the body's local frame."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    R_bw = data.xmat[body_id].reshape(3, 3).T
    return R_bw @ jacp


def get_body_jac_local_frame(model, data, body_id):
    """6×nv spatial Jacobian [ω; v] expressed in the body's local frame."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    R_bw = data.xmat[body_id].reshape(3, 3).T
    return np.vstack((R_bw @ jacr, R_bw @ jacp))


def get_current_joint_range(model, qk):
    """
    Compute per-DOF distance to joint limits relative to current config.

    Returns (d_min, d_max) arrays of shape (nv,).  The first 6 DOFs
    (free-joint translation + rotation) have ±inf bounds since the
    floating base is unconstrained.

    The G1's free joint is jnt_range[0]; all actuated joints start at
    jnt_range[1:].
    """
    lower_limits = model.jnt_range[1:, 0]
    upper_limits = model.jnt_range[1:, 1]

    limit_size = lower_limits.shape[0]
    nv = model.nv

    d_min = np.full(nv, -np.inf)
    d_max = np.full(nv, np.inf)

    d_min[nv - limit_size:] = lower_limits - qk[-limit_size:]
    d_max[nv - limit_size:] = upper_limits - qk[-limit_size:]

    return d_min, d_max
