"""
Layer 7 - Whole-Body Inverse Kinematics reference generator.

This is a weighted, prioritised-by-weight task-space (velocity-level) IK, the
same formulation used in the QP-IK lab (CoM task + foot tasks + posture), but
solved through the regularised normal equations instead of a generic QP
backend so it has no external solver dependency:

    v* = argmin  sum_i  w_i || J_i v - xi_i ||^2  +  lambda || v - v_post ||^2
                                                   +  mu || v ||^2

  =>  ( sum_i w_i J_i^T J_i + (lambda + mu) I ) v = sum_i w_i J_i^T xi_i + lambda v_post

where xi_i = k_i (x_des - x_cur) + xdot_ff   is the task-space velocity command.

CRITICAL DESIGN POINT (per the project's physics rules): this solver runs on a
*private* MjData clone (`self.ref`). It NEVER reads or writes the live
simulation state. Its only outputs are reference trajectories q_des, qd_des,
which the torque controller (Layer 8) then tracks. The simulation is advanced
only by MuJoCo physics under motor torques.

The reference floating base is re-seeded from the *estimated* base pose every
tick so the kinematic plan stays anchored to the real robot; balance feedback
enters through the CoM target (Layer 3), not by cheating the base.
"""

import numpy as np
import mujoco

from src.controllers.robot_model import RobotModel


def _rot_error(R_des, R_cur):
    """Rotation vector (axis*angle) taking R_cur to R_des, expressed in world."""
    R_err = R_des @ R_cur.T
    # log map of rotation matrix
    cos = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


class WholeBodyIK:
    def __init__(self, robot: RobotModel, params: dict):
        self.r = robot
        self.m = robot.model
        self.ref = mujoco.MjData(self.m)          # private reference state
        self.nv = self.m.nv

        p = params
        # task gains (task-space P gain, 1/s) and weights
        self.k_foot = p.get("k_foot", 20.0)
        self.k_com = p.get("k_com", 8.0)
        self.k_torso = p.get("k_torso", 6.0)
        self.w_support = p.get("w_support", 120.0)
        self.w_swing = p.get("w_swing", 30.0)
        self.w_com = p.get("w_com", 40.0)
        self.w_torso = p.get("w_torso", 8.0)
        self.lam_post = p.get("lambda_posture", 2.0)
        self.k_post = p.get("k_posture", 2.0)
        self.mu = p.get("mu_damping", 1.0)
        self.v_max = p.get("v_max", 6.0)          # rad/s clamp on reference velocity

        # scratch jacobians
        self._jp = np.zeros((3, self.nv))
        self._jr = np.zeros((3, self.nv))

        self.q_des = robot.q_home.copy()
        self.qd_des = np.zeros(robot.nu)
        self.info = {"com_err": 0.0, "vnorm": 0.0, "cond": 0.0}

    def reset(self, data: mujoco.MjData):
        """Seed the reference state from the current simulation state."""
        self.ref.qpos[:] = data.qpos
        self.ref.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.ref)
        self.q_des = self.r.joint_q(self.ref).copy()
        self.qd_des = np.zeros(self.r.nu)

    def solve(self, base_pos, base_quat, base_vel, base_angvel, targets, dt):
        """Run one IK step and return (q_des, qd_des) for the actuated joints.

        targets keys:
            com           (3,)  desired CoM position
            torso_R       (3,3) desired pelvis orientation
            support       'left'|'right'  which foot is planted
            support_pos   (3,)  desired support-foot (ankle) position
            support_R     (3,3) desired support-foot orientation
            swing_pos     (3,) | None   desired swing-foot position
            swing_R       (3,3)
        """
        m, ref = self.m, self.ref

        # The reference base is a FREE variable that the IK floats to satisfy the
        # tasks (it is seeded from the home pose in reset()). We deliberately do
        # NOT inject the measured, tilting base here: doing so feeds body tilt
        # back into the joint references and creates a destabilising loop. The IK
        # is a clean kinematic plan; disturbance rejection is the job of the
        # BalanceStabilizer, which uses the measured state.
        mujoco.mj_kinematics(m, ref)
        mujoco.mj_comPos(m, ref)

        A = np.zeros((self.nv, self.nv))
        b = np.zeros(self.nv)

        # --- CoM task (3D position) -------------------------------------
        mujoco.mj_jacSubtreeCom(m, ref, self._jp, 0)
        Jc = self._jp.copy()
        com_cur = ref.subtree_com[0]
        xi = self.k_com * (targets["com"] - com_cur)
        A += self.w_com * Jc.T @ Jc
        b += self.w_com * Jc.T @ xi

        # --- support foot task (6D, hold planted) -----------------------
        sup_bid = self.r.lfoot_bid if targets["support"] == "left" else self.r.rfoot_bid
        Js, ei = self._body_task(sup_bid, targets["support_pos"], targets["support_R"], self.k_foot)
        A += self.w_support * Js.T @ Js
        b += self.w_support * Js.T @ ei

        # --- swing foot task (6D) ---------------------------------------
        # In double support the 'swing' foot is really a second support, so it
        # gets the full support weight (no left/right bias during weight shift).
        if targets.get("swing_pos") is not None:
            sw_bid = self.r.rfoot_bid if targets["support"] == "left" else self.r.lfoot_bid
            Jw, ew = self._body_task(sw_bid, targets["swing_pos"], targets["swing_R"], self.k_foot)
            w_sw = self.w_support if targets.get("double") else self.w_swing
            A += w_sw * Jw.T @ Jw
            b += w_sw * Jw.T @ ew

        # --- torso / pelvis orientation task (3D) -----------------------
        mujoco.mj_jacBody(m, ref, self._jp, self._jr, self.r.pelvis_bid)
        Rp = ref.xmat[self.r.pelvis_bid].reshape(3, 3)
        xi = self.k_torso * _rot_error(targets["torso_R"], Rp)
        A += self.w_torso * self._jr.T @ self._jr
        b += self.w_torso * self._jr.T @ xi

        # --- posture regularisation (joints toward home) ----------------
        v_post = np.zeros(self.nv)
        q_now = self.r.joint_q(ref)
        v_post[self.r.act_dofadr] = self.k_post * (self.r.q_home - q_now)
        A += self.lam_post * np.eye(self.nv)
        b += self.lam_post * v_post

        # --- damping ----------------------------------------------------
        A += self.mu * np.eye(self.nv)

        # solve the weighted-least-squares (equality QP) normal equations
        v = np.linalg.solve(A, b)

        # clamp joint reference velocity for safety
        vj = v[self.r.act_dofadr]
        vj = np.clip(vj, -self.v_max, self.v_max)
        v[self.r.act_dofadr] = vj

        # integrate the reference configuration (kinematic, on the clone only)
        mujoco.mj_integratePos(m, ref.qpos, v, dt)

        # solver diagnostics (recomputed every tick -> proof it is live)
        self.info = {
            "com_err": float(np.linalg.norm(targets["com"] - com_cur)),  # m
            "vnorm": float(np.linalg.norm(vj)),                          # rad/s
            "cond": float(np.linalg.cond(A)),                            # QP conditioning
        }

        self.q_des = self.r.joint_q(ref).copy()
        self.qd_des = vj.copy()
        return self.q_des, self.qd_des

    def _body_task(self, body_id, pos_des, R_des, k):
        """Build a 6xnv body Jacobian and the 6D velocity command for a pose."""
        m, ref = self.m, self.ref
        mujoco.mj_jacBody(m, ref, self._jp, self._jr, body_id)
        J = np.vstack([self._jp, self._jr])
        pos_cur = ref.xpos[body_id]
        R_cur = ref.xmat[body_id].reshape(3, 3)
        xi = np.zeros(6)
        xi[0:3] = k * (pos_des - pos_cur)
        xi[3:6] = k * _rot_error(R_des, R_cur)
        return J, xi
