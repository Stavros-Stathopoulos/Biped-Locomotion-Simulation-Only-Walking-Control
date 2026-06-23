"""
Active balance stabilizer (Layer 3 CoM regulation + Layer 10 attitude hold).

The kinematic IK reference (Layer 7) cannot actively reject body tilt: the
pelvis is part of the *unactuated* floating base, so a pelvis-orientation IK
task only produces base velocity that we must discard. Real bipeds keep the
torso upright and the CoM over the foot by pushing on the ground with their
legs. We reproduce that with Jacobian-transpose ("virtual model") control:

    F_com = Kp_com * (com_des - com) - Kd_com * com_vel          (horizontal)
    M_pel = -Kp_ori * rpy           - Kd_ori * omega             (upright)

    tau_balance = J_com^T F_com + J_pelvisRot^T M_pel    (joint rows only)

J_com^T maps a desired horizontal force at the CoM to the joint torques that
generate it through the contact reactions; J_pelvisRot^T does the same for a
righting moment on the torso. Added on top of gravity compensation and the
posture-tracking PD, this gives the robot genuine, measurement-driven balance
authority while still using torques only.
"""

import numpy as np
import mujoco

from src.controllers.robot_model import RobotModel


class BalanceStabilizer:
    def __init__(self, robot: RobotModel, params: dict = None):
        self.r = robot
        self.m = robot.model
        p = params or {}
        # horizontal CoM force gains (N/m, N/(m/s)). With ~35 kg, kp~2000 gives
        # omega~7.5 rad/s and kd~600 is near-critical damping.
        # Secondary feed-forward GRF shaping. With the stiff posture PD owning the
        # joints (and the IK-reference CoM feedback owning active balance), this
        # term is a modest contact-force assist and pelvis-attitude damper.
        self.kp_com = np.array(p.get("kp_com_force", [400.0, 400.0]))
        self.kd_com = np.array(p.get("kd_com_force", [120.0, 120.0]))
        # pelvis righting moment gains (Nm/rad, Nm/(rad/s)) for roll, pitch, yaw
        self.kp_ori = np.array(p.get("kp_ori", [120.0, 120.0, 30.0]))
        self.kd_ori = np.array(p.get("kd_ori", [24.0, 24.0, 6.0]))
        # overall limit on the balance contribution per joint (Nm)
        self.tau_clip = p.get("tau_balance_clip", 100.0)

        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))

    def compute(self, data, com_des_xy, com, com_vel, rpy, omega):
        m = self.m
        # CoM Jacobian (live state) -> horizontal virtual force
        mujoco.mj_jacSubtreeCom(m, data, self._jp, 0)
        F = np.zeros(3)
        F[:2] = self.kp_com * (com_des_xy - com[:2]) - self.kd_com * com_vel[:2]
        tau_full = self._jp.T @ F

        # pelvis rotational Jacobian -> righting moment
        mujoco.mj_jacBody(m, data, self._jp, self._jr, self.r.pelvis_bid)
        M = -self.kp_ori * rpy - self.kd_ori * omega
        tau_full = tau_full + self._jr.T @ M

        tau = tau_full[self.r.act_dofadr]
        return np.clip(tau, -self.tau_clip, self.tau_clip)
