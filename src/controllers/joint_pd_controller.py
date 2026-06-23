"""
Layer 8 - Joint-space torque controller (PD + gravity/Coriolis compensation).

Implements exactly the control law documented in controllers/README.md:

    tau = kp * (q_ref - q) + kd * (qdot_ref - qdot) + tau_bias

where tau_bias = data.qfrc_bias projected onto the actuated DoFs. qfrc_bias is
MuJoCo's gravity + Coriolis + centrifugal generalised force, so adding it makes
the PD loop a feedforward-compensated tracker: the PD gains only have to fight
*tracking error*, not gravity. This is what lets moderate gains hold posture and
balance the floating-base robot with torques alone.

Output torques are clamped to each actuator's force range (Layer 9 awareness of
actuator limits). Commands are returned for assignment to data.ctrl; this class
never steps physics and never touches qpos/qvel.
"""

import numpy as np

from src.controllers.robot_model import RobotModel


class JointPDController:
    def __init__(self, robot: RobotModel, kp: np.ndarray, kd: np.ndarray,
                 gravity_comp: bool = True):
        self.r = robot
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.gravity_comp = gravity_comp
        assert self.kp.shape == (robot.nu,) and self.kd.shape == (robot.nu,)

    def compute_torques(self, data, q_ref: np.ndarray, qd_ref: np.ndarray):
        r = self.r
        q = data.qpos[r.act_qadr]
        qd = data.qvel[r.act_dofadr]

        tau = self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
        if self.gravity_comp:
            tau = tau + data.qfrc_bias[r.act_dofadr]

        # Layer 9: respect actuator torque limits.
        tau = np.clip(tau, -r.tau_limit, r.tau_limit)
        return tau

    def saturation(self, tau):
        """Fraction of |tau| relative to the limit, per actuator (diagnostics)."""
        return np.abs(tau) / np.maximum(self.r.tau_limit, 1e-9)
