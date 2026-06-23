"""
Top-level walking controller: orchestrates all layers into one update() call.

    estimate state (L1) -> gait FSM/planners (L2-L6,L10)
        -> whole-body IK reference (L7) -> PD+gravity torque (L8,L9) -> data.ctrl

The controller only ever returns torques. The simulation is advanced by MuJoCo
physics under those torques (in the run script); nothing here writes qpos/qvel
during the run. set_home() writes the initial pose once, before the loop.
"""

import numpy as np

from src.controllers.robot_model import RobotModel, StateEstimator
from src.controllers.wbik import WholeBodyIK
from src.controllers.gait import GaitController
from src.controllers.joint_pd_controller import JointPDController
from src.controllers.balance import BalanceStabilizer
from src.utils.terminal_logger import TerminalLogger as Logger


# Per-joint-group PD gains (gravity compensation does the heavy lifting, so
# these only need to track the smooth IK reference).
# Posture-tracking PD is deliberately moderate: it shapes the body toward the
# IK reference (and stiffly tracks the swing leg / arms), but is soft enough on
# the legs that the whole-body CoM/attitude balance feedback (BalanceStabilizer,
# acting through hip+knee+ankle) retains the authority to actually move the CoM.
_KP = {
    "hip": 400.0, "knee": 450.0, "ankle": 220.0,
    "waist_yaw": 350.0, "waist": 350.0,
    "shoulder": 80.0, "elbow": 80.0, "wrist": 30.0,
}
_KD = {
    "hip": 18.0, "knee": 20.0, "ankle": 11.0,
    "waist_yaw": 16.0, "waist": 16.0,
    "shoulder": 5.0, "elbow": 5.0, "wrist": 3.0,
}


def _group(name):
    if "knee" in name: return "knee"
    if "ankle" in name: return "ankle"
    if "hip" in name: return "hip"
    if "waist_yaw" in name: return "waist_yaw"
    if "waist" in name: return "waist"
    if "shoulder" in name: return "shoulder"
    if "elbow" in name: return "elbow"
    if "wrist" in name: return "wrist"
    return "hip"


class WalkingController:
    def __init__(self, model, data, params: dict = None):
        params = params or {}
        self.model = model
        self.data = data
        self.robot = RobotModel(model)
        self.estimator = StateEstimator(self.robot,
                                        params.get("contact_force_thresh", 15.0))
        self.ik = WholeBodyIK(self.robot, params.get("ik", {}))
        self.gait = GaitController(self.robot, params.get("gait", {}))
        self.balance = BalanceStabilizer(self.robot, params.get("balance", {}))

        # Give the gait a live terrain sensor so footsteps adapt to the surface
        # (stairs / uneven ground), unless disabled in params.
        if params.get("terrain_sensing", True):
            from src.controllers.terrain import ground_height
            self.gait.terrain_fn = lambda x, y: ground_height(self.model, self.data, x, y)

        from src.controllers.robot_model import JOINT_NAMES
        kp_map = dict(_KP); kp_map.update(params.get("kp", {}))
        kd_map = dict(_KD); kd_map.update(params.get("kd", {}))
        kp = np.array([kp_map[_group(n)] for n in JOINT_NAMES])
        kd = np.array([kd_map[_group(n)] for n in JOINT_NAMES])
        self.pd = JointPDController(self.robot, kp, kd, gravity_comp=True)

        # Whole-body inverse-dynamics QP (the live, online controller). When
        # enabled (default) it computes ALL joint torques from a QP each tick.
        from src.controllers.wbqp import WholeBodyQP
        self.use_wbqp = params.get("use_wbqp", True)
        self.wbqp = WholeBodyQP(self.robot, params.get("wbqp", {}))

        self.dt = model.opt.timestep
        self.last = {}
        self._com_des_prev = None
        self._swing_prev = None

    def reset(self):
        """Set the home pose (initial condition) and seed all sub-controllers."""
        self.robot.set_home(self.data)
        state = self.estimator.update(self.data)
        self.gait.reset(state)
        self.ik.reset(self.data)
        self._com_des_prev = None
        self._swing_prev = None
        Logger.info(f"WalkingController ready (home pose set, "
                    f"{'WBC-QP' if self.use_wbqp else 'IK+PD'} controller).")

    def update(self):
        """One control tick. Returns torque vector (nu,) for data.ctrl."""
        d = self.data
        state = self.estimator.update(d)
        targets, info = self.gait.update(state, self.dt)

        if self.use_wbqp:
            tau = self._update_wbqp(state, targets, info)
        else:
            tau = self._update_ik_pd(state, targets)

        self.last = {
            "state": info["state"], "step": info["step_count"],
            "support": state["support"], "com": state["com"], "com_vel": state["com_vel"],
            "com_target": info["com_target"], "swing_target": info["swing_target"],
            "base_rpy": state["base_rpy"],
            "lforce": state["lfoot_force"], "rforce": state["rfoot_force"],
            "tau": tau, "sat": np.abs(tau) / np.maximum(self.robot.tau_limit, 1e-9),
            "wbqp": dict(self.wbqp.info),
        }
        return tau

    def _update_wbqp(self, state, targets, info):
        """Drive the whole-body QP from the gait task references."""
        d = self.data
        com_des = targets["com"]
        # CoM velocity feed-forward is left at zero: the finite-difference of the
        # target is noisy and the QP is sensitive to it, and the slow gait makes
        # the lag negligible. The position PD (kp_com) does the tracking.
        com_vel_des = np.zeros(3)
        self._com_des_prev = com_des.copy()

        # contact set + swing target from the FSM state
        st = info["state"]
        if st == "RIGHT_SWING":
            contacts, sw_foot, sw_pos = ["left"], "right", targets.get("swing_pos")
        elif st == "LEFT_SWING":
            contacts, sw_foot, sw_pos = ["right"], "left", targets.get("swing_pos")
        else:
            contacts, sw_foot, sw_pos = ["left", "right"], None, None

        swing = None
        if sw_foot is not None and sw_pos is not None:
            sw_vel = (np.zeros(3) if self._swing_prev is None
                      else (sw_pos - self._swing_prev) / self.dt)
            swing = {"foot": sw_foot, "pos": sw_pos, "vel": sw_vel}
            self._swing_prev = np.asarray(sw_pos).copy()
        else:
            self._swing_prev = None

        return self.wbqp.compute(d, com_des, com_vel_des, targets["torso_R"],
                                 contacts, swing)

    def _update_ik_pd(self, state, targets):
        """Legacy IK-reference + PD + Jacobian-transpose balance path."""
        d = self.data
        q_des, qd_des = self.ik.solve(
            base_pos=d.qpos[0:3], base_quat=d.qpos[3:7],
            base_vel=d.qvel[0:3], base_angvel=d.qvel[3:6],
            targets=targets, dt=self.dt)
        tau = self.pd.compute_torques(d, q_des, qd_des)
        tau_bal = self.balance.compute(
            d, com_des_xy=targets["com"][:2], com=state["com"],
            com_vel=state["com_vel"], rpy=state["base_rpy"], omega=state["base_angvel"])
        return np.clip(tau + tau_bal, -self.robot.tau_limit, self.robot.tau_limit)
