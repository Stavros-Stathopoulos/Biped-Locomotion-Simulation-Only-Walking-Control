"""
Robot model wrapper and state estimation for the Unitree G1 (29 DoF).

This module is the single source of truth for *indexing* the MuJoCo model
(joint <-> actuator <-> qpos/qvel mapping), for the nominal "home" crouch
pose, and for Layer 1 state estimation (CoM, orientation, contacts, support
foot).

Index facts for this model (verified against the MJCF):
    nq = 36  (7 free-base + 29 hinges)
    nv = 35  (6 free-base + 29 hinges)
    nu = 29  (one <motor> per hinge, torque actuators)

Because every actuated joint is a hinge listed in body-tree order and the
<actuator> block lists them in the same order, the mapping is contiguous:

    actuator i  <->  qpos[7 + i]   <->  qvel[6 + i]   <->  qfrc_bias[6 + i]

We still resolve everything through mj_name2id / jnt_qposadr / jnt_dofadr so
the code stays correct even if the model is reordered.
"""

import numpy as np
import mujoco

from src.utils.terminal_logger import TerminalLogger as Logger


# Actuated joints in MuJoCo order (matches the <actuator> block).
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Nominal "home" crouch angles (rad). Bent knees give the legs authority to
# both lower the CoM and execute swing/stance without hitting joint limits.
# Foot stays flat because hip_pitch + knee + ankle_pitch = 0.
_HOME_ANGLES = {
    "left_hip_pitch_joint": -0.40, "left_knee_joint": 0.80, "left_ankle_pitch_joint": -0.40,
    "right_hip_pitch_joint": -0.40, "right_knee_joint": 0.80, "right_ankle_pitch_joint": -0.40,
    "left_shoulder_pitch_joint": 0.20, "left_shoulder_roll_joint": 0.20, "left_elbow_joint": 0.40,
    "right_shoulder_pitch_joint": 0.20, "right_shoulder_roll_joint": -0.20, "right_elbow_joint": 0.40,
}


def quat_to_rpy(quat):
    """MuJoCo quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians."""
    w, x, y, z = quat
    # roll (x-axis)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class RobotModel:
    """Static structural information + index maps for the G1."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.nu = model.nu
        self.nv = model.nv
        self.nq = model.nq

        # actuator i -> joint id / qpos addr / dof addr
        self.act_jnt_id = np.array(
            [model.actuator_trnid[i, 0] for i in range(self.nu)], dtype=int)
        self.act_qadr = np.array(
            [model.jnt_qposadr[j] for j in self.act_jnt_id], dtype=int)
        self.act_dofadr = np.array(
            [model.jnt_dofadr[j] for j in self.act_jnt_id], dtype=int)

        # name -> actuator index
        self.name2act = {n: i for i, n in enumerate(JOINT_NAMES)}

        # body ids we care about
        self.pelvis_bid = self._bid("pelvis")
        self.torso_bid = self._bid("torso_link")
        self.lfoot_bid = self._bid("left_ankle_roll_link")
        self.rfoot_bid = self._bid("right_ankle_roll_link")

        # torque limits (per actuator) from the joint actuatorfrcrange
        self.tau_limit = np.array(
            [max(abs(model.jnt_actfrcrange[j, 0]), abs(model.jnt_actfrcrange[j, 1]))
             for j in self.act_jnt_id])

        # contact geom ids per foot (the four corner spheres)
        self.lfoot_geoms = self._foot_geoms(self.lfoot_bid)
        self.rfoot_geoms = self._foot_geoms(self.rfoot_bid)
        self.floor_geom = self._gid("floor")

        # nominal joint vector (actuator order) and base height for home pose
        self.q_home = np.zeros(self.nu)
        for name, val in _HOME_ANGLES.items():
            self.q_home[self.name2act[name]] = val
        self.base_home_z = self._compute_home_height()

        # foot geometry (from the MJCF corner spheres, ankle-roll frame)
        # heel at x=-0.05, toe at x=+0.12, half-width y=0.03, sole at z=-0.035
        self.foot_len_back = 0.05
        self.foot_len_fwd = 0.12
        self.foot_half_width = 0.03
        self.foot_center_x = 0.5 * (self.foot_len_fwd - self.foot_len_back)  # ~0.035
        self.sole_offset_z = -0.035

        Logger.debug(f"RobotModel: nu={self.nu} nv={self.nv} mass={model.body_mass.sum():.2f}kg "
                     f"home_base_z={self.base_home_z:.4f}")

    # -- id helpers --------------------------------------------------------
    def _bid(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body '{name}' not found")
        return bid

    def _gid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _foot_geoms(self, body_id):
        return set(g for g in range(self.model.ngeom)
                   if self.model.geom_bodyid[g] == body_id
                   and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                   and self.model.geom_contype[g] == 1)

    def _compute_home_height(self):
        """Place the pelvis so the home crouch rests the feet on the floor."""
        d = mujoco.MjData(self.model)
        d.qpos[3] = 1.0  # identity quaternion
        d.qpos[2] = 0.793
        for i in range(self.nu):
            d.qpos[self.act_qadr[i]] = self.q_home[i]
        mujoco.mj_forward(self.model, d)
        lo = np.inf
        for g in (self.lfoot_geoms | self.rfoot_geoms):
            z = d.geom_xpos[g][2] - self.model.geom_size[g][0]
            lo = min(lo, z)
        return 0.793 - lo + 0.002  # 2 mm clearance so we settle onto contact

    # -- pose setup --------------------------------------------------------
    def set_home(self, data: mujoco.MjData):
        """Write the home crouch pose into `data` (an initial condition, not a
        runtime override). Pelvis upright at the computed resting height."""
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[2] = self.base_home_z
        data.qpos[3] = 1.0
        for i in range(self.nu):
            data.qpos[self.act_qadr[i]] = self.q_home[i]
        mujoco.mj_forward(self.model, data)

    def joint_q(self, data):
        """Actuated joint positions (nu,) in actuator order."""
        return data.qpos[self.act_qadr]

    def joint_qd(self, data):
        """Actuated joint velocities (nu,) in actuator order."""
        return data.qvel[self.act_dofadr]


class StateEstimator:
    """Layer 1: reads MuJoCo state and produces the quantities the controller
    reasons about (CoM, CoM velocity, torso orientation, foot poses, contacts,
    support phase)."""

    DOUBLE = "double"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, robot: RobotModel, contact_force_thresh: float = 15.0):
        self.r = robot
        self.m = robot.model
        self.force_thresh = contact_force_thresh
        self._jacp = np.zeros((3, self.m.nv))

    def update(self, data: mujoco.MjData):
        m, r = self.m, self.r
        s = {}
        s["t"] = data.time

        # whole-body CoM (subtree of world body) and its velocity via Jacobian
        com = data.subtree_com[0].copy()
        mujoco.mj_jacSubtreeCom(m, data, self._jacp, 0)
        com_vel = self._jacp @ data.qvel
        s["com"] = com
        s["com_vel"] = com_vel

        # base / torso orientation
        base_quat = data.qpos[3:7].copy()
        s["base_rpy"] = quat_to_rpy(base_quat)
        s["base_angvel"] = data.qvel[3:6].copy()
        s["pelvis_pos"] = data.xpos[r.pelvis_bid].copy()

        # foot poses (ankle-roll body frames)
        s["lfoot_pos"] = data.xpos[r.lfoot_bid].copy()
        s["rfoot_pos"] = data.xpos[r.rfoot_bid].copy()
        s["lfoot_mat"] = data.xmat[r.lfoot_bid].reshape(3, 3).copy()
        s["rfoot_mat"] = data.xmat[r.rfoot_bid].reshape(3, 3).copy()

        # contact forces under each foot
        lf = self._foot_normal_force(data, r.lfoot_geoms)
        rf = self._foot_normal_force(data, r.rfoot_geoms)
        s["lfoot_force"] = lf
        s["rfoot_force"] = rf
        s["lfoot_contact"] = lf > self.force_thresh
        s["rfoot_contact"] = rf > self.force_thresh

        if s["lfoot_contact"] and s["rfoot_contact"]:
            s["support"] = self.DOUBLE
        elif s["lfoot_contact"]:
            s["support"] = self.LEFT
        elif s["rfoot_contact"]:
            s["support"] = self.RIGHT
        else:
            s["support"] = "flight"
        return s

    def _foot_normal_force(self, data, geom_set):
        total = 0.0
        forcetorque = np.zeros(6)
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 in geom_set or con.geom2 in geom_set:
                mujoco.mj_contactForce(self.m, data, c, forcetorque)
                total += abs(forcetorque[0])  # normal component (contact frame x)
        return total
