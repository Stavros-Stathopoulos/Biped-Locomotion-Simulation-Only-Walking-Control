"""
unitree_g1.py — Offline Whole-Body Walking Controller for the Unitree G1.

Adapted from the reference Talos implementation (test/talos.py) to the G1's
29-DOF kinematic tree with a 6-DOF floating base.

Strategy
--------
1.  Pre-compute the entire walk as a sequence of full-body joint
    configurations using a QP-based iterative inverse kinematics solver.
2.  Replay the trajectory either kinematically (visualize_traj) or with
    full physics via PD position tracking (position_control / simulate).

The IK formulation solves at each waypoint:

    min_v   ‖W·(J·v − e)‖²  +  λ·‖v‖²
    s.t.    q_min − q  ≤  step·v  ≤  q_max − q

where  v ∈ ℝ^nv  is the generalised velocity update, J is the stacked
spatial Jacobian (left foot + right foot + CoM), e the task-space error,
and W a diagonal weight matrix.  proxsuite solves this dense QP.

Body name mapping (G1)
----------------------
    Left foot  → left_ankle_roll_link
    Right foot → right_ankle_roll_link
    Torso      → pelvis
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import proxsuite

from src.utils import math_utils, mujoco_utils


# ── Verified crouch angles (from verify_stance.py) ───────────────────────────
_HP = -0.50   # hip pitch (rad)
_KN =  1.00   # knee flexion (rad)
_AP = -0.50   # ankle pitch (rad)

Q_CROUCH = np.array([
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,   # left leg
    _HP,   0.0,   0.0,  _KN,  _AP,   0.0,   # right leg
     0.0,  0.0,   0.3,                       # waist
     0.30,  0.30,  0.0,  0.50,  0.0, 0.0, 0.0,  # left arm
     0.30, -0.30,  0.0,  0.50,  0.0, 0.0, 0.0,  # right arm
], dtype=np.float64)

# Foot sphere geometry target Z (sphere radius ≈ 0.005 m → centre at 0.006 m)
_FOOT_SPHERE_TARGET_Z = 0.006


class UnitreeG1:
    """Self-contained walking controller for the Unitree G1 biped."""

    def __init__(self, scene_xml_path: str = None):
        # ── Resolve model path ────────────────────────────────────────────────
        if scene_xml_path is None:
            scene_xml_path = os.path.normpath(os.path.join(
                os.path.dirname(__file__), "../../assets/unitree_g1/scene.xml"
            ))

        if not os.path.exists(scene_xml_path):
            raise FileNotFoundError(f"Scene XML not found: {scene_xml_path}")

        # MuJoCo's C loader needs CWD next to the XML for <include> resolution
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(scene_xml_path))
        try:
            self.model = mujoco.MjModel.from_xml_path(os.path.basename(scene_xml_path))
        finally:
            os.chdir(original_cwd)

        self.data = mujoco.MjData(self.model)

        # Disable shadow rendering for performance
        self.model.vis.quality.shadowsize = 0

        # ── Body IDs ──────────────────────────────────────────────────────────
        self.left_foot_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.torso_id      = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        # ── Foot-sphere geom indices (radius ≈ 0.005 m) ──────────────────────
        self._foot_sphere_idx = [
            i for i in range(self.model.ngeom)
            if abs(self.model.geom_size[i, 0] - 0.005) < 1e-6
        ]

        # ── PD gains for position_control / simulate (per-actuator) ───────────
        self._kp = np.array([
            150.0, 100.0,  80.0, 200.0, 150.0,  60.0,   # left leg
            150.0, 100.0,  80.0, 200.0, 150.0,  60.0,   # right leg
            100.0,  80.0, 100.0,                         # waist
             50.0,  50.0,  30.0,  50.0,  25.0,  12.0, 12.0,  # left arm
             50.0,  50.0,  30.0,  50.0,  25.0,  12.0, 12.0,  # right arm
        ], dtype=np.float64)
        self._kd = np.array([
              8.0,  5.0,  4.0, 10.0,  8.0,  3.0,
              8.0,  5.0,  4.0, 10.0,  8.0,  3.0,
              5.0,  4.0,  5.0,
              2.0,  2.0,  1.5,  2.0,  1.0,  0.5, 0.5,
              2.0,  2.0,  1.5,  2.0,  1.0,  0.5, 0.5,
        ], dtype=np.float64)

        # ── Pre-compute actuator → qpos / dof index maps ─────────────────────
        joint_ids = self.model.actuator_trnid[:, 0]
        self._qpos_idx = self.model.jnt_qposadr[joint_ids].copy()
        self._dof_idx  = self.model.jnt_dofadr[joint_ids].copy()

        # ── Set the robot to the walk pose ────────────────────────────────────
        self._reset_to_walk_pose(self.data)

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_to_walk_pose(self, data):
        """
        Reset *data* to the verified crouch configuration and fine-tune
        pelvis height / X so that foot spheres touch the ground and the
        CoM is centred above the support polygon.
        """
        # Try to use the keyframe if available
        walk_pose_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "walk_pose")
        if walk_pose_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, data, walk_pose_id)
        else:
            mujoco.mj_resetData(self.model, data)
            data.qpos[7:36] = Q_CROUCH
        mujoco.mj_forward(self.model, data)

        # Fine-tune pelvis Z so lowest foot sphere just touches the ground
        if self._foot_sphere_idx:
            min_z = min(data.geom_xpos[i, 2] for i in self._foot_sphere_idx)
            data.qpos[2] += _FOOT_SPHERE_TARGET_Z - min_z
            mujoco.mj_forward(self.model, data)

            # Centre CoM X above foot contact centroid
            com_x  = data.subtree_com[self.torso_id, 0]
            foot_x = float(np.mean([data.geom_xpos[i, 0]
                                    for i in self._foot_sphere_idx]))
            data.qpos[0] += foot_x - com_x
            mujoco.mj_forward(self.model, data)

    # ──────────────────────────────────────────────────────────────────────────
    # QP-based Inverse Kinematics
    # ──────────────────────────────────────────────────────────────────────────

    def inverse_kinematics(self, data, feet_targets, feet_weights,
                           com_target, com_weight,
                           step=0.001, max_iters=5000):
        """
        Iterative whole-body IK via dense QP (proxsuite).

        Parameters
        ----------
        data          : MjData instance (modified in-place; qpos is updated).
        feet_targets  : [T_left_4x4, T_right_4x4] desired foot SE(3) poses.
        feet_weights  : [w_left(6,), w_right(6,)] per-axis task weights.
        com_target    : (3,) desired CoM world position.
        com_weight    : (3,) CoM task weights.
        step          : Integration step size for the configuration update.
        max_iters     : Maximum solver iterations.

        Returns
        -------
        solution : (nq,) converged joint configuration.
        """
        qp_dim  = self.model.nv
        qp_eq   = 0
        qp_ineq = self.model.nv
        qp = proxsuite.proxqp.dense.QP(qp_dim, qp_eq, qp_ineq)

        qk = data.qpos.copy()
        solution = qk.copy()
        feet_ids = [self.left_foot_id, self.right_foot_id]

        for it in range(max_iters + 1):
            # ── Foot errors and Jacobians (SE(3) log) ─────────────────────────
            errors, Jacobians = [], []
            for foot_id, foot_target in zip(feet_ids, feet_targets):
                T_wb     = mujoco_utils.get_body_full_transformation(self.model, data, foot_id)
                T_wb_inv = math_utils.invert_transformation(T_wb)
                foot_err = math_utils.log_transformation(T_wb_inv @ foot_target)
                errors.append(foot_err)
                Jacobians.append(
                    mujoco_utils.get_body_jac_local_frame(self.model, data, foot_id))

            # ── CoM error and Jacobian ────────────────────────────────────────
            com = data.subtree_com[0]
            errors.append((com_target - com).reshape(3, 1))

            J_com = np.zeros((3, self.model.nv))
            mujoco.mj_jacSubtreeCom(self.model, data, J_com, 0)
            Jacobians.append(J_com)

            # ── Stack ─────────────────────────────────────────────────────────
            error = np.vstack(errors)       # (15, 1)
            J     = np.vstack(Jacobians)    # (15, nv)

            # ── Convergence check ─────────────────────────────────────────────
            if np.all(np.abs(error) < 1e-5) or it == max_iters:
                break

            # ── Weighted QP ───────────────────────────────────────────────────
            weights_vec = np.concatenate([feet_weights[0], feet_weights[1], com_weight])
            W = np.diag(weights_vec) @ J
            error_w = np.diag(weights_vec) @ error

            Q = W.T @ W + 1e-4 * np.eye(qp_dim)
            q = -(W.T @ error_w).flatten()

            # Joint-limit inequality constraints
            C = step * np.eye(qp_dim)
            d_min, d_max = mujoco_utils.get_current_joint_range(self.model, qk)

            # Solve
            if it == 0:
                qp.init(Q, q, None, None, C, d_min, d_max)
            else:
                qp.update(Q, q, None, None, C, d_min, d_max)
            qp.solve()

            v = np.copy(qp.results.x)
            mujoco.mj_integratePos(self.model, qk, v, step)
            solution = qk.copy()
            data.qpos[:] = qk
            mujoco.mj_forward(self.model, data)

        return solution

    # ──────────────────────────────────────────────────────────────────────────
    # Walking primitives
    # ──────────────────────────────────────────────────────────────────────────

    def move_leg(self, data, dx, dy, swing_phase, Ryaw, move_left=True,
                 arc_height=0.05, arc_points=5):
        """
        Swing one leg along a cubic-spline foot arc while keeping the
        other foot and the CoM locked.

        Phase 1 — CoM pre-shift: both feet stay planted while the CoM
        moves over the stance foot.  Uses high task weights (feet=100,
        CoM=200) so the shift converges fully before any foot lifts.

        Phase 2 — Foot arc: the swing foot follows the cubic-spline
        arc while the stance foot and CoM stay locked.  CoM weight is
        kept at 15 (up from 5) to maintain balance during swing.
        """
        left_foot_pos  = data.xpos[self.left_foot_id].copy()
        right_foot_pos = data.xpos[self.right_foot_id].copy()
        com_target     = data.subtree_com[0].copy()

        if move_left:
            # Stance = right foot; shift CoM over it
            com_target[0] = right_foot_pos[0]
            com_target[1] = right_foot_pos[1]
            foot_target = left_foot_pos.copy()
            foot_target[0] += dx
            foot_target[1] += dy
            points = math_utils.generate_foot_arc(
                left_foot_pos, foot_target, arc_height, arc_points)
        else:
            # Stance = left foot; shift CoM over it
            com_target[0] = left_foot_pos[0]
            com_target[1] = left_foot_pos[1]
            foot_target = right_foot_pos.copy()
            foot_target[0] += dx
            foot_target[1] += dy
            points = math_utils.generate_foot_arc(
                right_foot_pos, foot_target, arc_height, arc_points)

        traj = []

        # ── Phase 1: CoM pre-shift (both feet locked, high weights) ───────
        # Without this, the G1's narrow stance (≈0.13 m) causes the CoM to
        # remain between the feet when the swing foot lifts → immediate fall.
        T_left_lock  = mujoco_utils.transformation(
            left_foot_pos,
            data.xmat[self.left_foot_id].copy().reshape(3, 3))
        T_right_lock = mujoco_utils.transformation(
            right_foot_pos,
            data.xmat[self.right_foot_id].copy().reshape(3, 3))

        q_start = data.qpos.copy()
        sol = self.inverse_kinematics(
            data, [T_left_lock, T_right_lock],
            [np.full(6, 100.0), np.full(6, 100.0)],   # high foot lock
            com_target, np.full(3, 200.0),              # high CoM pull
        )
        q_end = sol.copy()
        n_shift = max(1, int(swing_phase / 0.003))
        traj.extend(math_utils.interpolate_traj(q_start, q_end, n_shift))

        # ── Phase 2: foot arc (swing foot moves, stance + CoM locked) ─────
        for point in points:
            if move_left:
                T_left  = mujoco_utils.transformation(
                    point, Ryaw.copy() @ data.xmat[self.left_foot_id].copy().reshape(3, 3))
                T_right = mujoco_utils.transformation(
                    right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))
            else:
                T_left  = mujoco_utils.transformation(
                    left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
                T_right = mujoco_utils.transformation(
                    point, Ryaw.copy() @ data.xmat[self.right_foot_id].copy().reshape(3, 3))

            feet_targets = [T_left, T_right]
            w_feet = [np.full(6, 20.0), np.full(6, 20.0)]
            w_com  = np.full(3, 15.0)   # up from 5.0 — hold CoM during swing

            q_start = data.qpos.copy()
            sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)
            q_end = sol.copy()

            n_interp = int(swing_phase / 0.005)
            traj.extend(math_utils.interpolate_traj(q_start, q_end, n_interp))

        return traj

    def shift_com(self, data, x, y, shift_phase):
        """
        Smoothly shift the CoM to (x, y) while keeping both feet locked.

        Uses high task weights on the feet (100) and CoM (200) to ensure
        the body moves without slipping the feet.
        """
        left_foot_pos  = data.xpos[self.left_foot_id].copy()
        right_foot_pos = data.xpos[self.right_foot_id].copy()
        com_target     = data.subtree_com[0].copy()

        # Partial shift (50 %) for smoother transitions
        com_target[0] = com_target[0] + 0.5 * (x - com_target[0])
        com_target[1] = com_target[1] + 0.5 * (y - com_target[1])

        T_left  = mujoco_utils.transformation(
            left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
        T_right = mujoco_utils.transformation(
            right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))

        feet_targets = [T_left, T_right]
        w_feet = [np.full(6, 100.0), np.full(6, 100.0)]
        w_com  = np.full(3, 200.0)

        q_start = data.qpos.copy()
        sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

        return math_utils.interpolate_traj(q_start, sol, int(shift_phase / 0.001))

    # ──────────────────────────────────────────────────────────────────────────
    # High-level gaits
    # ──────────────────────────────────────────────────────────────────────────

    def march(self, n_steps=5, travel_distance=0.5, time_step=1.0, theta=0.0,
              arc_height=0.05):
        """
        Simple alternating-leg march (no explicit CoM shift phase).

        Parameters
        ----------
        n_steps         : Number of full left+right step cycles.
        travel_distance : Total forward distance (m) over all steps.
        time_step       : Controls interpolation density (s per arc waypoint).
        theta           : Heading angle (rad); 0 = forward.
        arc_height      : Foot lift height (m) during swing.
        """
        data_sim = mujoco.MjData(self.model)
        self._reset_to_walk_pose(data_sim)

        walk_trajectory = []
        Ryaw_mat = math_utils.Ryaw(theta / (n_steps * 5.0))

        dx = (travel_distance * np.cos(theta)) / n_steps
        dy = (travel_distance * np.sin(theta)) / n_steps

        for i in range(n_steps):
            print(f"  march step {i + 1}/{n_steps} ...")
            # Swing right leg
            walk_trajectory.extend(
                self.move_leg(data_sim, dx, dy, time_step, Ryaw_mat,
                              move_left=False, arc_height=arc_height))
            # Swing left leg
            walk_trajectory.extend(
                self.move_leg(data_sim, dx, dy, time_step, Ryaw_mat,
                              move_left=True, arc_height=arc_height))

        return walk_trajectory

    def walk(self, n_steps=5, step_length=0.5,
             left_swing_time=1.0, right_swing_time=1.0,
             shift_time=0.5, theta=0.0, arc_height=0.05):
        """
        Full walk with explicit CoM shift before each swing phase.

        This is the more stable variant: the CoM is moved over the stance
        foot before the swing leg lifts, ensuring quasi-static balance.
        """
        data_sim = mujoco.MjData(self.model)
        self._reset_to_walk_pose(data_sim)

        walk_trajectory = []
        Ryaw_mat = math_utils.Ryaw(theta / n_steps)

        dx = (step_length * np.cos(theta)) / n_steps
        dy = (step_length * np.sin(theta)) / n_steps

        for i in range(n_steps):
            print(f"  walk step {i + 1}/{n_steps} ...")
            left_foot_pos = data_sim.xpos[self.left_foot_id].copy()

            # 1. Shift CoM over left foot → prepare to swing right leg
            walk_trajectory.extend(
                self.shift_com(data_sim, left_foot_pos[0], left_foot_pos[1],
                               shift_time))
            # 2. Swing right leg
            walk_trajectory.extend(
                self.move_leg(data_sim, dx, dy, right_swing_time, Ryaw_mat,
                              move_left=False, arc_height=arc_height))

            right_foot_pos = data_sim.xpos[self.right_foot_id].copy()

            # 3. Shift CoM over right foot → prepare to swing left leg
            walk_trajectory.extend(
                self.shift_com(data_sim, right_foot_pos[0], right_foot_pos[1],
                               shift_time))
            # 4. Swing left leg
            walk_trajectory.extend(
                self.move_leg(data_sim, dx, dy, left_swing_time, Ryaw_mat,
                              move_left=True, arc_height=arc_height))

        return walk_trajectory

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory replay
    # ──────────────────────────────────────────────────────────────────────────

    def visualize_traj(self, traj):
        """
        Kinematic replay — sets qpos directly, no physics simulation.

        Use this to visually verify the IK trajectory before attempting
        physics-based playback.
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as vis:
            vis.cam.lookat[:] = [0, 0, 0.7]
            vis.cam.distance  = 3.0
            vis.cam.azimuth   = 180
            vis.cam.elevation = -20

            while vis.is_running():
                for q in traj:
                    self.data.qpos[:] = q
                    mujoco.mj_forward(self.model, self.data)
                    vis.sync()
                # Hold the final pose
                mujoco.mj_forward(self.model, self.data)
                vis.sync()

    def position_control(self, traj):
        """
        Physics-based replay using PD + gravity compensation torque control.

        Each trajectory waypoint is tracked via:
            τ = τ_grav + Kp·(q_des − q) − Kd·q̇
        where τ_grav = qfrc_bias (gravity + Coriolis compensation).
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as vis:
            vis.cam.lookat[:] = [0, 0, 0.7]
            vis.cam.distance  = 3.0
            vis.cam.azimuth   = 180
            vis.cam.elevation = -20

            while vis.is_running():
                for q_des in traj:
                    self._apply_pd_control(q_des)
                    mujoco.mj_step(self.model, self.data)
                    vis.sync()
                mujoco.mj_step(self.model, self.data)
                vis.sync()

    def simulate(self, traj):
        """
        Run trajectory with physics and compute a reward score.

        Reward terms:
            +1 per step survived
            −|pitch| per step (penalises torso tilt)
            −1 per foot-foot self-collision
            +10 × distance travelled (at the end)

        Returns (reward, final_xy_position).
        """
        reward = 0.0
        R_before = self.data.xmat[self.torso_id].reshape(3, 3).copy()
        init_pos = self.data.xpos[self.torso_id][:2].copy()

        left_foot_geoms  = np.where(
            self.model.geom_bodyid == self.left_foot_id)[0].tolist()
        right_foot_geoms = np.where(
            self.model.geom_bodyid == self.right_foot_id)[0].tolist()

        for q_des in traj:
            self._apply_pd_control(q_des)
            mujoco.mj_step(self.model, self.data)

            # Fall detection
            torso_pos = self.data.xpos[self.torso_id].copy()
            if torso_pos[2] < 0.4:
                break

            reward += 1.0

            # Pitch penalty
            R_after = self.data.xmat[self.torso_id].reshape(3, 3).copy()
            R_rel = R_before.T @ R_after
            _, pitch, _ = R.from_matrix(R_rel).as_euler('xyz')
            reward -= abs(pitch) * 1.0
            R_before = R_after.copy()

            # Self-collision penalty
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if ((c.geom[0] in left_foot_geoms and c.geom[1] in right_foot_geoms) or
                    (c.geom[1] in left_foot_geoms and c.geom[0] in right_foot_geoms)):
                    reward -= 1.0

        final_pos = self.data.xpos[self.torso_id][:2].copy()
        reward += np.linalg.norm(final_pos - init_pos) * 10.0

        return reward, final_pos.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal PD control
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_pd_control(self, q_des):
        """
        PD + gravity compensation torque control for one physics step.

        Maps from a full qpos target to actuator torques:
            τ_i = qfrc_bias[dof_i] + Kp_i·(q_des − q) − Kd_i·q̇
        """
        for i in range(self.model.nu):
            qpos_addr = self._qpos_idx[i]
            dof_addr  = self._dof_idx[i]

            pos_err = q_des[qpos_addr] - self.data.qpos[qpos_addr]
            vel     = self.data.qvel[dof_addr]
            grav    = self.data.qfrc_bias[dof_addr]

            self.data.ctrl[i] = grav + self._kp[i] * pos_err - self._kd[i] * vel
