import math
import mujoco
import numpy as np
from scipy.linalg.lapack import dgesv

from src.estimators.state_estimator import StateEstimator
from src.controllers.joint_pd_controller import JointPDController
from src.controllers.locomotion.gait_scheduler.gait_scheduler import GaitScheduler
from src.controllers.locomotion.footstepplanner.footstepplanner import FootstepPlanner
from src.controllers.locomotion.trajectory.swingtrajectory import SwingFootTrajectoryGenerator
from src.locomotion.analytical_com_planner import AnalyticalComPlanner


class BipedLocomotionController:
    """
    Unified Hierarchical Whole-Body Locomotion Engine for the Unitree G1.

    Orchestrates the full planning pipeline and maps task-space references to
    joint tracking commands via a stacked Damped-Least-Squares differential IK.

    Architecture contract (matches JointPDController)
    -------------------------------------------------
    * __init__                 : the only place that may allocate NumPy arrays.
    * advance_control()        : zero heap allocations on the 500 Hz hot path.
    * _inverse_kinematics_layer() : zero heap allocations; all intermediates use
                                 pre-allocated buffers and NumPy ufuncs with out=.
    """

    def __init__(
        self,
        model:           mujoco.MjModel,
        data:            mujoco.MjData,
        state_estimator: StateEstimator,
        low_level_pd:    JointPDController,
        step_duration:   float = 0.4,
        nominal_width:   float = 0.2,
        damping_factor:  float = 1e-2,
        com_height:      float = 0.66,
    ) -> None:
        self.model         = model
        self.data          = data
        self.estimator     = state_estimator
        self.pd_controller = low_level_pd

        self.dt             = model.opt.timestep
        self.nv             = model.nv
        self._step_duration = step_duration
        self._lambda_sq     = damping_factor

        # ── Planning sub-modules ───────────────────────────────────────────────
        self.scheduler   = GaitScheduler(
            step_duration          = step_duration,
            double_support_fraction = 0.1,
        )
        self.planner   = FootstepPlanner(
            nominal_step_width = nominal_width,
            step_duration      = step_duration,
        )
        self.swing_gen  = SwingFootTrajectoryGenerator(
            swing_duration = step_duration * 0.9,
        )
        self.com_planner = AnalyticalComPlanner(com_height=com_height)

        # ── One-time initialisation flag ───────────────────────────────────────
        self._is_initialized = False

        # ── Swing-trajectory liftoff anchors ───────────────────────────────────
        # Latched once per step at the moment the foot leaves the ground.
        self._p_start_l = np.zeros(3, dtype=np.float64)
        self._p_start_r = np.zeros(3, dtype=np.float64)

        # ── Immutable stance world anchors ─────────────────────────────────────
        # Latched at touchdown; kept fixed for the entire stance phase so that
        # the IK maintains stiffness toward a world-frame target rather than
        # drifting with small measured slips.
        self._p_stance_l = np.zeros(3, dtype=np.float64)
        self._p_stance_r = np.zeros(3, dtype=np.float64)

        # ── Task-space foot reference buffers ──────────────────────────────────
        self.foot_target_pos_l = np.zeros(3, dtype=np.float64)
        self.foot_target_vel_l = np.zeros(3, dtype=np.float64)
        self.foot_target_pos_r = np.zeros(3, dtype=np.float64)
        self.foot_target_vel_r = np.zeros(3, dtype=np.float64)

        # ── Task-space tracking gains ──────────────────────────────────────────
        self._kp_task = 20.0   # position gain  (1/s)
        self._ko_task = 15.0   # orientation gain (1/s)

        # ── MuJoCo body ID cache (string lookups done once here) ───────────────
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._l_hip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
        self._r_hip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hip_pitch_link")
        self._l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self._r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

        # ── IK Jacobian scratch-pads ───────────────────────────────────────────
        self._J_com     = np.zeros((3, self.nv), dtype=np.float64)
        self._J_torso   = np.zeros((3, self.nv), dtype=np.float64)
        self._J_foot_l  = np.zeros((3, self.nv), dtype=np.float64)
        self._J_foot_r  = np.zeros((3, self.nv), dtype=np.float64)
        self._J_stacked = np.zeros((12, self.nv), dtype=np.float64)
        self._v_task    = np.zeros(12, dtype=np.float64)

        # ── Normal-equation buffers (overwritten in-place by LAPACK dgesv) ─────
        self._A            = np.zeros((self.nv, self.nv), dtype=np.float64)
        self._b            = np.zeros(self.nv, dtype=np.float64)
        self._A_flat       = self._A.ravel()                            # 1-D view of A
        self._diag_lin     = np.arange(self.nv, dtype=np.intp) * (self.nv + 1)

        # ── Task-velocity error scratch buffers (zero-allocation assembly) ─────
        self._err_com    = np.zeros(3, dtype=np.float64)
        self._err_foot_l = np.zeros(3, dtype=np.float64)
        self._err_foot_r = np.zeros(3, dtype=np.float64)

        # ── Joint-space output and integration scratch buffers ─────────────────
        self._q_target    = np.zeros(model.nu, dtype=np.float64)
        self._qdot_target = np.zeros(model.nu, dtype=np.float64)
        self._q_delta     = np.zeros(model.nu, dtype=np.float64)  # qdot * dt

    # ──────────────────────────────────────────────────────────────────────────

    def advance_control(self, v_des: np.ndarray) -> np.ndarray:
        """
        Runs the complete execution cycle once at 500 Hz.

        Zero heap allocations on the hot path: all intermediates use pre-allocated
        buffers; scalar temporaries (tau, cosh, sinh, …) are Python-native floats.
        """

        # ── Lazy one-time initialisation ──────────────────────────────────────
        if not self._is_initialized:
            np.copyto(self._p_stance_l, self.data.xpos[self._l_foot_id])
            np.copyto(self._p_stance_r, self.data.xpos[self._r_foot_id])
            np.copyto(self._p_start_l,  self.data.xpos[self._l_foot_id])
            np.copyto(self._p_start_r,  self.data.xpos[self._r_foot_id])
            # Seed LIPM from current measured state; ZMP = foot midpoint
            com   = self.estimator.get_com()
            mid_x = 0.5 * (self._p_stance_l[0] + self._p_stance_r[0])
            mid_y = 0.5 * (self._p_stance_l[1] + self._p_stance_r[1])
            self._err_com[0] = mid_x   # borrow as ZMP proxy; overwritten before IK
            self._err_com[1] = mid_y
            self.com_planner.latch_step(
                initial_com_xy     = com,
                initial_com_vel_xy = self.data.qvel,
                stance_foot_xy     = self._err_com,
            )
            self._is_initialized = True

        # ── Gait clock ────────────────────────────────────────────────────────
        self.scheduler.update(self.data.time)

        # ── Step-transition edge: latch anchors, plan footstep, seed LIPM ─────
        if self.scheduler.new_step_triggered:
            if self.scheduler.current_step_index % 2 == 0:
                # Left foot becomes stance; right foot will swing.
                np.copyto(self._p_stance_l, self.data.xpos[self._l_foot_id])
                np.copyto(self._p_start_r,  self.data.xpos[self._r_foot_id])
                self.planner.compute_next_footstep(
                    hip_pos_l    = self.data.xpos[self._l_hip_id],
                    hip_pos_r    = self.data.xpos[self._r_hip_id],
                    body_vel     = self.data.qvel,
                    body_vel_des = v_des,
                )
                self.com_planner.latch_step(
                    initial_com_xy     = self.estimator.get_com(),
                    initial_com_vel_xy = self.data.qvel,
                    stance_foot_xy     = self._p_stance_l,
                )
            else:
                # Right foot becomes stance; left foot will swing.
                np.copyto(self._p_stance_r, self.data.xpos[self._r_foot_id])
                np.copyto(self._p_start_l,  self.data.xpos[self._l_foot_id])
                self.planner.compute_next_footstep(
                    hip_pos_l    = self.data.xpos[self._l_hip_id],
                    hip_pos_r    = self.data.xpos[self._r_hip_id],
                    body_vel     = self.data.qvel,
                    body_vel_des = v_des,
                )
                self.com_planner.latch_step(
                    initial_com_xy     = self.estimator.get_com(),
                    initial_com_vel_xy = self.data.qvel,
                    stance_foot_xy     = self._p_stance_r,
                )

        # ── LIPM reference update ──────────────────────────────────────────────
        # τ = time elapsed within the current step [0, T_step)
        tau = math.fmod(self.data.time, self._step_duration)
        self.com_planner.update(tau)

        # ── Foot reference resolution ──────────────────────────────────────────
        if self.scheduler.contact_states[0] == 0:       # Left foot: SWING
            self.swing_gen.compute_trajectory(
                phi      = self.scheduler.swing_phases[0],
                p_start  = self._p_start_l,
                p_target = self.planner.left_foot_target,
                out_pos  = self.foot_target_pos_l,
                out_vel  = self.foot_target_vel_l,
            )
        else:                                            # Left foot: STANCE — immutable anchor
            np.copyto(self.foot_target_pos_l, self._p_stance_l)
            self.foot_target_vel_l.fill(0.0)

        if self.scheduler.contact_states[1] == 0:       # Right foot: SWING
            self.swing_gen.compute_trajectory(
                phi      = self.scheduler.swing_phases[1],
                p_start  = self._p_start_r,
                p_target = self.planner.right_foot_target,
                out_pos  = self.foot_target_pos_r,
                out_vel  = self.foot_target_vel_r,
            )
        else:                                            # Right foot: STANCE — immutable anchor
            np.copyto(self.foot_target_pos_r, self._p_stance_r)
            self.foot_target_vel_r.fill(0.0)

        # ── Differential IK → joint references ────────────────────────────────
        self._inverse_kinematics_layer()

        return self.pd_controller.compute_torques(self._q_target, self._qdot_target)

    # ──────────────────────────────────────────────────────────────────────────

    def _inverse_kinematics_layer(self) -> None:
        """
        Stacked Damped-Least-Squares operational-space IK.

        Task stack (rows 0-11 of J_stacked and v_task):
            [0:3]  CoM position  — tracks analytical LIPM reference
            [3:6]  Torso angular — SO(3) skew-symmetric error → avoids
                                   positive-feedback pitch bug
            [6:9]  Left foot position
            [9:12] Right foot position

        All arithmetic uses pre-allocated buffers and NumPy ufuncs with out=
        parameters. Zero heap allocations.
        """

        # ── A. Jacobian evaluations ────────────────────────────────────────────
        mujoco.mj_jacSubtreeCom(
            self.model, self.data, self._J_com, self._pelvis_id,
        )
        mujoco.mj_jac(
            self.model, self.data,
            self._J_foot_l, None,
            self.data.xpos[self._l_foot_id], self._l_foot_id,
        )
        mujoco.mj_jac(
            self.model, self.data,
            self._J_foot_r, None,
            self.data.xpos[self._r_foot_id], self._r_foot_id,
        )
        mujoco.mj_jac(
            self.model, self.data,
            None, self._J_torso,
            self.data.xpos[self._pelvis_id], self._pelvis_id,
        )

        # ── B. Stack Jacobians ─────────────────────────────────────────────────
        self._J_stacked[0:3,  :] = self._J_com
        self._J_stacked[3:6,  :] = self._J_torso
        self._J_stacked[6:9,  :] = self._J_foot_l
        self._J_stacked[9:12, :] = self._J_foot_r

        # ── C. Task-space velocity commands ───────────────────────────────────
        # C1. CoM tracking:  v = v_ref + Kp·(p_ref − p_meas)
        np.subtract(self.com_planner.com_pos_ref, self.estimator.get_com(),
                    out=self._err_com)
        np.multiply(self._err_com, self._kp_task, out=self._err_com)
        np.add(self.com_planner.com_vel_ref, self._err_com, out=self._err_com)
        self._v_task[0] = self._err_com[0]
        self._v_task[1] = self._err_com[1]
        self._v_task[2] = self._err_com[2]

        # C2. Torso orientation:  SO(3) skew-symmetric error relative to identity R_des = I
        #
        #   e_so3 = 0.5 · skew_vex(R − Rᵀ)
        #         = 0.5 · [R₂₁−R₁₂,  R₀₂−R₂₀,  R₁₀−R₀₁]
        #
        # Row-major xmat layout:  r0=R₀₀  r1=R₀₁  r2=R₀₂
        #                          r3=R₁₀  r4=R₁₁  r5=R₁₂
        #                          r6=R₂₀  r7=R₂₁  r8=R₂₂
        #
        # Using the skew-vex elements avoids the sign error present in the
        # earlier scalar xmat[6]/xmat[7] approximation.
        r1 = self.data.xmat[self._pelvis_id, 1]   # R₀₁
        r2 = self.data.xmat[self._pelvis_id, 2]   # R₀₂
        r3 = self.data.xmat[self._pelvis_id, 3]   # R₁₀
        r5 = self.data.xmat[self._pelvis_id, 5]   # R₁₂
        r6 = self.data.xmat[self._pelvis_id, 6]   # R₂₀
        r7 = self.data.xmat[self._pelvis_id, 7]   # R₂₁
        self._v_task[3] = -0.5 * self._ko_task * (r7 - r5)   # ω_x  (roll)
        self._v_task[4] = -0.5 * self._ko_task * (r2 - r6)   # ω_y  (pitch)
        self._v_task[5] = -0.5 * self._ko_task * (r3 - r1)   # ω_z  (yaw)

        # C3. Left foot:  v = v_ref + Kp·(p_ref − p_meas)
        np.subtract(self.foot_target_pos_l, self.data.xpos[self._l_foot_id],
                    out=self._err_foot_l)
        np.multiply(self._err_foot_l, self._kp_task, out=self._err_foot_l)
        np.add(self.foot_target_vel_l, self._err_foot_l, out=self._err_foot_l)
        self._v_task[6]  = self._err_foot_l[0]
        self._v_task[7]  = self._err_foot_l[1]
        self._v_task[8]  = self._err_foot_l[2]

        # C4. Right foot:  v = v_ref + Kp·(p_ref − p_meas)
        np.subtract(self.foot_target_pos_r, self.data.xpos[self._r_foot_id],
                    out=self._err_foot_r)
        np.multiply(self._err_foot_r, self._kp_task, out=self._err_foot_r)
        np.add(self.foot_target_vel_r, self._err_foot_r, out=self._err_foot_r)
        self._v_task[9]  = self._err_foot_r[0]
        self._v_task[10] = self._err_foot_r[1]
        self._v_task[11] = self._err_foot_r[2]

        # ── D. Normal equations:  A = Jᵀ·J + λI,   b = Jᵀ·v ─────────────────
        np.matmul(self._J_stacked.T, self._J_stacked, out=self._A)
        np.matmul(self._J_stacked.T, self._v_task,    out=self._b)
        # Add Tikhonov diagonal (zero-allocation via flat view + linear indices)
        np.add.at(self._A_flat, self._diag_lin, self._lambda_sq)

        # ── E. In-place LAPACK solve  (no Python-side allocation) ─────────────
        # dgesv overwrites self._A with the LU factorisation and self._b with x.
        _, _, v_sol, info = dgesv(
            self._A, self._b, overwrite_a=True, overwrite_b=True,
        )

        # ── F. Joint-space integration:  q_ref = q + qdot·dt ─────────────────
        if info == 0:
            np.take(v_sol,          self.pd_controller._dof_idx,  out=self._qdot_target)
            np.take(self.data.qpos, self.pd_controller._qpos_idx, out=self._q_target)
            np.multiply(self._qdot_target, self.dt, out=self._q_delta)
            np.add(self._q_target, self._q_delta,   out=self._q_target)
        else:
            # Degenerate Jacobian — freeze at current position so the PD layer
            # applies gravity compensation with zero tracking error rather than
            # driving the robot toward the all-zero default pose.
            np.take(self.data.qpos, self.pd_controller._qpos_idx, out=self._q_target)
            self._qdot_target.fill(0.0)
