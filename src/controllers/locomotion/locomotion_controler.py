import math
import mujoco
import numpy as np
from scipy.linalg.lapack import dgesv

from src.estimators.state_estimator import StateEstimator
from src.controllers.joint_pd_controller import JointPDController
from src.controllers.locomotion.gait_scheduler.gait_scheduler import GaitScheduler
from src.controllers.locomotion.footstepplanner.footstepplanner import FootstepPlanner
from src.controllers.locomotion.trajectory.swingtrajectory import SwingFootTrajectoryGenerator
from src.controllers.locomotion.comtrajectory.dcmTrajectoryPlanner import DcmTrajectoryPlanner

class BipedLocomotionController:
    """
    Unified Hierarchical Whole-Body Locomotion Engine for the Unitree G1.
    
    Orchestrates the entire planning pipeline and maps task-space curves down 
    to tracking references via a high-performance stacked Differential IK layer.
    """
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state_estimator: StateEstimator,
        low_level_pd: JointPDController,
        step_duration: float = 0.4,
        nominal_width: float = 0.2,
        damping_factor: float = 1e-2
    ) -> None:
        self.model = model
        self.data = data
        self.estimator = state_estimator
        self.pd_controller = low_level_pd

        self.dt = model.opt.timestep
        self.nv = model.nv
        self._step_duration = step_duration
        self._lambda_sq = damping_factor
        
        # ── Core Sub-Modules ────────────────────────────────────────────────
        self.scheduler = GaitScheduler(step_duration=step_duration, double_support_fraction=0.1)
        self.planner = FootstepPlanner(nominal_step_width=nominal_width, step_duration=step_duration)
        self.swing_gen = SwingFootTrajectoryGenerator(swing_duration=step_duration * 0.9)
        self.com_planner = DcmTrajectoryPlanner(dt=self.dt)
        
        # ── State Initialization Flag ───────────────────────────────────────
        self._is_initialized = False
        
        # ── Pre-allocated Task Space Operational Buffers ────────────────────
        self._p_start_l = np.zeros(3, dtype=np.float64)
        self._p_start_r = np.zeros(3, dtype=np.float64)
        
        self.foot_target_pos_l = np.zeros(3, dtype=np.float64)
        self.foot_target_vel_l = np.zeros(3, dtype=np.float64)
        self.foot_target_pos_r = np.zeros(3, dtype=np.float64)
        self.foot_target_vel_r = np.zeros(3, dtype=np.float64)
        
        # Tracking Feedback Gains
        self._kp_task = 20.0
        self._ko_task = 15.0
        
        # ── MuJoCo Frame Mapping Constants ──────────────────────────────────
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._l_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
        self._r_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hip_pitch_link")
        self._l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self._r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        
        # ── Mathematical Scratch-Pads (Zero allocation on hot-path) ─────────
        self._J_com = np.zeros((3, self.nv), dtype=np.float64)
        self._J_torso = np.zeros((3, self.nv), dtype=np.float64)
        self._J_foot_l = np.zeros((3, self.nv), dtype=np.float64)
        self._J_foot_r = np.zeros((3, self.nv), dtype=np.float64)
        
        self._J_stacked = np.zeros((12, self.nv), dtype=np.float64)
        self._v_task = np.zeros(12, dtype=np.float64)
        
        self._A = np.zeros((self.nv, self.nv), dtype=np.float64)
        self._b = np.zeros(self.nv, dtype=np.float64)
        self._diag_indices = np.diag_indices(self.nv)
        
        self._q_target = np.zeros(model.nu, dtype=np.float64)
        self._qdot_target = np.zeros(model.nu, dtype=np.float64)

    def advance_control(self, v_des: np.ndarray) -> np.ndarray:
        """Runs the complete execution cycle at 500 Hz entirely in memory."""
        if not self._is_initialized:
            self.com_planner.initialize_states(self.estimator.get_com())
            self._is_initialized = True
            
        self.scheduler.update(self.data.time)
        body_vel = self.data.qvel[0:2]
        
        # 1. Edge-Trigger Latching Execution Loop
        if self.scheduler.new_step_triggered:
            if self.scheduler.current_step_index % 2 == 0:
                # Left stance, right swings
                np.copyto(self._p_start_r, self.data.xpos[self._r_foot_id])
                self.planner.compute_next_footstep(
                    hip_pos_l=self.data.xpos[self._l_hip_id],
                    hip_pos_r=self.data.xpos[self._r_hip_id],
                    body_vel=body_vel,
                    body_vel_des=v_des
                )
                self.com_planner.reset_for_step(
                    stance_foot=self.data.xpos[self._l_foot_id],
                    swing_foot_target=self.planner.right_foot_target,
                    T_step=self._step_duration
                )
            else:
                # Right stance, left swings
                np.copyto(self._p_start_l, self.data.xpos[self._l_foot_id])
                self.planner.compute_next_footstep(
                    hip_pos_l=self.data.xpos[self._l_hip_id],
                    hip_pos_r=self.data.xpos[self._r_hip_id],
                    body_vel=body_vel,
                    body_vel_des=v_des
                )
                self.com_planner.reset_for_step(
                    stance_foot=self.data.xpos[self._r_foot_id],
                    swing_foot_target=self.planner.left_foot_target,
                    T_step=self._step_duration
                )

        # 2. Advance CoM / ZMP Linear Inverted Pendulum dynamics
        self.com_planner.update_trajectory(
            contact_states=self.scheduler.contact_states,
            left_foot_pos=self.data.xpos[self._l_foot_id],
            right_foot_pos=self.data.xpos[self._r_foot_id],
            next_footstep_target=self.planner.left_foot_target if (self.scheduler.current_step_index % 2 == 0) else self.planner.right_foot_target
        )

        # 3. Resolve Operational Space Foot Paths
        if self.scheduler.contact_states[0] == 0:  # Left Foot Swing
            self.swing_gen.compute_trajectory(
                phi=self.scheduler.swing_phases[0], p_start=self._p_start_l,
                p_target=self.planner.left_foot_target, out_pos=self.foot_target_pos_l, out_vel=self.foot_target_vel_l
            )
        else:
            np.copyto(self.foot_target_pos_l, self.data.xpos[self._l_foot_id])
            self.foot_target_vel_l.fill(0.0)

        if self.scheduler.contact_states[1] == 0:  # Right Foot Swing
            self.swing_gen.compute_trajectory(
                phi=self.scheduler.swing_phases[1], p_start=self._p_start_r,
                p_target=self.planner.right_foot_target, out_pos=self.foot_target_pos_r, out_vel=self.foot_target_vel_r
            )
        else:
            np.copyto(self.foot_target_pos_r, self.data.xpos[self._r_foot_id])
            self.foot_target_vel_r.fill(0.0)

        # 4. Stack Mathematical Transformations & Execute Inverse Kinematics Mapping
        self._inverse_kinematics_layer()

        return self.pd_controller.compute_torques(self._q_target, self._qdot_target)

    def _inverse_kinematics_layer(self) -> None:
        """
        Executes stacked Regularized Levenberg-Marquardt operational space projections 
        using memory-contig raw C-LAPACK bindings.
        """
        # A. Analytical Jacobian Evaluation Frame Extractions
        mujoco.mj_jacSubtreeCom(self.model, self.data, self._J_com, self._pelvis_id)
        mujoco.mj_jac(self.model, self.data, self._J_foot_l, None, self.data.xpos[self._l_foot_id], self._l_foot_id)
        mujoco.mj_jac(self.model, self.data, self._J_foot_r, None, self.data.xpos[self._r_foot_id], self._r_foot_id)
        mujoco.mj_jac(self.model, self.data, None, self._J_torso, self.data.xpos[self._pelvis_id], self._pelvis_id)
        
        # B. Construct Stacked Operational Jacobian Layout
        self._J_stacked[0:3, :] = self._J_com
        self._J_stacked[3:6, :] = self._J_torso
        self._J_stacked[6:9, :] = self._J_foot_l
        self._J_stacked[9:12, :] = self._J_foot_r
        
        # C. Resolve Task-Space Operational Velocities with Closed-Loop Proportional Correction
        # CoM Tracking Task
        self._v_task[0:3] = self.com_planner.com_vel_ref + self._kp_task * (self.com_planner.com_pos_ref - self.estimator.get_com())
        
        # Torso Orientation Task (Tracks identity upright orientation matrix using simple skew error)
        self._v_task[3] = -self._ko_task * self.data.xmat[self._pelvis_id, 7]   # Roll correction component
        self._v_task[4] = -self._ko_task * self.data.xmat[self._pelvis_id, 6]   # Pitch correction component
        self._v_task[5] = -self._ko_task * math.asin(max(-1.0, min(1.0, self.data.xmat[self._pelvis_id, 1]))) # Yaw tracking alignment
        
        # Feet Spatial Velocity Tracking Tasks
        self._v_task[6:9] = self.foot_target_vel_l + self._kp_task * (self.foot_target_pos_l - self.data.xpos[self._l_foot_id])
        self._v_task[9:12] = self.foot_target_vel_r + self._kp_task * (self.foot_target_pos_r - self.data.xpos[self._r_foot_id])
        
        # D. Map Normal Equations in-place: A = J^T * J, b = J^T * v_task
        np.matmul(self._J_stacked.T, self._J_stacked, out=self._A)
        np.matmul(self._J_stacked.T, self._v_task, out=self._b)
        
        # Apply Tikhonov Damped Least Squares Diagonal Regularization
        self._A[self._diag_indices] += self._lambda_sq
        
        # E. Execute raw LAPACK solver over the buffers (No allocations triggered)
        # dgesv overwrites self._A with LU-factored variants and self._b with the raw velocity solution vector
        _, _, v_sol, info = dgesv(self._A, self._b, overwrite_a=True, overwrite_b=True)
        
        if info == 0:
            # F. Extract Actuated Velocity Subcomponents & Perform Numerical Model Integration
            np.take(v_sol, self.pd_controller._dof_idx, out=self._qdot_target)
            np.take(self.data.qpos, self.pd_controller._qpos_idx, out=self._q_target)
            np.add(self._q_target, self._qdot_target * self.dt, out=self._q_target)
        else:
            # Solver failed (degenerate Jacobian): freeze joints at current position so
            # the PD layer applies pure gravity compensation with zero tracking error,
            # rather than commanding zeros and driving the robot to the default pose.
            np.take(self.data.qpos, self.pd_controller._qpos_idx, out=self._q_target)
            self._qdot_target.fill(0.0)