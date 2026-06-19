import numpy as np

class FootstepPlanner:
    """
    Vectorized 2D Footstep Planner using Raibert Velocity Feedback Heuristics.
    
    Calculates dynamically balanced foot placement targets based on 
    instantaneous linear base velocity deviations.
    
    Mathematical Foundation (Slide 14, Lecture 10):
    -------------------------------------------------------
    x_foot = x_hip + (T_step / 2) * v_x + k_x * (v_x - v_x_des)
    y_foot = y_hip + s * (W / 2) + (T_step / 2) * v_y + k_y * (v_y - v_y_des)
    
    where s = +1 for Left Foot, s = -1 for Right Foot.
    """
    def __init__(
        self,
        nominal_step_width: float,
        step_duration: float,
        gain_x: float = 0.1,
        gain_y: float = 0.1
    ) -> None:
        self.w_nominal = nominal_step_width
        self.t_step = step_duration
        
        # Proportional feedback gains for capturing velocity disturbances
        self._k_x = gain_x
        self._k_y = gain_y
        
        # ── Pre-allocated Hot-Path Output Buffers ───────────────────────────
        # Target step coordinates structured as [X, Y, Z] positions in World frame
        self.left_foot_target = np.zeros(3, dtype=np.float64)
        self.right_foot_target = np.zeros(3, dtype=np.float64)
        
        # Working buffers to completely avoid dynamic allocation overhead
        self._vel_error = np.zeros(2, dtype=np.float64)
        self._raibert_offset = np.zeros(2, dtype=np.float64)

    def compute_next_footstep(
        self,
        hip_pos_l: np.ndarray,
        hip_pos_r: np.ndarray,
        body_vel: np.ndarray,
        body_vel_des: np.ndarray,
        ground_clearance_z: float = 0.006  # Aligned with _FOOT_SPHERE_TARGET_Z
    ) -> None:
        """
        Executes zero-allocation vector arithmetic to update footstep landing targets.
        
        Parameters:
        -----------
        hip_pos_l : Array (3,) specifying the current Left Hip position in world coordinates.
        hip_pos_r : Array (3,) specifying the current Right Hip position in world coordinates.
        body_vel  : Array (2,) specifying current [v_x, v_y] linear velocity.
        body_vel_des : Array (2,) specifying target commanded [v_x_des, v_y_des] velocity.
        ground_clearance_z : Base target height of the floor contact surface.
        """
        # Calculate velocity tracking deviations: (v - v_des)
        np.subtract(body_vel[:2], body_vel_des[:2], out=self._vel_error)
        
        # ── Forward / X-Axis Placement ──────────────────────────────────────
        # Combined Feedforward Neutral Point + Proportional Correction
        # x_offset = (T / 2) * v_x + k_x * x_error
        self._raibert_offset[0] = (self.t_step / 2.0) * body_vel[0] + self._k_x * self._vel_error[0]
        
        # ── Lateral / Y-Axis Placement ──────────────────────────────────────
        # y_offset = (T / 2) * v_y + k_y * y_error
        self._raibert_offset[1] = (self.t_step / 2.0) * body_vel[1] + self._k_y * self._vel_error[1]
        
        # ── Apply Left Foot Vector Projections ──────────────────────────────
        # World X position = Left Hip X + Raibert X offset
        self.left_foot_target[0] = hip_pos_l[0] + self._raibert_offset[0]
        # World Y position = Left Hip Y + Nominal Width Offset + Raibert Y offset
        self.left_foot_target[1] = hip_pos_l[1] + (self.w_nominal / 2.0) + self._raibert_offset[1]
        # Maintain rigid ground contact baseline plane reference
        self.left_foot_target[2] = ground_clearance_z
        
        # ── Apply Right Foot Vector Projections ─────────────────────────────
        self.right_foot_target[0] = hip_pos_r[0] + self._raibert_offset[0]
        # World Y position for right side uses negative width offset configuration
        self.right_foot_target[1] = hip_pos_r[1] - (self.w_nominal / 2.0) + self._raibert_offset[1]
        self.right_foot_target[2] = ground_clearance_z