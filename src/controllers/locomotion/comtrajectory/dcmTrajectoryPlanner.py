import numpy as np
import math

class DcmTrajectoryPlanner:
    """
    Zero-Allocation Center of Mass & ZMP Trajectory Planner.
    
    Implements the discrete-time reduced-order Linear Inverted Pendulum Model (LIPM) 
    and Divergent Component of Motion (DCM) split dynamics.
    
    Strict Architecture Contract:
    -----------------------------
    * Allocations are restricted entirely to __init__.
    * update_trajectory() operates in-place at 500 Hz.
    """
    def __init__(self, dt: float, com_height: float = 0.66) -> None:
        self.dt = dt
        self.z_c = com_height
        
        # LIPM constant natural frequency: omega = sqrt(g / z_c)
        self.omega = math.sqrt(9.81 / self.z_c)
        
        # Discrete-time state transition coefficients (Slides 36 & 39)
        self.a = math.exp(self.omega * self.dt)
        self.beta = math.exp(-self.omega * self.dt)
        
        # ── Pre-allocated Reference State Buffers ───────────────────────────
        self.com_pos_ref = np.zeros(3, dtype=np.float64)
        self.com_vel_ref = np.zeros(3, dtype=np.float64)
        self.dcm_pos_ref = np.zeros(3, dtype=np.float64)
        self.zmp_pos_ref = np.zeros(3, dtype=np.float64)

    def initialize_states(self, initial_com: np.ndarray) -> None:
        """Sets up the initial boundary conditions for the predictive filters."""
        np.copyto(self.com_pos_ref, initial_com)
        np.copyto(self.dcm_pos_ref, initial_com)
        self.com_pos_ref[2] = self.z_c
        self.dcm_pos_ref[2] = self.z_c

    def reset_for_step(
        self,
        stance_foot: np.ndarray,
        swing_foot_target: np.ndarray,
        T_step: float
    ) -> None:
        """
        Seeds the DCM at the start of each single-support phase via backward planning.

        Terminal constraint: DCM must equal swing_foot_target at touchdown (t = T_step).
        Working backwards through  ξ̇ = ω(ξ − p_zmp)  gives the required initial condition:

            ξ₀ = p_stance + (p_swing_target − p_stance) · exp(−ω · T_step)

        Without this reset the forward DCM integration diverges by ≈ exp(ω·T) ≈ 3.9× per
        step, generating a lateral velocity reference that tips the robot in the first gait
        cycle.
        """
        decay = math.exp(-self.omega * T_step)
        self.dcm_pos_ref[0] = stance_foot[0] + (swing_foot_target[0] - stance_foot[0]) * decay
        self.dcm_pos_ref[1] = stance_foot[1] + (swing_foot_target[1] - stance_foot[1]) * decay
        # Z is kept at z_c throughout; height is managed by update_trajectory.

    def update_trajectory(
        self,
        contact_states: np.ndarray,
        left_foot_pos: np.ndarray,
        right_foot_pos: np.ndarray,
        next_footstep_target: np.ndarray
    ) -> None:
        """
        Updates the reference CoM state vectors using exponential filters in-place.
        """
        # 1. Resolve target ZMP position based on hybrid contact configuration
        if contact_states[0] == 1 and contact_states[1] == 1:
            # Double Support Phase: Anchor ZMP between both support feet
            self.zmp_pos_ref[0] = 0.5 * (left_foot_pos[0] + right_foot_pos[0])
            self.zmp_pos_ref[1] = 0.5 * (left_foot_pos[1] + right_foot_pos[1])
        elif contact_states[0] == 1:
            # Left Single Support: ZMP must stay locked inside the left sole
            self.zmp_pos_ref[0] = left_foot_pos[0]
            self.zmp_pos_ref[1] = left_foot_pos[1]
        else:
            # Right Single Support: ZMP must stay locked inside the right sole
            self.zmp_pos_ref[0] = right_foot_pos[0]
            self.zmp_pos_ref[1] = right_foot_pos[1]
        self.zmp_pos_ref[2] = 0.0  # Kept on the flat ground plane

        # 2. Propagate Discrete Unstable DCM Profile: xi_k+1 = a*xi_k + (1-a)*p_zmp
        self.dcm_pos_ref[0] = self.a * self.dcm_pos_ref[0] + (1.0 - self.a) * self.zmp_pos_ref[0]
        self.dcm_pos_ref[1] = self.a * self.dcm_pos_ref[1] + (1.0 - self.a) * self.zmp_pos_ref[1]

        # 3. Propagate Discrete Stable CoM Profile: p_k+1 = beta*p_k + (1-beta)*xi_k
        self.com_pos_ref[0] = self.beta * self.com_pos_ref[0] + (1.0 - self.beta) * self.dcm_pos_ref[0]
        self.com_pos_ref[1] = self.beta * self.com_pos_ref[1] + (1.0 - self.beta) * self.dcm_pos_ref[1]
        self.com_pos_ref[2] = self.z_c

        # 4. Reconstruct Kinematic Target Velocities: p_dot = omega * (xi - p)
        self.com_vel_ref[0] = self.omega * (self.dcm_pos_ref[0] - self.com_pos_ref[0])
        self.com_vel_ref[1] = self.omega * (self.dcm_pos_ref[1] - self.com_pos_ref[1])
        self.com_vel_ref[2] = 0.0