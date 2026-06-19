import numpy as np

class SwingFootTrajectoryGenerator:
    """
    Zero-Allocation 3D Swing Foot Trajectory Generator.
    
    Uses combined Cubic-Hermite and Quartic bounding polynomials to ensure 
    smooth spatial tracking with guaranteed zero-velocity boundary conditions 
    at both liftoff and touchdown phases.
    
    Strict Architecture Contract:
    -----------------------------
    * Allocations are restricted entirely to __init__.
    * compute_trajectory() executes on the hot path (500 Hz) using zero-copy 
      in-place arithmetic operations.
    """
    def __init__(self, swing_duration: float, peak_clearance: float = 0.08) -> None:
        self.t_swing = swing_duration
        self.z_clear = peak_clearance
        
        # Pre-allocated structural difference buffers to eliminate heap garbage collection
        self._pos_delta = np.zeros(3, dtype=np.float64)

    def compute_trajectory(
        self,
        phi: float,
        p_start: np.ndarray,
        p_target: np.ndarray,
        out_pos: np.ndarray,
        out_vel: np.ndarray
    ) -> None:
        """
        Computes the target 3D task-space position and velocity vectors in-place.
        
        Parameters:
        -----------
        phi : Normalized swing phase scalar clamped to [0.0, 1.0).
        p_start : Coordinates [X, Y, Z] of the foot at liftoff frame.
        p_target : Coordinates [X, Y, Z] of the targeted footprint placement.
        out_pos : Output mutable buffer view for target position vector (shape: 3).
        out_vel : Output mutable buffer view for target linear velocity vector (shape: 3).
        """
        # Defensive boundary clipping to prevent arithmetic instability beyond the domain
        phi = max(0.0, min(1.0, phi))
        
        # Calculate the direct spatial distance vector: (p_target - p_start)
        np.subtract(p_target, p_start, out=self._pos_delta)
        
        # ── 1. Calculate Polynomial Scalings ────────────────────────────────
        phi_sq = phi * phi
        phi_cube = phi_sq * phi
        
        # Horizontal path scaling: s = 3*phi^2 - 2*phi^3
        s = 3.0 * phi_sq - 2.0 * phi_cube
        # Time derivative modifier: ds_dphi = 6*phi - 6*phi^2
        ds_dphi = 6.0 * phi - 6.0 * phi_sq
        
        # Vertical clearance hump scaling: h = 16 * phi^2 * (1 - phi)^2
        one_minus_phi = 1.0 - phi
        h = 16.0 * phi_sq * (one_minus_phi * one_minus_phi)
        # Time derivative modifier: dh_dphi = 32 * phi * (1 - phi) * (1 - 2*phi)
        dh_dphi = 32.0 * phi * one_minus_phi * (1.0 - 2.0 * phi)
        
        # ── 2. Resolve Task-Space Positions ─────────────────────────────────
        # X and Y Interpolation
        out_pos[0] = p_start[0] + self._pos_delta[0] * s
        out_pos[1] = p_start[1] + self._pos_delta[1] * s
        # Z Interpolation with added vertical clearance hump
        out_pos[2] = p_start[2] + self._pos_delta[2] * s + self.z_clear * h
        
        # ── 3. Resolve Task-Space Linear Velocities ─────────────────────────
        # Convert phase derivatives to physical velocity (m/s) by scaling with (1 / T_swing)
        phase_dot = 1.0 / self.t_swing
        
        out_vel[0] = self._pos_delta[0] * ds_dphi * phase_dot
        out_vel[1] = self._pos_delta[1] * ds_dphi * phase_dot
        out_vel[2] = (self._pos_delta[2] * ds_dphi + self.z_clear * dh_dphi) * phase_dot