import numpy as np


class AnalyticalComPlanner:
    """
    C1-Continuous Cubic Hermite Spline CoM Trajectory Planner.

    Replaces the LIPM closed-form planner with a Hermite polynomial that
    guarantees smooth, shock-free reference signals across every step
    transition.

    Root causes fixed
    -----------------
    1. Reference velocity starvation
       The old LIPM planner re-seeded (x₀, ẋ₀) from the live estimator at
       every step boundary.  When the robot lagged behind the reference,
       the new step started from a lower-momentum seed and the LIPM had to
       re-accelerate, creating a repeating "brake–accelerate" cycle.

    2. Step-jump position discontinuity
       Seeding x₀ from estimator.get_com() injected the tracking error
       into the reference: if the robot was 20 mm behind, the new
       reference snapped 20 mm backward → torque spike → motor saturation.

    Fix
    ---
    * C1 continuity  — p₀, v₀ are ALWAYS the final output of the last
      compute_reference() call, never from live sensors.  The reference is
      a smooth function of time even through step transitions.

    * Non-zero terminal velocity  — v₁ = v_des.  The trajectory smoothly
      blends toward the commanded walking speed rather than braking to zero
      at the end of each step.

    Cubic Hermite formulation
    ─────────────────────────
    For normalised phase φ ∈ [0, 1] over step duration T:

        p(φ) = H₀₀(φ)·p₀  +  H₁₀(φ)·T·v₀
             + H₀₁(φ)·p₁  +  H₁₁(φ)·T·v₁

        ṗ(φ) = (1/T)·[ H̃₀₀(φ)·p₀ + H̃₁₀(φ)·T·v₀
                      + H̃₀₁(φ)·p₁ + H̃₁₁(φ)·T·v₁ ]

    Basis polynomials and their φ-derivatives:

        H₀₀ = 2φ³ − 3φ² + 1     H̃₀₀ = 6φ² − 6φ
        H₁₀ = φ³ − 2φ² + φ      H̃₁₀ = 3φ² − 4φ + 1
        H₀₁ = −2φ³ + 3φ²        H̃₀₁ = −6φ² + 6φ
        H₁₁ = φ³ − φ²            H̃₁₁ = 3φ² − 2φ

    Boundary conditions set once per step (in start_new_step):
        p₀, v₀  latched from previous com_pos_ref / com_vel_ref output
        p₁      = stance_foot_xy + (T/2)·v_des   (Raibert capture point)
        v₁      = v_des                           (commanded velocity)

    Verification at φ = 0:  p(0) = p₀, ṗ(0) = v₀  ✓  (C1 continuity)
    Verification at φ = 1:  p(1) = p₁, ṗ(1) = v₁  ✓  (smooth arrival)

    Architecture contract
    ─────────────────────
    * __init__           : only place allowed to allocate NumPy arrays.
    * start_new_step()   : called once per step (~3 Hz); zero heap allocs.
    * compute_reference(): 500 Hz hot path; zero heap allocations —
                          only Python-native scalar temporaries and
                          element-wise writes into pre-allocated buffers.
    """

    def __init__(self, com_height: float = 0.66, step_duration: float = 0.4) -> None:
        self.z_c = com_height
        self._T  = step_duration

        # ── Start boundary conditions (read from previous reference output) ──
        self._p0 = np.zeros(2, dtype=np.float64)   # CoM XY at φ = 0
        self._v0 = np.zeros(2, dtype=np.float64)   # CoM vel XY at φ = 0

        # ── End boundary conditions (set from stance foot + v_des) ───────────
        self._p1 = np.zeros(2, dtype=np.float64)   # Raibert capture point XY
        self._v1 = np.zeros(2, dtype=np.float64)   # Commanded velocity XY

        # ── Public outputs — read by the controller on every tick ────────────
        self.com_pos_ref = np.zeros(3, dtype=np.float64)
        self.com_vel_ref = np.zeros(3, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────

    def start_new_step(
        self,
        stance_foot_xy: np.ndarray,
        v_des:          np.ndarray,
        step_duration:  float = None,
    ) -> None:
        """
        Latch boundary conditions at the start of each single-support phase.

        Called exactly once when new_step_triggered is True, before the
        first compute_reference() call of the new step.

        C1 continuity is enforced by reading p₀/v₀ from com_pos_ref /
        com_vel_ref — the last output of the PREVIOUS step's
        compute_reference() — rather than from live sensor data.

        Parameters
        ----------
        stance_foot_xy : XY world position of the new stance foot (ZMP proxy).
                         Indexable with [0] = X, [1] = Y.
        v_des          : Commanded walking velocity [v_x, v_y] in m/s.
                         Indexable with [0] = v_x, [1] = v_y.
        step_duration  : Optional per-step override (seconds).  When None,
                         the value set in __init__ is reused unchanged.
        """
        if step_duration is not None:
            self._T = step_duration

        # ── C1 start: latch from the last reference output (not sensors) ─────
        self._p0[0] = self.com_pos_ref[0]
        self._p0[1] = self.com_pos_ref[1]
        self._v0[0] = self.com_vel_ref[0]
        self._v0[1] = self.com_vel_ref[1]

        # ── Terminal position: Raibert capture point ─────────────────────────
        # At the end of this step the CoM should be at the dynamic equilibrium
        # point for the new stance foot:  p₁ = p_stance + (T/2)·v_des.
        # This prevents the Hermite from under- or over-shooting.
        self._p1[0] = stance_foot_xy[0] + 0.5 * self._T * v_des[0]
        self._p1[1] = stance_foot_xy[1] + 0.5 * self._T * v_des[1]

        # ── Terminal velocity: commanded velocity (no braking to zero) ───────
        self._v1[0] = v_des[0]
        self._v1[1] = v_des[1]

    # ──────────────────────────────────────────────────────────────────────────

    def compute_reference(self, phi: float) -> None:
        """
        Evaluate the cubic Hermite spline at normalised step phase φ ∈ [0, 1].

        φ = tau / T_step   where  tau = math.fmod(sim_time, T_step).

        Must be called on every physics tick (500 Hz hot path).

        Zero heap allocations: phi, phi2, phi3, h-values and Tv-scalars are
        all Python native floats.  The two XY output components are written
        element-wise into pre-allocated buffers via direct index assignment.
        """
        phi = max(0.0, min(1.0, phi))

        # ── Cubic basis polynomials at φ ──────────────────────────────────────
        phi2 = phi * phi
        phi3 = phi2 * phi

        h00 =  2.0 * phi3 - 3.0 * phi2 + 1.0   # H₀₀
        h10 =  phi3 - 2.0 * phi2 + phi          # H₁₀
        h01 = -2.0 * phi3 + 3.0 * phi2          # H₀₁
        h11 =  phi3 - phi2                        # H₁₁

        # ── φ-derivatives (scaled by 1/T for physical velocity) ───────────────
        dh00 =  6.0 * phi2 - 6.0 * phi          # H̃₀₀
        dh10 =  3.0 * phi2 - 4.0 * phi + 1.0    # H̃₁₀
        dh01 = -6.0 * phi2 + 6.0 * phi           # H̃₀₁
        dh11 =  3.0 * phi2 - 2.0 * phi           # H̃₁₁

        inv_T = 1.0 / self._T

        # ── In-place element-wise writes into pre-allocated output buffers ────
        for i in range(2):
            Tv0_i = self._T * self._v0[i]   # T·v₀ᵢ — scalar, no allocation
            Tv1_i = self._T * self._v1[i]   # T·v₁ᵢ — scalar, no allocation

            self.com_pos_ref[i] = (
                h00 * self._p0[i]
                + h10 * Tv0_i
                + h01 * self._p1[i]
                + h11 * Tv1_i
            )
            self.com_vel_ref[i] = (
                dh00 * self._p0[i]
                + dh10 * Tv0_i
                + dh01 * self._p1[i]
                + dh11 * Tv1_i
            ) * inv_T

        # Height and vertical velocity are fixed by rigid-leg constraint
        self.com_pos_ref[2] = self.z_c
        self.com_vel_ref[2] = 0.0
