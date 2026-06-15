import math
import numpy as np


class AnalyticalComPlanner:
    """
    Closed-form LIPM trajectory generator for bipedal locomotion.

    Replaces the unstable forward-DCM filter with the exact analytical solution
    of the Linear Inverted Pendulum Model:

        ẍ = ω²(x − p_zmp),   ω = sqrt(g / z_c)

    Given initial state (x₀, ẋ₀) and a constant ZMP p held over one step, the
    exact solution is:

        x(τ)  = p + (x₀ − p)·cosh(ω·τ) + (ẋ₀/ω)·sinh(ω·τ)
        ẋ(τ)  = ω·(x₀ − p)·sinh(ω·τ)   + ẋ₀·cosh(ω·τ)

    Latching the MEASURED (x₀, ẋ₀) at every step transition eliminates the
    reference-velocity discontinuity that was causing the second-step collapse:
    the reference begins exactly at the measured state, so there is no abrupt
    jump in commanded velocity across the gait boundary.

    Architecture contract
    --------------------
    * __init__     : the only place allowed to allocate NumPy arrays.
    * latch_step() : called once per step at ~3 Hz; zero heap allocations.
    * update()     : called on the 500 Hz hot path; zero heap allocations —
                     only scalar temporaries and element-wise writes into
                     pre-allocated buffers.
    """

    def __init__(self, com_height: float = 0.66) -> None:
        self.z_c       = com_height
        self.omega     = math.sqrt(9.81 / com_height)
        self._inv_omega = 1.0 / self.omega

        # ── Latched boundary conditions (overwritten once per step) ──────────
        self._x0    = np.zeros(2, dtype=np.float64)   # CoM XY at step start
        self._xdot0 = np.zeros(2, dtype=np.float64)   # CoM vel XY at step start
        self._p_zmp = np.zeros(2, dtype=np.float64)   # Stance foot XY (ZMP proxy)

        # ── Public outputs — read by the controller on every tick ────────────
        self.com_pos_ref = np.zeros(3, dtype=np.float64)
        self.com_vel_ref = np.zeros(3, dtype=np.float64)

    def latch_step(
        self,
        initial_com_xy:     np.ndarray,
        initial_com_vel_xy: np.ndarray,
        stance_foot_xy:     np.ndarray,
    ) -> None:
        """
        Latches boundary conditions at the start of each single-support phase.

        Must be called exactly once when new_step_triggered is True, BEFORE
        update() is invoked for τ = 0.

        Parameters
        ----------
        initial_com_xy     : Any indexable with [0],[1] giving the measured
                             CoM XY position (e.g., estimator.get_com()).
        initial_com_vel_xy : Any indexable with [0],[1] giving the CoM XY vel
                             (e.g., data.qvel — the free-joint XY velocity).
        stance_foot_xy     : Any indexable with [0],[1] giving the XY world
                             position of the stance foot (ZMP proxy).
        """
        self._x0[0]    = initial_com_xy[0]
        self._x0[1]    = initial_com_xy[1]
        self._xdot0[0] = initial_com_vel_xy[0]
        self._xdot0[1] = initial_com_vel_xy[1]
        self._p_zmp[0] = stance_foot_xy[0]
        self._p_zmp[1] = stance_foot_xy[1]

        # Seed reference at τ = 0  (analytical solution identity: cosh(0)=1, sinh(0)=0)
        self.com_pos_ref[0] = initial_com_xy[0]
        self.com_pos_ref[1] = initial_com_xy[1]
        self.com_pos_ref[2] = self.z_c
        self.com_vel_ref[0] = initial_com_vel_xy[0]
        self.com_vel_ref[1] = initial_com_vel_xy[1]
        self.com_vel_ref[2] = 0.0

    def update(self, tau: float) -> None:
        """
        Evaluates the closed-form LIPM solution at elapsed step time τ.

        τ = math.fmod(sim_time, T_step) — seconds elapsed since the last
        step transition.  Must be called every physics tick on the 500 Hz path.

        Zero heap allocations: cosh/sinh are scalar, the loop over i ∈ {0,1}
        operates entirely on pre-allocated buffer elements.
        """
        cosh_t = math.cosh(self.omega * tau)
        sinh_t = math.sinh(self.omega * tau)

        for i in range(2):
            dx0_i = self._x0[i] - self._p_zmp[i]
            self.com_pos_ref[i] = (
                self._p_zmp[i]
                + dx0_i * cosh_t
                + self._xdot0[i] * sinh_t * self._inv_omega
            )
            self.com_vel_ref[i] = (
                self.omega * dx0_i * sinh_t
                + self._xdot0[i] * cosh_t
            )

        # Height and vertical velocity are fixed by the LIPM rigid-leg assumption
        self.com_pos_ref[2] = self.z_c
        self.com_vel_ref[2] = 0.0
