"""
dcm_trajectory_planner.py — Stateless LIPM/DCM offline CoM trajectory planner.

Implements the two-stage integration pipeline from Lecture 10:
  1. Backward DCM integration (exact closed-form for piecewise-constant ZMP)
  2. Forward CoM integration  (exact exponential integrator, unconditionally stable)

This module has NO MuJoCo dependency — it is pure NumPy math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.interpolate import CubicSpline


# ══════════════════════════════════════════════════════════════════════════════
# Data types — frozen (treat as immutable)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LipmConfig:
    """Physical constants for the Linear Inverted Pendulum Model."""
    gravity_mps2: float = 9.81    # gravitational acceleration  [m/s²]
    com_height_m: float = 0.75    # assumed constant CoM height [m]

    @property
    def omega_rps(self) -> float:
        """Natural frequency ω = √(g / z_c)  [rad/s]."""
        return np.sqrt(self.gravity_mps2 / self.com_height_m)


@dataclass(frozen=True)
class GaitTiming:
    """Timing parameters for one gait cycle."""
    initial_dsp_s: float = 0.8    # settling double-support at start     [s]
    single_support_s: float = 0.4 # single-support (swing) duration      [s]
    double_support_s: float = 0.2 # weight-transfer double-support       [s]
    final_dsp_s: float = 0.8      # settling double-support at end       [s]
    dt_s: float = 0.005           # planning discretisation timestep     [s]


@dataclass(frozen=True)
class Footstep:
    """A single foot placement in the world frame."""
    position_m: np.ndarray   # (3,) XYZ centre of foot contact  [m]
    yaw_rad: float           # heading angle                    [rad]
    foot: str                # "left" | "right"

    def __post_init__(self):
        if self.foot not in ("left", "right"):
            raise ValueError(f"foot must be 'left' or 'right', got '{self.foot}'")
        pos = np.asarray(self.position_m, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"position_m must be shape (3,), got {pos.shape}")
        object.__setattr__(self, "position_m", pos)


@dataclass
class DcmTrajectory:
    """Complete output of the planner.  All arrays are (N, …)."""
    time_s: np.ndarray          # (N,)    timestamps           [s]
    com_m: np.ndarray           # (N, 3)  CoM XYZ trajectory   [m]
    dcm_m: np.ndarray           # (N, 2)  DCM XY trajectory    [m]
    zmp_ref_m: np.ndarray       # (N, 2)  reference ZMP XY     [m]
    left_foot_m: np.ndarray     # (N, 3)  left foot positions  [m]
    right_foot_m: np.ndarray    # (N, 3)  right foot positions [m]
    left_yaw_rad: np.ndarray    # (N,)    left foot heading    [rad]
    right_yaw_rad: np.ndarray   # (N,)    right foot heading   [rad]


# ══════════════════════════════════════════════════════════════════════════════
# Internal phase descriptor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Phase:
    kind: str               # "ds_init" | "ssp" | "dsp" | "ds_final"
    duration_s: float
    zmp_start: np.ndarray   # (2,) XY
    zmp_end: np.ndarray     # (2,) XY
    swing_foot: str | None  # "left" | "right" | None
    left_start: np.ndarray  # (3,)
    left_end: np.ndarray    # (3,)
    right_start: np.ndarray # (3,)
    right_end: np.ndarray   # (3,)
    left_yaw_s: float
    left_yaw_e: float
    right_yaw_s: float
    right_yaw_e: float


# ══════════════════════════════════════════════════════════════════════════════
# Planner
# ══════════════════════════════════════════════════════════════════════════════

class DcmTrajectoryPlanner:
    """
    Stateless offline trajectory planner using LIPM + DCM theory.

    Usage
    -----
    >>> cfg  = LipmConfig(com_height_m=0.75)
    >>> tmg  = GaitTiming()
    >>> plan = DcmTrajectoryPlanner(cfg)
    >>> traj = plan.plan(footsteps, tmg, initial_com_xy)
    """

    def __init__(self, config: LipmConfig) -> None:
        self._omega: float = config.omega_rps          # [rad/s]
        self._z_c: float   = config.com_height_m       # [m]
        self._g: float     = config.gravity_mps2       # [m/s²]

    # ── public entry point ────────────────────────────────────────────────────

    def plan(
        self,
        initial_left: Footstep,
        initial_right: Footstep,
        swing_sequence: Sequence[Footstep],
        timing: GaitTiming,
        initial_com_xy_m: np.ndarray,
        arc_height_m: float = 0.05,
    ) -> DcmTrajectory:
        """
        Generate a complete dynamically-feasible walking trajectory.

        Parameters
        ----------
        initial_left     : planted left-foot pose at t = 0.
        initial_right    : planted right-foot pose at t = 0.
        swing_sequence   : ordered list of swing Footsteps (alternating feet).
        timing           : gait timing configuration.
        initial_com_xy_m : (2,) initial CoM XY position [m].
        arc_height_m     : swing-foot lift height [m].

        Returns
        -------
        DcmTrajectory with all discretised reference signals.
        """
        com0 = np.asarray(initial_com_xy_m, dtype=np.float64).ravel()
        assert com0.shape == (2,), f"initial_com_xy_m must be (2,), got {com0.shape}"
        assert len(swing_sequence) >= 1, "Need at least one swing step"

        dt = timing.dt_s
        assert dt > 0, f"dt must be positive, got {dt}"

        # 1. Build gait-phase timeline
        phases = self._build_phases(initial_left, initial_right,
                                    swing_sequence, timing)

        # 2. Discretise all reference signals
        zmp, lf, rf, ly, ry, t = self._discretise(phases, dt, arc_height_m)

        N = len(t)
        assert N >= 2, "Trajectory too short — increase durations or reduce dt"

        # 3. Backward DCM integration  (exact for piecewise-constant ZMP)
        dcm = self._integrate_dcm_backward(zmp, dt)

        # 4. Forward CoM integration   (exact exponential integrator)
        com_xy = self._integrate_com_forward(dcm, com0, dt)

        # 5. Assemble 3-D CoM (constant height)
        com_3d = np.column_stack([com_xy, np.full(N, self._z_c)])

        return DcmTrajectory(
            time_s=t, com_m=com_3d, dcm_m=dcm, zmp_ref_m=zmp,
            left_foot_m=lf, right_foot_m=rf,
            left_yaw_rad=ly, right_yaw_rad=ry,
        )

    # ── phase construction ────────────────────────────────────────────────────

    def _build_phases(
        self,
        init_l: Footstep, init_r: Footstep,
        swings: Sequence[Footstep],
        t: GaitTiming,
    ) -> List[_Phase]:

        phases: List[_Phase] = []
        cur_l, cur_r = init_l.position_m.copy(), init_r.position_m.copy()
        yl, yr = init_l.yaw_rad, init_r.yaw_rad

        # Determine first stance foot (opposite of first swing)
        first_stance_pos = cur_l if swings[0].foot == "right" else cur_r

        # ── Initial double-support: shift ZMP from centre to first stance ─────
        centre_xy = 0.5 * (cur_l[:2] + cur_r[:2])
        phases.append(_Phase(
            kind="ds_init", duration_s=t.initial_dsp_s,
            zmp_start=centre_xy.copy(), zmp_end=first_stance_pos[:2].copy(),
            swing_foot=None,
            left_start=cur_l.copy(), left_end=cur_l.copy(),
            right_start=cur_r.copy(), right_end=cur_r.copy(),
            left_yaw_s=yl, left_yaw_e=yl,
            right_yaw_s=yr, right_yaw_e=yr,
        ))

        # ── Per-step: SSP then DSP ────────────────────────────────────────────
        for k, step in enumerate(swings):
            is_last = (k == len(swings) - 1)
            sf = step.foot
            tgt = step.position_m.copy()
            ty = step.yaw_rad

            if sf == "right":
                stance_xy = cur_l[:2].copy()
                sw_start = cur_r.copy()
            else:
                stance_xy = cur_r[:2].copy()
                sw_start = cur_l.copy()

            # SSP — ZMP fixed at stance foot
            phases.append(_Phase(
                kind="ssp", duration_s=t.single_support_s,
                zmp_start=stance_xy.copy(), zmp_end=stance_xy.copy(),
                swing_foot=sf,
                left_start=cur_l.copy(),
                left_end=(tgt.copy() if sf == "left" else cur_l.copy()),
                right_start=cur_r.copy(),
                right_end=(tgt.copy() if sf == "right" else cur_r.copy()),
                left_yaw_s=yl,
                left_yaw_e=(ty if sf == "left" else yl),
                right_yaw_s=yr,
                right_yaw_e=(ty if sf == "right" else yr),
            ))

            # Update current foot positions
            if sf == "left":
                cur_l, yl = tgt.copy(), ty
            else:
                cur_r, yr = tgt.copy(), ty

            # DSP transition or final settle
            if not is_last:
                next_stance = tgt[:2].copy()
                phases.append(_Phase(
                    kind="dsp", duration_s=t.double_support_s,
                    zmp_start=stance_xy.copy(), zmp_end=next_stance.copy(),
                    swing_foot=None,
                    left_start=cur_l.copy(), left_end=cur_l.copy(),
                    right_start=cur_r.copy(), right_end=cur_r.copy(),
                    left_yaw_s=yl, left_yaw_e=yl,
                    right_yaw_s=yr, right_yaw_e=yr,
                ))
            else:
                final_centre = 0.5 * (cur_l[:2] + cur_r[:2])
                phases.append(_Phase(
                    kind="ds_final", duration_s=t.final_dsp_s,
                    zmp_start=tgt[:2].copy(), zmp_end=final_centre.copy(),
                    swing_foot=None,
                    left_start=cur_l.copy(), left_end=cur_l.copy(),
                    right_start=cur_r.copy(), right_end=cur_r.copy(),
                    left_yaw_s=yl, left_yaw_e=yl,
                    right_yaw_s=yr, right_yaw_e=yr,
                ))
        return phases

    # ── discretisation ────────────────────────────────────────────────────────

    def _discretise(self, phases, dt, arc_h):
        zmp_list, lf_list, rf_list = [], [], []
        ly_list, ry_list, t_list = [], [], []
        t_acc = 0.0

        for ph in phases:
            n = max(1, int(round(ph.duration_s / dt)))
            alphas = np.linspace(0.0, 1.0, n, endpoint=False)

            for a in alphas:
                t_list.append(t_acc)
                t_acc += dt
                zmp_list.append(ph.zmp_start + a * (ph.zmp_end - ph.zmp_start))
                ly_list.append(ph.left_yaw_s + a * (ph.left_yaw_e - ph.left_yaw_s))
                ry_list.append(ph.right_yaw_s + a * (ph.right_yaw_e - ph.right_yaw_s))

            # Foot trajectories
            if ph.kind == "ssp" and ph.swing_foot is not None:
                if ph.swing_foot == "left":
                    arc = self._foot_arc(ph.left_start, ph.left_end, arc_h, n)
                    lf_list.extend(arc)
                    rf_list.extend([ph.right_start.copy() for _ in range(n)])
                else:
                    arc = self._foot_arc(ph.right_start, ph.right_end, arc_h, n)
                    rf_list.extend(arc)
                    lf_list.extend([ph.left_start.copy() for _ in range(n)])
            else:
                lf_list.extend([ph.left_start.copy() for _ in range(n)])
                rf_list.extend([ph.right_start.copy() for _ in range(n)])

        # Append terminal sample
        last = phases[-1]
        t_list.append(t_acc)
        zmp_list.append(last.zmp_end.copy())
        lf_list.append(last.left_end.copy())
        rf_list.append(last.right_end.copy())
        ly_list.append(last.left_yaw_e)
        ry_list.append(last.right_yaw_e)

        return (np.array(zmp_list), np.array(lf_list), np.array(rf_list),
                np.array(ly_list), np.array(ry_list), np.array(t_list))

    # ── backward DCM integration (exact closed-form) ─────────────────────────

    def _integrate_dcm_backward(self, zmp: np.ndarray, dt: float) -> np.ndarray:
        """
        Exact backward recursion for piecewise-constant ZMP:
            ξ_k = p_ZMP_k + e^{-ω·Δt} · (ξ_{k+1} − p_ZMP_k)

        Terminal boundary: ξ_N = p_ZMP_N  (bounded, no divergence).
        """
        N = len(zmp)
        dcm = np.zeros((N, 2), dtype=np.float64)
        dcm[-1] = zmp[-1]                          # terminal BC

        decay = np.exp(-self._omega * dt)           # e^{-ω·Δt} < 1, stable

        for k in range(N - 2, -1, -1):
            dcm[k] = zmp[k] + decay * (dcm[k + 1] - zmp[k])

        return dcm

    # ── forward CoM integration (exact exponential integrator) ────────────────

    def _integrate_com_forward(
        self, dcm: np.ndarray, com0: np.ndarray, dt: float,
    ) -> np.ndarray:
        """
        Exact solution for ẋ = ω(ξ − x) with ξ constant over [t_k, t_{k+1}]:
            x_{k+1} = ξ_k + (x_k − ξ_k) · e^{-ω·Δt}

        Unconditionally stable because |e^{-ω·Δt}| < 1  ∀  Δt > 0.
        """
        N = len(dcm)
        com = np.zeros((N, 2), dtype=np.float64)
        com[0] = com0

        decay = np.exp(-self._omega * dt)

        for k in range(N - 1):
            com[k + 1] = dcm[k] + decay * (com[k] - dcm[k])

        return com

    # ── swing-foot cubic spline arc ───────────────────────────────────────────

    @staticmethod
    def _foot_arc(start, end, height_m, n_points):
        """Cubic-spline 3-D arc with a mid-point hump of `height_m`."""
        t_knots = np.array([0.0, 0.5, 1.0])
        mid = 0.5 * (start + end)
        mid_lifted = mid.copy()
        mid_lifted[2] += height_m
        pts = np.array([start, mid_lifted, end])
        cs = CubicSpline(t_knots, pts, axis=0)
        samples = np.linspace(0.0, 1.0, n_points)
        return list(cs(samples))


# ══════════════════════════════════════════════════════════════════════════════
# Footstep generation utility (pure function)
# ══════════════════════════════════════════════════════════════════════════════

def generate_footstep_sequence(
    left_init_m: np.ndarray,
    right_init_m: np.ndarray,
    n_steps: int,
    travel_distance_m: float,
    theta_rad: float = 0.0,
) -> tuple[Footstep, Footstep, list[Footstep]]:
    """
    Generate an alternating L/R footstep sequence for straight-line walking.

    Returns (initial_left, initial_right, swing_sequence).
    """
    l0 = np.asarray(left_init_m, dtype=np.float64).ravel()
    r0 = np.asarray(right_init_m, dtype=np.float64).ravel()

    init_l = Footstep(position_m=l0.copy(), yaw_rad=0.0, foot="left")
    init_r = Footstep(position_m=r0.copy(), yaw_rad=0.0, foot="right")

    fwd = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    lat = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0.0])
    half_w = abs(l0[1] - r0[1]) / 2.0

    centre = 0.5 * (l0 + r0)
    step_len = travel_distance_m / (2 * n_steps) if n_steps > 0 else 0.0

    swings: list[Footstep] = []
    for s in range(1, 2 * n_steps + 1):
        centre = centre + step_len * fwd
        if s % 2 == 1:
            pos = centre - half_w * lat
            swings.append(Footstep(pos.copy(), theta_rad, "right"))
        else:
            pos = centre + half_w * lat
            swings.append(Footstep(pos.copy(), theta_rad, "left"))

    # Alignment step
    centre = centre + step_len * fwd
    swings.append(Footstep((centre - half_w * lat).copy(), theta_rad, "right"))

    return init_l, init_r, swings
