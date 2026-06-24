"""
DCM (Divergent Component of Motion / capture point) walking pattern generator.

This is the *dynamic* gait that lets the G1 walk continuously without falling.
It replaces the quasi-static weight-shift planner (gait.py) for the walking task.
The whole-body QP (wbqp.py) is reused unchanged as the tracking controller.

Why DCM and not a quasi-static CoM-over-foot shift?  The G1's feet are tiny
(17 cm long, 6 cm wide) and the ankle-roll torque is weak (+-50 Nm), so lateral
balance CANNOT be held by the ankle.  It must come from where the swing foot is
placed.  The capture point tells us exactly where to step to arrest the falling
CoM, and the DCM feedback law tells us what ground-reaction (ZMP) to command in
between.  Together they give a stable walking limit cycle.

Model (Linear Inverted Pendulum, per horizontal axis, CoM height z_c constant):

    omega = sqrt(g / z_c)                      natural frequency
    xi    = c + c_dot / omega                  DCM (capture point)
    xi_dot = omega (xi - p)                    DCM dynamics, p = ZMP/CoP
    c_dot  = omega (xi - c)                     CoM follows the DCM

During a step the ZMP is held at the stance foot p_stance and the reference DCM
travels from the stance foot toward the next footstep:

    xi_ref(tau) = p_stance + (p_next - p_stance) * exp(omega (tau - T_ss))

DCM tracking feedback turns DCM error into a commanded ZMP (stable error dynamics
xi_dot_err = -k_dcm * xi_err):

    p_cmd = p_stance + (1 + k_dcm/omega) (xi_meas - xi_ref)

which we send to the QP as a CoM acceleration feed-forward (the QP realises it
through the contact forces, i.e. it places the centre of pressure):

    c_ddot_cmd = omega^2 (c - p_cmd)

Foot placement (capture-point step adjustment): during single support we predict
where the DCM will be at the end of the step and steer the swing foot toward it,
blended with the nominal alternating footstep that sustains the side-to-side sway:

    xi_pred_eos = p_stance + (xi_meas - p_stance) exp(omega (T_ss - tau))
    p_next      = (1 - k_cap) * p_nominal + k_cap * xi_pred_eos
"""

import numpy as np

from src.controllers.robot_model import RobotModel
from src.utils.terminal_logger import TerminalLogger as Logger


def _quintic(s):
    """Minimum-jerk scalar interpolation s in [0,1] -> [0,1] (zero vel/acc ends)."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _quintic_deriv(s):
    """d/ds of _quintic: 30 s^2 (1-s)^2."""
    s = np.clip(s, 0.0, 1.0)
    return 30 * s**2 * (1 - s)**2


class DCMWalkingGait:
    # FSM phases
    STAND = "STAND"
    START = "START"    # one-off initial weight shift onto the first stance foot
    DS = "DS"          # double support: transfer weight onto the next stance foot
    SS = "SS"          # single support: swing the other foot to its capture target

    def __init__(self, robot: RobotModel, params: dict = None):
        self.r = robot
        p = params or {}
        self.g = 9.81

        # timing
        self.t_stand = p.get("t_stand", 1.2)
        self.t_start = p.get("t_start", 1.0)   # initial weight shift onto 1st stance
        self.t_ds = p.get("t_ds", 0.12)        # double-support transfer
        self.t_ss = p.get("t_ss", 0.50)        # single-support / swing

        # footstep geometry
        self.step_length = p.get("step_length", 0.0)    # forward (m); 0 = in place
        self.step_width = p.get("step_width", 0.22)      # lateral foot separation (m)
        self.step_height = p.get("step_height", 0.15)    # swing-foot lift (m)

        # DCM control
        self.n_preview = p.get("n_preview", 6)  # footstep preview horizon (steps)
        self.k_dcm = p.get("k_dcm", 2.5)        # DCM feedback gain (>0 stabilises)
        self.k_cap = p.get("k_cap", 0.8)        # lateral DCM step-adjustment gain
        self.max_dy = p.get("max_dy", 0.06)     # clamp on lateral foot correction (m)
        self.max_dx = p.get("max_dx", 0.04)     # clamp on sagittal foot correction (m)
        self.k_center = p.get("k_center", 0.0)  # centreline-restoring foot bias
        self.zmp_margin = p.get("zmp_margin", 0.02)  # CoP clamp inside foot (m)

        # how many steps before stopping into a permanent stand (None = forever)
        self.max_steps = p.get("max_steps", None)

        # safety abort thresholds
        self.max_tilt = p.get("max_tilt", 0.5)  # rad roll/pitch -> recover/stop


        # runtime (set in reset)
        self.z_c = None
        self.omega = None
        self.state = self.STAND
        self.t_state = 0.0
        self.step_count = 0
        self._init = False

    # ------------------------------------------------------------------ reset
    def reset(self, state):
        self.foot_R = state["lfoot_mat"].copy()
        self.torso_R = np.eye(3)
        # planted foot positions (ankle xy + sole z)
        self.foot = {"left": state["lfoot_pos"].copy(),
                     "right": state["rfoot_pos"].copy()}
        self.z_c = state["com"][2]
        self.omega = np.sqrt(self.g / self.z_c)
        # CoM height target tracks the mean foot height so it climbs with terrain
        self._foot_z_ref = 0.5 * (self.foot["left"][2] + self.foot["right"][2])

        self.state = self.STAND
        self.t_state = 0.0
        self.step_count = 0
        self.walking = True
        # upcoming stance foot for the first step (start by standing on the left,
        # swinging the right). Symmetric; either works.
        self.stance = "left"
        # Footsteps are planned at ABSOLUTE positions so the gait does not drift:
        #   x_k = x0 + k * step_length ,   y_k = +d (even k / left) or -d (odd k).
        # stance_idx is the footstep index of the current stance foot (left=0).
        self.x0 = 0.5 * (self._foot_center_xy(self.foot["left"])[0] +
                         self._foot_center_xy(self.foot["right"])[0])
        self.stance_idx = 0
        # DCM-consistent reference CoM (xy), integrated from the reference DCM and
        # tracked by the whole-body QP's CoM PD.
        self.c_ref = state["com"][:2].copy()
        self.swing_start = None
        self.swing_goal = None      # 3D landing pose of the swing foot
        self.p_next = None          # xy ZMP target of the next stance (= swing land)
        self._init = True
        Logger.debug(f"DCMWalkingGait reset: z_c={self.z_c:.3f} omega={self.omega:.2f} "
                     f"T_ss={self.t_ss} T_ds={self.t_ds} step=({self.step_length},"
                     f"{self.step_width})")

    # -------------------------------------------------------- foot/ZMP helpers
    def _foot_center_xy(self, ankle_xyz):
        """ZMP reference point: the foot geometric centre, foot_center_x ahead of
        the ankle frame origin."""
        return np.array([ankle_xyz[0] + self.r.foot_center_x, ankle_xyz[1]])

    def _support_polygon(self, contacts):
        """Axis-aligned CoP bounds (xlo,xhi,ylo,yhi) over the contact feet, shrunk
        by zmp_margin. Used to clamp the commanded ZMP so it stays realisable."""
        xs, ys = [], []
        for f in contacts:
            c = self._foot_center_xy(self.foot[f])
            xs += [c[0] - self.r.foot_len_back, c[0] + self.r.foot_len_fwd]
            ys += [c[1] - self.r.foot_half_width, c[1] + self.r.foot_half_width]
        m = self.zmp_margin
        return (min(xs) + m, max(xs) - m, min(ys) + m, max(ys) - m)

    def _clamp_zmp(self, p, contacts):
        xlo, xhi, ylo, yhi = self._support_polygon(contacts)
        return np.array([np.clip(p[0], xlo, xhi), np.clip(p[1], ylo, yhi)])

    def _com_z(self):
        """Constant CoM height above the mean foot level (rises with terrain)."""
        mean_foot_z = 0.5 * (self.foot["left"][2] + self.foot["right"][2])
        return self.z_c + max(0.0, mean_foot_z - self._foot_z_ref)

    # ----------------------------------------------------------- swing motion
    def _swing_pose(self, frac):
        s = _quintic(frac)
        xy = (1 - s) * self.swing_start[:2] + s * self.swing_goal[:2]
        z0, zg = self.swing_start[2], self.swing_goal[2]
        rise = max(0.0, zg - z0)
        arch = (self.step_height + rise) * np.sin(np.pi * np.clip(frac, 0.0, 1.0))
        z = (1 - s) * z0 + s * zg + arch
        return np.array([xy[0], xy[1], z])

    # ------------------------------------------------------- footstep planning
    def _nominal_next(self, stance):
        """Nominal landing (world xy) of the foot that swings while `stance` is
        planted: kept at +-step_width/2 laterally, advanced step_length forward
        from the current stance foot."""
        swing = "right" if stance == "left" else "left"
        side = +1.0 if swing == "left" else -1.0      # left foot at +y
        x = self._foot_center_xy(self.foot[stance])[0] + self.step_length
        y = side * (self.step_width / 2.0)
        return np.array([x, y])

    def _future_footsteps(self, stance_idx, p_st_xy, n):
        """The next `n` footstep ZMPs after the current stance foot, at ABSOLUTE
        positions: x = x0 + k*step_length, y = +-d (alternating by parity).

        Absolute anchoring is essential: it pins the forward walking speed and the
        stance width so the gait cannot drift or run away. (Coupling the footstep
        to the measured DCM instead creates positive feedback through the DCM
        reference and accelerates the CoM -- so foot placement is NOT used to set
        forward speed here; the speed is the plan's, and lateral balance uses the
        DCM step adjustment.)"""
        d = self.step_width / 2.0
        steps = []
        for i in range(1, n + 1):
            k = stance_idx + i
            steps.append(np.array([self.x0 + k * self.step_length,
                                   d if (k % 2 == 0) else -d]))
        return steps

    # --------------------------------------------------------- DCM references
    def _xi_eos_current(self, p_st_xy, future):
        """End-of-step DCM for the CURRENT step, from a backward recursion over the
        preview footsteps. This is THE fix for lateral drift: it makes the per-step
        reference DCM continuous (xi_eos of one step == xi_ini of the next), which
        for a periodic gait converges to the classic +-d*tanh(omega*T/2) sway.

            xi_eos,N = p_N                      (rest on the last previewed foot)
            xi_ini,i = p_i + (xi_eos,i - p_i) e^{-omega T}
            xi_eos,i-1 = xi_ini,i
        """
        E = np.exp(-self.omega * self.t_ss)
        seq = [p_st_xy] + list(future)          # seq[0] = current stance ZMP
        xi = seq[-1].copy()                     # terminal DCM = last footstep
        for i in range(len(seq) - 1, 0, -1):    # i = last .. 1
            xi = seq[i] + (xi - seq[i]) * E      # xi_ini,i == xi_eos,{i-1}
        return xi                                # xi_eos for the current step

    def _current_xi_eos(self):
        """(p_stance_xy, nominal_next_xy, xi_eos) for the current stance foot.

        The stance ZMP used for planning is LATERALLY ANCHORED to the ideal
        centreline foothold (y = +-d by parity) rather than the actual, possibly
        drifted, foot. This keeps the reference DCM / CoM trajectory pinned to the
        straight centreline so the gait does not slowly veer sideways and cross its
        legs. The forward (x) coordinate still tracks the actual foot so forward
        progress accumulates."""
        p_act = self._foot_center_xy(self.foot[self.stance])
        d = self.step_width / 2.0
        ideal_y = d if (self.stance_idx % 2 == 0) else -d
        p_st = np.array([p_act[0], ideal_y])      # lateral-anchored stance
        future = self._future_footsteps(self.stance_idx, p_st, self.n_preview)
        xi_eos = self._xi_eos_current(p_st, future)
        return p_st, future[0], xi_eos

    def _xi_ref(self, p_stance, xi_eos, tau):
        """Reference DCM during the current step: p_stance -> xi_eos."""
        return p_stance + (xi_eos - p_stance) * np.exp(self.omega * (tau - self.t_ss))

    def _xi_ini(self, p_stance, xi_eos):
        """Initial DCM of the current step (value of _xi_ref at tau=0)."""
        return p_stance + (xi_eos - p_stance) * np.exp(-self.omega * self.t_ss)

    def _capture_adjust(self, p_nominal, p_st, xi_eos_plan, com, com_vel, tau):
        """DCM step adjustment (Englsberger): correct the nominal footstep by the
        DCM tracking error so the post-step DCM rejoins the plan. Anchoring to the
        nominal footstep (which is absolute) keeps x from drifting, while the error
        term provides the lateral catch that prevents a sideways fall:

            xi_pred_eos = p_st + (xi_meas - p_st) exp(omega (T_ss - tau))
            dy_corr     = clip( k_cap (xi_pred_eos.y - xi_eos_plan.y), +-max_dy )
            p_next.y    = p_nominal.y + dy_corr            (p_nominal.y is ABSOLUTE)

        k_cap in [0,1] (0 = open-loop plan, 1 = full correction). The correction is
        CLAMPED to +-max_dy of the absolute nominal foothold: without the clamp a
        sideways CoM excursion makes the foot chase the absolute DCM (positive
        feedback), so the whole gait slowly walks sideways, the feet converge to
        the centreline and cross, and it tips. Clamping pins the footholds to the
        centreline-anchored +-d pattern while still allowing a bounded lateral
        catch. The lateral side is also clamped so the feet never cross."""
        xi = com[:2] + com_vel[:2] / self.omega
        xi_pred_eos = p_st + (xi - p_st) * np.exp(self.omega * (self.t_ss - tau))
        p_next = p_nominal.copy()
        # Forward (x): bounded sagittal correction so the robot adapts to slopes
        # and velocity disturbances rather than blindly following the absolute plan.
        dx = np.clip(self.k_cap * (xi_pred_eos[0] - xi_eos_plan[0]),
                     -self.max_dx, self.max_dx)
        p_next[0] = p_nominal[0] + dx
        # Lateral (y): bounded DCM step adjustment about the absolute foothold,
        # plus a slow centreline-restoring bias. The capture term gives the local
        # balance catch; the restoring term (-k_center * com_y) walks the footholds
        # back toward y=0 whenever the body has drifted sideways, so the gait holds
        # a straight line instead of slowly veering off and crossing its legs.
        dy = np.clip(self.k_cap * (xi_pred_eos[1] - xi_eos_plan[1]),
                     -self.max_dy, self.max_dy)
        p_next[1] = p_nominal[1] + dy - self.k_center * com[1]
        swing = "right" if self.stance == "left" else "left"
        if swing == "left":
            p_next[1] = max(p_next[1], p_st[1] + 0.08)
        else:
            p_next[1] = min(p_next[1], p_st[1] - 0.08)
        return p_next

    def _build_refs(self, com, com_vel, p_ref, xi_ref, contacts, swing, dt, z_des):
        """Integrate the DCM-consistent reference CoM and pack the QP references.

        The reference CoM follows the reference DCM and obeys the LIPM:

            c_ref_dot  = omega (xi_ref - c_ref)
            c_ref_ddot = omega^2 (c_ref - p_ref)

        We hand the QP a CoM position+velocity reference (c_ref, c_ref_dot) plus the
        acceleration feed-forward c_ref_ddot. The QP's CoM PD then actively servos
        the real CoM onto c_ref -- this position regulation removes the slow drift
        that pure acceleration feed-forward could not hold. Lateral balance is
        supplied by capture-point foot placement. A DCM-feedback ZMP (p_cmd) is
        also computed for diagnostics.
        """
        cd_ref = self.omega * (xi_ref - self.c_ref)
        self.c_ref = self.c_ref + dt * cd_ref
        cdd_ref = self.omega**2 * (self.c_ref - p_ref)

        refs = {
            "com_des": np.array([self.c_ref[0], self.c_ref[1], z_des]),
            "com_vel_des": np.array([cd_ref[0], cd_ref[1], 0.0]),
            "com_acc_ff": np.array([cdd_ref[0], cdd_ref[1], 0.0]),
            "torso_R": self.torso_R,
            "contacts": contacts,
            "swing": swing,
        }
        xi = com[:2] + com_vel[:2] / self.omega
        p_cmd = self._clamp_zmp(
            p_ref + (1.0 + self.k_dcm / self.omega) * (xi - xi_ref), contacts)
        return refs, xi, p_cmd

    # ------------------------------------------------------------------ update
    def update(self, state, dt):
        """Advance the FSM and return (refs, info).

        refs: dict with com_des(3), com_vel_des(3), com_acc_ff(3), torso_R,
              contacts(list), swing(dict|None) -- consumed by WholeBodyQP.compute.
        """
        assert self._init, "call reset() first"
        self.t_state += dt
        com = state["com"]
        com_vel = state["com_vel"]
        rpy = state["base_rpy"]

        z_des = self._com_z()

        # ---- safety: a big tilt means we lost it; stop walking and try to stand
        tilt = max(abs(rpy[0]), abs(rpy[1]))
        if self.walking and tilt > self.max_tilt and self.state != self.STAND:
            Logger.warning(f"[dcm] abort {self.state}: tilt={tilt:.2f} -> hold stand")
            self.walking = False
            self._enter(self.STAND)

        # ===================== STAND =====================
        if self.state == self.STAND:
            mid = 0.5 * (self._foot_center_xy(self.foot["left"]) +
                         self._foot_center_xy(self.foot["right"]))
            contacts = ["left", "right"]
            # constant DCM reference (mid) -> consistent reference ZMP is mid
            refs, xi, p_cmd = self._build_refs(com, com_vel, mid, mid,
                                               contacts, None, dt, z_des)
            if self.walking and self.t_state > self.t_stand:
                self._begin_first_step()
            return refs, self._info(p_cmd, xi, mid)

        # ===================== START (initial weight shift) =====================
        if self.state == self.START:
            # Both feet down. Drive the CoM/DCM from the centre onto the first
            # stance foot, ending at this step's initial DCM, so single support
            # begins on the correct side (otherwise the very first lift topples
            # the robot toward the unsupported side).
            contacts = ["left", "right"]
            p_st, nominal_next, xi_eos = self._current_xi_eos()
            self.p_next = nominal_next
            mid = 0.5 * (self._foot_center_xy(self.foot["left"]) +
                         self._foot_center_xy(self.foot["right"]))
            xi_goal = self._xi_ini(p_st, xi_eos)
            u = self.t_state / self.t_start
            s = _quintic(u)
            xi_ref = (1 - s) * mid + s * xi_goal
            # consistent reference ZMP for a moving DCM ref: p_ref = xi_ref - xi_ref_dot/omega
            xi_ref_dot = (_quintic_deriv(u) / self.t_start) * (xi_goal - mid)
            p_ref = xi_ref - xi_ref_dot / self.omega
            refs, xi, p_cmd = self._build_refs(com, com_vel, p_ref, xi_ref,
                                               contacts, None, dt, z_des)
            if self.t_state >= self.t_start:
                self.swing_start = self.foot["right" if self.stance == "left"
                                             else "left"].copy()
                self._enter(self.SS)
            return refs, self._info(p_cmd, xi, xi_ref)

        # ===================== DS (weight transfer) =====================
        if self.state == self.DS:
            # Hold the reference DCM at the upcoming single-support initial DCM so
            # the swing begins consistently with the limit cycle.
            contacts = ["left", "right"]
            p_st, nominal_next, xi_eos = self._current_xi_eos()
            self.p_next = nominal_next
            xi_ref = self._xi_ini(p_st, xi_eos)
            # holding a constant DCM target -> consistent reference ZMP is xi_ref
            refs, xi, p_cmd = self._build_refs(com, com_vel, xi_ref, xi_ref,
                                               contacts, None, dt, z_des)
            if self.t_state >= self.t_ds:
                self._enter(self.SS)
            return refs, self._info(p_cmd, xi, xi_ref)

        # ===================== SS (single support + swing) =====================
        if self.state == self.SS:
            swing = "right" if self.stance == "left" else "left"
            contacts = [self.stance]
            tau = self.t_state
            frac = tau / self.t_ss
            p_st, nominal_next, xi_eos = self._current_xi_eos()

            # reference DCM follows the continuous preview plan; the physical
            # landing is the nominal (absolute) footstep, corrected laterally by
            # the DCM tracking error for disturbance rejection (k_cap)
            self.p_next = self._capture_adjust(nominal_next, p_st, xi_eos,
                                               com, com_vel, tau)
            self.swing_goal = np.array([self.p_next[0], self.p_next[1],
                                        self.foot[self.stance][2]])
            xi_ref = self._xi_ref(p_st, xi_eos, tau)
            swing_pos = self._swing_pose(frac)
            sw = {"foot": swing, "pos": swing_pos, "vel": np.zeros(3),
                  "R": self.foot_R}      # keep the swing foot level & forward
            refs, xi, p_cmd = self._build_refs(com, com_vel, p_st, xi_ref,
                                               contacts, sw, dt, z_des)

            sw_contact = state[f"{swing[0]}foot_contact"]
            if frac >= 1.0 or (frac > 0.6 and sw_contact):
                # plant the swung foot where it landed and switch stance
                self.foot[swing] = np.array([self.swing_goal[0], self.swing_goal[1],
                                             self.foot[self.stance][2]])
                self.step_count += 1
                if self.max_steps is not None and self.step_count >= self.max_steps:
                    self.walking = False
                    self._enter(self.STAND)
                else:
                    self.stance = swing          # the foot that just landed
                    self.stance_idx += 1         # advance the absolute footstep index
                    self._begin_step()
            return refs, self._info(p_cmd, xi, xi_ref, swing_pos)

        # fallback (should not be reached)
        mid = 0.5 * (self._foot_center_xy(self.foot["left"]) +
                     self._foot_center_xy(self.foot["right"]))
        refs, xi, p_cmd = self._build_refs(com, com_vel, mid, mid,
                                           ["left", "right"], None, dt, z_des)
        return refs, self._info(p_cmd, xi, mid)

    # ----------------------------------------------------------- step helpers
    def _begin_first_step(self):
        """Set up the very first step: plan the nominal landing and run the
        start-up weight shift (START) before lifting."""
        self.p_next = self._nominal_next(self.stance)
        self._enter(self.START)

    def _begin_step(self):
        """Set up a subsequent step: latch swing start, plan landing, enter DS.
        After the first step the CoM has already swayed over the new stance foot,
        so only a short double-support transfer is needed."""
        swing = "right" if self.stance == "left" else "left"
        self.swing_start = self.foot[swing].copy()
        self.p_next = self._nominal_next(self.stance)
        self._enter(self.DS)

    def _enter(self, new_state):
        Logger.debug(f"[DCM-FSM] {self.state} -> {new_state} "
                     f"(t={self.t_state:.2f} step={self.step_count} stance={self.stance})")
        self.state = new_state
        self.t_state = 0.0

    # ----------------------------------------------------------- ref builders
    def _info(self, p_cmd, xi, xi_ref, swing_pos=None):
        return {
            "state": self.state,
            "step_count": self.step_count,
            "stance": self.stance,
            "zmp_cmd": np.asarray(p_cmd).copy(),
            "dcm": np.asarray(xi).copy(),
            "dcm_ref": np.asarray(xi_ref).copy(),
            "swing_target": swing_pos,
        }
