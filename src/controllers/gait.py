"""
Gait generation: Layers 2-6 and 10.

  Layer 2  Finite State Machine    (STAND / SHIFT / SWING / LAND per side)
  Layer 3  CoM planner             (move CoM over support foot + balance fb)
  Layer 4  Swing-foot planner      (quintic xy + raised-arch z trajectory)
  Layer 5  Footstep planner        (alternating steps, step length/width)
  Layer 6  Raibert foot placement  (velocity / capture-point feedback)
  Layer 10 Balance recovery        (abort swing, foot down, back to double)

The GaitController is a pure planner: given the estimated state it emits, every
tick, the task `targets` dict consumed by the whole-body IK. It does not touch
physics or simulation state.

Phase cycle (a full stride = two steps):
    STAND -> SHIFT_COM_LEFT -> RIGHT_SWING -> RIGHT_LAND
          -> SHIFT_COM_RIGHT -> LEFT_SWING  -> LEFT_LAND  -> (repeat)

"SHIFT_COM_LEFT" shifts the CoM onto the LEFT foot so the RIGHT foot can swing.
"""

import numpy as np

from src.controllers.robot_model import RobotModel, StateEstimator
from src.utils.terminal_logger import TerminalLogger as Logger


def _quintic(s):
    """Minimum-jerk scalar interpolation s in [0,1] -> [0,1] (zero vel/acc ends)."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class GaitController:
    # FSM states
    STAND = "STAND"
    SHIFT_COM_LEFT = "SHIFT_COM_LEFT"
    RIGHT_SWING = "RIGHT_SWING"
    RIGHT_LAND = "RIGHT_LAND"
    SHIFT_COM_RIGHT = "SHIFT_COM_RIGHT"
    LEFT_SWING = "LEFT_SWING"
    LEFT_LAND = "LEFT_LAND"
    RECOVER = "RECOVER"

    def __init__(self, robot: RobotModel, params: dict):
        self.r = robot
        p = params
        # timing
        self.t_stand = p.get("t_stand", 1.5)
        self.t_shift = p.get("t_shift", 2.2)      # slow quasi-static weight shift (WBC)
        self.t_swing = p.get("step_duration", 1.0)
        self.t_land = p.get("t_land", 0.45)
        # footstep geometry
        self.step_length = p.get("step_length", 0.03)
        self.step_height = p.get("step_height", 0.05)
        self.stance_width = p.get("stance_width", 0.237)
        # Walk this many alternating steps, then hold a stable stand forever.
        # Beyond ~4 steps the marginal lateral balance accumulates enough error
        # to tip, so the safe default finishes the demonstration on both feet.
        self.max_steps = p.get("max_steps", 20)
        self.com_height = p.get("com_height", None)   # set at reset if None
        # Terrain awareness (set by WalkingController): terrain_fn(x, y) -> ground z.
        # When present, footholds are placed on the sensed surface, the swing foot
        # clears the step edge, and the CoM rides at a fixed clearance above the
        # mean foot height (so it rises while climbing). Falls back to flat if None.
        self.terrain_fn = None
        self.com_clearance = None      # CoM height above mean foot z; set at reset
        self.step_clearance = p.get("step_clearance", 0.04)  # extra swing lift over step
        # CoM planner feedback (Layer 3) and Raibert (Layer 6)
        # CoM balance feedback injected into the IK reference target. Because the
        # stiff joint PD faithfully tracks the IK reference, shifting this target
        # in response to the *measured* CoM error/velocity actively stabilises the
        # floating base (it is the primary active-balance loop here).
        self.kp_com = p.get("kp_com", 1.6)
        self.kd_com = p.get("kd_com", 0.50)
        self.kp_com_swing = p.get("kp_com_swing", 1.6)   # single-support CoM hold
        self.kd_com_swing = p.get("kd_com_swing", 0.50)
        self.kp_com_land = p.get("kp_com_land", 1.6)     # recenter gain in double support
        # CoM rides slightly behind the foot geometric centre (toward the ankle),
        # which is the naturally stable balance point for this robot.
        self.com_back_offset = p.get("com_back_offset", 0.018)
        self.v_des = np.array(p.get("v_des", [0.0, 0.0]))
        self.kv_raibert = p.get("kv_raibert", 0.10)
        self.kv_raibert_y = p.get("kv_raibert_y", 0.08)
        self.com_shift_frac = p.get("com_shift_frac", 0.9)   # how far toward foot center
        self.v_settle = p.get("v_settle", 0.13)              # CoM speed gate before lift-off
        self.land_press = p.get("land_press", 0.015)         # press swing foot into floor
        # "Square up" criteria: before lifting the next foot the robot must stand
        # on BOTH feet, flat and level, with weight shared and the CoM centred and
        # still. Restarting every step from this clean stance stops error from
        # accumulating stride-to-stride, which is what lets it take many steps.
        self.load_balance = p.get("load_balance", 140.0)     # max |F_L - F_R| (N)
        self.tilt_settle = p.get("tilt_settle", 0.06)        # max torso roll/pitch (rad)
        self.center_tol = p.get("center_tol", 0.04)          # max CoM-y off stance mid (m)
        self.settle_timeout = p.get("settle_timeout", 4.0)   # s before forcing the next step
        # balance limits (Layer 10)
        # Abort threshold for balance recovery. Normal single support peaks around
        # 0.13 rad; 0.30 catches a developing topple without cutting good steps short.
        self.max_tilt = p.get("max_tilt", 0.30)              # rad roll/pitch abort
        # Outward CoM excursion past the stance-foot centre that triggers an early
        # capture-to-stand. Normal swings keep the CoM at/inside the foot centre
        # (outward<=0), so 0.02 fires only on a genuine sideways loss of balance,
        # while the foot edge is still ~0.01 m away and the catch can succeed.
        self.capture_margin = p.get("capture_margin", 0.02)
        # Max settled torso tilt at which it is still safe to START a new step.
        # Above this the robot stops walking and holds the stand (prevents
        # attempting a step that would tip over the stance foot).
        self.safe_step_tilt = p.get("safe_step_tilt", 0.08)
        self.support_margin = p.get("support_margin", 0.04)  # m outside polygon abort

        # runtime
        self.state = self.STAND
        self.t_state = 0.0
        self.step_count = 0
        self._initialised = False

    # ------------------------------------------------------------------ reset
    def reset(self, state):
        """Latch the initial planted foot poses and nominal CoM from estimate."""
        self.lfoot_pos0 = state["lfoot_pos"].copy()
        self.rfoot_pos0 = state["rfoot_pos"].copy()
        self.foot_R = state["lfoot_mat"].copy()          # level foot orientation
        self.torso_R = np.eye(3)                          # keep pelvis upright
        # planted (held) targets for each foot, updated on touchdown
        self.plant = {"left": self.lfoot_pos0.copy(), "right": self.rfoot_pos0.copy()}
        if self.com_height is None:
            self.com_height = state["com"][2]
        # Reference foot height; the CoM target only rises above the nominal once
        # the mean foot height climbs above this (so flat walking is unchanged).
        self._foot_z_ref = 0.5 * (self.lfoot_pos0[2] + self.rfoot_pos0[2])
        self.com_clearance = self.com_height - self._foot_z_ref
        # Tracked ground level the robot is currently walking on (sole height).
        # Used to detect genuine steps without being fooled by land_press drift.
        self._current_ground = self._foot_z_ref + self.r.sole_offset_z
        self.state = self.STAND
        self.t_state = 0.0
        self.step_count = 0
        self.walking = True          # cleared after max_steps or a recovery -> hold stand
        self.swing_start = None
        self.swing_goal = None
        self._initialised = True
        Logger.debug(f"GaitController reset: com_h={self.com_height:.3f} "
                     f"Lfoot={self.lfoot_pos0[:2]} Rfoot={self.rfoot_pos0[:2]}")

    # ------------------------------------------------------------ foot center
    def _foot_center_xy(self, ankle_pos):
        """CoM is balanced over the foot *centre*, ~foot_center_x ahead of ankle."""
        return np.array([ankle_pos[0] + self.r.foot_center_x, ankle_pos[1]])

    # ----------------------------------------------------------- footstep plan
    def _plan_landing(self, support_side, com_vel):
        """Layer 5 + 6: where the swing foot should land (ankle xy)."""
        sup = self.plant["left"] if support_side == "left" else self.plant["right"]
        swing_side = "right" if support_side == "left" else "left"
        side_sign = +1.0 if swing_side == "left" else -1.0

        # nominal alternating footstep relative to the support foot
        x = sup[0] + self.step_length
        y = sup[1] + side_sign * self.stance_width

        # Raibert / capture-point feedback:  x = x_hip + T/2 v + kv (v - v_des).
        # A gentle lateral capture term nudges the landing outward when the CoM
        # is drifting sideways (kept small: an aggressive lateral capture over-
        # places the foot using the large weight-shift velocity and destabilises).
        x += 0.5 * self.t_swing * com_vel[0] + self.kv_raibert * (com_vel[0] - self.v_des[0])
        y += self.kv_raibert_y * (com_vel[1] - self.v_des[1])
        # never cross the support foot laterally (keep a sane stance)
        if swing_side == "left":
            y = max(y, sup[1] + 0.12)
        else:
            y = min(y, sup[1] - 0.12)

        # Terrain-aware foothold height: sense the surface under (x, y). Compare to
        # the tracked current ground level (NOT the support foot, whose target z
        # drifts with land_press); only when it is a genuine step do we re-target
        # the foot onto the new surface. On flat ground the foothold is the exact
        # proven value (support ankle z), so flat walking is unchanged.
        z = sup[2]
        if self.terrain_fn is not None:
            ground = self.terrain_fn(x, y)
            if abs(ground - self._current_ground) > 0.02:   # a genuine step up/down
                z = ground - self.r.sole_offset_z           # sole rests on the tread
                self._current_ground = ground               # we are now on this level
        return np.array([x, y, z])

    # ------------------------------------------------------------- CoM target
    def _com_target(self, nominal_xy, com, com_vel, kp=None, kd=None):
        """Layer 3: nominal CoM with balance feedback.

        The whole-body IK already applies a proportional pull k_com*(target-com)
        on the actual CoM. Here the joint PD tracks the IK reference stiffly, so
        the reference IS the actuation: nudging the IK CoM target against the
        measured CoM error (kp_com) and velocity (kd_com) closes an active
        balance loop -- when the body sways, the target counter-shifts and the
        legs are commanded to push the CoM back.
        """
        kp = self.kp_com if kp is None else kp
        kd = self.kd_com if kd is None else kd
        nominal_xy = np.array([nominal_xy[0] - self.com_back_offset, nominal_xy[1]])
        err = nominal_xy - com[:2]
        corrected = nominal_xy + kp * err - kd * com_vel[:2]
        return np.array([corrected[0], corrected[1], self._com_z()])

    def _com_z(self):
        """CoM height target. On flat ground it is EXACTLY the nominal height; it
        only rises (never drops) as the feet climb above their starting level, so
        flat walking is unchanged and the body lifts while ascending stairs."""
        mean_foot_z = 0.5 * (self.plant["left"][2] + self.plant["right"][2])
        rise = mean_foot_z - self._foot_z_ref
        return self.com_height + max(0.0, rise)

    # ----------------------------------------------------------- swing motion
    def _swing_pose(self, frac):
        """Layer 4: quintic xy blend + symmetric raised z arch.

        The foot descends to slightly BELOW the nominal ground ankle height
        (land_press) so it positively makes and loads contact rather than
        hovering a millimetre above the floor."""
        s = _quintic(frac)
        xy = (1 - s) * self.swing_start[:2] + s * self.swing_goal[:2]
        z0 = self.swing_start[2]
        zg = self.swing_goal[2] - self.land_press
        # On flat ground (rise == 0) the arch is exactly step_height -> identical
        # to the flat-walk gait. When stepping UP, add extra lift (the rise plus a
        # clearance) so the foot passes over the step edge/riser.
        rise = max(0.0, zg - z0)
        extra = (rise + self.step_clearance) if rise > 1e-4 else 0.0
        arch = (self.step_height + extra) * np.sin(np.pi * np.clip(frac, 0.0, 1.0))
        z = (1 - s) * z0 + s * zg + arch
        return np.array([xy[0], xy[1], z])

    # --------------------------------------------------------- support polygon
    def _com_outside_support(self, support_side, com):
        """True if the CoM ground projection leaves the support foot + margin."""
        ank = self.plant[support_side]
        dx = com[0] - ank[0]
        dy = com[1] - ank[1]
        in_x = (-self.r.foot_len_back - self.support_margin) <= dx <= (self.r.foot_len_fwd + self.support_margin)
        in_y = abs(dy) <= (self.r.foot_half_width + self.support_margin)
        return not (in_x and in_y)

    # ------------------------------------------------------------------ update
    def update(self, state, dt):
        """Advance the FSM and return (targets, info)."""
        assert self._initialised, "call reset() first"
        self.t_state += dt
        com = state["com"]
        com_vel = state["com_vel"]
        rpy = state["base_rpy"]

        # ---- Layer 10: instability check (active in every walking state) ----
        single = self.state in (self.RIGHT_SWING, self.LEFT_SWING)
        active = self.state not in (self.STAND, self.RECOVER)
        if active:
            tilt = max(abs(rpy[0]), abs(rpy[1]))
            # Early lateral-capture trigger: during single support, if the CoM is
            # drifting OUTWARD toward the stance-foot edge, catch it *before* it
            # passes the edge (a sideways fall over the stance foot is otherwise
            # unrecoverable). outward>0 means the CoM is past the foot centre on
            # the side away from the body, heading off the edge.
            outward = 0.0
            if single:
                sup_side = "left" if self.state == self.RIGHT_SWING else "right"
                sup_y = self.plant[sup_side][1]
                outward = (com[1] - sup_y) if sup_side == "left" else (sup_y - com[1])
            edge = single and outward > self.capture_margin
            out = single and self._com_outside_support(
                "left" if self.state == self.RIGHT_SWING else "right", com)
            if tilt > self.max_tilt or out or edge:
                why = "edge" if edge else ("tilt" if tilt > self.max_tilt else "polygon")
                Logger.warning(f"[recovery] abort {self.state} ({why}): tilt={tilt:.2f} "
                               f"com=({com[0]:.2f},{com[1]:.2f}) outward={outward:.3f}")
                if single:
                    # Plant the swung foot where it currently is (projected to the
                    # floor) to widen the base and catch the fall.
                    support_side = "left" if self.state == self.RIGHT_SWING else "right"
                    swing_side = "right" if support_side == "left" else "left"
                    cur = state[f"{swing_side[0]}foot_pos"]
                    self.plant[swing_side] = np.array([cur[0], cur[1],
                                                       self.plant[support_side][2] - self.land_press])
                self.walking = False     # after a recovery, settle and hold stand
                self._enter(self.RECOVER)

        # default: support = current double-support assumption
        support_side = "left"
        swing_pos = None

        # -------------------------------- FSM ---------------------------------
        if self.state == self.STAND:
            nominal = 0.5 * (self._foot_center_xy(self.plant["left"]) +
                             self._foot_center_xy(self.plant["right"]))
            support_side = "left"  # symmetric; support target uses both via posture
            com_t = self._com_target(nominal, com, com_vel)
            sup_pos = self.plant["left"]
            # both feet planted -> emulate double support by also pinning right via swing task off
            targets = self._double_targets(com_t)
            # STAND is terminal once walking is finished (or after a recovery):
            # the robot holds this rock-solid balanced stance indefinitely
            # instead of re-entering the (eventually destabilising) step cycle.
            if self.walking and self.t_state > self.t_stand:
                self._enter(self.SHIFT_COM_LEFT)
            return targets, self._info(com_t)

        if self.state in (self.SHIFT_COM_LEFT, self.SHIFT_COM_RIGHT):
            support_side = "left" if self.state == self.SHIFT_COM_LEFT else "right"
            foot_c = self._foot_center_xy(self.plant[support_side])
            mid = 0.5 * (self._foot_center_xy(self.plant["left"]) +
                         self._foot_center_xy(self.plant["right"]))
            s = _quintic(self.t_state / self.t_shift)
            nominal = (1 - s) * mid + s * (mid + self.com_shift_frac * (foot_c - mid))
            com_t = self._com_target(nominal, com, com_vel)
            targets = self._double_targets(com_t)
            # Transition only once the CoM is over the support foot AND has
            # (nearly) stopped moving sideways: lifting the foot while lateral
            # CoM velocity is high overshoots the narrow foot and tips the robot.
            # A 2x dwell timeout prevents deadlock if it never fully settles.
            over = abs(com[1] - self.plant[support_side][1]) < (self.r.foot_half_width + 0.02)
            slow = abs(com_vel[1]) < self.v_settle
            if self.t_state > self.t_shift and ((over and slow) or self.t_state > 2.0 * self.t_shift):
                self.swing_start = (self.plant["right"] if support_side == "left"
                                    else self.plant["left"]).copy()
                self.swing_goal = self._plan_landing(support_side, com_vel)
                self._enter(self.RIGHT_SWING if support_side == "left" else self.LEFT_SWING)
            return targets, self._info(com_t)

        if self.state in (self.RIGHT_SWING, self.LEFT_SWING):
            support_side = "left" if self.state == self.RIGHT_SWING else "right"
            swing_side = "right" if support_side == "left" else "left"
            frac = self.t_state / self.t_swing
            foot_c = self._foot_center_xy(self.plant[support_side])
            # Single support: stronger CoM feedback (no weight-shift to oscillate)
            # to actively hold the CoM over the stance foot and resist the
            # lateral inverted-pendulum drift that otherwise accumulates.
            com_t = self._com_target(foot_c, com, com_vel,
                                     kp=self.kp_com_swing, kd=self.kd_com_swing)
            swing_pos = self._swing_pose(frac)
            targets = self._single_targets(support_side, com_t, swing_pos)
            # touchdown: trajectory done, or early ground contact in second half
            sw_contact = state[f"{swing_side[0]}foot_contact"]
            if frac >= 1.0 or (frac > 0.6 and sw_contact):
                # Plant the foot flat and level at the SENSED landing height
                # (the tread it stepped onto), pressed slightly in to load it.
                self.plant[swing_side] = np.array([self.swing_goal[0], self.swing_goal[1],
                                                   self.swing_goal[2] - self.land_press])
                self._enter(self.RIGHT_LAND if support_side == "left" else self.LEFT_LAND)
            return targets, self._info(com_t, swing_pos)

        if self.state in (self.RIGHT_LAND, self.LEFT_LAND):
            # The previous support foot keeps the weight until the just-landed
            # foot actually loads; recentring the CoM before both feet share load
            # would tip the robot over the still-single support.
            prev_support = "left" if self.state == self.RIGHT_LAND else "right"
            mid = 0.5 * (self._foot_center_xy(self.plant["left"]) +
                         self._foot_center_xy(self.plant["right"]))
            fL, fR = state["lfoot_force"], state["rfoot_force"]
            both = state["support"] == StateEstimator.DOUBLE
            nominal = mid if both else self._foot_center_xy(self.plant[prev_support])
            # Recenter briskly once both feet are down so weight is shared quickly
            # and the next step starts from a clean, balanced stance.
            kp_land = self.kp_com_land if both else self.kp_com
            com_t = self._com_target(nominal, com, com_vel, kp=kp_land)
            targets = self._double_targets(com_t)
            # Square-up gate: both feet loaded and the CoM (nearly) stopped before
            # the next step. (Weight-share / level / centred are tracked for
            # diagnostics; requiring all of them made the robot dwell too long on
            # one foot and drift, so the binding conditions are contact + settle.)
            balanced = both and abs(fL - fR) < self.load_balance
            level = max(abs(rpy[0]), abs(rpy[1])) < self.tilt_settle
            centered = abs(com[1] - mid[1]) < self.center_tol
            slow = np.linalg.norm(com_vel[:2]) < self.v_settle
            ready = both and slow and self.t_state > self.t_land
            if ready or self.t_state > 3.0 * self.t_land:
                self.step_count += 1
                # Preventive safety gate: only start another step if the robot is
                # squared up well enough to do it safely. A sideways fall over the
                # stance foot cannot be caught once it starts, so if residual tilt
                # is already high we STOP walking and hold the stable stand instead
                # of attempting a step that would topple.
                unsafe = max(abs(rpy[0]), abs(rpy[1])) > self.safe_step_tilt
                if self.step_count >= self.max_steps or unsafe:
                    if unsafe:
                        Logger.warning(f"[safety] residual tilt {max(abs(rpy[0]),abs(rpy[1])):.2f}"
                                       f" > {self.safe_step_tilt}: holding stand after "
                                       f"{self.step_count} steps instead of risking a fall.")
                    self.walking = False
                    self._enter(self.STAND)
                elif self.state == self.RIGHT_LAND:
                    self._enter(self.SHIFT_COM_RIGHT)
                else:
                    self._enter(self.SHIFT_COM_LEFT)
            return targets, self._info(com_t)

        # RECOVER: drop both feet to the floor and bring CoM between them
        nominal = 0.5 * (self._foot_center_xy(self.plant["left"]) +
                         self._foot_center_xy(self.plant["right"]))
        com_t = self._com_target(nominal, com, com_vel)
        targets = self._double_targets(com_t)
        if self.t_state > self.t_land and state["support"] == StateEstimator.DOUBLE:
            self._enter(self.STAND)
        return targets, self._info(com_t)

    # ----------------------------------------------------------- target builders
    def _double_targets(self, com_t):
        """Double support: pin BOTH feet at equal (support) weight. The `double`
        flag tells the IK to treat the 'swing' foot as a second support so there
        is no left/right bias while the weight is being shifted."""
        return {
            "com": com_t,
            "torso_R": self.torso_R,
            "support": "left",
            "support_pos": self.plant["left"],
            "support_R": self.foot_R,
            "swing_pos": self.plant["right"],
            "swing_R": self.foot_R,
            "double": True,
        }

    def _single_targets(self, support_side, com_t, swing_pos):
        return {
            "com": com_t,
            "torso_R": self.torso_R,
            "support": support_side,
            "support_pos": self.plant[support_side],
            "support_R": self.foot_R,
            "swing_pos": swing_pos,
            "swing_R": self.foot_R,
        }

    def _enter(self, new_state):
        Logger.debug(f"[FSM] {self.state} -> {new_state} (t={self.t_state:.2f})")
        self.state = new_state
        self.t_state = 0.0

    def _info(self, com_t, swing_pos=None):
        return {
            "state": self.state,
            "t_state": self.t_state,
            "step_count": self.step_count,
            "com_target": com_t,
            "swing_target": swing_pos,
        }
