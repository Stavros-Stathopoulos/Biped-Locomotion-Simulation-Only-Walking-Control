"""
fall_recovery_controller.py — Active Balance Recovery for Unitree G1.

Architecture
------------
Sits ABOVE the locomotion controller as a torque-overlay layer.

    locomotion_tau = locomotion_controller.advance_control(v_des)
    final_tau      = recovery.process(locomotion_tau)
    env.data.ctrl[:] = final_tau

The recovery controller never allocates on the hot path. It owns its own
output buffer so it does not mutate the PD controller's internal state.

State machine
-------------
STABLE     → all sensors within normal bounds; recovery overlay = 0
RECOVERING → at least one soft threshold breached; corrective torques added
FALLEN     → hard limit or CoM collapse detected; locomotion torques replaced
             with gravity compensation; latched until reset() is called

Physics conventions (G1 free joint, MuJoCo)
-------------------------------------------
qpos[3:7]  = [qw, qx, qy, qz]  pelvis quaternion (world frame)
qvel[0:3]  = [vx, vy, vz]      pelvis linear velocity (world frame)
qvel[3:6]  = [ωx, ωy, ωz]      pelvis angular velocity (world frame)

Pitch sign: positive = robot tips FORWARD (validated against verify_stance.py)
Roll  sign: positive = robot tips RIGHT
"""

from __future__ import annotations
import math
from enum import IntEnum

import numpy as np
import mujoco


class FallState(IntEnum):
    STABLE     = 0
    RECOVERING = 1
    FALLEN     = 2


class FallRecoveryController:
    """
    Monitors pelvis orientation and CoM state; injects corrective torques
    when the robot begins to fall.

    Corrective strategy
    -------------------
    Pitch (forward/backward): PD on ankle-pitch joints (primary) plus
    hip-pitch joints (secondary at 30 % gain).  A forward tip (positive
    pitch) receives a positive ankle torque — plantarflexion — which
    pushes the toes into the ground and pushes the CoM rearward.

    Roll (lateral): asymmetric ankle-roll torques.  A rightward tip
    (positive roll) receives a positive left-ankle-roll torque and a
    negative right-ankle-roll torque, pushing the pelvis back left.
    Hip-roll joints assist at 30 % gain.

    Parameters
    ----------
    model, data         : MuJoCo model and data handles.
    com_height_nominal  : Expected CoM height during stance (m).
    pitch_soft          : Pitch angle (rad) that triggers RECOVERING.
    pitch_hard          : Pitch angle (rad) that triggers FALLEN (latched).
    roll_soft / hard    : Same for roll axis.
    ang_rate_soft / hard: Angular rate thresholds (rad/s) for pitch and roll.
    com_height_min      : CoM Z below this → FALLEN (robot has collapsed).
    com_speed_max       : |vy| above this triggers RECOVERING.
    Kp_pitch / Kd_pitch : PD gains for pitch recovery (Nm/rad, Nm·s/rad).
    Kp_roll  / Kd_roll  : PD gains for roll recovery.
    stable_settle_ticks : Consecutive ticks within soft bounds before
                          RECOVERING reverts to STABLE.
    """

    # ── Actuator indices — order matches g1_29dof.xml <actuator> block ────────
    _IDX_LHP: int = 0   # left  hip pitch   ±88 Nm
    _IDX_LHR: int = 1   # left  hip roll    ±88 Nm
    _IDX_LAP: int = 4   # left  ankle pitch ±50 Nm
    _IDX_LAR: int = 5   # left  ankle roll  ±50 Nm
    _IDX_RHP: int = 6   # right hip pitch   ±88 Nm
    _IDX_RHR: int = 7   # right hip roll    ±88 Nm
    _IDX_RAP: int = 10  # right ankle pitch ±50 Nm
    _IDX_RAR: int = 11  # right ankle roll  ±50 Nm

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        com_height_nominal: float = 0.66,
        pitch_soft: float = 0.15,
        pitch_hard: float = 0.50,
        roll_soft:  float = 0.12,
        roll_hard:  float = 0.40,
        ang_rate_soft: float = 1.5,
        ang_rate_hard: float = 5.0,
        com_height_min: float = 0.45,
        com_speed_max:  float = 1.5,
        Kp_pitch: float = 200.0,
        Kd_pitch: float = 12.0,
        Kp_roll:  float = 100.0,
        Kd_roll:  float = 8.0,
        stable_settle_ticks: int = 100,
    ) -> None:
        self.model = model
        self.data  = data

        # ── Thresholds ────────────────────────────────────────────────────────
        self._pitch_soft    = pitch_soft
        self._pitch_hard    = pitch_hard
        self._roll_soft     = roll_soft
        self._roll_hard     = roll_hard
        self._ang_rate_soft = ang_rate_soft
        self._ang_rate_hard = ang_rate_hard
        self._com_height_min = com_height_min
        self._com_speed_max  = com_speed_max

        # ── Gains ─────────────────────────────────────────────────────────────
        self._Kp_pitch = Kp_pitch
        self._Kd_pitch = Kd_pitch
        self._Kp_roll  = Kp_roll
        self._Kd_roll  = Kd_roll

        # ── State machine ─────────────────────────────────────────────────────
        self.fall_state: FallState = FallState.STABLE
        self._stable_count: int    = 0
        self._stable_settle: int   = stable_settle_ticks

        # ── Body IDs (resolved once at construction) ───────────────────────────
        self._pelvis_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # ── DOF indices into qvel / qfrc_bias for the 29 actuated joints ──────
        joint_ids      = model.actuator_trnid[:, 0]           # (nu,)
        self._dof_idx  = model.jnt_dofadr[joint_ids].copy()   # into nv-dim arrays

        # ── Actuator force limits from model (mirrors JointPDController) ──────
        frc = model.jnt_actfrcrange[joint_ids]                 # (nu, 2)
        if np.all(frc[:, 1] > frc[:, 0]):
            self._tau_min = frc[:, 0].copy()
            self._tau_max = frc[:, 1].copy()
        else:
            self._tau_min = np.full(model.nu, -np.inf)
            self._tau_max = np.full(model.nu,  np.inf)

        # ── Pre-allocated hot-path buffers (zero allocation contract) ─────────
        self._output_tau   = np.zeros(model.nu, dtype=np.float64)
        self._recovery_tau = np.zeros(model.nu, dtype=np.float64)

        # ── Public monitoring state (updated every process() call) ────────────
        self.pitch: float              = 0.0
        self.roll:  float              = 0.0
        self.pitch_rate: float         = 0.0
        self.roll_rate:  float         = 0.0
        self.com_height: float         = com_height_nominal
        self.com_speed_lateral: float  = 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def process(self, locomotion_tau: np.ndarray) -> np.ndarray:
        """
        Intercept locomotion torques and apply fall-recovery overlay.

        Called once per physics tick (500 Hz hot path).  Zero heap
        allocations: all arithmetic writes into pre-allocated buffers.

        Parameters
        ----------
        locomotion_tau : (nu,) array from BipedLocomotionController.advance_control().
                         Not modified in-place — the PD controller's internal
                         buffer is not mutated.

        Returns
        -------
        (nu,) corrected torque array backed by self._output_tau.
        The returned reference is valid until the next process() call.
        """
        self._read_state()
        self._update_state_machine()

        if self.fall_state == FallState.FALLEN:
            # Robot has collapsed: discard locomotion torques and apply only
            # gravity compensation to hold joints without thrashing.
            np.take(self.data.qfrc_bias, self._dof_idx, out=self._output_tau)
            return self._output_tau

        # Start from locomotion torques (copy avoids mutating the PD buffer)
        np.copyto(self._output_tau, locomotion_tau)

        if self.fall_state == FallState.RECOVERING:
            self._recovery_tau.fill(0.0)
            self._compute_recovery_torques()
            np.add(self._output_tau, self._recovery_tau, out=self._output_tau)

        np.clip(self._output_tau, self._tau_min, self._tau_max, out=self._output_tau)
        return self._output_tau

    def reset(self) -> None:
        """Clear FALLEN latch and return to STABLE (e.g. after robot is righted)."""
        self.fall_state    = FallState.STABLE
        self._stable_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Private — state observation
    # ──────────────────────────────────────────────────────────────────────────

    def _read_state(self) -> None:
        """Read pelvis orientation and CoM kinematics from MuJoCo buffers.

        All accesses are scalar reads from zero-copy MuJoCo array views.
        No NumPy operations here — Python native arithmetic only.
        """
        # Free-joint quaternion at qpos[3:7] = [qw, qx, qy, qz]
        qw = float(self.data.qpos[3])
        qx = float(self.data.qpos[4])
        qy = float(self.data.qpos[5])
        qz = float(self.data.qpos[6])

        # Pitch: positive = robot tips FORWARD
        # Formula validated in verify_stance.py against live MuJoCo telemetry.
        t_pitch = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        self.pitch = math.asin(t_pitch)

        # Roll: positive = robot tips RIGHT (standard ZYX Euler, small-angle valid)
        t_roll = max(-1.0, min(1.0, 2.0 * (qw * qx + qy * qz)))
        self.roll = math.asin(t_roll)

        # Pelvis angular velocity in world frame (qvel[3]=ωx, qvel[4]=ωy)
        self.pitch_rate = float(self.data.qvel[4])
        self.roll_rate  = float(self.data.qvel[3])

        # CoM height from subtree_com (zero-copy view into MuJoCo buffer)
        self.com_height        = float(self.data.subtree_com[self._pelvis_id, 2])
        self.com_speed_lateral = abs(float(self.data.qvel[1]))

    # ──────────────────────────────────────────────────────────────────────────
    # Private — state machine
    # ──────────────────────────────────────────────────────────────────────────

    def _update_state_machine(self) -> None:
        # Hard limits → FALLEN (latched; requires explicit reset() to clear)
        if (abs(self.pitch)      > self._pitch_hard
                or abs(self.roll)       > self._roll_hard
                or abs(self.pitch_rate) > self._ang_rate_hard
                or abs(self.roll_rate)  > self._ang_rate_hard
                or self.com_height      < self._com_height_min):
            self.fall_state    = FallState.FALLEN
            self._stable_count = 0
            return

        # Soft limits → RECOVERING
        is_unstable = (
            abs(self.pitch)      > self._pitch_soft
            or abs(self.roll)       > self._roll_soft
            or abs(self.pitch_rate) > self._ang_rate_soft
            or abs(self.roll_rate)  > self._ang_rate_soft
            or self.com_speed_lateral > self._com_speed_max
        )

        if is_unstable:
            self.fall_state    = FallState.RECOVERING
            self._stable_count = 0
        elif self.fall_state == FallState.RECOVERING:
            # Hysteresis: stay in RECOVERING until stable for _stable_settle ticks
            self._stable_count += 1
            if self._stable_count >= self._stable_settle:
                self.fall_state    = FallState.STABLE
                self._stable_count = 0
        # STABLE → STABLE: nothing to do

    # ──────────────────────────────────────────────────────────────────────────
    # Private — corrective torques
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_recovery_torques(self) -> None:
        """
        Populate self._recovery_tau with PD corrective torques.

        Called only when fall_state == RECOVERING.  self._recovery_tau has
        been zeroed by the caller before this method is invoked.

        Pitch
        -----
        pitch_cmd = Kp·θ_pitch + Kd·ω_pitch

        Added to both ankle-pitch joints (primary actuators) and both
        hip-pitch joints (secondary, 30 % gain).  Positive pitch_cmd
        increases plantarflexion → toes push ground → CoM moves rearward.

        Roll
        ----
        roll_cmd = Kp·θ_roll + Kd·ω_roll

        Applied asymmetrically: left ankle +roll_cmd, right ankle −roll_cmd.
        For rightward tip (positive roll) this drives the left foot to push
        the pelvis back toward centre.  Hip-roll joints assist at 30 % gain.

        Note on roll sign
        -----------------
        The sign assumes left_ankle_roll_joint positive rotation about its
        local X-axis corresponds to eversion (outer-edge-up).  If the robot
        tilts toward the correction direction during roll recovery, negate
        Kp_roll and Kd_roll at construction time.
        """
        # ── Pitch correction ──────────────────────────────────────────────────
        pitch_cmd = self._Kp_pitch * self.pitch + self._Kd_pitch * self.pitch_rate
        self._recovery_tau[self._IDX_LAP]  = pitch_cmd           # left  ankle pitch
        self._recovery_tau[self._IDX_RAP]  = pitch_cmd           # right ankle pitch
        self._recovery_tau[self._IDX_LHP]  = 0.3 * pitch_cmd     # left  hip pitch
        self._recovery_tau[self._IDX_RHP]  = 0.3 * pitch_cmd     # right hip pitch

        # ── Roll correction ───────────────────────────────────────────────────
        roll_cmd = self._Kp_roll * self.roll + self._Kd_roll * self.roll_rate
        self._recovery_tau[self._IDX_LAR]  =  roll_cmd           # left  ankle roll
        self._recovery_tau[self._IDX_RAR]  = -roll_cmd           # right ankle roll
        self._recovery_tau[self._IDX_LHR]  =  0.3 * roll_cmd    # left  hip roll
        self._recovery_tau[self._IDX_RHR]  = -0.3 * roll_cmd    # right hip roll
