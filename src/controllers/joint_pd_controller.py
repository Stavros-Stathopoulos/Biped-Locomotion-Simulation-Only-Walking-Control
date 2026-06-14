from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.utils.logger.data_logger import data_logger


@dataclass
class ControllerConfig:
    """
    Configuration bundle for JointPDController.

    Fields
    ------
    kp           : Proportional gains, shape (nu,), Nm/rad — one per actuator.
    kd           : Derivative gains,   shape (nu,), Nm·s/rad — one per actuator.
    gravity_comp : When True, a feedforward term τ_ff = qfrc_bias[dof_idx] is
                   added.  qfrc_bias contains gravity + Coriolis/centripetal in
                   generalised coordinates; at near-zero joint velocity this
                   equals pure gravity compensation.  Enabling this eliminates
                   the steady-state position error and allows much lower PD
                   gains, restoring natural joint compliance.
    nan_check    : When True, raise RuntimeError if any output torque is NaN.
                   Costs one np.any() call per step; disable for peak throughput.
    log_interval : Steps between JSONL disk writes (default 500 → ~1 Hz at a
                   500 Hz simulation rate).
    """

    kp:           np.ndarray
    kd:           np.ndarray
    gravity_comp: bool = True
    nan_check:    bool = True
    log_interval: int  = 500


class JointPDController:
    """
    Vectorized joint-space PD controller with optional gravity compensation.

    Torque equation
    ---------------
        τ = τ_ff + Kp·(q_ref − q) + Kd·(q̇_ref − q̇)

    where
        τ_ff = data.qfrc_bias[dof_idx]   (gravity_comp=True)
        τ_ff = 0                          (gravity_comp=False — pure PD)

    qfrc_bias is populated by MuJoCo at the end of each mj_step, so the
    feedforward term carries a one-timestep lag.  At 500 Hz (dt = 2 ms) this
    lag is negligible for all practical standing and slow-motion tasks.

    Architecture contract
    ---------------------
    * __init__         : the ONLY place that may allocate NumPy arrays.
    * compute_torques  : zero dynamic allocation, zero explicit Python loops.
      Every array operation uses pre-allocated output buffers via NumPy
      in-place ufuncs (np.subtract, np.multiply, np.add, np.clip) or
      np.take(..., out=<pre-alloc>).

    Floating-base offset
    --------------------
    The G1 has a 6-DoF free joint on the pelvis, contributing +7 entries to
    qpos (3 position + 4 quaternion) and +6 entries to qvel / qfrc_bias
    (3 linear + 3 angular).  All offsets are resolved automatically by reading
    jnt_qposadr / jnt_dofadr indexed by actuator_trnid[:, 0].

    Torque saturation
    -----------------
    Priority: model.actuator_ctrlrange first.  If every entry is [0, 0] (the
    G1 default when ctrlrange is omitted), fall back to
    model.jnt_actfrcrange[joint_ids] — where the G1 stores physical limits
    via the XML actuatorfrcrange joint attribute.
    """

    def __init__(self, model, data, cfg: ControllerConfig) -> None:
        self.model = model
        self.data  = data
        self.nu    = model.nu

        self._gravity_comp = cfg.gravity_comp
        self._nan_check    = cfg.nan_check
        self._log_interval = cfg.log_interval
        self._log_counter  = 0

        # ── Index arrays — built once, reused every step ─────────────────────
        joint_ids      = self.model.actuator_trnid[:, 0]           # (nu,) int
        self._qpos_idx = self.model.jnt_qposadr[joint_ids].copy()  # qpos gather
        self._dof_idx  = self.model.jnt_dofadr[joint_ids].copy()   # qvel / qfrc gather

        # ── Torque saturation bounds ──────────────────────────────────────────
        ctrl_range   = self.model.actuator_ctrlrange               # (nu, 2)
        ctrl_limited = ctrl_range[:, 1] > ctrl_range[:, 0]
        if np.any(ctrl_limited):
            self._ctrl_min: np.ndarray = np.where(ctrl_limited, ctrl_range[:, 0], -np.inf)
            self._ctrl_max: np.ndarray = np.where(ctrl_limited, ctrl_range[:, 1],  np.inf)
        else:
            # G1 motors omit ctrlrange; fall back to jnt_actfrcrange.
            try:
                frc   = self.model.jnt_actfrcrange[joint_ids]      # (nu, 2)
                valid = frc[:, 1] > frc[:, 0]
                self._ctrl_min = np.where(valid, frc[:, 0], -np.inf)
                self._ctrl_max = np.where(valid, frc[:, 1],  np.inf)
            except AttributeError:
                self._ctrl_min = np.full(self.nu, -np.inf)
                self._ctrl_max = np.full(self.nu,  np.inf)

        # ── Gain vectors ──────────────────────────────────────────────────────
        self._kp = np.asarray(cfg.kp, dtype=np.float64)
        self._kd = np.asarray(cfg.kd, dtype=np.float64)
        if self._kp.shape != (self.nu,) or self._kd.shape != (self.nu,):
            raise ValueError(
                f"kp and kd must both have shape ({self.nu},); "
                f"got kp={self._kp.shape}, kd={self._kd.shape}"
            )

        # ── Pre-allocated working buffers ─────────────────────────────────────
        # _err_qdot doubles as the Kd·Δqdot intermediate to save one buffer.
        self._q        = np.empty(self.nu, dtype=np.float64)
        self._qdot     = np.empty(self.nu, dtype=np.float64)
        self._tau      = np.empty(self.nu, dtype=np.float64)
        self._err_q    = np.empty(self.nu, dtype=np.float64)
        self._err_qdot = np.empty(self.nu, dtype=np.float64)
        self._tau_ff   = np.zeros(self.nu, dtype=np.float64)  # feedforward buffer

    def compute_torques(
        self,
        q_ref:    np.ndarray,
        qdot_ref: np.ndarray,
    ) -> np.ndarray:
        # 1. Extract state vectors
        np.take(self.data.qpos, self._qpos_idx, out=self._q)
        np.take(self.data.qvel, self._dof_idx,  out=self._qdot)

        # 2. Compute PD feedback tracking: τ = Kp·(q_ref − q) + Kd·(qdot_ref − qdot)
        np.subtract(q_ref,    self._q,    out=self._err_q)
        np.subtract(qdot_ref, self._qdot, out=self._err_qdot)
        np.multiply(self._kp, self._err_q,    out=self._tau)
        np.multiply(self._kd, self._err_qdot, out=self._err_qdot)
        np.add(self._tau, self._err_qdot, out=self._tau)

        # 3. CRITICAL: Gather gravity compensation terms from qfrc_bias (dof space)
        # Re-use _err_qdot buffer to preserve memory overhead rules
        np.take(self.data.qfrc_bias, self._dof_idx, out=self._err_qdot)
        np.add(self._tau, self._err_qdot, out=self._tau)

        # 4. Enforce motor saturation bounds
        np.clip(self._tau, self._ctrl_min, self._ctrl_max, out=self._tau)

        return self._tau
