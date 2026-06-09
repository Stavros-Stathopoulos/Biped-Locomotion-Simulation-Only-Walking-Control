import numpy as np

from src.utils.logger.data_logger import data_logger


class JointPDController:
    """
    Vectorized joint-space PD controller with gravity/Coriolis compensation.

    Architecture contract
    ---------------------
    * __init__  : the ONLY place that may allocate NumPy arrays.
    * compute_torques : zero dynamic allocation, zero explicit Python loops.
      Every array operation uses a pre-allocated output buffer via NumPy
      in-place ufuncs (np.subtract, np.multiply, np.add, np.clip) or
      np.take(..., out=<pre-alloc>).

    Floating-base offset
    --------------------
    The G1 has a 6-DoF free joint on the pelvis, which contributes
      +7 entries to qpos  (3 position + 4 quaternion components)
      +6 entries to qvel  (3 linear + 3 angular velocity components)
    These offsets are resolved automatically by reading jnt_qposadr and
    jnt_dofadr for the joint IDs referenced by actuator_trnid, so no
    manual offset arithmetic is needed or embedded here.

    Torque saturation
    -----------------
    The G1 XML sets actuatorfrcrange on each joint but leaves the motor
    actuator's ctrlrange / forcerange unset (both read as [0,0] from
    model.actuator_ctrlrange / actuator_forcerange). The physical limits are
    therefore sourced from model.jnt_actfrcrange, indexed by the joint IDs
    associated with each actuator.

    Logging
    -------
    DataLogger writes to a JSONL file on disk. At 500 Hz, logging every step
    would saturate I/O. Writes are gated to every `log_interval` steps
    (default 500 → ~1 Hz at 500 Hz sim rate).
    """

    def __init__(
        self,
        model,
        data,
        kp: np.ndarray,
        kd: np.ndarray,
        log_interval: int = 500,
    ) -> None:
        self.model = model
        self.data = data
        self.nu = model.nu

        self._log_interval = log_interval
        self._log_counter = 0

        # ── Index arrays — built once, used every step ─────────────────────
        # actuator_trnid[:, 0] : joint ID for actuator i (shape nu)
        # jnt_qposadr[j]       : first qpos index for joint j (1-DOF joints → one entry)
        # jnt_dofadr[j]        : first dof  index for joint j (same mapping for qvel/qfrc_bias)
        joint_ids = self.model.actuator_trnid[:, 0]           # (nu,) int
        self._qpos_idx: np.ndarray = self.model.jnt_qposadr[joint_ids].copy()
        self._dof_idx:  np.ndarray = self.model.jnt_dofadr[joint_ids].copy()

        # ── Torque saturation bounds ────────────────────────────────────────
        # model.jnt_actfrcrange holds the [min, max] actuator force range for
        # each joint, populated from the XML actuatorfrcrange attribute.
        try:
            frc = self.model.jnt_actfrcrange[joint_ids]   # (nu, 2)
            valid = frc[:, 1] > frc[:, 0]
            self._ctrl_min: np.ndarray = np.where(valid, frc[:, 0], -np.inf)
            self._ctrl_max: np.ndarray = np.where(valid, frc[:, 1],  np.inf)
        except AttributeError:
            self._ctrl_min = np.full(self.nu, -np.inf)
            self._ctrl_max = np.full(self.nu,  np.inf)

        # ── Diagonal gain vectors ───────────────────────────────────────────
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        if self._kp.shape != (self.nu,) or self._kd.shape != (self.nu,):
            raise ValueError(
                f"kp and kd must both have shape ({self.nu},); "
                f"got kp={self._kp.shape}, kd={self._kd.shape}"
            )

        # ── Pre-allocated working buffers ───────────────────────────────────
        # Six distinct buffers eliminate all temporary allocations on the hot path.
        # _err_qdot is reused as the Kd*err_qdot intermediate product.
        self._q        = np.empty(self.nu, dtype=np.float64)
        self._qdot     = np.empty(self.nu, dtype=np.float64)
        self._tau_bias = np.empty(self.nu, dtype=np.float64)
        self._tau      = np.empty(self.nu, dtype=np.float64)
        self._err_q    = np.empty(self.nu, dtype=np.float64)
        self._err_qdot = np.empty(self.nu, dtype=np.float64)

    def compute_torques(
        self,
        q_ref:    np.ndarray,
        qdot_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Compute τ = Kp·(q_ref − q) + Kd·(qdot_ref − qdot) + τ_bias,
        then clamp to [ctrl_min, ctrl_max].

        Hot-path constraints
        --------------------
        * No Python loops.
        * No heap allocations (no np.zeros / np.array / operator + etc.).
        * np.take(..., out=) performs a vectorised gather into a pre-allocated
          destination without creating an intermediate array.
        * All arithmetic uses the ufunc `out=` parameter to write results
          directly into pre-allocated buffers.

        Return value
        ------------
        A *view* of the internal self._tau buffer. The caller must copy it
        into data.ctrl before the next call (e.g. `data.ctrl[:] = torques`).
        """
        # State extraction — vectorised gather, no allocation
        np.take(self.data.qpos,      self._qpos_idx, out=self._q)
        np.take(self.data.qvel,      self._dof_idx,  out=self._qdot)
        np.take(self.data.qfrc_bias, self._dof_idx,  out=self._tau_bias)

        # τ_pd = Kp·(q_ref − q) + Kd·(qdot_ref − qdot)
        np.subtract(q_ref,    self._q,    out=self._err_q)
        np.subtract(qdot_ref, self._qdot, out=self._err_qdot)
        np.multiply(self._kp, self._err_q,    out=self._tau)      # Kp·Δq → τ
        np.multiply(self._kd, self._err_qdot, out=self._err_qdot) # Kd·Δqdot (reuse buf)
        np.add(self._tau, self._err_qdot, out=self._tau)           # τ += Kd·Δqdot
        np.add(self._tau, self._tau_bias, out=self._tau)           # τ += τ_bias

        # Enforce physical motor limits
        np.clip(self._tau, self._ctrl_min, self._ctrl_max, out=self._tau)

        # Guarded disk I/O — fires at ~1 Hz, not every step
        self._log_counter += 1
        if self._log_counter >= self._log_interval:
            self._log_counter = 0
            data_logger.log_input("PD Torques",               self._tau.tolist())
            data_logger.log_input("Gravity/Coriolis Torques", self._tau_bias.tolist())

        return self._tau
