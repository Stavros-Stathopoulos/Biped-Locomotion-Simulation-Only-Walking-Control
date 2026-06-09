import numpy as np

from src.utils.logger.data_logger import data_logger


class JointPDController:
    """
    Vectorized joint-space PD controller.

    Architecture contract
    ---------------------
    * __init__  : the ONLY place that may allocate NumPy arrays.
    * compute_torques : zero dynamic allocation, zero explicit Python loops.
      Every array operation uses a pre-allocated output buffer via NumPy
      in-place ufuncs (np.subtract, np.multiply, np.add, np.clip) or
      np.take(..., out=<pre-alloc>).

    Design intent
    -------------
    No gravity/Coriolis feed-forward is applied. Gravity is the disturbance;
    the PD feedback is the mechanism that must reject it. Pre-cancelling
    qfrc_bias would short-circuit the control loop and hide whether the gains
    are actually sufficient to hold the robot against the load.

    At steady state the robot settles where Kp·(q_ref − q_eq) = τ_gravity(q_eq),
    so there will be a small position offset proportional to τ_gravity/Kp. High
    Kp minimises this offset; adding an integral term would eliminate it.

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
    Priority: model.actuator_ctrlrange first (populated when the XML motor
    element carries an explicit ctrlrange attribute). If every entry is [0,0]
    (MuJoCo default when ctrlrange is omitted, as on G1 motors), fall back to
    model.jnt_actfrcrange indexed by the joint IDs — this is where the G1 XML
    stores physical limits via the actuatorfrcrange joint attribute.

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
        # jnt_dofadr[j]        : first dof  index for joint j
        joint_ids = self.model.actuator_trnid[:, 0]           # (nu,) int
        self._qpos_idx: np.ndarray = self.model.jnt_qposadr[joint_ids].copy()
        self._dof_idx:  np.ndarray = self.model.jnt_dofadr[joint_ids].copy()

        # ── Torque saturation bounds ────────────────────────────────────────
        ctrl_range = self.model.actuator_ctrlrange          # (nu, 2)
        ctrl_limited = ctrl_range[:, 1] > ctrl_range[:, 0]
        if np.any(ctrl_limited):
            self._ctrl_min: np.ndarray = np.where(ctrl_limited, ctrl_range[:, 0], -np.inf)
            self._ctrl_max: np.ndarray = np.where(ctrl_limited, ctrl_range[:, 1],  np.inf)
        else:
            # G1 motors have no ctrlrange set; fall back to jnt_actfrcrange.
            try:
                frc = self.model.jnt_actfrcrange[joint_ids]   # (nu, 2)
                valid = frc[:, 1] > frc[:, 0]
                self._ctrl_min = np.where(valid, frc[:, 0], -np.inf)
                self._ctrl_max = np.where(valid, frc[:, 1],  np.inf)
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
        # Five buffers. _err_qdot is reused as the Kd*err_qdot intermediate.
        self._q        = np.empty(self.nu, dtype=np.float64)
        self._qdot     = np.empty(self.nu, dtype=np.float64)
        self._tau      = np.empty(self.nu, dtype=np.float64)
        self._err_q    = np.empty(self.nu, dtype=np.float64)
        self._err_qdot = np.empty(self.nu, dtype=np.float64)

    def compute_torques(
        self,
        q_ref:    np.ndarray,
        qdot_ref: np.ndarray,
    ) -> np.ndarray:
        """
        Compute τ = Kp·(q_ref − q) + Kd·(qdot_ref − qdot),
        then clamp to [ctrl_min, ctrl_max].

        Gravity is a disturbance rejected by the feedback terms above — no
        feed-forward cancellation is applied.

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
        np.take(self.data.qpos, self._qpos_idx, out=self._q)
        np.take(self.data.qvel, self._dof_idx,  out=self._qdot)

        # τ = Kp·(q_ref − q) + Kd·(qdot_ref − qdot)
        np.subtract(q_ref,    self._q,    out=self._err_q)
        np.subtract(qdot_ref, self._qdot, out=self._err_qdot)
        np.multiply(self._kp, self._err_q,    out=self._tau)       # Kp·Δq → τ
        np.multiply(self._kd, self._err_qdot, out=self._err_qdot)  # Kd·Δqdot (reuse buf)
        np.add(self._tau, self._err_qdot, out=self._tau)            # τ += Kd·Δqdot

        # Enforce physical motor limits
        np.clip(self._tau, self._ctrl_min, self._ctrl_max, out=self._tau)

        # Guarded disk I/O — fires at ~1 Hz, not every step
        self._log_counter += 1
        if self._log_counter >= self._log_interval:
            self._log_counter = 0
            data_logger.log_input("PD Torques", self._tau.tolist())

        return self._tau
