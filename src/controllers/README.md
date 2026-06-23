# Controllers Module

Joint-space and task-space control algorithms for the Unitree G1 humanoid.

## Rules

- Controllers must **never** import or call `mujoco.mj_step` directly. Physics stepping is the responsibility of `MujocoEnv`.
- Controllers receive references to the live `mujoco.MjModel` and `mujoco.MjData` objects from the active environment.
- No `if __name__ == "__main__":` blocks. These are library modules.

---

## Walking controller architecture

`WalkingController` (`walking_controller.py`) orchestrates the full torque-only
locomotion pipeline. One `update()` call returns the torque vector for
`data.ctrl`; physics is advanced by `MujocoEnv.step()`. The robot is **never**
pinned, welded, teleported, or stabilised by external forces, and `qpos`/`qvel`
are never written during the run — only `set_home()` writes the initial pose.

| File | Layer(s) | Role |
|------|----------|------|
| `robot_model.py` | 1 | `RobotModel` (index maps, torque limits, foot geometry, home pose) and `StateEstimator` (CoM + CoM velocity via `mj_jacSubtreeCom`, torso RPY, per-foot contact force, support phase). |
| `gait.py` | 2-6, 10 | `GaitController`: FSM (`STAND / SHIFT_COM_* / *_SWING / *_LAND / RECOVER`), CoM planner with measured-CoM balance feedback, quintic swing-foot planner, footstep + Raibert/capture placement, balance-recovery abort. |
| `wbik.py` | 7 | `WholeBodyIK`: weighted task-space (velocity) IK on a **private** `MjData` clone — CoM + both-foot + torso + posture tasks. Emits `q_des, qd_des`; never touches the sim. |
| `joint_pd_controller.py` | 8, 9 | `JointPDController`: `tau = kp(q_ref-q) + kd(qd_ref-qd) + qfrc_bias`, clamped to actuator force limits. |
| `balance.py` | 3, 10 | `BalanceStabilizer`: Jacobian-transpose CoM-force + pelvis-attitude assist. |

### Key design decisions (learned from tuning)

- **Gravity compensation needs stiff legs.** `qfrc_bias` only approximates the
  static hold for a floating base; the residual is carried by joint stiffness, so
  leg gains are high (hip 400 / knee 450 / ankle 220).
- **The IK is a clean kinematic plan** (its base floats freely, seeded from home).
  Feeding the measured tilting base back into the IK destabilises it — disturbance
  rejection is done elsewhere.
- **Active balance is injected into the IK CoM target** from the *measured* CoM
  error/velocity. Because the stiff PD faithfully tracks the IK reference, moving
  that target moves the robot — this is the primary balance loop.
- **Lift-off is gated on a settled CoM**; the swing foot presses slightly into
  the floor and the FSM waits for confirmed double-support contact before
  recentring, so single support is entered/exited with low momentum.

---

## JointPDController

`src/controllers/joint_pd_controller.py`

Implements a joint-space PD controller with gravity and Coriolis compensation.

### Control Law

```
τ = kp * (q_ref - q) + kd * (qdot_ref - qdot) + τ_bias
```

- `q`, `qdot` — current joint positions and velocities, extracted from `data.qpos` and `data.qvel`
- `q_ref`, `qdot_ref` — reference joint positions and velocities
- `τ_bias` — gravity and Coriolis forces from `data.qfrc_bias`
- `kp`, `kd` — proportional and derivative gain arrays, shape `(nu,)`

### Constructor

```python
JointPDController(model, data, kp: np.ndarray, kd: np.ndarray)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `mujoco.MjModel` | Model from the active `MujocoEnv` |
| `data` | `mujoco.MjData` | Data from the active `MujocoEnv` |
| `kp` | `np.ndarray` shape `(nu,)` | Proportional gains per actuator |
| `kd` | `np.ndarray` shape `(nu,)` | Derivative gains per actuator |

`nu = model.nu` — number of actuated motors (floating-base DOFs are excluded).

### Method: `compute_torques`

```python
compute_torques(q_ref: np.ndarray, qdot_ref: np.ndarray) -> np.ndarray
```

Returns torques of shape `(nu,)` ready to assign to `data.ctrl`.

**Joint-to-actuator index mapping** (handles non-contiguous MuJoCo addressing):

| MuJoCo field | Used for |
|---|---|
| `model.actuator_trnid[i, 0]` | Joint index for actuator `i` |
| `model.jnt_qposadr[joint_id]` | Index into `data.qpos` |
| `model.jnt_dofadr[joint_id]` | Index into `data.qvel` and `data.qfrc_bias` |

Each call logs the computed PD torques and gravity/Coriolis torques via `DataLogger`.

### Example

```python
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.joint_pd_controller import JointPDController
import os

env = MujocoEnv(os.path.abspath("assets/unitree_g1/scene.xml"), rate_hz=500.0)
env.reset(keystring="stand")

nu = env.model.nu
kp = np.full(nu, 100.0)
kd = np.full(nu, 10.0)

controller = JointPDController(env.model, env.data, kp, kd)

q_ref = env.data.qpos[7:7 + nu].copy()  # hold current pose
qdot_ref = np.zeros(nu)

while True:
    tau = controller.compute_torques(q_ref, qdot_ref)
    env.data.ctrl[:] = tau
    env.step()
```
