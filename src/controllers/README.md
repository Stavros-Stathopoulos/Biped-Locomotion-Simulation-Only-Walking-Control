# Controllers Module

Joint-space and task-space control algorithms for the Unitree G1 humanoid.

## Rules

- Controllers must **never** import or call `mujoco.mj_step` directly. Physics stepping is the responsibility of `MujocoEnv`.
- Controllers receive references to the live `mujoco.MjModel` and `mujoco.MjData` objects from the active environment.
- No `if __name__ == "__main__":` blocks. These are library modules.

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
