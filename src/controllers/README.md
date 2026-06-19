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
τ = τ_ff + Kp·(q_ref − q) + Kd·(q̇_ref − q̇)
```

- `q`, `qdot` — current joint positions and velocities, extracted from `data.qpos` and `data.qvel`
- `q_ref`, `qdot_ref` — reference joint positions and velocities
- `τ_ff` — feedforward term: `data.qfrc_bias[dof_idx]` when `gravity_comp=True` (gravity + Coriolis at near-zero velocity = pure gravity compensation); zero otherwise
- `Kp`, `Kd` — proportional and derivative gain arrays, shape `(nu,)`, defined in `ControllerConfig`

### Configuration

```python
from src.controllers.joint_pd_controller import ControllerConfig

cfg = ControllerConfig(
    kp=np.full(nu, 100.0),   # Nm/rad
    kd=np.full(nu, 10.0),    # Nm·s/rad
    gravity_comp=True,        # adds τ_ff = qfrc_bias[dof_idx]
    nan_check=True,           # raises RuntimeError on NaN output
    log_interval=500,         # disk-log every N steps
)
```

### Constructor

```python
JointPDController(model, data, cfg: ControllerConfig)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `model` | `mujoco.MjModel` | Model from the active `MujocoEnv` |
| `data` | `mujoco.MjData` | Data from the active `MujocoEnv` |
| `cfg` | `ControllerConfig` | Gain bundle and feature flags |

`nu = model.nu` — number of actuated motors (floating-base DOFs are excluded automatically via `jnt_dofadr`).

### Method: `compute_torques`

```python
compute_torques(q_ref: np.ndarray, qdot_ref: np.ndarray) -> np.ndarray
```

Returns torques of shape `(nu,)` ready to assign to `data.ctrl`.

**Joint-to-actuator index mapping** (handles non-contiguous MuJoCo addressing):

| MuJoCo field | Used for |
| --- | --- |
| `model.actuator_trnid[i, 0]` | Joint index for actuator `i` |
| `model.jnt_qposadr[joint_id]` | Index into `data.qpos` |
| `model.jnt_dofadr[joint_id]` | Index into `data.qvel` and `data.qfrc_bias` |

### Example

```python
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.joint_pd_controller import ControllerConfig, JointPDController
from src.utils.config_parser import SimConfig

config = SimConfig("config/simulation.yaml")
env = MujocoEnv(config)

nu = env.model.nu
cfg = ControllerConfig(
    kp=np.full(nu, 100.0),
    kd=np.full(nu, 10.0),
    gravity_comp=True,
)
controller = JointPDController(env.model, env.data, cfg)

q_ref    = env.data.qpos[7:7 + nu].copy()
qdot_ref = np.zeros(nu)

while True:
    env.data.ctrl[:] = controller.compute_torques(q_ref, qdot_ref)
    env.step()
```
