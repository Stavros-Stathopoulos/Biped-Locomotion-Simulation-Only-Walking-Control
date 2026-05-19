# Environment Module

Wraps the raw MuJoCo C-bindings into a clean, reusable Python API. Scripts and controllers should interact with the simulation exclusively through this module.

## MujocoEnv

`src/env/mujoco_env.py`

### Constructor

```python
MujocoEnv(xml_path: str, rate_hz: float = 500.0)
```

Loads an MJCF scene file and initializes the MuJoCo model and data structures.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xml_path` | `str` | required | Absolute path to the MJCF `.xml` scene file |
| `rate_hz` | `float` | `500.0` | Simulation rate in Hz — sets `model.opt.timestep = 1 / rate_hz` |

Raises `FileNotFoundError` if the XML path does not exist.

> **Windows note:** The constructor temporarily changes the working directory to the asset folder before calling `mujoco.MjModel.from_xml_path`. This is a workaround for a path-encoding limitation in MuJoCo's C++ `fopen` on Windows. Always pass an absolute path when instantiating.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `mujoco.MjModel` | Compiled model — read-only geometry and physics constants |
| `data` | `mujoco.MjData` | Mutable simulation state (`qpos`, `qvel`, `ctrl`, `qfrc_bias`, etc.) |
| `viewer` | `mujoco.viewer` or `None` | Viewer handle; `None` until `init_viewer()` is called |

### Methods

#### `step()`

```python
env.step()
```

Advances physics by one integration timestep by calling `mujoco.mj_step`. Call this once per control loop iteration.

---

#### `reset(keystring=None)`

```python
env.reset(keystring: str = None)
```

Resets the simulation to its initial state. If `keystring` is provided, the simulation is placed in the matching keyframe defined in the MJCF file (e.g., `"stand"`). Calls `mj_forward` after reset to recompute derived quantities.

---

#### `init_viewer()`

```python
env.init_viewer()
```

Launches the native MuJoCo passive viewer in a background thread. Has no effect if a viewer is already open.

---

#### `sync_viewer()`

```python
env.sync_viewer()
```

Pushes the current simulation state to the viewer display. Call this once per step when visualization is active. No-ops if no viewer is running.

---

#### `close_viewer()`

```python
env.close_viewer()
```

Closes the viewer window and sets `self.viewer = None`.

### Example

```python
from src.env.mujoco_env import MujocoEnv
import os

scene_xml = os.path.abspath("assets/unitree_g1/scene.xml")
env = MujocoEnv(xml_path=scene_xml, rate_hz=500.0)
env.init_viewer()
env.reset(keystring="stand")

while env.viewer.is_running():
    env.data.ctrl[:] = 0.0  # zero torque
    env.step()
    env.sync_viewer()

env.close_viewer()
```
