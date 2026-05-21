# Source Directory

This is the core library for the bipedal locomotion project. It contains all reusable modules, algorithms, and wrappers. Nothing in this directory is a script — these are modules meant to be imported.

## Structure

```
src/
├── env/
│   └── mujoco_env.py       # MuJoCo simulation environment wrapper
├── controllers/
│   └── joint_pd_controller.py  # Joint-space PD + gravity compensation controller
└── utils/
    ├── terminal_logger.py  # Colored terminal logger
    └── data_logger.py      # JSONL telemetry logger
```

## Modules

### `env/`

Contains `MujocoEnv` — a thin wrapper over the raw MuJoCo C-bindings that exposes a clean Python API: `reset()`, `step()`, `init_viewer()`, `sync_viewer()`, and `close_viewer()`. See [env/README.md](env/README.md).

### `controllers/`

Joint-space and task-space control algorithms. Currently implements `JointPDController`, a PD controller with gravity and Coriolis compensation.

**Rules:**
- Controllers must never import or call `mujoco.mj_step` directly. Physics stepping is delegated to `MujocoEnv`.
- Controllers receive references to the live `model` and `data` objects and return torque arrays.

See [controllers/README.md](controllers/README.md).

### `utils/`

Shared helper modules. Currently provides:
- `TerminalLogger` — colored, formatted terminal logging
- `DataLogger` — JSONL-format telemetry logging to disk

See [utils/README.md](utils/README.md).

## Rules

- **No execution code.** Do not put `if __name__ == "__main__":` blocks in these files.
- Controllers must not call `mujoco.mj_step` directly.
- All parameters (gains, rates, thresholds) must live in `config/` YAML files, not hardcoded here.
