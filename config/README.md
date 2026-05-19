# Configuration Directory

Centralizes all runtime parameters so the Python source code stays free of magic numbers. All tunable values — gains, rates, thresholds — live here.

## Rules

- **Controller gains:** PD gains, whole-body control weights, MPC horizons, and similar tuning values belong in YAML files (e.g., `controller_params.yaml`), not hardcoded in `src/`.
- **Environment settings:** Simulation rate, control rate (if decoupled from simulation), and logging frequencies are defined here.
- **Sweeps and optimization:** Keeping parameters in YAML enables automated parameter sweeps and optimization scripts without touching core logic.

---

## Files

### `simulation.yaml`

Physics and environment parameters loaded at simulation startup.

```yaml
physics:
  gravity: [0, 0, -9.81]       # m/s² — standard Earth gravity (x, y, z)
  solver_iterations: 50         # MuJoCo constraint solver iterations

  override_joint_damping: true  # If true, replaces per-joint damping from the MJCF/URDF
  default_damping: 15.0         # N·s/rad — applied to all joints when override is active
```

**`override_joint_damping`** — when `true`, a uniform damping coefficient (`default_damping`) is applied to all joints, overriding whatever values are defined in the robot model files. This is currently set to `true` and `15.0 N·s/rad` to give the G1 enough passive resistance to stand upright without active control, which is required for the passive stability test to pass.
