# Source Directory

This is the core library for the bipedal locomotion project. It contains all the reusable modules, algorithms, and wrappers.

## Architecture
- `env/`: Contains the `MujocoEnv` class. This wrapper abstracts away the raw MuJoCo C-bindings. It provides a clean API: `reset()`, `step()`, and state-fetching methods.
- `controllers/`: The brains of the robot. 
  - Must inherit from a `BaseController` class.
  - Controllers must take in a defined `State` object (positions, velocities, CoM) and return a `Command` object (torques or target positions).
  - **Rule:** Controllers must never directly import or call `mujoco.mj_step`.
- `utils/`: Helper functions for homogeneous transformations, rigid body dynamics math, and data logging.

## Logic & Rules
- **No execution code here.** Do not put `if __name__ == "__main__":` blocks in these files. They are libraries meant to be imported.