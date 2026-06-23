# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuJoCo-based bipedal walking controller for the Unitree G1 humanoid robot (29 DoF). University course final project (ECE_DK801 Robotics Systems I). The robot must walk using only motor torques through real physics — no pinning, no teleporting joints, no external forces.

## Running

Python 3.10+ with a virtualenv at `.venv/`. Dependencies: `mujoco`, `numpy`, `pyyaml`, `qpsolvers`.

```bash
# Activate venv
source .venv/bin/activate

# Main walking controller (quasi-static FSM + whole-body QP)
python scripts/run_walking.py                    # interactive MuJoCo viewer
python scripts/run_walking.py --headless         # no viewer, prints diagnostics
python scripts/run_walking.py --steps 6          # walk N steps then stand
python scripts/run_walking.py --kick 0.15        # initial disturbance test

# DCM (capture-point) continuous walking controller
python scripts/run_dcm_walk.py
python scripts/run_dcm_walk.py --step-length 0.06 --step-width 0.22

# Passive stability test (robot stands with zero control)
python scripts/run_passive_test.py
```

All scripts are in `scripts/` and import from `src/` via `sys.path.append`. There is no package install — run from project root.

## Configuration

`config/simulation.yaml` contains all tunable parameters: physics settings, PD gains, WBQP task weights, and gait timing. The controller reads this at startup; CLI flags like `--steps` override specific values.

## Architecture

Two independent control pipelines share the same robot model and environment:

### Pipeline 1: Quasi-static FSM (`run_walking.py`)

```
WalkingController.update() each tick:
  StateEstimator  →  GaitController (FSM)  →  WholeBodyIK (reference)
                                           →  WholeBodyQP (torques)  →  data.ctrl
                                      or   →  JointPDController (fallback if use_wbqp=false)
```

- `src/controllers/walking_controller.py` — top-level orchestrator
- `src/controllers/gait.py` — FSM (STAND→SHIFT→SWING→LAND cycle), CoM planner, swing-foot trajectory, Raibert footstep correction, balance recovery
- `src/controllers/wbqp.py` — acceleration-level whole-body QP (decision vars: qddot + contact forces; constraints: floating-base dynamics, friction cones, torque limits)
- `src/controllers/wbik.py` — velocity-level whole-body IK (weighted least-squares QP, used as reference generator or as fallback controller)
- `src/controllers/joint_pd_controller.py` — PD + gravity compensation torque controller
- `src/controllers/balance.py` — CoM force feedback + orientation stabilizer (IK+PD path only)

### Pipeline 2: DCM walking (`run_dcm_walk.py`)

```
StateEstimator  →  DCMWalkingGait  →  WholeBodyQP  →  data.ctrl
```

- `src/controllers/dcm_gait.py` — LIPM-based DCM/capture-point pattern generator with online foot placement adjustment. Produces CoM acceleration feed-forward for the QP.
- Reuses `wbqp.py` and `robot_model.py` from Pipeline 1.

### Shared components

- `src/controllers/robot_model.py` — `RobotModel` (joint indexing, home pose, Jacobians, torque limits) and `StateEstimator` (CoM, contacts, orientation). Single source of truth for MuJoCo model indexing: `actuator i ↔ qpos[7+i] ↔ qvel[6+i]`.
- `src/controllers/terrain.py` — ray-cast ground height sensing for stair climbing
- `src/env/mujoco_env.py` — thin MuJoCo wrapper (load XML, step, viewer)
- `src/utils/terminal_logger.py` — colored terminal logging
- `src/utils/data_logger.py` — CSV data logging
- `src/utils/config_parser.py` — YAML config loader

### Robot model

MJCF scene: `assets/unitree_g1/scene.xml` (includes `g1_29dof.xml`). All 29 actuators are torque motors — `data.ctrl[:]` expects torques directly. The robot has a floating base (free joint), so `nq=36`, `nv=35`, `nu=29`.

## Hard Constraints (never violate)

- **Only `data.ctrl[:]`** may be written during the simulation loop. Never overwrite `data.qpos` or `data.qvel` after initialization.
- **No cheating**: no base welding, no pelvis pinning, no external forces, no teleportation. The robot must balance through MuJoCo physics (gravity, contacts, friction, inertia).
- **Torques must be clipped** to actuator limits before applying.
- **The QP must run online** every control tick — no pre-recorded trajectories.
- Prefer a slow, physically valid controller over a visually impressive fake animation.
