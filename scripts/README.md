# Scripts Directory

This directory contains the executable entry points. These are the files you actually run from the terminal.

## Logic & Rules
- **Dependency Injection:** Scripts are responsible for loading the configurations from `config/`, initializing the environment from `src/env/`, instantiating the controller from `src/controllers/`, and running the main loop.
- **Single Purpose:** Keep scripts focused.
  - `run_passive_test.py`: Loads the scene at the `stand` keyframe and checks passive dynamics (zero control).
  - `run_walking.py`: Main entry. Runs the full `WalkingController` (FSM weight-shifting / stepping gait) with the interactive viewer and live diagnostics. Options: `--headless`, `--duration N`, `--steps N` (number of alternating steps before holding a stable stand; the robot squares up flat on both feet and settles between every step), `--kick V [--seed N]` (initial disturbance / shows the sim is reactive, not scripted). The lateral balance is reliable to ~4 steps, after which it holds the stand; beyond that it can tip.
  - `test_balance.py [duration_s]`: Headless balance check — holds the standing pose on torques only; PASS if it stays up.
  - `test_walking.py [step_length_m] [duration_s]`: Headless gait check — reports completed alternating steps, survival time and forward travel. `step_length 0.0` = step in place.
- **Clean Loops:** The main while-loop inside these scripts should be easy to read: fetch state -> compute control -> apply control -> step sim.

## Quick start

```
py -3.12 scripts/run_walking.py                 # watch it walk (viewer)
py -3.12 scripts/test_balance.py 10             # verify it balances 10 s
py -3.12 scripts/test_walking.py 0.03 45        # verify alternating forward steps
```