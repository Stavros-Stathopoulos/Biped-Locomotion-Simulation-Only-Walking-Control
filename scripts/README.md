# Scripts Directory

This directory contains the executable entry points. These are the files you actually run from the terminal.

## Logic & Rules
- **Dependency Injection:** Scripts are responsible for loading the configurations from `config/`, initializing the environment from `src/env/`, instantiating the controller from `src/controllers/`, and running the main loop.
- **Single Purpose:** Keep scripts focused. 
  - `run_passive_test.py`: Tests basic environment loading.
  - `run_lipm_walking.py`: Executes the basic Linear Inverted Pendulum Model walking.
  - `tune_gains.py`: A script specifically for testing joint-level PD tracking.
- **Clean Loops:** The main while-loop inside these scripts should be easy to read: fetch state -> compute control -> apply control -> step sim.