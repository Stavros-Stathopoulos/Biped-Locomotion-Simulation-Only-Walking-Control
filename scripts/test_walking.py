"""
Headless walking test: run the full FSM gait and report how many alternating
steps the robot completes and how long it survives.

Usage:
    py -3.12 scripts/test_walking.py [step_length_m] [duration_s]

    step_length 0.0  -> in-place stepping (weight shift + foot lift, alternating)
    step_length >0   -> forward walking

Prints a PASS/summary line with the number of completed steps and survival time.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.walking_controller import WalkingController

scene = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/unitree_g1/scene.xml'))
step_length = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

env = MujocoEnv(scene, rate_hz=500.0)
ctrl = WalkingController(env.model, env.data,
                        {"gait": {"step_length": step_length, "max_steps": 9999}})
ctrl.reset()
com0 = ctrl.last.get("com", env.data.subtree_com[0].copy())

steps = 0
while env.data.time < dur:
    env.data.ctrl[:] = ctrl.update()
    env.step()
    steps = max(steps, ctrl.last["step"])
    if env.data.qpos[2] < 0.45:
        print(f"FELL: t={env.data.time:.2f}s steps={steps} "
              f"forward_travel={env.data.subtree_com[0][0]-com0[0]:+.3f} m")
        sys.exit(0)
print(f"SURVIVED {dur:.0f}s steps={steps} pelvis_z={env.data.qpos[2]:.3f} "
      f"forward_travel={env.data.subtree_com[0][0]-com0[0]:+.3f} m")
