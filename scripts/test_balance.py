"""
Headless balance test: hold the standing (double-support) pose and verify the
robot stays upright on motor torques alone for N seconds.

Usage:
    py -3.12 scripts/test_balance.py [duration_s]

Prints CoM, torso roll/pitch and pelvis height periodically. PASS if the pelvis
never drops below 0.45 m.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.walking_controller import WalkingController

scene = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/unitree_g1/scene.xml'))
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

env = MujocoEnv(scene, rate_hz=500.0)
# keep the FSM in STAND for the whole test
ctrl = WalkingController(env.model, env.data, {"gait": {"t_stand": dur + 10.0}})
ctrl.reset()

while env.data.time < dur:
    env.data.ctrl[:] = ctrl.update()
    env.step()
    if int(env.data.time * 500) % 250 == 0:
        com = ctrl.last["com"]; rpy = ctrl.last["base_rpy"]
        print(f"t={env.data.time:5.2f} com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) "
              f"roll={rpy[0]:+.3f} pitch={rpy[1]:+.3f} pelvis_z={env.data.qpos[2]:.3f} "
              f"tau_max={np.abs(ctrl.last['tau']).max():.1f}")
    if env.data.qpos[2] < 0.45:
        print(f"FAIL: fell at t={env.data.time:.2f}"); sys.exit(1)
print(f"PASS: balanced {dur:.0f}s on torques only (pelvis_z={env.data.qpos[2]:.3f}).")
