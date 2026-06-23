"""
Terrain-aware stair-climbing test.

Runs the walking controller on the staircase scene. The controller senses each
tread height with downward ray casts, places the foot on the surface it reaches,
lifts the swing foot over the riser, and raises the CoM while ascending.

Usage:
    py -3.12 scripts/test_stairs.py                 # viewer, step_length 0.12
    py -3.12 scripts/test_stairs.py 0.10 --headless # choose stride, no viewer

Note: bigger strides are needed to reach the treads, which pushes the (quasi-
static) lateral balance; the robot climbs a few risers and currently tips after
~2-3 steps. Sustained climbing needs the dynamic capture-point controller.
"""
import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.walking_controller import WalkingController
from src.utils.terminal_logger import TerminalLogger as logger

sl = 0.12
headless = "--headless" in sys.argv
for a in sys.argv[1:]:
    if not a.startswith("--"):
        sl = float(a)

scene = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     '../assets/unitree_g1/scene_stairs.xml'))
env = MujocoEnv(scene, rate_hz=500.0)
c = WalkingController(env.model, env.data, {"gait": {"step_length": sl, "max_steps": 30}})
c.reset()
if not headless:
    env.init_viewer()

lf, rf = c.robot.lfoot_bid, c.robot.rfoot_bid
max_foot, fell = 0.0, False
start = env.data.time
while env.data.time - start < 25:
    t0 = time.time()
    env.data.ctrl[:] = c.update(); env.step()
    max_foot = max(max_foot, env.data.xpos[lf][2], env.data.xpos[rf][2])
    if env.data.qpos[2] < 0.45:
        logger.error(f"fell at t={env.data.time-start:.2f}s"); fell = True; break
    if not headless:
        if not env.viewer.is_running():
            break
        env.sync_viewer()
        dt = time.time() - t0
        if dt < env.model.opt.timestep:
            time.sleep(env.model.opt.timestep - dt)
logger.info(f"SUMMARY stairs: stride={sl} steps={c.last['step']} "
            f"highest_foot_z={max_foot:.3f} com_x={c.last['com'][0]:.3f} fell={fell}")
if not headless:
    time.sleep(0.5); env.close_viewer()
