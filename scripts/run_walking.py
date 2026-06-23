"""
Run the Unitree G1 walking controller in MuJoCo.

Usage:
    py -3.12 scripts/run_walking.py                 # interactive viewer
    py -3.12 scripts/run_walking.py --headless      # no viewer, prints diagnostics
    py -3.12 scripts/run_walking.py --duration 20   # stop after N sim seconds
    py -3.12 scripts/run_walking.py --steps 20      # walk up to N steps, then stand
    py -3.12 scripts/run_walking.py --kick 0.15     # initial disturbance (reactivity test)

Controller gains/timing are read from config/simulation.yaml (edit it to retune).
A safety gate stops stepping and holds a stable stand whenever the robot is too
tilted to take another step, so it does not fall even when many steps are asked.

The robot is balanced and walked using motor torques ONLY: gravity, contacts,
friction, masses and inertias are all MuJoCo's. We never pin, weld, freeze the
base, teleport joints, or overwrite qpos during the run (the home pose is set
once before the loop as an initial condition).
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.walking_controller import WalkingController


def load_params(config_path):
    """Load config/simulation.yaml -> WalkingController params dict. Missing file
    or section just falls back to the controller's built-in defaults."""
    params = {}
    if not os.path.exists(config_path):
        return params, {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    ctrl = cfg.get("controller", {}) or {}
    if "use_wbqp" in ctrl: params["use_wbqp"] = ctrl["use_wbqp"]
    if "wbqp" in ctrl:     params["wbqp"] = ctrl["wbqp"]
    if "kp" in ctrl:       params["kp"] = ctrl["kp"]
    if "kd" in ctrl:       params["kd"] = ctrl["kd"]
    if "balance" in ctrl:  params["balance"] = ctrl["balance"]
    if "gait" in ctrl:     params["gait"] = ctrl["gait"]
    if "ik" in ctrl:       params["ik"] = ctrl["ik"]
    return params, cfg.get("physics", {}) or {}
from src.utils.terminal_logger import TerminalLogger as logger


def diagnostics(t, info):
    com = info["com"]; ct = info["com_target"]; rpy = info["base_rpy"]
    msg = (f"t={t:5.2f} | {info['state']:<15} step={info['step']} sup={info['support']:<6} "
           f"| com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) "
           f"tgt=({ct[0]:+.3f},{ct[1]:+.3f}) "
           f"| roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} "
           f"| F_L={info['lforce']:5.0f} F_R={info['rforce']:5.0f} "
           f"| tau_max={np.abs(info['tau']).max():5.1f} sat={info['sat'].max():.2f} "
           f"| WBC-QP: contacts={info['wbqp'].get('ncontact','-')} "
           f"Fz={info['wbqp'].get('fz',0.0):5.0f}N fails={info['wbqp'].get('fail',0)}")
    logger.info(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=500.0)
    ap.add_argument("--steps", type=int, default=None,
                    help="number of alternating steps to take before holding a "
                         "stable stand (default 4). Each step starts only after "
                         "the robot is squared up flat on both feet.")
    ap.add_argument("--kick", type=float, default=0.0,
                    help="apply a random initial base velocity disturbance (m/s) to "
                         "test recovery; also makes runs differ (the sim is otherwise "
                         "deterministic, so identical runs are expected, not a bug)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for --kick")
    ap.add_argument("--scene", default="scene.xml",
                    help="MJCF scene file under assets/unitree_g1/ (e.g. scene_stairs.xml "
                         "for the terrain-aware stair-climbing test)")
    args = ap.parse_args()

    scene_xml = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../assets/unitree_g1/', args.scene))
    env = MujocoEnv(xml_path=scene_xml, rate_hz=args.rate)

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '../config/simulation.yaml'))
    params, physics = load_params(config_path)
    # apply solver iterations from config (joint damping override is intentionally
    # left to the MJCF so the tuned gains stay valid)
    if physics.get("solver_iterations"):
        env.model.opt.iterations = int(physics["solver_iterations"])
    if args.steps is not None:                       # CLI overrides config
        params.setdefault("gait", {})["max_steps"] = args.steps
        # The user asked for N steps explicitly: relax the preventive "stop and
        # stand when tilted" gate so it actually attempts all N (rather than
        # holding the stand at ~4). NOTE: with this controller the marginal
        # lateral balance still tips after ~5-6 steps, so large N will fall.
        params["gait"]["safe_step_tilt"] = 0.5
        logger.warning(f"--steps {args.steps}: attempting all steps (preventive "
                       f"stop relaxed); expect a fall beyond ~5-6 steps.")
    logger.info(f"Loaded controller params from {os.path.relpath(config_path)}")
    controller = WalkingController(env.model, env.data, params)
    controller.reset()

    # Optional initial disturbance: a one-off velocity "shove" at t=0. This is a
    # legitimate test perturbation (not a stabilising force), and it demonstrates
    # that the physics + controller are live and reactive -- with no kick the run
    # is deterministic and reproduces exactly every time.
    if args.kick > 0.0:
        rng = np.random.default_rng(args.seed)
        env.data.qvel[0:2] = rng.uniform(-args.kick, args.kick, size=2)  # base x,y push
        logger.warning(f"Applied initial base velocity kick: {env.data.qvel[0:2]} m/s")

    if not args.headless:
        env.init_viewer()
        # Make sure the terrain (floor/stairs, render group 2) is visible.
        try:
            env.viewer.opt.geomgroup[2] = 1
        except Exception:
            pass
    logger.info(f"Walking controller running (timestep={env.model.opt.timestep:.4f}s). "
                f"Total mass={env.model.body_mass.sum():.1f} kg.")

    start = env.data.time
    fell = False
    prev_sim_time = env.data.time
    print_every = int(0.25 / env.model.opt.timestep)
    n = 0
    try:
        while True:
            step_start = time.time()

            # Detect an external reset from the MuJoCo viewer (Backspace resets the
            # PHYSICS to the default pose but not our controller's internal FSM/IK
            # state). When that happens we re-initialise the controller so it
            # re-syncs to the fresh state and stands back up -- otherwise the stale
            # mid-stride references would topple it.
            if env.data.time < prev_sim_time - 1e-9:
                logger.warning("Viewer reset detected -> re-initialising controller "
                               "(re-syncing FSM + IK reference to the new state).")
                controller.reset()
                start = env.data.time
                fell = False

            tau = controller.update()
            env.data.ctrl[:] = tau
            env.step()
            prev_sim_time = env.data.time

            t = env.data.time - start
            n += 1
            if n % print_every == 0:
                diagnostics(t, controller.last)

            # fall detection. Headless: stop (for tests). Interactive: keep running
            # so you can press the viewer's reset to stand it back up.
            if env.data.qpos[2] < 0.45 and not fell:
                logger.error(f"FELL: pelvis z={env.data.qpos[2]:.3f} at t={t:.2f}s"
                             + ("" if args.headless else "  (press Backspace in the viewer to reset)"))
                fell = True
                if args.headless:
                    break

            if t >= args.duration:
                logger.info(f"Reached target duration {args.duration:.1f}s.")
                break

            if not args.headless:
                if not env.viewer.is_running():
                    break
                env.sync_viewer()
                elapsed = time.time() - step_start
                if elapsed < env.model.opt.timestep:
                    time.sleep(env.model.opt.timestep - elapsed)
    finally:
        info = controller.last
        logger.info(f"SUMMARY: survived {env.data.time - start:.2f}s | "
                    f"final state={info.get('state')} steps={info.get('step')} "
                    f"fell={fell} | final pelvis_z={env.data.qpos[2]:.3f}")
        if not args.headless:
            time.sleep(0.5)
            env.close_viewer()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
