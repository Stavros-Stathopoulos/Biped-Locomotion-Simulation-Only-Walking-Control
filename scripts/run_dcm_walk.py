"""
DCM (capture-point) continuous-walking demo for the Unitree G1.

    py -3.11 scripts/run_dcm_walk.py                  # interactive viewer
    py -3.11 scripts/run_dcm_walk.py --headless       # no viewer, prints diagnostics
    py -3.11 scripts/run_dcm_walk.py --duration 30
    py -3.11 scripts/run_dcm_walk.py --step-length 0.06 --step-width 0.22

Pipeline each control tick (torques only; gravity, contacts, friction are MuJoCo's;
nothing is pinned and qpos is never overwritten during the run):

    state  = estimator.update(data)
    refs   = dcm.update(state, dt)             # DCM/LIPM pattern + capture stepping
    tau    = wbqp.compute(..., com_acc_ff=refs['com_acc_ff'])
    data.ctrl[:] = tau ; mj_step
"""

import sys, os, time, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.env.mujoco_env import MujocoEnv
from src.controllers.robot_model import RobotModel, StateEstimator
from src.controllers.dcm_gait import DCMWalkingGait
from src.controllers.wbqp import WholeBodyQP
from src.utils.terminal_logger import TerminalLogger as logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=500.0)
    ap.add_argument("--scene", default="scene.xml")
    # Defaults are the STABLE continuous-walking config: forward steps at a fast
    # cadence. (In-place stepping, --step-length 0, is much harder for this robot
    # and is not the default.)
    ap.add_argument("--step-length", type=float, default=0.03)
    ap.add_argument("--step-width", type=float, default=0.22)
    ap.add_argument("--t-ss", type=float, default=0.30)
    ap.add_argument("--t-ds", type=float, default=0.12)
    ap.add_argument("--k-dcm", type=float, default=2.5)
    ap.add_argument("--k-cap", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--kick", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    scene = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../assets/unitree_g1/', args.scene))
    env = MujocoEnv(scene, rate_hz=args.rate)
    robot = RobotModel(env.model)
    est = StateEstimator(robot)
    gait = DCMWalkingGait(robot, {
        "step_length": args.step_length, "step_width": args.step_width,
        "t_ss": args.t_ss, "t_ds": args.t_ds,
        "k_dcm": args.k_dcm, "k_cap": args.k_cap, "max_steps": args.max_steps,
    })
    wbqp = WholeBodyQP(robot, {
        # tighten pelvis/torso orientation so the body does not pitch away while
        # the QP produces the commanded CoM acceleration (the point-mass DCM model
        # ignores this rotational coupling, so the tracker must suppress it)
        "kp_torso": 250.0, "kd_torso": 30.0, "w_torso": 12.0,
        # stronger sagittal CoM tracking + velocity damping to hold the forward
        # speed on the plan (stops the slow forward/backward creep)
        "kp_com": [90.0, 60.0, 90.0], "kd_com": [28.0, 16.0, 19.0],
    })

    robot.set_home(env.data)
    gait.reset(est.update(env.data))
    dt = env.model.opt.timestep

    if args.kick > 0.0:
        rng = np.random.default_rng(args.seed)
        env.data.qvel[0:2] = rng.uniform(-args.kick, args.kick, size=2)
        logger.warning(f"kick: base vel {env.data.qvel[0:2]} m/s")

    if not args.headless:
        env.init_viewer()
        try:
            env.viewer.opt.geomgroup[2] = 1
        except Exception:
            pass

    logger.info(f"DCM walking: mass={env.model.body_mass.sum():.1f}kg dt={dt:.4f} "
                f"step_len={args.step_length} step_w={args.step_width} "
                f"T_ss={args.t_ss} T_ds={args.t_ds} k_dcm={args.k_dcm} k_cap={args.k_cap}")

    start = env.data.time
    n = 0
    fell = False
    print_every = int(0.25 / dt)
    com0 = env.data.subtree_com[0].copy()
    try:
        while True:
            t0 = time.time()
            state = est.update(env.data)
            refs, info = gait.update(state, dt)
            tau = wbqp.compute(env.data, refs["com_des"], refs["com_vel_des"],
                               refs["torso_R"], refs["contacts"], refs["swing"],
                               com_acc_ff=refs["com_acc_ff"])
            env.data.ctrl[:] = tau
            env.step()

            t = env.data.time - start
            n += 1
            if n % print_every == 0:
                com = state["com"]; rpy = state["base_rpy"]
                z = info["zmp_cmd"]; xi = info["dcm"]
                logger.info(
                    f"t={t:5.2f} | {info['state']:<5} step={info['step_count']} "
                    f"stance={info['stance']:<5} | com=({com[0]:+.3f},{com[1]:+.3f},{com[2]:.3f}) "
                    f"dcm=({xi[0]:+.3f},{xi[1]:+.3f}) zmp=({z[0]:+.3f},{z[1]:+.3f}) "
                    f"| roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} "
                    f"| F_L={state['lfoot_force']:4.0f} F_R={state['rfoot_force']:4.0f} "
                    f"| tau={np.abs(tau).max():4.0f} sat={np.max(np.abs(tau)/np.maximum(robot.tau_limit,1e-9)):.2f} "
                    f"fails={wbqp.info.get('fail',0)}")

            if env.data.qpos[2] < 0.45 and not fell:
                logger.error(f"FELL: pelvis z={env.data.qpos[2]:.3f} at t={t:.2f}s "
                             f"steps={info['step_count']}")
                fell = True
                if args.headless:
                    break

            if t >= args.duration:
                logger.info(f"Reached duration {args.duration:.1f}s.")
                break

            if not args.headless:
                if not env.viewer.is_running():
                    break
                env.sync_viewer()
                el = time.time() - t0
                if el < dt:
                    time.sleep(dt - el)
    finally:
        travel = env.data.subtree_com[0] - com0
        logger.info(f"SUMMARY: survived {env.data.time-start:.2f}s | "
                    f"steps={gait.step_count} fell={fell} "
                    f"pelvis_z={env.data.qpos[2]:.3f} "
                    f"travel=({travel[0]:+.3f},{travel[1]:+.3f}) m")
        if not args.headless:
            time.sleep(0.3); env.close_viewer()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("interrupted")
