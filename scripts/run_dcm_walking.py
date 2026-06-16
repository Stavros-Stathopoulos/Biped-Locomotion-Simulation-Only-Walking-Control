"""
run_dcm_walking.py — Offline DCM-based walking for the Unitree G1.

Pipeline:  footstep plan → DCM/LIPM trajectory → QP IK → MuJoCo replay.

Usage
-----
    python scripts/run_dcm_walking.py [--physics] [--steps 5] [--distance 0.5]
"""

import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.controllers.unitree_g1 import UnitreeG1
from src.controllers.dcm_walking_controller import plan_dcm_walk, replay_physics
from src.planners.dcm_trajectory_planner import GaitTiming


def main() -> None:
    p = argparse.ArgumentParser(description="G1 DCM walking simulation")
    p.add_argument("--physics", action="store_true",
                   help="Replay with PD torque control (default: kinematic)")
    p.add_argument("--steps", type=int, default=5,
                   help="Number of full step cycles")
    p.add_argument("--distance", type=float, default=0.5,
                   help="Total travel distance [m]")
    p.add_argument("--theta", type=float, default=0.0,
                   help="Heading angle [rad]")
    p.add_argument("--arc-height", type=float, default=0.05,
                   help="Swing foot lift height [m]")
    p.add_argument("--ssp", type=float, default=0.4,
                   help="Single support duration [s]")
    p.add_argument("--dsp", type=float, default=0.2,
                   help="Double support duration [s]")
    p.add_argument("--subsample", type=int, default=1,
                   help="IK subsampling factor (2 = half waypoints)")
    args = p.parse_args()

    print("=" * 60)
    print("  Unitree G1 — DCM/LIPM Walking Trajectory Generator")
    print("=" * 60)

    # 1. Load robot
    print("\n[1/3] Loading G1 model ...")
    robot = UnitreeG1()

    # 2. Plan trajectory
    timing = GaitTiming(
        single_support_s=args.ssp,
        double_support_s=args.dsp,
    )
    print(f"\n[2/3] Planning DCM trajectory ({args.steps} cycles, "
          f"{args.distance:.2f} m) ...")

    traj, planning_dt = plan_dcm_walk(
        robot,
        n_steps=args.steps,
        travel_distance_m=args.distance,
        theta_rad=args.theta,
        arc_height_m=args.arc_height,
        timing=timing,
        subsample=args.subsample,
    )
    print(f"\n  → {len(traj)} joint-space waypoints (dt={planning_dt:.4f}s)")

    # 3. Replay
    mode = "physics (PD)" if args.physics else "kinematic"
    print(f"\n[3/3] Replaying ({mode}) — close the viewer to exit.\n")

    if args.physics:
        replay_physics(robot, traj, planning_dt)
    else:
        robot.visualize_traj(traj)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nFatal: {exc}")
        raise
