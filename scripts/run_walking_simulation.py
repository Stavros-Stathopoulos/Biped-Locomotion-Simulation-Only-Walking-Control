"""
run_walking_simulation.py — Offline Trajectory Walking for the Unitree G1.

Generates a complete walking trajectory via QP-based inverse kinematics
(offline), then replays it in the MuJoCo viewer.

Usage
-----
    python scripts/run_walking_simulation.py [--mode march|walk] [--physics]

Modes:
    march   : Simple alternating legs, no explicit CoM shift (default)
    walk    : Full walk with CoM shift before each swing phase

Replay:
    --physics : Use PD torque control with MuJoCo physics (default: kinematic)
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.controllers.unitree_g1 import UnitreeG1


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 offline walking simulation")
    parser.add_argument("--mode", choices=["march", "walk"], default="march",
                        help="Gait mode: 'march' (simple) or 'walk' (with CoM shift)")
    parser.add_argument("--physics", action="store_true",
                        help="Replay with PD torque control (default: kinematic)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of full step cycles")
    parser.add_argument("--distance", type=float, default=0.5,
                        help="Total travel distance in metres")
    parser.add_argument("--theta", type=float, default=0.0,
                        help="Heading angle in radians (0 = forward)")
    parser.add_argument("--arc-height", type=float, default=0.05,
                        help="Foot lift height during swing (m)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Unitree G1 — Offline Walking Trajectory Generator")
    print("=" * 60)

    # 1. Instantiate the robot controller (loads model, sets initial pose)
    print("\n[1/3] Loading G1 model and setting walk pose ...")
    robot = UnitreeG1()

    # 2. Generate the walking trajectory offline
    print(f"\n[2/3] Computing {args.mode} trajectory "
          f"({args.steps} steps, {args.distance:.2f} m, θ={args.theta:.2f} rad) ...")

    if args.mode == "march":
        traj = robot.march(
            n_steps=args.steps,
            travel_distance=args.distance,
            time_step=1.0,
            theta=args.theta,
            arc_height=args.arc_height,
        )
    else:
        traj = robot.walk(
            n_steps=args.steps,
            step_length=args.distance,
            left_swing_time=1.0,
            right_swing_time=1.0,
            shift_time=0.5,
            theta=args.theta,
            arc_height=args.arc_height,
        )

    print(f"\n  → Trajectory computed: {len(traj)} waypoints")

    # 3. Replay in viewer
    replay_mode = "physics (PD control)" if args.physics else "kinematic"
    print(f"\n[3/3] Replaying trajectory ({replay_mode}) ...")
    print("       Close the viewer window to exit.\n")

    if args.physics:
        robot.position_control(traj)
    else:
        robot.visualize_traj(traj)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise