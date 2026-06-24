# ECE_DK801-Robotics-Systems-I

This repository contains the final project for the course ECE_DK801 Robotics Systems I for the academic year 2025 - 2026

---

## Contents

- [Team](#team)
- [Project Description](#project-description)
  - [Robot](#robot)
  - [Deployment](#deployment)
  - [Description](#description)
  - [Core Tasks](#core-tasks)
  - [Relevant Methods](#relevant-methods)
  - [Evaluation](#evaluation)
- [Runbook](#runbook)
- [References](#references)

## Team

Team 5 

### Team Members

- Mpantekas Nikolaos up1092562
- Stathopoulos Stavros up1101069

---

## Project Description

### Project 5: 

Biped Locomotion Challenge - Simulation-only Walking Control

#### Robot

`Unitree Robotics G1 humanoid` or `Pal Robotics Talos humanoid`

#### Deployment

MuJoCo simulation environment

#### Description

In this project, teams will develop a simulated bipedal walking controller capable of generating stable locomotion through dynamic balance and center-of-mass control. The project is simulation-only and focuses on the fundamental principles of humanoid walking, including balance maintenance, weight shifting, and stable foot-to-foot transitions. 

The core objective is to enable the humanoid robot to walk by continuously moving its center of mass (CoM) over the supporting foot during each step. Teams must design controllers that coordinate the robot's posture, balance, and stepping behavior so that stable single-support phases can be achieved without falling.

---

#### Core Tasks

- Generate stable walking motions for a simulated humanoid robot.
- Plan and control center-of-mass (CoM) trajectories to achieve stable locomotion.
- Shift the robot's weight appropriately to stabilize over the supporting foot during single-support phases.
- Coordinate foot placement and body motion to enable continuous stepping and balance maintenance.
- Maintain stability during transitions between double-support and single-support phases.
- Recover from small disturbances, balance errors, or imperfect foot placements during walking.

---

#### Relevant Methods

Finite-state machines, center-of-mass planning, zero-moment point (ZMP) control, linear inverted pendulum models (LIPM), capture point methods, model predictive control, inverse kinematics, whole-body control, trajectory optimization, and stability analysis.

---

#### Evaluation

Teams will be evaluated based on their ability to achieve stable continuous walking, successfully stabilize over a single support foot, and maintain balance during support transitions. Additional evaluation criteria include walking distance and duration before falling, smoothness and naturalness of the generated gait, robustness to disturbances or modeling errors, and performance on unseen walking scenarios or terrain variations (e.g. small stairs).

---

## Runbook

### Prerequisites
* **Linux Host Engine** with Docker installed.
* **X11 Display Server** running natively for 3D physics engine visualization.
* **Direct Rendering Infrastructure (`/dev/dri`)** for hardware-accelerated graphics pipelines.

>[!WARNING]
> It is tested and well working in Debian based operating systems like Ubuntu

---

### Execution Sequence

Execute the following commands sequentially inside your host terminal:

```bash
# 1. Pull the concrete snapshot image from GHCR
docker pull ghcr.io/stavros-stathopoulos/biped-locomotion-simulation-only-walking-control:sha-f67cee8

# 2. Grant the local container authority to connect to your host's display server
xhost +local:root

# 3. Execute the simulation loop using explicit system and graphic-device mappings
docker run -it --rm \
  --name biped_simulation \
  --ipc=host \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e MUJOCO_GL=glx \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri:/dev/dri \
  ghcr.io/stavros-stathopoulos/biped-locomotion-simulation-only-walking-control:sha-f67cee8 \
  scripts/run_dcm_walk.py --scene scene.xml

# 4. Immediately revoke screen permissions once the simulation exits
xhost -local:root
```
>[!TIP]
>Instead of `--scene scene.xml` you can use also `--scene scene_stairs.xml` or `--scene scene_tilted.xml`
---

## References

- Unitree G1 Robot: https://www.unitree.com/g1/
- MuJoCo Physics Engine: https://mujoco.readthedocs.io/

