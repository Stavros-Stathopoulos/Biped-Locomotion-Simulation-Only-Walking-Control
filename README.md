# ECE_DK801-Robotics-Systems-I

This repository contains the final project for the course ECE_DK801 Robotics Systems I for the academic year 2025 - 2026

---

## Contents

- [Team](#team)
- [Project Description](#project-description)
  - [Robot](#robot)
  - [Deployment](#deployment)
  - [Desription](#description)
  - [Core Tasks](#core-tasks)
  - [Relevant Methods](#relevant-methods)
  - [Evaluation](#evaluation)
- [Runbook](#runbook)
- [Refrences](#refrences)

## Team

Team 5 

### Teammembers

- Mpantekas Nikolaos up1092562
- Stathopoulos Stavros up1101069

---

## Project Description

### Project 5: 

Biped Locomotion Challenge - Simulation - only Walking Control

#### Robot

`Unitree Robotics G1 humanoid` or `Pal Robotics Talos humanoid`

#### Deployment

MuJoCo simulation environment

#### Description

In this project, teams will develop a simulated bipedal walking controller capable of generating stable locomotion through dynamic balance and centerof-mass control. The project is simulation-only and focuses on the fundamental principles of humanoid walking, including balance maintenance, weight shifting, and stable foot-to-foot transitions. 

The core objective is to enable the humanoid robot to walk by continuously moving its center of mass (CoM) over the supporting foot during each step. Teams must design controllers that coordinate the robot’s posture, balance, and stepping behavior so that stable single-support phases can be achieved without falling.

---

#### Core Tasks

- Generate stable walking motions for a simulated humanoid robot.
- Plan and control center-of-mass (CoM) trajectories to achieve stable locomotion.
- Shift the robot’s weight appropriately to stabilize over the supporting foot during single-support phases.
- Coordinate foot placement and body motion to enable continuous stepping and balance maintenance.
- Maintain stability during transitions between double-support and singlesupport phases.
- Recover from small disturbances, balance errors, or imperfect foot placements during walking.

---

#### Relevant Methods

Finite-state machines, center-of-mass planning, zeromoment point (ZMP) control, linear inverted pendulum models (LIPM), capture point methods, model predictive control, inverse kinematics, whole-body control, trajectory optimization, and stability analysis.

---

#### Evaluation

Teams will be evaluated based on their ability to achieve stable continuous walking, successfully stabilize over a single support foot, and maintain balance during support transitions. Additional evaluation criteria include walking distance and duration before falling, smoothness and naturalness of the generated gait, robustness to disturbances or modeling errors, and performance on unseen walking scenarios or terrain variations (e.g. small stairs).

---

## Runbook

---

## Refrences

---
