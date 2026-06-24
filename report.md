# Biped Locomotion Challenge -- Simulation-Only Walking Control

## Project Report

**Robot:** Unitree Robotics G1 humanoid robot (29 DoF)  
**Deployment:** MuJoCo simulation environment  
**Total mass:** 35.11 kg  
**Control rate:** 500 Hz (dt = 0.002 s)

---

## 1. System Overview

The controller produces stable continuous walking on a floating-base humanoid driven entirely by motor torques. No part of the robot is welded or pinned; gravity, contacts, friction, and inertia are handled by MuJoCo. The only runtime input is `data.ctrl[:]` -- the 29-dimensional torque vector.

The pipeline that runs at every 2 ms control tick is (`run_dcm_walk.py`):

```python
state  = est.update(env.data)                          # Layer 1: state estimation
refs, info = gait.update(state, dt)                    # Layer 2: DCM gait FSM + references
tau = wbqp.compute(env.data, refs["com_des"],          # Layer 3: whole-body QP
                   refs["com_vel_des"], refs["torso_R"],
                   refs["contacts"], refs["swing"],
                   com_acc_ff=refs["com_acc_ff"])
env.data.ctrl[:] = tau                                 # Layer 4: apply torques
mujoco.mj_step(model, data)                            # MuJoCo physics
```

No `data.qpos` overwrite occurs during the simulation loop.

---

## 2. Robot Model and Actuation

The G1 is modeled in MuJoCo XML (`g1_29dof.xml`) with a floating base (6 unactuated DoF) and 29 torque-controlled motor actuators:

| Joint group | Joints | Torque limit |
|---|---|---|
| Hip pitch/roll/yaw | 6 (3 per leg) | 88 Nm |
| Knee | 2 | 139 Nm |
| Ankle pitch/roll | 4 (2 per leg) | 50 Nm |
| Waist yaw/roll/pitch | 3 | 50--88 Nm |
| Shoulder/elbow/wrist | 14 (7 per arm) | 5--25 Nm |

All actuators are declared as `<motor>` elements (pure torque control):

```xml
<motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
```

The floor is a plane geom with default friction. Gravity is the MuJoCo default (0, 0, -9.81).

Foot contact geometry consists of four corner spheres on each ankle-roll link (`geom_type == SPHERE`, `contype == 1`). Foot dimensions: 17 cm long (heel -5 cm, toe +12 cm from ankle), 6 cm wide.

---

## 3. State Estimation (Layer 1)

Implemented in `robot_model.py : StateEstimator`. At each tick it reads the MuJoCo state and computes:

| Quantity | Source |
|---|---|
| CoM position | `data.subtree_com[0]` |
| CoM velocity | `J_com @ data.qvel` (Jacobian from `mj_jacSubtreeCom`) |
| Base orientation (roll, pitch, yaw) | quaternion `data.qpos[3:7]` converted via `quat_to_rpy` |
| Foot poses | `data.xpos[foot_bid]`, `data.xmat[foot_bid]` |
| Foot contact (boolean) | Normal force > 15 N threshold |
| Support phase | `DOUBLE`, `LEFT`, `RIGHT`, or `FLIGHT` |

Contact detection iterates over active MuJoCo contacts, sums the contact-frame normal force per foot, and thresholds:

```python
def _foot_normal_force(self, data, geom_set):
    total = 0.0
    forcetorque = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        if con.geom1 in geom_set or con.geom2 in geom_set:
            mujoco.mj_contactForce(self.m, data, c, forcetorque)
            total += abs(forcetorque[0])
    return total
```

---

## 4. Gait Pattern Generator -- DCM/Capture Point (Layer 2)

Implemented in `dcm_gait.py : DCMWalkingGait`.

### 4.1 Why DCM instead of quasi-static CoM shifting

The G1's feet are small (17x6 cm) and ankle-roll torque is limited to +/-50 Nm. Lateral balance cannot be held by the ankle alone -- it must come from where the swing foot is placed. The Divergent Component of Motion (capture point) tells the controller exactly where to step to arrest the falling CoM, and the DCM feedback law determines the ZMP to command between steps. Together they produce a stable walking limit cycle.

### 4.2 Linear Inverted Pendulum Model

The underlying dynamics model is the LIPM (constant CoM height):

```
omega = sqrt(g / z_c)           # natural frequency (~3.80 rad/s at z_c = 0.679 m)
xi    = c + c_dot / omega       # DCM (capture point)
xi_dot = omega * (xi - p)       # DCM dynamics (p = ZMP)
c_dot  = omega * (xi - c)       # CoM follows the DCM
```

### 4.3 Finite State Machine

The FSM has four states:

```
STAND  ->  START  ->  SS  <->  DS
                      ^        |
                      |--------|
```

| State | Duration | Contacts | Description |
|---|---|---|---|
| `STAND` | 1.2 s | Both feet | CoM held at midpoint between feet. Waits before initiating walk. |
| `START` | 1.0 s | Both feet | One-off weight shift onto the first stance foot. Uses a quintic (minimum-jerk) interpolation from the midpoint DCM to the first step's initial DCM. |
| `DS` (double support) | 0.12 s | Both feet | Holds the DCM reference at the upcoming single-support initial DCM so the swing begins consistently with the limit cycle. |
| `SS` (single support) | 0.30 s | Stance foot only | Swing foot tracks a trajectory to its capture target; DCM follows the reference exponential from stance to next footstep. |

Transition logic (not purely time-based):

```python
# SS -> DS: end of swing time OR early contact after 60% of swing
sw_contact = state[f"{swing[0]}foot_contact"]
if frac >= 1.0 or (frac > 0.6 and sw_contact):
    # plant foot, switch stance, enter DS
```

Safety abort (tilt-triggered):

```python
tilt = max(abs(rpy[0]), abs(rpy[1]))
if self.walking and tilt > self.max_tilt and self.state != self.STAND:
    self.walking = False
    self._enter(self.STAND)
```

### 4.4 DCM Reference Generation

During single support, the reference DCM travels exponentially from the stance foot toward the next footstep (backward recursion over a 6-step preview horizon):

```python
def _xi_eos_current(self, p_st_xy, future):
    E = np.exp(-self.omega * self.t_ss)
    seq = [p_st_xy] + list(future)
    xi = seq[-1].copy()
    for i in range(len(seq) - 1, 0, -1):
        xi = seq[i] + (xi - seq[i]) * E
    return xi

def _xi_ref(self, p_stance, xi_eos, tau):
    return p_stance + (xi_eos - p_stance) * np.exp(self.omega * (tau - self.t_ss))
```

The reference CoM is integrated from the reference DCM each tick:

```python
cd_ref = self.omega * (xi_ref - self.c_ref)
self.c_ref = self.c_ref + dt * cd_ref
cdd_ref = self.omega**2 * (self.c_ref - p_ref)
```

The acceleration feed-forward `cdd_ref` is the signal that makes the QP command the correct ground-reaction force (i.e. ZMP control).

### 4.5 ZMP Feedback

DCM tracking error is converted into a commanded ZMP (Englsberger's law):

```python
p_cmd = p_ref + (1.0 + self.k_dcm / self.omega) * (xi_meas - xi_ref)
```

with `k_dcm = 2.5`. This is clamped to the support polygon and used for diagnostics. The CoM acceleration feed-forward is the primary signal to the QP.

### 4.6 Capture-Point Foot Placement

During single support, the controller predicts where the DCM will be at end-of-step and adjusts the swing foot landing site in both sagittal (x) and lateral (y) directions:

```python
xi = com[:2] + com_vel[:2] / self.omega
xi_pred_eos = p_st + (xi - p_st) * np.exp(self.omega * (self.t_ss - tau))

# Forward correction (adapts to slopes and velocity disturbances)
dx = np.clip(self.k_cap * (xi_pred_eos[0] - xi_eos_plan[0]),
             -self.max_dx, self.max_dx)      # max_dx = 0.04 m
p_next[0] = p_nominal[0] + dx

# Lateral correction (prevents sideways fall)
dy = np.clip(self.k_cap * (xi_pred_eos[1] - xi_eos_plan[1]),
             -self.max_dy, self.max_dy)      # max_dy = 0.06 m
p_next[1] = p_nominal[1] + dy
```

with `k_cap = 0.8`. The nominal footstep uses absolute positions (`x0 + k * step_length`) so the gait does not drift forward.

### 4.7 Swing Foot Trajectory

The swing foot follows a quintic (minimum-jerk) trajectory with a sinusoidal arch:

```python
def _swing_pose(self, frac):
    s = _quintic(frac)                             # 10s^3 - 15s^4 + 6s^5
    xy = (1 - s) * self.swing_start[:2] + s * self.swing_goal[:2]
    arch = (self.step_height + rise) * np.sin(np.pi * frac)
    z = (1 - s) * z0 + s * zg + arch
    return np.array([xy[0], xy[1], z])
```

Default step height is 0.15 m (enough clearance to avoid scuffing).

### 4.8 Footstep Planning

The alternating footstep pattern is:

```
left foot:  y = +step_width / 2  (even indices)
right foot: y = -step_width / 2  (odd indices)
x_k = x0 + k * step_length
```

Default parameters: `step_length = 0.03 m`, `step_width = 0.22 m`.

**Note on Raibert-style feedback:** The forward footstep correction (Section 4.6) serves a similar purpose to the classical Raibert heuristic `x_foot = x_hip + 0.5*T*v_x + k*(v_x - v_des)`. However, it is formulated through the DCM prediction rather than a velocity-proportional rule. The Raibert formula per se is not explicitly implemented; the capture-point adjustment achieves the same effect within the DCM framework.

---

## 5. Whole-Body QP Controller (Layer 3)

Implemented in `wbqp.py : WholeBodyQP`. This is a true online inverse-dynamics QP solved every 2 ms tick.

### 5.1 Decision Variables

```
x = [ qddot (35),  lambda (3 per active contact point) ]
```

### 5.2 Objective (Weighted Least Squares)

```
min  sum_t  w_t || J_t * qddot - a_t* ||^2
     + r_qdd * ||qddot||^2
     + r_f * ||lambda||^2
```

Tasks in priority order (by weight):

| Task | Jacobian | Desired acceleration | Weight |
|---|---|---|---|
| CoM tracking | `J_com` (3 x nv) | `kp*(com_des - com) + kd*(v_des - v) + acc_ff` | 30.0 |
| Torso orientation | `J_rot_pelvis` (3 x nv) | `kp*rot_error(R_des, R) + kd*(-omega)` | 12.0 |
| Swing foot position | `J_pos_foot` (3 x nv) | `kp*(pos_des - pos) + kd*(vel_des - vel)` | 50.0 |
| Swing foot orientation | `J_rot_foot` (3 x nv) | `kp*rot_error(R_des, R) + kd*(-omega)` | 3.0 |
| Posture regularisation | `I_actuated` (29 x nv) | `kp*(q_home - q) + kd*(-qdot)` | 0.1 (hip_yaw: 9.0) |

### 5.3 Equality Constraints (as stiff soft penalties, w_eq = 10000)

Since the `quadprog` solver does not natively support equality constraints, they are folded into the objective as a very stiff least-squares penalty:

**Floating-base dynamics (6 rows):**

```python
M[:6] @ qddot + h[:6] = Jc[:, :6].T @ lambda
```

**Stance foot no-slip (Baumgarte-damped):**

```python
Jc @ qddot = -kd_contact * (Jc @ qdot)
```

### 5.4 Inequality Constraints

**Friction pyramid** (5 rows per contact point):

```python
|fx| <= mu * fz,   |fy| <= mu * fz,   fz >= fz_min
```

with `mu = 0.7`, `fz_min = 2.0 N`.

**Torque limits** (from the MJCF `actuatorfrcrange`):

```python
-tau_max <= M[6:] @ qddot + h[6:] - Jc[:,6:].T @ lambda <= tau_max
```

### 5.5 Torque Recovery

After solving, joint torques are extracted from the actuated rows of the dynamics equation:

```python
tau = M[6:] @ qddot + h[6:] - Jc[:,6:].T @ lambda
tau = np.clip(tau, -tau_limit, tau_limit)
```

If the QP fails, the controller falls back to gravity compensation: `tau = h[act_dofadr]`.

**Note:** There is no separate torque control layer with an explicit `kp*(q_des - q) + kd*(qd_des - qd)` joint-PD plus gravity compensation, as suggested in some formulations. The QP directly outputs torques that incorporate the dynamics, task objectives, and constraints in a single solve. This is a design choice: the QP already accounts for the full mass matrix and Coriolis terms, so a downstream PD would double-count them.

---

## 6. Terrain Adaptation

### 6.1 Terrain Sensing Module

A ray-casting terrain sensor is available in `terrain.py`:

```python
def ground_height(model, data, x, y, z_start=3.0, default=0.0):
    pnt = np.array([x, y, z_start], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, _GEOMGROUP, 1, -1, geomid)
    if dist < 0 or geomid[0] < 0:
        return default
    return z_start - dist
```

It casts a ray straight down from a starting height and intersects only terrain geoms (group 2), ignoring the robot's own body.

**Note:** This terrain sensor is available and used by an older quasi-static gait planner (`gait.py` / `walking_controller.py`), but the DCM gait (`dcm_gait.py`) does **not** currently integrate it for foot landing height or CoM height adaptation. The DCM controller handles slopes through its capture-point step adjustment (Section 4.6) rather than explicit terrain height measurement. Integrating terrain sensing into the DCM gait was attempted but found to destabilize lateral balance -- the measured foot heights created coupling between the sagittal and lateral planes that the LIPM model does not account for.

### 6.2 Test Scenes

| Scene | Description |
|---|---|
| `scene.xml` | Flat ground plane. The robot walks 30+ seconds, 66 steps, 2+ m forward without falling. |
| `scene_tilted.xml` | Uphill ramp (3 degrees, 1.1 m) followed by a downhill ramp (3 degrees, 1.1 m). The robot walks the full 30 seconds without falling, using the sagittal capture-point adjustment to adapt its step length on the slopes. |

---

## 7. CoM Height Control

The CoM height target tracks the mean foot height so it rises on uphill terrain:

```python
def _com_z(self):
    mean_foot_z = 0.5 * (self.foot["left"][2] + self.foot["right"][2])
    return self.z_c + max(0.0, mean_foot_z - self._foot_z_ref)
```

**Note:** The `max(0.0, ...)` means the height only increases from the initial reference, never decreases. On downhill terrain, the CoM target stays at the peak height. The robot's ability to handle downhill slopes comes from the sagittal capture-point step adjustment (which adjusts foot placement forward/backward) rather than from adapting the CoM height downward. This is a known limitation: very steep or very long downhill sections would eventually cause the legs to over-extend.

---

## 8. Safety and Recovery

The controller monitors for excessive tilt:

```python
tilt = max(abs(rpy[0]), abs(rpy[1]))
if self.walking and tilt > self.max_tilt and self.state != self.STAND:
    self.walking = False
    self._enter(self.STAND)
```

The default `max_tilt` is 0.5 rad (~29 degrees). When triggered, the FSM transitions to `STAND`, both feet are treated as contacts, and the CoM reference returns to the midpoint.

Fall detection is based on pelvis height:

```python
if env.data.qpos[2] < 0.45 and not fell:
    logger.error(f"FELL: pelvis z={env.data.qpos[2]:.3f}")
```

**Note on MPC:** Model Predictive Control is listed among the relevant methods in the assignment but is **not** used in this implementation. The DCM preview recursion over 6 future footsteps serves a conceptually similar role (planning ahead), but it is not a true receding-horizon optimisation -- it computes an analytical closed-form reference from the preview footsteps rather than solving an optimisation problem at each step.

---

## 9. Diagnostic Output

The simulation logs a diagnostic line every 0.25 seconds:

```
t= 5.00 | SS  step=12 stance=left  | com=(+0.260,+0.010,0.706) dcm=(+0.270,-0.024)
         zmp=(+0.261,+0.097) | roll=-0.06 pitch=+0.02 | F_L= 297 F_R=   0
         | tau=  17 sat=0.19 fails=0
```

Fields include: FSM state, step count, stance foot, CoM position, DCM, commanded ZMP, torso roll/pitch, left/right foot normal force (N), max absolute torque, torque saturation ratio, and QP solver failures.

---

## 10. Results Summary

| Metric | Flat ground | Tilted ramps (3 deg) |
|---|---|---|
| Duration before fall | 30.0 s (no fall) | 30.0 s (no fall) |
| Steps completed | 66 | 66 |
| Forward travel | ~2.0 m | ~2.0 m |
| Max torque saturation | 0.40 | 1.00 (brief, on slope transition) |
| Max roll | ~0.07 rad | ~0.07 rad |
| Max pitch | ~0.05 rad | ~0.05 rad (steady state) |

---

## 11. Project Structure

```
project_root/
+-- scripts/
|   +-- run_dcm_walk.py         # Main simulation loop
+-- src/
|   +-- controllers/
|   |   +-- dcm_gait.py         # DCM gait FSM + reference generation
|   |   +-- wbqp.py             # Whole-body inverse-dynamics QP
|   |   +-- robot_model.py      # Robot model wrapper + state estimator
|   |   +-- terrain.py          # Ray-casting terrain sensor
|   |   +-- gait.py             # Older quasi-static gait (not used for walking)
|   |   +-- wbik.py             # Whole-body IK (not used for walking)
|   |   +-- balance.py          # Balance utilities
|   |   +-- walking_controller.py # Older controller (not used for walking)
|   +-- env/
|   |   +-- mujoco_env.py       # MuJoCo environment wrapper
|   +-- utils/
|       +-- terminal_logger.py  # Logging
|       +-- config_parser.py    # Configuration
+-- assets/unitree_g1/
|   +-- g1_29dof.xml            # Robot MJCF model (29 DoF, torque motors)
|   +-- scene.xml               # Flat ground scene
|   +-- scene_tilted.xml        # Uphill + downhill ramp scene
+-- CLAUDE.md                   # Controller design specification
+-- report.md                   # This file
```

---

## 12. What Is and Is Not Implemented

| Assignment requirement | Status | Notes |
|---|---|---|
| Stable walking motions | Implemented | 66 steps, 30 s, 2 m forward on flat ground |
| CoM trajectory planning | Implemented | DCM/LIPM-based reference CoM integrated from DCM reference |
| Weight shifting over support foot | Implemented | START phase shifts CoM over first stance; DS phase transfers between stances |
| Foot placement + body coordination | Implemented | Capture-point step adjustment (both sagittal and lateral) |
| DS/SS transition stability | Implemented | FSM with timed + contact-aware transitions |
| Disturbance recovery | Partially | Capture-point feedback corrects steps; tilt abort stops walking. No explicit push-recovery or step-in-place recovery. |
| ZMP control | Implemented | DCM feedback law produces commanded ZMP; QP realises it via CoM acceleration feed-forward |
| LIPM | Implemented | Core of the DCM gait planner |
| Capture point methods | Implemented | DCM is the capture point; step adjustment uses predicted end-of-step DCM |
| Finite state machine | Implemented | STAND -> START -> SS <-> DS |
| Whole-body control | Implemented | Full inverse-dynamics QP with dynamics, friction, and torque constraints |
| Inverse kinematics | Not used for walking | A WBIK module (`wbik.py`) exists but the walking controller uses inverse dynamics (WBQP) directly |
| Model predictive control | Not implemented | DCM preview recursion is used instead (analytical, not optimisation-based) |
| Trajectory optimisation | Not implemented | Trajectories are generated analytically (quintic interpolation + DCM exponentials) |
| Terrain adaptation | Partial | Sagittal capture stepping handles slopes; explicit terrain height sensing is available but not integrated into DCM gait |
| Walking on small stairs | Available but limited | Terrain ray-casting exists; the older quasi-static gait uses it. The DCM gait handles the tilted ramp scene through step adjustment. |
