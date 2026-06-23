# CLAUDE.md

## Project Mission

This repository contains a MuJoCo humanoid locomotion project for either:

- Unitree Robotics G1 humanoid robot
- PAL Robotics Talos humanoid robot

The goal is to implement a physically valid biped controller that can:

1. Stand upright under gravity.
2. Shift the robot center of mass (CoM) over one support foot.
3. Raise the opposite leg clearly off the floor.
4. Lower the leg back to the ground.
5. Alternate legs.
6. Extend the behavior into small forward steps.

The robot must use a **full-body controller** and must solve the walking/balance problem online using **QP-based whole-body control**.

This is not an animation task. The robot must balance through real MuJoCo physics.

---

## Hard Physical Requirements

The robot must stand and move using **only motor torques**.

Do **not**:

- weld the robot base to the world,
- pin the pelvis,
- add external stabilizing forces,
- apply artificial forces to the torso/base,
- teleport the feet,
- directly overwrite `data.qpos` during the live simulation,
- directly impose support constraints outside MuJoCo contact physics,
- hold the robot upright with invisible constraints.

The only valid runtime control input is:

```python
data.ctrl[:]
```

or the appropriate MuJoCo actuator command interface.

The simulation must use:

- gravity,
- floating-base dynamics,
- MuJoCo contacts,
- foot-ground friction,
- body masses,
- body inertias,
- actuator limits,
- joint limits,
- physical ground reaction forces.

The robot is underactuated. The floating base is not directly actuated.

---

## Important Course Context

The course material emphasizes that humanoid walking is hard because:

- the robot is underactuated,
- contacts appear and disappear,
- impacts create non-smooth dynamics,
- balance is fragile,
- friction, terrain, delay, and modeling errors matter.

The controller must therefore be written as a hybrid locomotion controller with contact phases, not as a single open-loop pose animation.

The expected walking structure should include:

- finite-state machine,
- double-support phase,
- single-support phase,
- swing-foot trajectory,
- footstep placement,
- CoM control,
- feedback stabilization.

---

## Existing Code Context

The existing lab/project code already contains QP-based inverse kinematics ideas.

The whole-body IK formulation tracks:

- left foot pose,
- right foot pose,
- center of mass.

The existing stacked task structure is:

```text
e = [
    e_left_foot
    e_right_foot
    e_CoM
]

J = [
    J_left_foot
    J_right_foot
    J_CoM
]
```

The weighted QP form is:

```text
min_v 1/2 || W (J v - e) ||^2
```

Expanded as:

```text
Q = (WJ)^T (WJ) + lambda I
q = -(WJ)^T W e
```

This existing IK/QP code should be reused, but only as part of a physically valid controller.

Critical distinction:

> IK may generate references, but IK must not directly teleport the robot during live physics.

---

## Required Controller Architecture

Implement a layered full-body controller.

---

# 1. State Estimation Layer

Read the current MuJoCo state:

```python
qpos = data.qpos.copy()
qvel = data.qvel.copy()
```

Compute:

- joint positions,
- joint velocities,
- floating-base pose,
- torso orientation,
- torso angular velocity,
- CoM position,
- CoM velocity,
- left foot pose,
- right foot pose,
- left foot velocity,
- right foot velocity,
- left/right foot contact states.

Foot contact must be detected from actual MuJoCo contacts.

Example logic:

```python
def foot_in_contact(model, data, foot_geom_ids, ground_geom_id):
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in foot_geom_ids and c.geom2 == ground_geom_id) or \
           (c.geom2 in foot_geom_ids and c.geom1 == ground_geom_id):
            return True
    return False
```

Use contact state to identify:

```text
DOUBLE_SUPPORT
LEFT_SUPPORT
RIGHT_SUPPORT
FLIGHT_OR_FAILURE
```

---

# 2. Finite State Machine

Implement walking as a finite-state machine.

Required states:

```text
STAND
SHIFT_COM_LEFT
RIGHT_LEG_LIFT
RIGHT_LEG_HOLD
RIGHT_LEG_LOWER
SHIFT_COM_RIGHT
LEFT_LEG_LIFT
LEFT_LEG_HOLD
LEFT_LEG_LOWER
```

Optional forward-step states:

```text
RIGHT_LEG_SWING_FORWARD
LEFT_LEG_SWING_FORWARD
```

Each state must have:

- start time,
- duration,
- support foot,
- swing foot,
- CoM target,
- foot target,
- transition condition.

Transitions should depend on:

- elapsed time,
- foot contact,
- CoM error,
- torso roll/pitch,
- safety checks.

Do not run a purely open-loop animation. The FSM should react to contact and balance state.

---

# 3. Reference Generation Layer

Generate smooth references for:

- CoM position,
- pelvis height,
- torso orientation,
- support foot pose,
- swing foot pose,
- posture regularization.

Use conservative initial values:

```python
step_height = 0.04       # meters
step_length = 0.05       # meters
step_width = 0.18        # tune to robot
shift_time = 1.0         # seconds
swing_time = 1.0         # seconds
hold_time = 0.3          # seconds
com_lateral_shift = 0.04 # meters, tune carefully
```

For single-support balance:

- If left foot supports, move CoM toward left foot.
- If right foot supports, move CoM toward right foot.

The CoM target should be near the support foot center, not between the feet.

---

# 4. Swing Foot Trajectory

Use a smooth swing-foot trajectory, preferably cubic Hermite or another smooth polynomial.

The trajectory should:

- start at the current swing foot pose,
- lift vertically,
- optionally move forward,
- land softly,
- avoid foot scuffing.

Example vertical profile:

```python
def smoothstep(s):
    return 3*s**2 - 2*s**3

def swing_height_profile(s, h):
    return h * 4.0 * s * (1.0 - s)
```

Example horizontal profile:

```python
p_des = (1.0 - alpha) * p_start + alpha * p_goal
p_des[2] += swing_height_profile(alpha, step_height)
```

where:

```python
alpha = smoothstep(phase_time / swing_time)
```

---

# 5. Footstep Planning

Footsteps define where and when contacts happen.

Use a simple alternating pattern first:

```text
left foot:  y = +step_width / 2
right foot: y = -step_width / 2
x_next = x_current + step_length
```

Then add feedback.

Use a Raibert-style velocity-based foot placement rule:

```python
x_foot = x_hip + 0.5 * T * v_body_x + k_v * (v_body_x - v_des_x)
```

Interpretation:

- if the robot moves too fast, place the next foot farther forward,
- if it moves too slowly, place it less far forward.

Keep the correction small at first.

Example:

```python
k_v = 0.05
max_step_correction = 0.04
```

---

# 6. Online QP Full-Body Controller

Use QP online at every control step or at a slower control frequency.

The controller should solve for desired generalized velocities or accelerations that satisfy whole-body tasks.

At minimum include these tasks:

## Highest priority

Support foot pose should remain fixed relative to the world.

```text
J_support v = e_support
```

## High priority

CoM tracking:

```text
J_com v = e_com
```

## High priority

Torso upright orientation:

```text
J_torso_rot v = e_torso_rot
```

## Medium priority

Swing foot pose tracking:

```text
J_swing v = e_swing
```

## Low priority

Posture regularization:

```text
q -> q_nominal
```

Use weighted least squares inside the QP.

Example task weights:

```python
w_support_foot = np.full(6, 100.0)
w_com = np.array([30.0, 30.0, 40.0])
w_torso = np.array([50.0, 50.0, 20.0])
w_swing_foot = np.full(6, 30.0)
w_posture = np.full(nv, 1.0)
```

The QP should produce reference motion:

```python
v_des
q_des
qd_des
```

Do not directly write:

```python
data.qpos[:] = q_des
```

during the live simulation.

---

# 7. Torque Control Layer

Convert whole-body references into actuator torques.

Basic torque controller:

```python
tau = kp * (q_des - q) + kd * (qd_des - qd)
```

Add gravity/bias compensation if the actuator model allows torque control:

```python
tau += data.qfrc_bias[dof_indices]
```

Then clip to actuator limits:

```python
tau = np.clip(tau, tau_min, tau_max)
```

Apply only through:

```python
data.ctrl[:] = tau_to_ctrl(tau)
```

The exact mapping depends on the XML actuator definitions.

You must inspect the XML:

- motor actuator names,
- joint names,
- gear ratios,
- ctrlrange,
- forcerange,
- actuator-to-joint mapping.

If actuators are position servos instead of motors, either:

1. convert them to torque motors in XML, or
2. send position references while acknowledging this is not pure torque control.

For this challenge, prefer torque motors.

---

# 8. MuJoCo Physics Requirements

Check the XML for:

```xml
<option gravity="0 0 -9.81"/>
```

Ground and foot geoms must have realistic friction.

Example:

```xml
<geom name="floor" type="plane" friction="1.0 0.005 0.0001"/>
```

Foot geoms should also have reasonable friction.

Avoid unrealistically high friction unless debugging.

Recommended initial friction:

```text
sliding friction: 0.8 to 1.2
torsional friction: small but nonzero
rolling friction: small
```

Make sure collisions are active between feet and floor.

---

# 9. Balance Feedback

Continuously monitor:

- CoM position,
- CoM velocity,
- torso roll,
- torso pitch,
- support foot contact,
- foot slip,
- torque saturation.

Use CoM feedback:

```python
com_error = com_des - com
com_vel_error = com_vel_des - com_vel

com_des_corrected = com_des \
                    + kp_com_feedback * com_error \
                    + kd_com_feedback * com_vel_error
```

Use capture point for debugging and optional correction:

```python
omega = np.sqrt(9.81 / com_height)
capture_point = com_xy + com_vel_xy / omega
```

If capture point moves too far from support foot, lower the swing leg and return to double support.

---

# 10. Safety and Recovery

Add failure detection:

```python
if abs(torso_roll) > roll_limit:
    recovery()

if abs(torso_pitch) > pitch_limit:
    recovery()

if not support_foot_contact:
    recovery()

if torque_saturation_ratio > 0.5:
    slow_down()
```

Initial limits:

```python
roll_limit = 0.35   # rad
pitch_limit = 0.35  # rad
```

Recovery behavior:

- stop stepping,
- command both feet to ground,
- move CoM back between feet,
- return to `STAND`.

---

# 11. Debug Output

Print or plot:

- FSM state,
- support foot,
- left contact,
- right contact,
- CoM position,
- CoM velocity,
- capture point,
- torso roll/pitch,
- foot target,
- torque saturation percentage,
- support foot slip velocity.

Example print line:

```python
print(
    f"state={state}, "
    f"L_contact={left_contact}, R_contact={right_contact}, "
    f"com={com}, "
    f"roll={roll:.3f}, pitch={pitch:.3f}, "
    f"tau_sat={tau_sat:.2f}"
)
```

---

# 12. Development Plan

Implement in this order.

## Step 1: Verify torque control

- Load robot standing pose.
- Apply gravity.
- Do not step.
- Use joint PD + gravity compensation.
- Confirm the robot is not pinned or externally supported.

## Step 2: Stable standing

- Tune PD gains.
- Keep both feet on floor.
- Keep torso upright.
- Keep CoM between feet.

## Step 3: CoM shift

- Move CoM slowly toward left foot.
- Then toward right foot.
- No leg lifting yet.

## Step 4: Single leg lift

- Shift CoM over left foot.
- Keep left foot fixed.
- Raise right foot 4 cm.
- Hold briefly.
- Lower right foot.

## Step 5: Alternate

- Repeat on both sides.

## Step 6: Small forward step

- Add 5 cm forward swing.
- Land softly.
- Return to double support.

## Step 7: Feedback foot placement

- Add Raibert correction.
- Add capture point debug signal.

---

# 13. What To Implement In This Repository

Search for the existing files that likely contain:

- Talos robot class,
- inverse kinematics method,
- MuJoCo utilities,
- walking test script,
- XML robot model,
- viewer loop.

Likely files/folders may include:

```text
talos.py
test_walk.py
mujoco_utils.py
math_utils.py
models/
xml/
robots/
controllers/
```

Do not rewrite the project from scratch.

Add new files only where appropriate.

Suggested structure:

```text
project_root/
├── CLAUDE.md
├── test_walk_torque.py
├── controllers/
│   ├── full_body_qp_controller.py
│   ├── walking_fsm.py
│   ├── trajectory_generators.py
│   └── torque_control.py
├── utils/
│   ├── contact_utils.py
│   └── debug_utils.py
```

If the project has no `controllers/` folder, create it.

If the project has existing controller folders, follow the existing style.

---

# 14. Required New Components

Implement:

## `walking_fsm.py`

Contains:

```python
class WalkingFSM:
    def __init__(self, params):
        ...

    def update(self, state_estimate, t):
        ...

    def get_phase(self):
        ...
```

## `trajectory_generators.py`

Contains:

```python
def smoothstep(s):
    ...

def swing_foot_trajectory(p0, p1, s, step_height):
    ...

def com_shift_trajectory(com0, com1, s):
    ...
```

## `full_body_qp_controller.py`

Contains:

```python
class FullBodyQPController:
    def __init__(self, model, robot, params):
        ...

    def solve(self, data, references):
        # Build task errors
        # Build task Jacobians
        # Build weighted QP
        # Return q_des, qd_des, diagnostics
        ...
```

## `torque_control.py`

Contains:

```python
class TorqueController:
    def __init__(self, model, params):
        ...

    def compute(self, data, q_des, qd_des):
        # PD + gravity/bias compensation
        # torque clipping
        # actuator mapping
        return ctrl, diagnostics
```

## `test_walk_torque.py`

Runs the full simulation:

```python
while viewer.is_running():
    estimate = estimator.update(model, data)
    phase = fsm.update(estimate, data.time)
    references = planner.compute(estimate, phase)
    q_des, qd_des, qp_diag = wbc.solve(data, references)
    ctrl, tau_diag = torque_controller.compute(data, q_des, qd_des)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
```

---

# 15. Verification Checklist

Before claiming success, verify:

- `data.qpos` is not overwritten inside the simulation loop.
- The base is not welded or pinned.
- No external force is applied to the pelvis or torso.
- Gravity is enabled.
- Contacts are active.
- Feet collide with the ground.
- Friction is nonzero.
- The robot falls if all actuator commands are zero.
- The robot stands only when motor commands are active.
- Foot lift is generated by joint torques.
- CoM motion is generated by joint torques and contact forces.
- Torques are clipped to actuator limits.
- The QP runs online during simulation.

---

# 16. Immediate First Target

Do not attempt fast walking first.

The first successful demo should be:

```text
stand for 2 seconds
shift CoM left for 1 second
raise right foot 4 cm
hold for 0.3 seconds
lower right foot
shift CoM right for 1 second
raise left foot 4 cm
hold for 0.3 seconds
lower left foot
repeat
```

Once this works, add:

```text
5 cm forward step
```

---

# 17. Instructions For Claude

When modifying the code:

1. Inspect the XML actuator definitions first.
2. Determine whether `data.ctrl` expects torque, position, or velocity commands.
3. If it is not torque controlled, explain exactly what XML changes are needed.
4. Reuse existing IK/QP functions where possible.
5. Do not create fake constraints.
6. Do not overwrite MuJoCo physics.
7. Implement the FSM and torque controller.
8. Add diagnostics.
9. Keep parameters conservative.
10. Prefer a working physically valid slow controller over a visually impressive fake animation.

The final answer should include actual code patches, not just theory.
