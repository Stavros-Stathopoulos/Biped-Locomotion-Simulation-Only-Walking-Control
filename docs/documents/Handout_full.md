# Robotic Systems I – Lab 4

## Inverse Kinematics with Quadratic Programming

### Introduction

In the previous lab, we used the **Extended Kalman Filter SLAM** algorithm to estimate both the robot pose and the map simultaneously.

In this week’s lab, we move to a different robotics problem:

> _How do we move a robot arm so that the end-effector reaches a desired target pose?_

This problem is called **Inverse Kinematics (IK)**. We will formulate IK as a **Quadratic Program (QP)**.

This is extremely useful because QPs allow us to:

- solve IK as an optimization problem,
- include joint limits,
- include velocity limits,
- add additional constraints,
- and later extend the formulation to more advanced whole-body control methods.

Your task is to complete the missing QP formulation inside the functions:

- `poe_ik_dls_qp` in `position_IK_solve_methods.py`
- `poe_ik_dls_qp` in `fullpose_IK_solve_methods.py`

Your job is to build the optimization problem correctly.

# Part I – Position-Only IK

In the first exercise we only care about the **position** of the end-effector.

We ignore orientation completely.

The desired task is:

$$
p(q) = p_d
$$

Where:

- $p(q)$ is the current end-effector position,
- $p_d$ is the desired target position.

The position error is:

$$
e_p = p_d - p(q)
$$

Using differential kinematics:

$$
\dot{p} = J_p(q)\dot{q}
$$

where:

- $J_p$ is the translational part of the Jacobian.

The Damped Least Squares IK update minimizes:

$$
\min_{\dot{q}} \; \frac{1}{2}\|J_p\dot{q} - e_p\|^2 + \frac{\lambda^2}{2}\|\dot{q}\|^2
$$

Expanding this quadratic objective gives the standard QP form:

$$
\min_{\dot{q}} \; \frac{1}{2}\dot{q}^T Q \dot{q} + q_{lin}^T \dot{q}
$$

with:

$$
Q = J_p^T J_p + \lambda^2 I
$$

and

$$
q_{lin} = -J_p^T e_p
$$

where $q_{lin}$ is the QP linear term, not to be confused with the joints positions.

## Joint Constraints

We also want to respect:

- joint position limits,
- joint velocity limits.

The next joint configuration after one timestep is:

$$
q_{k+1} = q_k + \Delta t \; \dot{q}
$$

To stay within joint limits:

$$
q_{lb} \le q_k + \Delta t \; \dot{q} \le q_{ub}
$$

This can be rewritten as linear inequalities:

$$
-\Delta t \; \dot{q} \le q_k - q_{lb}
$$

$$
\Delta t \; \dot{q} \le q_{ub} - q_k
$$

We also constrain joint velocities:

$$
q_{\dot{lb}} \le \dot{q} \le q_{\dot{ub}}
$$

All constraints together become:

$$
C\dot{q} \le d
$$

# Part II – Full Pose IK

In the second exercise we solve IK for the **full end-effector pose**.

Now we care about both position and orientation.

The pose error is represented as a body twist using the matrix logarithm:

$$
V_b = \text{Log}(T_d T^{-1})
$$

Where:

- $T_d$ is the desired pose in world or space frame,
- $T$ is the current end-effector pose in world or space frame.

Using differential kinematics:

$$
V_b = J(q)\dot{q}
$$

The optimization problem becomes:

$$
\min_{\dot{q}} \; \frac{1}{2}\|J\dot{q} - V_b\|^2 + \frac{\lambda^2}{2}\|\dot{q}\|^2
$$

which again leads to:

$$
Q = J^T J + \lambda^2 I
$$

$$
q_{lin} = -J^T V_b
$$

The constraints remain exactly the same as before.

# Position IK Code

```python
def poe_ik_dls_qp(
        q0,
        pd,
        frame_in = "space",
        frame_ref = "world",
        twist_dt = 1.0,
        max_ik_iter = 100,
        error_tol = 1e-5,
        damping = 1e-3,
        qp_solver = "custom",
):
    q = np.copy(q0)

    count_iter = 0

    for _ in range(max_ik_iter):

        count_iter += 1

        T = poe_fk(q, frame_in, frame_ref if frame_ref in ("world", "space") else "world")

        p = T[:3, 3]

        bb_error = (pd.reshape(3) - p.reshape(3)).reshape(-1, 1)

        Vb_linear = bb_error / twist_dt

        if np.linalg.norm(bb_error) < error_tol:
            break

        J = jacobian(q, frame_in, "local_world_aligned")

        Jp = J[3:, :]  # get only translational part

        JpT = Jp.T

        ##### ENTER CODE HERE #####
        n = q.size
        I_n = np.eye(n)

        Q_mat = JpT @ Jp + (damping ** 2) * I_n
        q_lin = -JpT @ Vb_linear

        C_pos_lb = -twist_dt * I_n
        d_pos_lb = (q - q_lb).reshape(-1, 1)
        C_pos_ub = twist_dt * I_n
        d_pos_ub = (q_ub - q).reshape(-1, 1)

        C_vel_lb = -I_n
        d_vel_lb = (-qdot_lb).reshape(-1, 1)
        C_vel_ub = I_n
        d_vel_ub = qdot_ub.reshape(-1, 1)

        C = np.vstack([C_pos_lb, C_pos_ub, C_vel_lb, C_vel_ub])
        d = np.vstack([d_pos_lb, d_pos_ub, d_vel_lb, d_vel_ub])

        qp = QP(Q_mat, q_lin, C=C, d=d)
        
        ###########################

        if qp_solver == "custom":
            v_linear, _ = qp.QP_custom_solver()
        else:
            v_linear = qp.QP_lib_solver(qp_solver)

        if v_linear is None:
            break

        q += twist_dt * v_linear.flatten()

    return q, count_iter
```

# Full Pose IK Code

```python
def poe_ik_dls_qp(
        q0,
        Td,
        frame_in = "space",
        frame_ref = "world",
        twist_dt = 1.0,
        max_ik_iter = 100,
        error_tol = 1e-5,
        damping = 1e-3,
        qp_solver = "custom",
):
    q = np.copy(q0)

    count_iter = 0

    for _ in range(max_ik_iter):

        count_iter += 1

        if frame_ref == "world" or frame_ref == "space":
            Trefb = poe_fk(q, frame_in, frame_ref)
            bb_error = Log(Td @ np.linalg.inv(Trefb))
            Vb = bb_error / twist_dt
        elif frame_ref == "body":
            Twb = poe_fk(q, frame_in, "world")
            bb_error = Log(np.linalg.inv(Twb) @ Td)
            Vb = bb_error / twist_dt

        if np.linalg.norm(bb_error) < error_tol:
            break

        J = jacobian(q, frame_in, frame_ref)

        JT = J.T

        ##### ENTER CODE HERE #####
        n = q.size
        I_n = np.eye(n)

        # QP cost: Q = J^T J + lambda^2 I,  q_lin = -J^T * Vb
        Q_mat = JT @ J + (damping ** 2) * I_n
        q_lin = -JT @ Vb

        # Joint position limit constraints: q_lb <= q + dt*qdot <= q_ub
        C_pos_lb = -twist_dt * I_n
        d_pos_lb = (q - q_lb).reshape(-1, 1)
        C_pos_ub = twist_dt * I_n
        d_pos_ub = (q_ub - q).reshape(-1, 1)

        # Joint velocity limit constraints: qdot_lb <= qdot <= qdot_ub
        C_vel_lb = -I_n
        d_vel_lb = (-qdot_lb).reshape(-1, 1)
        C_vel_ub = I_n
        d_vel_ub = qdot_ub.reshape(-1, 1)

        C = np.vstack([C_pos_lb, C_pos_ub, C_vel_lb, C_vel_ub])
        d = np.vstack([d_pos_lb, d_pos_ub, d_vel_lb, d_vel_ub])

        qp = QP(Q_mat, q_lin, C=C, d=d)

        ###########################

        if qp_solver == "custom":
            v,_ = qp.QP_custom_solver()
        else:
            v = qp.QP_lib_solver(qp_solver)

        if v is None:
            break

        q += twist_dt * v.flatten()

    return q, count_iter
```

---

# Student TODOs

> - **Student TODO I:** In `position_IK_solve_methods.py`, formulate the QP cost matrix `Q` and the linear term `q`. Use the translational Jacobian `Jp` and the position error `Vb_linear`.

> - **Student TODO II:** Create the inequality constraint matrices `C` and `d` for:
>     - joint position limits,
>     - joint velocity limits.

> - **Student TODO III:** Construct the `QP` object.

> - **Student TODO IV:** Repeat the same procedure for the full-pose IK solver in `fullpose_IK_solve_methods.py`, but now use the full Jacobian `J` and the full body twist error `Vb`.

> - **Student TODO V:** Run the examples `try_fullpose_IK.py` and `try_position_IK.py` and compare the behavior of:
>     - Newton IK,
>     - Damped Least Squares,
>     - QP-based IK.

---

# Part III – Whole Body IK for Humanoid Walking

In the second part of the lab we will extend the same ideas to a **humanoid robot**.

Instead of controlling only a robot arm end-effector, we now want to control:

- both feet,
- the center of mass (CoM),
- and eventually the whole body posture.

We will use the Talos humanoid robot model.

The goal is to generate walking motions by repeatedly solving a **whole-body inverse kinematics QP**.

The example script: `test_walk.py` will generate and visualize a walking trajectory.

Your task is to complete the QP formulation inside `Talos.inverse_kinematics()` inside the file `talos.py`.

# Whole-Body IK Formulation

For walking, we simultaneously optimize multiple tasks:

- left foot pose tracking,
- right foot pose tracking,
- center of mass tracking.

Each task produces:

- a task-space error,
- and a Jacobian.

All task errors are stacked together:

$$
e =
\begin{bmatrix}
e_{left\_foot} \\
e_{right\_foot} \\
e_{CoM}
\end{bmatrix}
$$

and similarly for the Jacobians:

$$
J =
\begin{bmatrix}
J_{left\_foot} \\
J_{right\_foot} \\
J_{CoM}
\end{bmatrix}
$$

The weighted least-squares optimization becomes:

$$
\min_v \frac{1}{2}\|W(Jv - e)\|^2
$$

where:

- $v$ is the generalized joint velocity vector,
- $W$ is a weighting matrix.

Expanding again into QP form:

$$
\min_v \frac{1}{2} v^T Q v + q^T v
$$

with:

$$
Q = (WJ)^T (WJ) + \lambda I
$$

and:

$$
q = -(WJ)^T We
$$

where $\lambda$ is a regularization term.

# Joint Constraints

The humanoid robot also has joint limits.

The QP solver uses inequality constraints:

$$
d_{min} \le Cv \le d_{max}
$$

In this lab we use:

$$
C = \Delta t I
$$

which constrains the integrated configuration update.

The vectors:

- `d_min`
- `d_max`

are automatically computed using:

```python
mujoco_utils.get_current_joint_range()
```

# Whole Body IK Code

```python
    def inverse_kinematics(self, data, feet_targets, feet_weights, com_target, com_weight, step = 0.001, max_iters = 5000):
        qp_dim = self.model.nv
        qp_eq = 0
        qp_ineq = self.model.nv
        qp = proxsuite.proxqp.dense.QP(qp_dim, qp_eq, qp_ineq)

        it = 0

        qk = data.qpos.copy()

        solution = qk.copy()

        feet_ids = [self.left_foot_id, self.right_foot_id]

        while True:
            # Calculate feet errors and Jacobians
            errors = []
            Jacobians = []
            for foot_id, foot_target in zip(feet_ids, feet_targets):
                T_wb = mujoco_utils.get_body_full_transformation(self.model, data, foot_id)
                T_wb_inv = math_utils.invert_transformation(T_wb)
                foot_error = math_utils.log_transformation(T_wb_inv @ foot_target)
                errors.append(foot_error)

                Jacobians.append(mujoco_utils.get_body_jac_local_frame(self.model, data, foot_id))

            # Calculate CoM error and Jacobian
            com = data.subtree_com[0]
            errors.append((com_target - com).reshape(3, 1))

            J_b_com = np.zeros((3, self.model.nv))
            mujoco.mj_jacSubtreeCom(self.model, data, J_b_com, 0)
            Jacobians.append(J_b_com)

            # Stack errors and jacobians
            ##### ENTER CODE HERE #####

            error = np.vstack(errors)       # (15, 1): [left_foot(6), right_foot(6), com(3)]
            J = np.vstack(Jacobians)        # (15, nv)

            ###########################

            # Check for convergence
            if np.all(error < 1e-5) or it == max_iters:
                break

            # Setup the QP problem
            ##### ENTER CODE HERE #####

            weights_vec = np.concatenate([feet_weights[0], feet_weights[1], com_weight])
            weights = np.diag(weights_vec)  # (15, 15) diagonal weight matrix

            ###########################
            W = np.dot(weights, J)
            error = weights @ error

            ##### ENTER CODE HERE #####

            Q = W.T @ W + 1e-4 * np.eye(qp_dim)
            q = -(W.T @ error).flatten()

            ###########################
            ##### ENTER CODE HERE #####

            C = step * np.eye(qp_dim)
            d_min, d_max = mujoco_utils.get_current_joint_range(self.model, qk)

            ###########################

            # Solve

            if it == 0:
                qp.init(Q, q, None, None, C, d_min, d_max)
            else:
                qp.update(Q, q, None, None, C, d_min, d_max)
            qp.solve()

            v = np.copy(qp.results.x)

            mujoco.mj_integratePos(self.model, qk, v, step)
            solution = qk.copy()
            data.qpos[:] = qk
            mujoco.mj_forward(self.model, data)

            it += 1
        
        return solution

```

# Understanding the Tasks

## Foot Tasks

Each foot task is a **6D pose task**:

$$
e_{foot} \in \mathbb{R}^6
$$

The pose error is computed using:

```python
math_utils.log_transformation()
```

which returns a body twist error.

The Jacobian is computed with:

```python
mujoco_utils.get_body_jac_local_frame()
```

## Center of Mass Task

The CoM task is a **3D position task**:

$$
e_{CoM} \in \mathbb{R}^3
$$

The CoM Jacobian is computed using:

```python
mujoco.mj_jacSubtreeCom()
```

# Weighting Tasks

Different tasks can be given different priorities using weights.

Large weights make the optimizer care more about a task.

For example:

```python
w_feet = [np.full(6, 20.), np.full(6, 20.)]
w_com = np.full(3, 5.)
```

means:

- foot tracking is more important,
- CoM tracking is softer.

# Student TODOs – Whole Body IK

> - **Student TODO VI:** Stack the task Jacobians and task errors into a single arrays.

> - **Student TODO VII:** Construct the weighting matrix `weights` by horizontally stacking left feet wieghts, right feet weights and center of mass weight.

> - **Student TODO VIII:** Formulate the QP matrices `Q` and `q` for the whole-body IK problem using 1e-4 as the regularization term $\lambda$.

> - **Student TODO IX:** Construct the constraint matrix `C` and create/get the min and max joint ranges `d_min` and `d_max`.

> - **Student TODO X:** Run `test_walk.py` and visualize the Talos robot walking.
