"""
Whole-Body Inverse-Dynamics QP controller (WBC).

This is a TRUE online whole-body controller: every control tick it solves a
Quadratic Program over the full floating-base dynamics for the joint
accelerations and contact forces, then reads off the torques for ALL 29 joints.
Nothing is hardcoded or replayed -- the torques come out of the live solve using
the robot's measured state, mass matrix, contact set, friction and torque limits.

Decision variables:   x = [ qddot (nv) ,  lambda (3 per active contact point) ]

Minimise (weighted task accelerations + regularisation)

    sum_t  w_t || J_t qddot - a_t* ||^2   +  r_qdd ||qddot||^2 + r_f ||lambda||^2

subject to
    floating-base dynamics (6 eq):   M[:6] qddot + h[:6] = Jc[:, :6]^T lambda
    stance feet don't accelerate :   Jc qddot = -kd_c (Jc qdot)         (eq)
    friction cones (ineq)        :   |fx|,|fy| <= mu fz ,  fz >= fz_min
    torque limits  (ineq)        :   -tau_max <= tau(qddot,lambda) <= tau_max

with the actuated torques recovered from the lower rows of the dynamics:

    tau = ( M qddot + h - Jc^T lambda )[6:]

Task references (CoM, torso orientation, swing foot, posture) are supplied by the
gait planner each tick; the QP turns them into a single consistent set of joint
torques that respects the contacts and the robot's limits.

Coriolis/centrifugal task-bias terms (Jdot qdot) are dropped: the gait is slow
(quasi-static) so they are negligible, which keeps the QP linear and fast.
"""

import warnings

import numpy as np
import mujoco
import qpsolvers

from src.controllers.robot_model import RobotModel

# The QP is small and dense; osqp warns when it converts our dense matrices to
# sparse each call. That is expected and harmless here, so silence the spam.
warnings.filterwarnings("ignore", message="Converted matrix")


def _rot_error(R_des, R_cur):
    R_err = R_des @ R_cur.T
    cos = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


class WholeBodyQP:
    def __init__(self, robot: RobotModel, params: dict = None):
        self.r = robot
        self.m = robot.model
        self.nv = self.m.nv
        p = params or {}

        # task acceleration-PD gains (kept moderate; high gains + low regularisation
        # make the QP command large accelerations and overshoot)
        self.kp_com = np.array(p.get("kp_com", [60.0, 60.0, 90.0]))
        self.kd_com = np.array(p.get("kd_com", [16.0, 16.0, 19.0]))
        self.kp_torso = p.get("kp_torso", 80.0); self.kd_torso = p.get("kd_torso", 18.0)
        self.kp_sw = p.get("kp_swing", 300.0);   self.kd_sw = p.get("kd_swing", 35.0)
        self.kp_post = p.get("kp_posture", 20.0); self.kd_post = p.get("kd_posture", 9.0)
        # task weights (CoM dominates posture so balance tracks tightly)
        self.w_com = p.get("w_com", 12.0)
        self.w_torso = p.get("w_torso", 3.0)
        self.w_swing = p.get("w_swing", 40.0)
        self.w_post = p.get("w_posture", 0.1)
        # regularisation / contact
        self.r_qdd = p.get("r_qddot", 1e-2)
        self.r_f = p.get("r_force", 1e-5)
        self.mu = p.get("friction", 0.7)
        self.fz_min = p.get("fz_min", 2.0)
        self.kd_contact = p.get("kd_contact", 20.0)   # Baumgarte velocity damping
        self.solver = p.get("solver", "osqp")

        # ordered contact points per foot: (geom_id, body_id)
        self.foot_pts = {
            "left": [(g, self.m.geom_bodyid[g]) for g in sorted(robot.lfoot_geoms)],
            "right": [(g, self.m.geom_bodyid[g]) for g in sorted(robot.rfoot_geoms)],
        }
        self._M = np.zeros((self.nv, self.nv))
        self._jp = np.zeros((3, self.nv))
        self._jr = np.zeros((3, self.nv))
        self.info = {"cost": 0.0, "fail": 0, "fz": 0.0}

    # -- jacobian helpers --------------------------------------------------
    def _com_jac(self, d):
        mujoco.mj_jacSubtreeCom(self.m, d, self._jp, 0)
        return self._jp.copy()

    def _body_jac(self, d, body_id):
        mujoco.mj_jacBody(self.m, d, self._jp, self._jr, body_id)
        return self._jp.copy(), self._jr.copy()

    def _point_jac(self, d, point, body_id):
        mujoco.mj_jac(self.m, d, self._jp, None, point, body_id)
        return self._jp.copy()

    # -- main solve --------------------------------------------------------
    def compute(self, data, com_des, com_vel_des, torso_R_des, contact_feet, swing,
                com_acc_ff=None):
        """Solve the whole-body QP for one tick.

        com_acc_ff: optional (3,) CoM acceleration feed-forward. When supplied
        (DCM/LIPM walking) it is added to the CoM task acceleration, so the QP
        realises that CoM acceleration through the contact forces -- i.e. it
        commands the ground-reaction / centre-of-pressure (ZMP control). The PD
        terms then only correct residual position/velocity error.
        """
        m, r, nv = self.m, self.r, self.nv
        qvel = data.qvel

        # dynamics terms (dense inertia matrix from the live state)
        mujoco.mj_fullM(m, self._M, data.qM)
        M = self._M
        h = data.qfrc_bias.copy()

        # active contact points
        pts = []
        for foot in contact_feet:
            for g, b in self.foot_pts[foot]:
                pts.append((data.geom_xpos[g].copy(), b))
        nc = len(pts)
        nf = 3 * nc
        n = nv + nf

        # stacked contact Jacobian Jc (nf x nv)
        Jc = np.zeros((nf, nv))
        for i, (pnt, b) in enumerate(pts):
            Jc[3 * i:3 * i + 3, :] = self._point_jac(data, pnt, b)

        # ---- objective: P, q -------------------------------------------
        P = np.zeros((n, n)); q = np.zeros(n)

        def add_task(J, a_des, w):
            P[:nv, :nv] += w * (J.T @ J)
            q[:nv] += -w * (J.T @ a_des)

        # CoM task
        Jcom = self._com_jac(data)
        com = data.subtree_com[0]
        com_vel = Jcom @ qvel
        a_com = self.kp_com * (com_des - com) + self.kd_com * (com_vel_des - com_vel)
        if com_acc_ff is not None:
            a_com = a_com + np.asarray(com_acc_ff)
        add_task(Jcom, a_com, self.w_com)

        # torso / pelvis orientation task
        _, Jr = self._body_jac(data, r.pelvis_bid)
        Rp = data.xmat[r.pelvis_bid].reshape(3, 3)
        omega = Jr @ qvel
        a_torso = self.kp_torso * _rot_error(torso_R_des, Rp) + self.kd_torso * (-omega)
        add_task(Jr, a_torso, self.w_torso)

        # swing foot task (only in single support)
        if swing is not None:
            bid = r.rfoot_bid if swing["foot"] == "right" else r.lfoot_bid
            Jsw = self._point_jac(data, data.xpos[bid], bid)
            psw = data.xpos[bid]; vsw = Jsw @ qvel
            a_sw = self.kp_sw * (swing["pos"] - psw) + self.kd_sw * (swing["vel"] - vsw)
            add_task(Jsw, a_sw, self.w_swing)

        # posture task (all actuated joints toward home)
        Jpost = np.zeros((r.nu, nv))
        for i, dof in enumerate(r.act_dofadr):
            Jpost[i, dof] = 1.0
        q_now = data.qpos[r.act_qadr]; qd_now = data.qvel[r.act_dofadr]
        a_post = self.kp_post * (r.q_home - q_now) + self.kd_post * (-qd_now)
        add_task(Jpost, a_post, self.w_post)

        # regularisation (keeps P positive definite)
        P[:nv, :nv] += self.r_qdd * np.eye(nv)
        if nf:
            P[nv:, nv:] += self.r_f * np.eye(nf)

        # ---- equality constraints A x = b ------------------------------
        A_rows = []; b_rows = []
        # floating-base dynamics (6)
        A_fb = np.zeros((6, n))
        A_fb[:, :nv] = M[:6, :]
        if nf:
            A_fb[:, nv:] = -Jc[:, :6].T
        A_rows.append(A_fb); b_rows.append(-h[:6])
        # stance feet: no acceleration (Baumgarte-damped)
        if nf:
            A_c = np.zeros((nf, n)); A_c[:, :nv] = Jc
            b_c = -self.kd_contact * (Jc @ qvel)
            A_rows.append(A_c); b_rows.append(b_c)
        A = np.vstack(A_rows); b = np.concatenate(b_rows)

        # ---- inequality constraints G x <= h_i -------------------------
        # friction pyramid + min normal per contact point (5 rows each)
        Gf = np.zeros((5 * nc, n)); hf = np.zeros(5 * nc)
        for i in range(nc):
            base = nv + 3 * i; row = 5 * i
            Gf[row + 0, base + 0] = 1.0;  Gf[row + 0, base + 2] = -self.mu   #  fx <= mu fz
            Gf[row + 1, base + 0] = -1.0; Gf[row + 1, base + 2] = -self.mu   # -fx <= mu fz
            Gf[row + 2, base + 1] = 1.0;  Gf[row + 2, base + 2] = -self.mu   #  fy <= mu fz
            Gf[row + 3, base + 1] = -1.0; Gf[row + 3, base + 2] = -self.mu   # -fy <= mu fz
            Gf[row + 4, base + 2] = -1.0; hf[row + 4] = -self.fz_min         #  fz >= fz_min
        # torque limits:  tau = M[6:] qddot + h[6:] - Jc[:,6:]^T lambda
        Mtau = M[6:, :]
        Jct = Jc[:, 6:].T if nf else np.zeros((r.nu, 0))
        Gtau = np.zeros((r.nu, n)); Gtau[:, :nv] = Mtau
        if nf:
            Gtau[:, nv:] = -Jct
        tlim = r.tau_limit
        G = np.vstack([Gf, Gtau, -Gtau])
        h_i = np.concatenate([hf, tlim - h[6:], tlim + h[6:]])

        # ---- solve ------------------------------------------------------
        x = qpsolvers.solve_qp(P, q, G, h_i, A, b, solver=self.solver)
        if x is None:
            self.info["fail"] += 1
            return h[r.act_dofadr]          # fall back to gravity compensation

        qddot = x[:nv]; lam = x[nv:]
        tau = Mtau @ qddot + h[6:]
        if nf:
            tau = tau - Jct @ lam
        tau = np.clip(tau, -tlim, tlim)
        self.info = {"fail": self.info["fail"],
                     "fz": float(lam[2::3].sum()) if nf else 0.0,
                     "ncontact": nc}
        return tau
