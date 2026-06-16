"""
math_utils.py — Lie-group geometry and trajectory utilities.

Ported from the reference Talos implementation (test/math_utils.py).
Provides SE(3)/SO(3) logarithmic maps, trajectory interpolation,
cubic-spline foot arcs, and yaw rotation helpers used by the
offline whole-body IK walking pipeline.
"""

import numpy as np
from scipy.interpolate import CubicSpline


# ── Trajectory helpers ────────────────────────────────────────────────────────

def interpolate_traj(q_start, q_end, n_steps):
    """Linear interpolation between two joint configurations."""
    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    return [q_start + a * (q_end - q_start) for a in alphas]


def generate_foot_arc(start, end, height=0.05, n_points=20):
    """Cubic-spline arc between two 3-D foot positions with a mid-point hump."""
    t = np.array([0.0, 0.5, 1.0])

    mid = (start + end) / 2.0
    mid[2] += height

    spline_x = CubicSpline(t, [start[0], mid[0], end[0]])
    spline_y = CubicSpline(t, [start[1], mid[1], end[1]])
    spline_z = CubicSpline(t, [start[2], mid[2], end[2]])

    samples = np.linspace(0.0, 1.0, n_points)
    return np.stack([spline_x(samples), spline_y(samples), spline_z(samples)], axis=1)


# ── Rotation / yaw ───────────────────────────────────────────────────────────

def Ryaw(theta):
    """3×3 rotation matrix about the Z (yaw) axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def normalize_angle(theta):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(theta), np.cos(theta))


# ── Skew-symmetric (hat / unhat) ─────────────────────────────────────────────

def hat(vec):
    """so(3) hat map: ℝ³ → 3×3 skew-symmetric matrix."""
    v = vec.reshape((3,))
    return np.array([[0.0,  -v[2],  v[1]],
                     [v[2],  0.0,  -v[0]],
                     [-v[1], v[0],  0.0]])


def unhat(mat):
    """so(3) vee map: 3×3 skew-symmetric → ℝ³ column vector."""
    return np.array([[mat[2, 1], mat[0, 2], mat[1, 0]]]).T


# ── SO(3) logarithm and left-Jacobian inverse ────────────────────────────────

def J_l_inv(q, epsilon=1e-8):
    """Inverse of the left Jacobian of SO(3)."""
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    c, s = np.cos(n), np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    return np.eye(3) - 0.5 * hat_q + hat_q_sq * (1.0 / n_sq - (1.0 + c) / (2.0 * n * s))


def log_rotation(R):
    """SO(3) logarithm: rotation matrix → axis-angle column vector (3×1)."""
    theta = np.arccos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))

    if np.isclose(theta, 0.0):
        return np.zeros((3, 1))
    elif np.isclose(theta, np.pi):
        r00, r11, r22 = R[0, 0], R[1, 1], R[2, 2]
        r02, r12 = R[0, 2], R[1, 2]
        r01, r21 = R[0, 1], R[2, 1]
        r10, r20 = R[1, 0], R[2, 0]
        if not np.isclose(r22, -1.0):
            m = theta / np.sqrt(2.0 * (1.0 + r22))
            return m * np.array([[r02, r12, 1.0 + r22]]).T
        elif not np.isclose(r11, -1.0):
            m = theta / np.sqrt(2.0 * (1.0 + r11))
            return m * np.array([[r01, 1.0 + r11, r21]]).T
        elif not np.isclose(r00, -1.0):
            m = theta / np.sqrt(2.0 * (1.0 + r00))
            return m * np.array([[1.0 + r00, r10, r20]]).T

    mat = R - R.T
    r = unhat(mat)
    return theta / (2.0 * np.sin(theta)) * r


# ── SE(3) helpers ─────────────────────────────────────────────────────────────

def invert_transformation(T):
    """Invert a 4×4 homogeneous transformation matrix."""
    R = T[:3, :3]
    p = T[:3, 3]
    return np.block([
        [R.T, -R.T @ p.reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))]
    ])


def log_transformation(T):
    """SE(3) logarithm: 4×4 transform → 6×1 twist coordinates [ω; v]."""
    R = T[:3, :3]
    p = T[:3, 3]
    phi = log_rotation(R)
    rho = J_l_inv(phi) @ p
    return np.vstack((phi, rho.reshape(3, 1)))
