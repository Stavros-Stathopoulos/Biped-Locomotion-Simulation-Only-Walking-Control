import numpy as np
import mujoco
from scipy.spatial import ConvexHull, QhullError
from matplotlib.path import Path


class StateEstimator:
    """
    Estimates whole-body kinematics state from MuJoCo mjData buffers.

    All methods operate directly on MuJoCo's internal memory via zero-copy views
    or pre-allocated buffers to avoid heap allocation on the high-frequency control
    path. Callers must not mutate the returned arrays; values are valid until the
    next mj_step or mj_forward call updates the underlying buffers.

    Prerequisites: mj_step (or mj_forward) must have been called at least once
    before any estimator method is invoked, so that the kinematics buffers
    (subtree_com, contact) are fully populated.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Pelvis is the G1's floating-base root — the sole direct child of worldbody
        # with a free joint. Its subtree spans the entire kinematic tree of the robot.
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if self.pelvis_id == -1:
            raise ValueError("Could not find 'pelvis' body in the MJCF model.")

        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane"
        )
        if self.ground_geom_id == -1:
            raise ValueError(
                "Could not find 'ground_plane' geom. Ensure scene.xml is loaded."
            )

        # Pre-allocate a contact-point buffer sized to the model's maximum simultaneous
        # contact count. Writing in-place into this buffer on every timestep avoids
        # Python list growth and np.array() construction on the hot path.
        # nconmax defaults to 500 when not set in the MJCF; the G1 generates at most
        # 8 ground contacts (4 corner sphere geoms per ankle_roll_link × 2 feet).
        _max_contacts = getattr(self.model, "nconmax", 500)
        if _max_contacts <= 0:
            _max_contacts = 500
        self._contact_buf = np.zeros((_max_contacts, 2), dtype=np.float64)
        self._n_ground_contacts: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_com(self) -> np.ndarray:
        """
        Returns the 3D Center of Mass of the entire robot in the world frame.

        Buffer mechanics — data.subtree_com layout:
            Shape  : (nbody, 3)
            Updated: mj_kinematics, called internally by mj_step and mj_forward.
            Entry i: mass-weighted mean position of all bodies in the subtree
                     rooted at body i, expressed in the world frame.

        Index choice — pelvis_id vs. world body (index 0):
            Body 0 is the MuJoCo worldbody pseudo-body. subtree_com[0] would also
            return the full-robot CoM here because the ground_plane geom carries
            zero inertia (no <inertial> tag). However, in scenes with multiple
            robots or massive static objects, subtree_com[0] would be polluted by
            those bodies. Using self.pelvis_id — the floating-base root whose
            subtree is exactly the G1 kinematic chain — is semantically correct
            and robust to multi-body scenes.

        Returns a direct view into MuJoCo's internal buffer (zero-copy). The
        returned array must not be mutated by the caller.
        """
        return self.data.subtree_com[self.pelvis_id]

    def get_contact_points_2d(self) -> np.ndarray:
        """
        Extracts the XY world-frame coordinates of all active contacts with the ground.

        Buffer mechanics — data.contact layout:
            Type   : array of mjContact C-structs, capacity model.nconmax.
            Active : first data.ncon entries are populated this timestep.
            Fields used:
              .geom1 / .geom2  — integer IDs of the two colliding geoms.
              .pos[3]          — world-frame contact point (geometric midpoint
                                 of the penetration gap, at the collision normal).

        Filtering: a contact is classified as a ground contact when either geom
        slot matches ground_geom_id. This handles both orderings (geom1=ground,
        geom2=foot and vice versa) transparently, and extends gracefully to
        multi-geom contacts produced by mesh decomposition or composite feet.

        Performance: XY components are written in-place into self._contact_buf,
        a pre-allocated (nconmax, 2) buffer, avoiding any heap allocation.

        Returns a zero-copy slice view self._contact_buf[0:n], valid until the
        next call to this method.
        """
        self._n_ground_contacts = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ground_geom_id or c.geom2 == self.ground_geom_id:
                self._contact_buf[self._n_ground_contacts, 0] = c.pos[0]
                self._contact_buf[self._n_ground_contacts, 1] = c.pos[1]
                self._n_ground_contacts += 1
        return self._contact_buf[: self._n_ground_contacts]

    def get_support_polygon(self):
        """
        Computes the convex hull of active ground contact patches in the XY plane.

        The support polygon is defined as the smallest convex set that contains all
        ground contact points projected onto the horizontal XY plane. It represents
        the region within which the CoM ground projection must fall for quasi-static
        stability (necessary condition of the ZMP/CoP criterion).

        Degenerate contact states are handled without raising exceptions:

            N = 0  (flight phase)         → return (None, empty_array)
            N = 1  (point contact)        → return (None, pts)
            N = 2  (line/edge contact)    → return (None, pts)
            N ≥ 3, all collinear          → ConvexHull raises QhullError because the
                                            2D hull has zero area; caught here and
                                            mapped to (None, pts).
            N ≥ 3, non-collinear (normal) → return (hull_vertices_ccw, pts)

        scipy.spatial.ConvexHull is backed by Qhull. hull.vertices gives the indices
        of the convex hull vertices in counter-clockwise order, which is the ordering
        expected by matplotlib.path.Path for correct winding.

        Returns:
            hull_points : np.ndarray | None
                Shape (K, 2). CCW-ordered XY vertices of the support polygon, or None
                when a non-degenerate 2D polygon cannot be formed.
            raw_pts : np.ndarray
                Shape (N, 2). All ground contact XY positions this timestep.
        """
        pts = self.get_contact_points_2d()

        if self._n_ground_contacts < 3:
            return None, pts

        try:
            hull = ConvexHull(pts)
        except QhullError:
            # All contact points are collinear → degenerate hull with zero area.
            # This occurs during single-foot heel/toe transitions or near-liftoff.
            return None, pts

        return pts[hull.vertices], pts

    def is_com_stable(self) -> bool:
        """
        Tests whether the CoM ground projection lies strictly inside the support polygon.

        Stability criterion (quasi-static, planar):
            The XY projection of the whole-body CoM must lie strictly inside the
            convex hull of ground contact patches. This is a necessary condition for
            static balance: if the CoM projection exits the support polygon, the net
            gravitational moment about every contact edge has the same sign and the
            robot will topple without corrective action.

        Containment test:
            Uses matplotlib.path.Path (even-odd ray-casting algorithm). The behavior
            of `contains_point` for points exactly on the polygon boundary is
            implementation-defined (depends on floating-point rounding and ray
            direction). To enforce strict-interior semantics, a small negative
            radius=-1e-10 is passed, which contracts the effective test polygon by
            ~0.1 nm. This ensures that boundary-coincident points consistently
            return False, matching the strict-inside requirement, at zero practical
            cost for the intended stability margin of a walking robot.

        Access pattern:
            subtree_com is read directly (bypassing get_com()) to avoid an extra
            Python function-call frame on the hot path. Both accesses refer to the
            same underlying buffer row, so there is no inconsistency.

        Returns:
            True  — CoM projection is strictly inside the support polygon.
            False — degenerate contact state (< 3 non-collinear contacts), or CoM
                    projection lies on or outside the polygon boundary.
        """
        hull_points, _ = self.get_support_polygon()

        if hull_points is None:
            return False

        com_xy = self.data.subtree_com[self.pelvis_id, :2]
        return bool(Path(hull_points).contains_point(com_xy, radius=-1e-10))
