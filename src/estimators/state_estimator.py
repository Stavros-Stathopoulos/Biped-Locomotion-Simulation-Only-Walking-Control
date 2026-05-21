import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from matplotlib.path import Path

class StateEstimator:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Identify the root body of the robot to get the full-body CoM
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id == -1:
            raise ValueError("Could not find 'pelvis' body in the MJCF model.")
            
        # Identify the ground to filter contacts
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane")
        if self.ground_geom_id == -1:
            raise ValueError("Could not find 'ground_plane'. Ensure scene.xml is correct.")

    def get_com(self) -> np.ndarray:
        """Returns the 3D Center of Mass of the entire robot in world coordinates."""
        # subtree_com is calculated automatically by mj_step / mj_kinematics
        return self.data.subtree_com[self.pelvis_id].copy()

    def get_contact_points_2d(self) -> np.ndarray:
        """Extracts the X-Y coordinates of all active contacts with the ground."""
        points = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check if either of the colliding geoms is the ground
            if contact.geom1 == self.ground_geom_id or contact.geom2 == self.ground_geom_id:
                points.append(contact.pos[:2]) # Keep only X and Y
                
        return np.array(points)

    def get_support_polygon(self):
        """
        Computes the support polygon from active ground contacts.
        Returns:
            hull_points (np.ndarray): Ordered 2D vertices of the polygon.
            raw_points (np.ndarray): All contact points (for debugging).
        """
        pts = self.get_contact_points_2d()
        
        # A polygon requires at least 3 points. 
        # (Fewer means the robot is balancing on a line or point, or falling).
        if len(pts) < 3:
            return None, pts
            
        # Compute the convex hull of the contact patch
        hull = ConvexHull(pts)
        hull_points = pts[hull.vertices]
        
        return hull_points, pts

    def is_com_stable(self) -> bool:
        """Checks if the CoM projects strictly inside the support polygon."""
        com = self.get_com()
        hull_points, _ = self.get_support_polygon()
        
        if hull_points is None:
            return False
            
        # Use matplotlib's Path to easily check if the point is inside the hull
        polygon = Path(hull_points)
        return polygon.contains_point(com[:2])