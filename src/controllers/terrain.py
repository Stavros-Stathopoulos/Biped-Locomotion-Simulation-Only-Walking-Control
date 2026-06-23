"""
Terrain sensing via MuJoCo ray casting.

`ground_height(model, data, x, y)` returns the height (world z) of the support
surface at a horizontal point, by casting a ray straight down and intersecting
the *terrain* geoms only. This is what makes the controller respond to a dynamic
environment: each footstep is planned onto whatever surface is actually there
(floor, a stair tread, a kerb, uneven ground), sensed live every step rather than
assumed flat.

Terrain geoms (floor + stairs) are tagged with render group `TERRAIN_GROUP`; the
ray is masked to that group so it never reports the robot's own body as ground.
"""

import numpy as np
import mujoco

TERRAIN_GROUP = 2   # rendered by default in the viewer, and unused by the robot
_GEOMGROUP = np.zeros(int(mujoco.mjNGROUP), dtype=np.uint8)
_GEOMGROUP[TERRAIN_GROUP] = 1


def ground_height(model, data, x, y, z_start: float = 3.0, default: float = 0.0) -> float:
    """World z of the terrain surface under (x, y), or `default` if nothing hit."""
    pnt = np.array([x, y, z_start], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, _GEOMGROUP, 1, -1, geomid)
    if dist < 0 or geomid[0] < 0:
        return default
    return z_start - dist
