import os
import time
import mujoco
import mujoco.viewer
import numpy as np

from src.utils.config_parser import SimConfig
from src.utils.terminal_logger import TerminalLogger as Logger

class MujocoEnv:
    def __init__(self, config: SimConfig):
        xml_path = config.scene_xml_path
        if not os.path.exists(xml_path):
            Logger.error(f"MJCF model file missing at: {xml_path}")
            raise FileNotFoundError(f"MJCF model file missing at: {xml_path}")

        Logger.debug(f"Loading MJCF model from: {xml_path}")
        # Workaround for path encoding issues with C++ fopen on Windows
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(xml_path))
        try:
            self.model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
        finally:
            os.chdir(original_cwd)

        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = config.sim_timestep
        self.model.opt.gravity[:] = config.gravity

        if config.override_damping:
            self.model.dof_damping[:] = config.default_damping
            Logger.debug(f"Joint damping overridden to {config.default_damping} for all DOFs.")

        self.viewer = None

    def step(self):
        """Advances physics by one integration step."""
        mujoco.mj_step(self.model, self.data)

    def reset(self, keystring: str = None):
        """Resets simulation to initial state or specific keyframe."""
        mujoco.mj_resetData(self.model, self.data)
        if keystring:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keystring)
            if key_id != -1:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def init_viewer(self):
        """Launches the native interactive visualizer thread."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Force the viewer to use our tracking camera instead of the free camera
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track_com_cam")
            if cam_id != -1:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = cam_id

    def sync_viewer(self):
        """Synchronizes current data state to visualizer."""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close_viewer(self):
        """Gracefully tears down the viewer instance."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
