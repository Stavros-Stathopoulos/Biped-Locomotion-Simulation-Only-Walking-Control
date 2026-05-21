import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_parser import SimConfig
from src.env.mujoco_env import MujocoEnv
from src.estimators.state_estimator import StateEstimator
from utils.logger.terminal_logger import TerminalLogger as Logger



def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/simulation.yaml'))
    config = SimConfig(config_path)
    env = MujocoEnv(config)
    env.init_viewer()
    
    # Initialize the estimator with the memory-mapped model and data
    estimator = StateEstimator(env.model, env.data)
    
    Logger.info("Letting robot settle into neutral stance...")
    
    # Run the simulation for 1.5 seconds to let the robot hit the floor
    # and establish solid double-support contacts.
    while env.data.time < 1.5:
        env.step()
        env.sync_viewer()
        
    # Execute Acceptance Criteria Checks
    Logger.info("\n--- Kinematics Verification ---")
    
    # Criteria 1: CoM Position
    com = estimator.get_com()
    Logger.debug(f"[X] CoM Position Read: X={com[0]:.4f}, Y={com[1]:.4f}, Z={com[2]:.4f}")
    
    # Criteria 2: Support Polygon
    hull_pts, raw_pts = estimator.get_support_polygon()
    if hull_pts is not None:
        Logger.debug(f"[X] Support Polygon Computed. Generated {len(hull_pts)} perimeter vertices from {len(raw_pts)} contact points.")
    else:
        Logger.error("[ ] FAILED: Could not compute support polygon. Not enough contacts.")
        
    # Criteria 3: CoM Stability Verification
    is_stable = estimator.is_com_stable()
    if is_stable:
        Logger.info("[X] STABLE: CoM ground projection is safely INSIDE the support polygon.")
    else:
        Logger.error("[ ] FAILED: CoM is OUTSIDE the support polygon.")

    Logger.info("-------------------------------\n")
    Logger.info("Verification complete. Closing in 5 seconds...")
    time.sleep(5)
    env.close_viewer()

if __name__ == "__main__":
    main()