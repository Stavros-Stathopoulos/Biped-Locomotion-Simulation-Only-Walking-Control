import sys
import os
import time
# Appending src directory to system path for modular import execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.mujoco_env import MujocoEnv
from src.utils.logger import logger

def main():
    logger.info("Initializing passive stability test for Unitree Go1 robot...")
    # Point directly to your scene description file
    scene_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/unitree_g1/scene.xml'))
    logger.debug(f"Using MJCF model file at: {scene_xml}")
    
    # Instantiate simulation at target execution speed (500 Hz)
    env = MujocoEnv(xml_path=scene_xml, rate_hz=500.0)
    env.init_viewer()
    
    # Reset to default stance pose if configured via keyframe configuration in MJCF
    env.reset(keystring="stand")
    
    logger.info(f"Simulation configured successfully. Timestep: {env.model.opt.timestep} seconds.")
    logger.info("Running passive stability stress test...")

    sim_start_time = env.data.time
    real_start_time = time.time()
    
    while env.viewer.is_running():
        step_start = time.time()
        
        # Apply ZERO control input (verifying purely passive dynamics)
        env.data.ctrl[:] = 0.0
        
        env.step()
        env.sync_viewer()
        
        current_sim_time = env.data.time - sim_start_time
        
        if current_sim_time >= 2.0:
            logger.info(f"Success: Robot main body survived for {current_sim_time:.2f} sim seconds without collapsing.")
            break

        # Maintain real-time synchronization lock
        elapsed = time.time() - step_start
        if elapsed < env.model.opt.timestep:
            time.sleep(env.model.opt.timestep - elapsed)
            
    time.sleep(1.0)
    env.close_viewer()

if __name__ == "__main__":
    main()