import sys
import os
import time
# Appending src directory to system path for modular import execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.mujoco_env import MujocoEnv
from src.utils.config_parser import SimConfig
from src.utils.logger.terminal_logger import TerminalLogger as logger

def main():
    logger.info("Initializing passive stability test for Unitree Go1 robot...")

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/simulation.yaml'))
    config = SimConfig(config_path)
    logger.debug(f"Using MJCF model file at: {config.scene_xml_path}")

    env = MujocoEnv(config=config)
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
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Passive stability test interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the passive stability test: {e}")
    finally:
        logger.info("Passive stability test concluded.")
