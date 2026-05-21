import os
import yaml

class SimConfig:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file missing at: {config_path}")
            
        with open(config_path, 'r') as file:
            self._raw_cfg = yaml.safe_load(file)
            
        # Extract sub-dictionaries with safe fallbacks
        self.env_cfg = self._raw_cfg.get('environment', {})
        self.timing = self._raw_cfg.get('timing', {})
        self.physics = self._raw_cfg.get('physics', {})
        
        # Resolve scene_xml_path relative to the config file's directory
        config_dir = os.path.dirname(os.path.abspath(config_path))
        raw_path = self.env_cfg.get('scene_xml_path')
        if raw_path and not os.path.isabs(raw_path):
            self.scene_xml_path = os.path.normpath(os.path.join(config_dir, raw_path))
        else:
            self.scene_xml_path = raw_path
        if not self.scene_xml_path or not os.path.exists(self.scene_xml_path):
            raise ValueError(f"Invalid scene_xml_path in config: {self.scene_xml_path}")

    @property
    def sim_timestep(self) -> float:
        """Helper to convert Hz to seconds for the MuJoCo engine."""
        hz = self.timing.get('sim_frequency_hz', 1000)
        return 1.0 / float(hz)
        
    @property
    def gravity(self) -> list:
        return self.physics.get('gravity', [0.0, 0.0, -9.81])
        
    @property
    def override_damping(self) -> bool:
        return self.physics.get('override_joint_damping', False)
        
    @property
    def default_damping(self) -> float:
        return self.physics.get('default_damping', 0.0)