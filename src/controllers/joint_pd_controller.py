import numpy as np

from utils.logger.data_logger import DataLogger

class JointPDController:
    def __init__(self, model, data, kp: np.ndarray, kd: np.ndarray):
        """
        Initializes the controller for the actuated Degrees of Freedom (DoF).
        """
        self.model = model
        self.data = data
        self.kp = kp
        self.kd = kd
        
        # Number of actuated motors (excludes the floating base)
        self.nu = model.nu

    def compute_torques(self, q_ref: np.ndarray, qdot_ref: np.ndarray) -> np.ndarray:
        """
        Calculates PD + Gravity Compensation torques for all motors.
        """
        q = np.zeros(self.nu)
        qdot = np.zeros(self.nu)
        tau_bias = np.zeros(self.nu)
        
        # Safely extract state data mapped specifically to the actuators
        for i in range(self.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            dof_adr = self.model.jnt_dofadr[joint_id]
            
            q[i] = self.data.qpos[qpos_adr]
            qdot[i] = self.data.qvel[dof_adr]
            
            # Extract gravity and Coriolis forces for this specific joint
            tau_bias[i] = self.data.qfrc_bias[dof_adr]

        # Apply the control law
        tau_pd = self.kp * (q_ref - q) + self.kd * (qdot_ref - qdot)
        
        DataLogger.log_input(context="PD Torques", data=tau_pd)
        DataLogger.log_input(context="Gravity/Coriolis Torques", data=tau_bias)
        return tau_pd + tau_bias