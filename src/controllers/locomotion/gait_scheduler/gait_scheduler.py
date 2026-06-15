import math
import numpy as np

class GaitScheduler:
    def __init__(self, step_duration: float, double_support_fraction: float = 0.1) -> None:
        self.t_step = step_duration
        self.t_stride = 2.0 * self.t_step
        self.dsp_fraction = double_support_fraction
        
        if self.dsp_fraction < 0.0 or self.dsp_fraction >= 0.5:
            raise ValueError("Double support fraction must be in [0.0, 0.5).")
            
        # ── Pre-allocated Hot-Path Buffers ──────────────────────────────────
        # contact_states: [Left Foot, Right Foot] -> 1 = Ground Contact, 0 = Flight Phase
        self.contact_states = np.ones(2, dtype=np.int32)
        
        # swing_phases: Normalized progress [0.0, 1.0) for the airborne foot
        self.swing_phases = np.zeros(2, dtype=np.float64)
        
        # Internal tracking scalars
        self.stride_phase = 0.0
        self.current_step_index = 0
        self._last_step_index = -1
        self.new_step_triggered = False

    def update(self, sim_time: float) -> None:
        """
        Processes master clock time and resolves phase state layout.
        
        Must be executed on every physics iteration step (500 Hz).
        """
        # Determine continuous phase tracking within the full stride cycle [0, 1)
        # Using math.fmod to bypass accumulation errors found in iterative addition
        self.stride_phase = math.fmod(sim_time, self.t_stride) / self.t_stride
        if self.stride_phase < 0.0:
            self.stride_phase += 1.0
            
        # Extract global discrete step count
        self.current_step_index = int(sim_time / self.t_step)
        
        # Check for step transition edge (latch condition for footstep planner)
        if self.current_step_index != self._last_step_index:
            self.new_step_triggered = True
            self._last_step_index = self.current_step_index
        else:
            self.new_step_triggered = False

        # Resolve phase inside the current active step interval [0.0, 1.0)
        step_phase = (self.stride_phase * 2.0) % 1.0
        
        # Determine which leg is designated as the nominal leader
        # Step index even -> Left leg stance, Right leg swing
        # Step index odd  -> Right leg stance, Left leg swing
        is_left_stance_step = (self.current_step_index % 2 == 0)
        
        # Default resetting state
        self.contact_states[0] = 1
        self.contact_states[1] = 1
        self.swing_phases[0] = 0.0
        self.swing_phases[1] = 0.0
        
        # Process Single Support Phase splitting logic
        if step_phase >= self.dsp_fraction:
            # We are outside the double-support window; calculate swing progression
            normalized_swing = (step_phase - self.dsp_fraction) / (1.0 - self.dsp_fraction)
            
            if is_left_stance_step:
                self.contact_states[1] = 0  # Right foot airborne
                self.swing_phases[1] = normalized_swing
            else:
                self.contact_states[0] = 0  # Left foot airborne
                self.swing_phases[0] = normalized_swing