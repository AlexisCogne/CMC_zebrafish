"""Network controller"""

import numpy as np
from farms_amphibious.control.network import AnimatNetwork


class ZebrafishController(AnimatNetwork):
    """zebrafish controller"""

    def __init__(self, animat_data, controller):
        self.n_iterations = np.shape(animat_data.state.array)[0]
        super().__init__(data=animat_data, n_iterations=self.n_iterations)
        self.offsets = np.zeros(15)  # zero offsets
        self.controller = controller

    def step(self, iteration, time, timestep):
        """Control step"""
        if iteration >= self.n_iterations-1:
            return
        pos = np.array(self.data.sensors.joints.positions(iteration))
        self.data.state.array[iteration] = np.concatenate(
            [self.controller.step(iteration, time, timestep, pos=pos), self.offsets])

