"""Network controller"""

import numpy as np
import farms_pylog as pylog


class WaveController:

    """Test controller"""

    def __init__(self, pars):
        self.pars = pars
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            pars.n_iterations *
            pars.timestep,
            pars.n_iterations)
        self.n_joints = pars.n_joints

        # state array for recording all the variables
        self.state = np.zeros((pars.n_iterations, 2*self.n_joints))

        pylog.warning(
            "Implement below the step function following the instructions here and in the report")

        # indexes of the left muscle activations (optional)
        self.muscle_l = 2*np.arange(15)
        # indexes of the right muscle activations (optional)
        self.muscle_r = self.muscle_l+1

    def step(self, iteration, time, timestep, pos=None, epsilon = 1.25, A = 0.75, freq = 3):
        """
        Step function. This function passes the activation functions of the muscle model
        Inputs:
        - iteration - iteration index
        - time - time vector
        - timestep - integration timestep
        - pos (not used) - joint angle positions

        Implement here the control step function,
        it should return an array of 2*n_joint=30 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations

        In addition to returning the activation functions, store
        them in self.state for later use offline
        """
        # The lines below with if iteration < x is to change the parameters during the simulation. 
        # This will be erased later but allows to test different parameters during the simulation and see effects.
        if iteration < 5000:
            A = 0.75
            freq = 1
            epsilon = 1.25

        if iteration >= 5000 and time < 10000:
            A = 0.75
            freq = 3
            epsilon = 1.25
        if iteration > 20000:
            A = 0.75
            freq = 1
            epsilon = 1.25
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            #print(f"i={i}, l_idx={l_index}, r_idx={r_index}")
            #self.state[iteration, l_index] = 0.5 
            self.state[iteration, l_index] = 0.5 + A/2*np.sin(2*np.pi*(freq*time-epsilon*i/self.n_joints))

            self.state[iteration, r_index] = 0.5 - A/2*np.sin(2*np.pi*(freq*time-epsilon*i/self.n_joints))
            #self.state[iteration, r_index] = 0.5
        return self.state[iteration, :]

