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
        self.A = pars.A
        self.freq = pars.freq
        self.epsilon = pars.epsilon
        self.steep = pars.steep
        self.gain = pars.gain
        # state array for recording all the variables
        self.state = np.zeros((pars.n_iterations, 2*self.n_joints))

        # indexes of the left muscle activations (optional)
        self.muscle_l = 2*np.arange(15)
        # indexes of the right muscle activations (optional)
        self.muscle_r = self.muscle_l+1


    def S_sin(self, iteration, time, timestep, pos=None):
        '''#ANTEAs
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
        '''
        # Implementation of sinusoidal gain function
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            #print(f"i={i}, l_idx={l_index}, r_idx={r_index}")
            #self.state[iteration, l_index] = 0.5 
            self.state[iteration, l_index] = 0.5 + self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))

            self.state[iteration, r_index] = 0.5 - self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            #self.state[iteration, r_index] = 0.5
            
        return self.state[iteration, :]


        
    def S_square(self, iteration, time, timestep, pos=None):
        """Square wave gain function.
        this function is optional. """

        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            sign_value = np.sign(np.sin(2*np.pi*(self.freq*time- self.epsilon*i/self.n_joints)))
            self.state[iteration, l_index] = 0.5 + 0.5 * self.A * sign_value 
            self.state[iteration, r_index] = 0.5 + 0.5 * self.A * sign_value 
        return self.state[iteration, :]


    def S_trapezoidal(self, iteration, time, timestep, pos=None):
        """Trapezoidal wave gain function. 
        This function allows to obtain a more natural-like swimming behavior for the zebrafish model.
        The parameter steepness allows to switch from a sinusoidal wave to a trapezoidal wave. """
        '''
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            # Compute the sin input with adjusted steepness
            sigmoid_input_l = self.steep * (0.5 + self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints)))
            sigmoid_input_r = self.steep * (0.5 - self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints)))
            # Compute the sigmoid activation function starting from the sin input with adapted steepness
            sigmoid_activation_l = 1 / (1 + np.exp(-sigmoid_input_l))
            sigmoid_activation_r = 1 / (1 + np.exp(-sigmoid_input_r))

            # Assign activations based on the muscle
            self.state[iteration, l_index] =  sigmoid_activation_l
            self.state[iteration, r_index] =  sigmoid_activation_r
        '''
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            # Compute the sin input with adjusted steepness
            sigmoid_input_l =   self.steep*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            sigmoid_input_r =   self.steep*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            # Compute the sigmoid activation function starting from the sin input with adapted steepness
            sigmoid_activation_l = 0.5 + self.A/2 * np.tanh(sigmoid_input_l ) #1 / (1 + np.exp(-sigmoid_input_l))
            sigmoid_activation_r = 0.5 - self.A/2 * np.tanh(sigmoid_input_r ) #1 / (1 + np.exp(-sigmoid_input_r))
            # Assign activations based on the muscle
            self.state[iteration, l_index] =  sigmoid_activation_l
            self.state[iteration, r_index] =  sigmoid_activation_r


        return self.state[iteration, :]

        
    def step(self, iteration, time, timestep, pos):
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
        It is possible to choose between a sinusoidal, a fully square and a trapezoidal wave controller
        """
        if self.gain == "sinusoidal":
            self.state[iteration, :] = self.S_sin(iteration, time, timestep, pos)
        elif self.gain == "squared":
            self.state[iteration, :] = self.S_square(iteration, time, timestep, pos)
        elif self.gain == "trapezoid":
            self.state[iteration, :]= self.S_trapezoidal(iteration, time, timestep, pos)

        return self.state[iteration, :]




    ##### Original code for the controller #####
    '''
    def step(self, iteration, time, timestep, pos=None):
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
        
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            #print(f"i={i}, l_idx={l_index}, r_idx={r_index}")
            #self.state[iteration, l_index] = 0.5 
            self.state[iteration, l_index] = 0.5 + self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))

            self.state[iteration, r_index] = 0.5 - self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            #self.state[iteration, r_index] = 0.5
        return self.state[iteration, :]
'''
