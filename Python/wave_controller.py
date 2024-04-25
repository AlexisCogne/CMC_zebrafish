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
        # Implementation of sinusoidal gain function 
        '''
        Implementation of sinusoidal gain function.
        Args: 
            - self
            - iteration: total number of iterations (simulation length)
            - time
            - timestep: integration timestep
        Output: muscles activation states
        '''
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]

            self.state[iteration, l_index] = 0.5 + self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            self.state[iteration, r_index] = 0.5 - self.A/2*np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            
        return self.state[iteration, :]


        
    def S_square(self, iteration, time, timestep, pos=None):
        '''
        Implementation of square wave gain function.
        NOTE: This function is not required in the assignment.
        
        Args: 
            - self
            - iteration: total number of iterations (simulation length)
            - time
            - timestep: integration timestep
        Output: muscles activation states
        '''
        
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            
            sign_value = np.sign(np.sin(2*np.pi*(self.freq*time- self.epsilon*i/self.n_joints)))
            
            self.state[iteration, l_index] = 0.5 + 0.5 * self.A * sign_value 
            self.state[iteration, r_index] = 0.5 + 0.5 * self.A * sign_value 
            
        return self.state[iteration, :]


    def S_trapezoidal(self, iteration, time, timestep, pos=None):
        '''
        Implementation of the trapezoidal wave gain function. 
        This function allows to obtain a more natural-like swimming behavior for the zebrafish model.
        The parameter steepness allows to switch from a sinusoidal wave to a trapezoidal wave. 
        Args: 
            - self
            - iteration: total number of iterations (simulation length)
            - time
            - timestep: integration timestep
        Output: muscles activation states
        
        '''
        
        for i in range(self.n_joints):
            l_index = self.muscle_l[i]
            r_index = self.muscle_r[i]
            # Compute the base sine propagation wave
            sigmoid_input =   np.sin(2*np.pi*(self.freq*time-self.epsilon*i/self.n_joints))
            # Compute the hyperbolic tangent (tanh) activation function starting from the sin input and adapting the steepness
            sigmoid_activation_l = 0.5 + self.A/2 * np.tanh(self.steep* sigmoid_input ) 
            sigmoid_activation_r = 0.5 - self.A/2 * np.tanh(self.steep* sigmoid_input ) 
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
        #elif self.gain == "squared":
        #    self.state[iteration, :] = self.S_square(iteration, time, timestep, pos)
        elif self.gain == "trapezoid":
            self.state[iteration, :]= self.S_trapezoidal(iteration, time, timestep, pos)

        return self.state[iteration, :]

