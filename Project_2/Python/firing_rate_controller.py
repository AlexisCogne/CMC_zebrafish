"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog


class FiringRateController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        self.n_iterations = pars.n_iterations
        self.n_neurons = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            self.n_iterations *
            self.timestep,
            self.n_iterations)
        self.pars = pars

        self.n_eq = self.n_neurons*4 + self.n_muscle_cells*2 + self.n_neurons * 2  # number of equations: number of CPG eq+muscle cells eq+sensors eq
        
        # vector of indexes for the CPG activity variables - modify this
        # according to your implementation
        #Setting the gain function based on lab4
        self.S = self.S_sqrt
        '''
        #Eventually can be combined with Project 1
        if self.pars.gain == "sqrt_max":
            self.S = self.S_sqrt
        #elif self.pars.gain == "sinusoidal":
        #    self.S = self.S_sin
        #elif self.pars.gain == "trapezoid":
        #    self.S = self.S_trapezoidal
        '''
        
        #-------------------------------------------------- implementation index vectors
        #Implementation of index vectors for the firing rate 
        self.all_v_left  = [i for i in range(n_neurons * 2) if i % 2 == 0]   # Left CPG activity indexes: 0,2,4,... 
                                                                             #range(0, self.n_neurons)
        self.all_v_right  = [i for i in range(n_neurons * 2) if i % 2 == 1]  # Right CPG activity indexes: 1,3,5,...
                                                                             #range(self.n_neurons, 2 * self.n_neurons)
        self.all_v= range(self.n_neurons*2)

        #Implementation of index vectors for the firing rate adaptation
        self.all_a_left  = [i for i in range(n_neurons * 2, n_neurons * 4) if i % 2 == 0]  # Left CPG adaptation indexes: 100,102,...
        self.all_a_right  = [i for i in range(n_neurons * 2, n_neurons * 4) if i % 2 == 1]  # Right CPG adaptation indexes: 101,103,...
        self.all_a= range(2 * self.n_neurons, 4 * self.n_neurons)  

        #Implementation of index vectors for the muscle cell
        self.muscle_l = 4*self.n_neurons + 2 * np.arange(0, self.n_muscle_cells)  # muscle cells left indexes
        self.muscle_r = self.muscle_l+1  # muscle cells right indexes
        self.all_muscles = 4*self.n_neurons + np.arange(0, 2*self.n_muscle_cells)  # all muscle cells indexes

        #Implementation of index vectors for next parts with sensory feedback
        sensor_start_index = 4 * self.n_neurons + 2 * self.n_muscle_cells
        self.all_s_left = [i for i in range(sensor_start_index, sensor_start_index + 2 * self.n_neurons) if i % 2 == 0]  # Left sensor indexes
        self.all_s_right= [i for i in range(sensor_start_index, sensor_start_index + 2 * self.n_neurons) if i % 2 == 1]  # Right sensor indexes
        self.all_s= range(4 * self.n_neurons, 4 * self.n_neurons + 2 * self.n_muscle_cells)
        

        #---------------------------------------------------
        #pylog.warning(
        #   "Implement here the vectorization indexed for the equation variables")

        self.state = np.zeros([self.n_iterations, self.n_eq])  # equation state
        self.dstate = np.zeros([self.n_eq])  # derivative state
        self.state[0] = np.random.rand(self.n_eq)  # set random initial state

        

        self.poses = np.array([
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
        ])  # active joint distances along the body (pos=0 is the tip of the head)
        self.poses_ext = np.linspace(
            self.poses[0], self.poses[-1], self.n_neurons)  # position of the sensors

        # initialize ode solver
        self.f = self.ode_rhs

        # stepper function selection
        if self.pars.method == "euler":
            self.step = self.step_euler
        elif self.pars.method == "noise":
            self.step = self.step_euler_maruyama
            # vector of noise for the CPG voltage equations (2*n_neurons)
            self.noise_vec = np.zeros(self.n_neurons*2)

        # zero vector activations to make first and last joints passive
        # pre-computed zero activity for the first 4 joints
        self.zeros8 = np.zeros(8)
        # pre-computed zero activity for the tail joint
        self.zeros2 = np.zeros(2)

    def get_ou_noise_process_dw(self, timestep, x_prev, sigma):
        """
        Implement here the integration of the Ornstein-Uhlenbeck processes
        dx_t = -0.5*x_t*dt+sigma*dW_t
        Parameters
        ----------
        timestep: <float>
            Timestep
        x_prev: <np.array>
            Previous time step OU process
        sigma: <float>
            noise level
        Returns
        -------
        x_t{n+1}: <np.array>
            The solution x_t{n+1} of the Euler Maruyama scheme
            x_new = x_prev-0.1*x_prev*dt+sigma*sqrt(dt)*Wiener
        """

        dx_process = np.zeros_like(x_prev)

    def step_euler(self, iteration, time, timestep, pos=None):
        """Euler step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def step_euler_maruyama(self, iteration, time, timestep, pos=None):
        """Euler Maruyama step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        self.noise_vec = self.get_ou_noise_process_dw(
            timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration+1, self.all_v] += self.noise_vec
        self.state[iteration+1,
                   self.all_muscles] = np.maximum(self.state[iteration+1,
                                                             self.all_muscles],
                                                  0)  # prevent from negative muscle activations
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def motor_output(self, iteration):
        """
        Here you have to final muscle activations for the 10 active joints.
        It should return an array of 2*n_muscle_cells=20 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations
        """
        return np.zeros(
            2 *
            self.n_muscle_cells)  # here you have to final active muscle equations for the 10 joints


    def S_sqrt(self, x):
        '''
        Gain function for closed loop firing rate controller 
        '''
        return np.sqrt(np.maximum(x,0))



    def ode_rhs(self,  _time, state, pos=None):
        """Network_ODE
        You should implement here the right hand side of the system of equations
        Parameters
        ----------
        _time: <float>
            Time
        state: <np.array>
            ODE states at time _time
        Returns
        -------
        dstate: <np.array>
            Returns derivative of state
        """

        #### Extracting the current state variables ####
        #rate states
        r_left = state[self.all_v_left]
        r_right = state[self.all_v_right]
        #rate adaptation states
        a_left = state[self.all_a_left]
        a_right = state[self.all_a_right]
        #muscle states
        m_left = state[self.muscle_l]
        m_right = state[self.muscle_r]
        #---------sensory feedback states-------- NEXT PART
        #s_left = state[self.all_s_left]
        #s_right = state[self.all_s_right]



        #### Implementing the closed loop system of equations 4-8 in a vectorial form ####

        # rate equations (inspired by lab 4)
        dstate[self.all_v_left] = ( -r_left + self.S (self.pars.I - self.pars.b * a_left - self.pars.gin * self.Win.dot(r_right))) / self.pars.tau
        dstate[self.all_v_right] = (-r_right + self.S(self.pars.I - self.pars.b * a_right - self.pars.gin * self.Win.dot(r_left))) / self.pars.tau

        #rate adaptation equations
        dstate[self.all_a_left] = (-a_left + self.pars.gamma * r_left) / self.pars.taua       #check if their gamma is rho (I think they used the wrong greek letter :)
        dstate[self.all_a_right] = (-a_right + self.pars.gamma * r_right) / self.pars.taua    #same to check

        # muscle cells equations
        dstate[self.muscle_l] = (self.pars.w_V2a2muscle * self.pars.Wcm.dot(r_left) * (1 - m_left) / self.pars.taum_a - m_left / self.pars.taum_d)      
        #not sure about this parameter w_V2a2muscle, but the values correspond and the fact that is for the muscle
        dstate[self.muscle_r] = (self.pars.w_V2a2muscle * self.pars.Wcm.dot(r_right) * (1 - m_right) / self.pars.taum_a - m_right / self.pars.taum_d)
        #same doubts as for the previous one

        #------------same equations with in addition the sensory feedback----------- NEXT PART
        #dstate[self.all_v_left] = ( -r_left + self.S (self.pars.I - self.pars.b * a_left - self.pars.gin * self.pars.Win.dot(r_right) - self.w_stretch * self.Wss.dot(s_right))) / self.pars.tau
        #dstate[self.all_v_right] = (-r_right + self.S(self.pars.I - self.pars.b * a_right - self.pars.gin * self.pars.Win.dot(r_left) - self.w_stretch * self.Wss.dot(s_left))) / self.pars.tau


        return self.dstate




    def general_connectivity_matrix(self, ndesc, nasc):
        '''
        Generate a connectivity matrix of given size based on the specific descending and ascending axonal branches

        Params:
            - ndesc : number of descending connections to the closest caudal CPG cells.
            - nasc : number of ascending connections to the closest rostral cells.

        Output:
            - the generated connectivity matrix.
        '''
        # Initialization of the connectivity matrix, the size corresponds to the total number of neurons
        connectivity_matrix = np.zeros((self.n_neurons, self.n_neurons))

        # define the system
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                # Condition when i <= j and the distance is within ndesc
                if i <= j and j - i <= ndesc:
                    connectivity_matrix[i, j] = 1 / (j - i + 1)
                # Condition when i > j and the distance is within nasc
                elif i > j and i - j <= nasc:
                    connectivity_matrix[i, j] = 1 / (i - j + 1)

        return connectivity_matrix
