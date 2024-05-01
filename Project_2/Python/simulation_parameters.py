
"""Simulation parameters"""


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        """
        Class containing all neuromechanical model parameters
        Inputs:
        kwargs: extra parameter arguments (these override previous declarations)
        """
        super(SimulationParameters, self).__init__()

        # pars for the sine controller

        # simulation parameters
        self.n_joints = 15  # number of joints
        self.timestep = 0.001  # integration time step
        self.n_iterations = 4001  # number of integration time steps

        # gui/recording parameters
        self.headless = True  # For headless mode (No GUI, could be faster)
        self.fast = False  # For fast mode (not real-time)
        self.video_record = False  # For saving the video
        self.video_speed = 0.25  # video speed
        # path where the simulation data will be stored (no simulation data
        # will be saved if the string is empty)
        self.log_path = ""
        # video name (saved in the log_path folder under the video_name name)
        self.video_name = "video"
        self.video_fps = 50  # frames per second
        self.camera_id = 0  # camera type: 0=angles top view, 1=top view, 2=side view, 3=back view
        self.show_progress = True  # show progress bar of running the simulation
        # simulation id (of log_path!="" saves the simulation in the log_path
        # folder under the name "simulation_i")
        self.simulation_i = 0
        # 0 = no metrics, 1 = neural metrics, 2 = mechanical metrics, 3 = all
        # metrics (metrics are stored in network.metrics)
        self.compute_metrics = 0
        self.print_metrics = True  # if True print all computed metrics
        # if True, run_single_sim will return the controller class
        self.return_network = False
        # (keep it False when running simulations on mutliple cpu cores,
        # or it will saturate the RAM memory)
        self.random_spine = True  # if True, initialize the joints angles in a random position

        # parameters of the firing rate controller
        # "sine" for using the WaveController (Project 1), "firing_rate" for using the FiringRateController (Project 2)
        self.controller = "firing_rate"
        self.n_neurons = 50  # number of CPG/sensory neurons
        self.n_muscle_cells = 10  # number of muscle cells
        self.method = "euler"  # integration method (euler or noise)

        # muscle cells parameters
        self.taum_a = 0.005  # muscle cell activation timescale
        self.taum_d = 0.02  # muscle cell deactivation timescale
        self.w_V2a2muscle = 0.3  # CPG to muscle cell connection weight
        # conversion weight from muscle cell activity to muscle activations
        self.act_strength = 0.3

        # CPG pars
        self.I = 10  # connstant input
        self.Idiff = 0.  # left-right input difference
        self.n_asc = 1  # number of ascending CPG connections
        self.n_desc = 2  # number of descending CPG connections
        self.tau = 0.002  # neuron timescale
        self.taua = 0.3  # adaptation timescale
        self.b = 10  # adaptation strength
        self.gamma = 0.5  # adaptation rate
        self.w_inh = 2  # inhibitory strength

        # stretch pars
        self.w_stretch = 0  # feedback strength
        self.n_asc_str = 10  # number of ascending stretch connections
        self.n_desc_str = 0  # number of descending stretch connections
        self.tau_str = 0.005  # stretch time scale

        # noise pars
        self.noise_sigma = 0  # sigma of the OU noise process

        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

