
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
        # "sine" for using the WaveController (Project 1), "firing_rate" for using the FiringRateController (Project 2)
        self.controller = "sine"
        self.n_joints = 15  # number of joints
        self.method = "ode"  # integration method (for Project 2)
        # sparse method could be benificial for many neurons and few
        # connections
        self.sparse = False
        self.timestep = 0.001  # integration time step
        self.n_iterations = 10001  # number of integration time steps

        # gui/recording parameters
        self.headless = True  # For headless mode (No GUI, could be faster)
        self.fast = False  # For fast mode (not real-time)
        self.video_record = False  # For saving the video
        self.video_speed = 0.33  # video speed
        # path where the simulation data will be stored (no simulation data
        # will be saved if the string is empty)
        self.log_path = ""
        # video name (saved in the log_path folder under the video_name name)
        self.video_name = "video"
        self.video_fps = 50  # frames per second
        self.camera_id = 1  # camera type: 0=angles top view, 1=top view, 2=side view, 3=back view
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

        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

