
from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
from wave_controller import WaveController
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import os
import numpy as np
import farms_pylog as pylog


def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    log_path = './logs/exercise1/'
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
            n_iterations=30001,
            controller="sine",
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            headless = False
        )
    pylog.info("Running the simulation")
    controller = run_single(
            all_pars
        )
    
    left_idx = controller.muscle_l
    right_idx = controller.muscle_r
    
    # example plot using plot_left_right
    plot_left_right(
        controller.times,
        controller.state,
        left_idx,
        right_idx,
        cm="green",
        offset=0.1)


if __name__ == '__main__':
    exercise1()
    plt.show()
