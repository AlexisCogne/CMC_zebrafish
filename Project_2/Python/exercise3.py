
from util.run_open_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
from plotting_common import  plot_trajectory, plot_left_right
import matplotlib.pyplot as plt


def exercise3(**kwargs):


    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")
    log_path = './logs/exercise3/'
    os.makedirs(log_path, exist_ok=True)

    pars_single = SimulationParameters(
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        **kwargs
        )
    
    pylog.info("Running the simulation")
    controller= run_single(pars_single)

    pylog.info("Plotting the result")

    left_idx = controller.muscle_l
    right_idx = controller.muscle_r    
    left_CPG = controller.all_v_left
    right_CPG = controller.all_v_right

    #Plot CPG activities
    plt.figure('CPG_activities_single')
    plot_left_right(
        controller.times,
        controller.state,
        left_CPG,
        right_CPG,
        cm="jet",
        offset=0.1)

    #Plot MC activities
    plt.figure('muscle_activities_single')
    plot_left_right(
        controller.times,
        controller.state,
        left_idx,
        right_idx,
        cm="jet",
        offset=0.1)

    
    #printing the metrics
    print(f'Parameters: frequency={np.round(0.5*controller.metrics["frequency"],5)}, Amp={np.round(0.5*controller.metrics["amp"],5)}, wavefrequency={np.round(controller.metrics["wavefrequency"],5)}')
    
    print("ptcc: ", "{:.2e}".format(controller.metrics["ptcc"]))





if __name__ == '__main__':
    exercise3()
    plt.show()

