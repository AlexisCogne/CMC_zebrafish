
from util.run_closed_loop import run_single, run_multiple, pretty
from simulation_parameters import SimulationParameters
from wave_controller import WaveController
import matplotlib.pyplot as plt
from plotting_common import plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows, save_figure
import os
import numpy as np
import farms_pylog as pylog


def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    log_path = './logs/exercise1/'
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
            n_iterations=10001,
            controller="sine",
            log_path=log_path,
            compute_metrics=3,
            A = 0.37,
            freq = 3,
            epsilon = 0.715,
            return_network=True,
            headless = False,
            video_record = False,
            camera_id = 1
        )
    pylog.info("Running the simulation")
    controller = run_single(
            all_pars
        )
    joint_angles = controller.joints_positions
    
    left_idx = controller.muscle_l
    right_idx = controller.muscle_r

    folder_path = "figures/"
    
    # 1.1 Plotting the activations of the left and right muscles
    name_figure = "1_fastest_2_left_right_activations.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("left_right_activations")
    plot_left_right(
        controller.times[3000:4000],
        controller.state[3000:4000,:],
        left_idx,
        right_idx,
        cm="jet",
        offset=0.1,
        save = True,
        file_path = file_path)
    
    # 1.2 Plotting the head trajectory 
    name_figure = "1_fastest_head_trajectory.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("head trajectory")
    plot_trajectory(controller, color = "red", save = True, path = file_path, xlabel=r"$\mathbf{x [m]}$", ylabel=r"$\mathbf{y [m]}$", title= "Head trajectory in the (x,y) plane")
    
    #1.3 Plotting the evolution of the joint angles
    name_figure = "1_fastest_ joint_angles.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("Joint angles")
    plot_time_histories(
        controller.times,
        joint_angles,
        ylabel="Joint angles [rad]",
        xlim = [3,4], #this is required since otherwise the y_lim is not respected as time is too long
        y_lim = [-1, 1],
        savepath = file_path,
        labels = [f"joint {i}" for i in range(joint_angles.shape[1])],
        loc = 8,
        ncol = 5
        )
    
    #1.4 Reporting performance metrics
    print(pretty(controller.metrics))


if __name__ == '__main__':
    exercise1()
    plt.show()
