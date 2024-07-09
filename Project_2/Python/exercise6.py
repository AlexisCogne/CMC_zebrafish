
from plotting_common import plot_time_histories, save_figure, plot_time_histories_multiple_windows_alexis
from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_single, run_multiple
import matplotlib.pyplot as plt
import numpy as np
import farms_pylog as pylog
import os



def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'
    os.makedirs(log_path, exist_ok=True)

    # Question A - Plotting of activities of CPG neurons, muscle cells and sensory neurons as well as joint angles
    gss = 2
    time_interval = [3000, 3500]
    neurons_interval = [0,10]
    pars_list_A = SimulationParameters(
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        w_stretch = gss,
        )
    #plotting_A(pars_list_A, gss, time_interval, neurons_interval) # Using w_stretch = 2 

    # Question B - Varying g_ss
    g_min = 0
    g_max = 15
    nsim = 16 
    pars_list_B = [SimulationParameters(
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        w_stretch = gss,
        )
        for i, gss in enumerate(np.linspace(g_min, g_max, nsim))
        ]
    plotting_ranges = (g_min, g_max, nsim)
    list__ranges = np.linspace(g_min, g_max, nsim)
    idx, best_speed = vary_gss(pars_list_B, plotting_ranges)
    print(f"Best g_ss value: {list__ranges[idx]} with speed {best_speed}")

def generate_color_gradient(n_colors): # Function to generate a color gradient
    colormap = plt.get_cmap('jet')
    colors = [colormap(i / n_colors) for i in range(n_colors)]
    return colors

""" Functions for plotting and varying g_ss"""
def plotting_A(parameters, gss, plot_interval, neurons_interval, folder_path = "figures/"):
    # Run the simulation
    colors = generate_color_gradient(10)
    controller = run_single(parameters)
    a, b = plot_interval[0], plot_interval[1] # Plotting for a certain time interval
    c, d = neurons_interval[0], neurons_interval[1] # Plotting only certain CPG and sensory neurons

    # Plotting CPG activities
    name_figure = f"6_CPG_activities_gss={gss}.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("CPG_activities")
    plot_time_histories(controller.times[a:b], controller.state[a:b, c:d:2], savepath = file_path)
    plt.show()

    # Plotting muscle cells activities
    left_idx = 200 + 2 * np.arange(0, 10)
    name_figure = f"6_muscle_cells_left_activities_gss={gss}.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("muscle_cells_activities_left")
    #plot_time_histories(controller.times, controller.state[:, 200:220], savepath = file_path)
    plot_time_histories_multiple_windows_alexis(controller.times[a:b], controller.state[a:b, left_idx], ylabels = np.arange(5, 15), ylim = [0,1], colors = colors, savepath = file_path)
    plt.show()

    right_idx = left_idx + 1
    name_figure = f"6_muscle_cells_right_activities_gss={gss}.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("muscle_cells_activities_right")
    #plot_time_histories(controller.times, controller.state[:, 200:220], savepath = file_path)
    plot_time_histories_multiple_windows_alexis(controller.times[a:b], controller.state[a:b, right_idx], ylabels = np.arange(5, 15), ylim = [0,1], colors = colors, savepath = file_path)
    plt.show()

    # Plotting sensory neurons activities
    name_figure = f"6_sensory_neurons_activities_gss={gss}.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("sensory_neurons_activities")
    plot_time_histories(controller.times[a:b], controller.state[a:b,220+c:220+d:2], savepath = file_path)
    plt.show()

    # Plotting joint angle positions
    name_figure = f"6_joint_angles_gss={gss}.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("joint_angles")
    joint_angles = controller.joints_positions
    plot_time_histories(
        controller.times[a:b], 
        joint_angles[a:b,:], 
        ylabel= "Joint angles [rad]", 
        labels = [f"joint {i}" for i in range(joint_angles.shape[1])],
        loc = 8, 
        ncol = 2, 
        specific_labels = [0,12,13,14],
        ylim = [-0.4, 0.3],
        savepath = file_path)
    plt.show()

def vary_gss(parameters, plotting_ranges, folder_path = "figures/"):
    controllers = run_multiple(parameters)
    frequency = []
    wavefrequency = []
    forward_speed = []
    g_min, g_max, nsim = plotting_ranges
    for i in range(nsim):
        frequency.append(controllers[i].metrics["frequency"])
        wavefrequency.append(controllers[i].metrics["wavefrequency"])
        forward_speed.append(controllers[i].metrics["fspeed_cycle"])
        print(f"Controller {i} - Frequency: {frequency[i]}, Wave Frequency: {wavefrequency[i]}, Forward Speed: {forward_speed[i]}")
    
    # Creating 3 subplots for the 3 metrics
    fig, axs =plt.subplots(1, 3, figsize=(12, 4))
    #fig.suptitle("Metrics for different g_ss values")

    axs[0].plot(np.linspace(g_min, g_max, nsim), frequency, color='red')
    axs[0].set_title("Frequency [Hz]", fontweight="bold")
    axs[0].set_xlabel("W_stretch")
    axs[0].set_xticks(np.arange(g_min, g_max+1, 3))

    axs[1].plot(np.linspace(g_min, g_max, nsim), wavefrequency, color='blue')
    axs[1].set_title("Wave Frequency [-]", fontweight="bold")
    axs[1].set_xlabel("W_stretch")
    axs[1].set_xticks(np.arange(g_min, g_max+1, 3))
   
    axs[2].plot(np.linspace(g_min, g_max, nsim), forward_speed, color='green')
    axs[2].set_title("Fcycle Speed [m/s]", fontweight="bold")
    axs[2].set_xlabel("W_stretch")
    axs[2].set_xticks(np.arange(g_min, g_max+1, 3))

    name_figure = "6_vary_gss_metrics.png"
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    plt.savefig(os.path.join(folder_path, name_figure))
    plt.show()

    idx, best_speed = max(enumerate(forward_speed), key=lambda x: x[1]) # Find the index and value of the best speed
    return idx, best_speed

if __name__ == '__main__':
    exercise6()