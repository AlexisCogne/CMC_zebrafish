
from plotting_common import plot_time_histories, save_figure
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
    pars_list_A = SimulationParameters(
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        )
    plotting_A(pars_list_A)

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
    vary_gss(pars_list_B, plotting_ranges)


""" Functions for plotting and varying g_ss"""
def plotting_A(parameters, folder_path = "figures/"):
    # Run the simulation
    controller = run_single(parameters)

    # Plotting CPG activities
    name_figure = "6_CPG_activities.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("CPG_activities")
    plot_time_histories(controller.times, controller.state[:, :100], savepath = file_path)
    plt.show()

    # Plotting muscle cells activities
    name_figure = "6_muscle_cells_activities.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("muscle_cells_activities")
    plot_time_histories(controller.times, controller.state[:, 200:220], savepath = file_path)
    plt.show()

    # Plotting sensory neurons activities
    name_figure = "6_sensory_neurons_activities.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("sensory_neurons_activities")
    plot_time_histories(controller.times, controller.state[:,220:320], savepath = file_path)
    plt.show()

    # Plotting joint angle positions
    name_figure = "6_joint_angles.png"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("joint_angles")
    plot_time_histories(controller.times, controller.joints_positions, savepath = file_path)
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
    fig.suptitle("Metrics for different g_ss values")

    axs[0].plot(np.linspace(g_min, g_max, nsim), frequency, color='red')
    axs[0].set_title("Frequency")
    axs[0].set_xlabel("g_ss")
    axs[0].set_ylabel("Frequency")
    axs[1].plot(np.linspace(g_min, g_max, nsim), wavefrequency, color='blue')
    axs[1].set_title("Wave Frequency")
    axs[1].set_xlabel("g_ss")
    axs[1].set_ylabel("Wave Frequency")
    axs[2].plot(np.linspace(g_min, g_max, nsim), forward_speed, color='green')
    axs[2].set_title("Forward Speed")
    axs[2].set_xlabel("g_ss")
    axs[2].set_ylabel("Forward Speed")

    name_figure = "6_vary_gss_metrics.png"
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, name_figure))
    plt.show()

if __name__ == '__main__':
    exercise6()