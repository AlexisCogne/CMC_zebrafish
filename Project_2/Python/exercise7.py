

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple, run_single
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import *
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    os.makedirs('./logs/exercise7', exist_ok=True)
    # Question B - Varying g_ss
    g_min = 0
    g_max = 15
    nsim = 20 
    Idiff_range = np.linspace(0, 10, nsim)
    gss_range = np.linspace(g_min, g_max, 5)
    parameters_combinations = [SimulationParameters(
        simulation_i=gss_range*nsim,
        n_iterations=5001,
        log_path="",
        compute_metrics=3,
        return_network=True,
        w_stretch = gss,
        Idiff=Idiff,
        print_metrics=False
        )
        for g, gss in enumerate(gss_range)
        for idiff, Idiff in enumerate(Idiff_range)
        ]
    plotting_ranges = (g_min, g_max, nsim, Idiff_range, gss_range)
    list__ranges = np.linspace(g_min, g_max, nsim)
    idx, best_speed = vary_gss_line(parameters_combinations, plotting_ranges)
    # for g, gss in enumerate(gss_range):
    #     for idiff, Idiff in enumerate(Idiff_range):
    #         print(f"Idiff: {Idiff}, gss: {gss}")
    print(f"Best g_ss value: {list__ranges[idx]} with speed {best_speed}")
def exercise7_I():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    os.makedirs('./logs/exercise7', exist_ok=True)
    # Question B - Varying g_ss
    g_min = 0
    g_max = 15
    nsim = 20 
    I_range = np.linspace(0, 30, nsim)
    gss_range = np.linspace(g_min, g_max, 5)
    parameters_combinations = [SimulationParameters(
        simulation_i=gss_range*nsim,
        n_iterations=5001,
        log_path="",
        compute_metrics=3,
        return_network=True,
        w_stretch = gss,
        I=I,
        print_metrics=False
        )
        for g, gss in enumerate(gss_range)
        for i, I in enumerate(I_range)
        ]
    plotting_ranges = (g_min, g_max, nsim, I_range, gss_range)
    list__ranges = np.linspace(g_min, g_max, nsim)
    idx, best_speed = vary_gss_line(parameters_combinations, plotting_ranges)
    # for g, gss in enumerate(gss_range):
    #     for idiff, Idiff in enumerate(Idiff_range):
    #         print(f"Idiff: {Idiff}, gss: {gss}")
    print(f"Best g_ss value: {list__ranges[idx]} with speed {best_speed}")


def main(ind=0, w_stretch=0):

    log_path = './logs/exercise7/w_stretch'+str(ind)+'/'
    os.makedirs(log_path, exist_ok=True)

def generate_color_gradient(n_colors,map = 'jet'): # Function to generate a color gradient
    colormap = plt.get_cmap(map)
    colors = [colormap(i / n_colors) for i in range(n_colors)]
    return colors

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
def vary_gss(parameters, plotting_ranges, folder_path = "figures/",plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project 2/figures/exercise7/'):
    
    # os.makedirs(log_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    controller = run_multiple(parameters)
    frequency = []
    wavefrequency = []
    forward_speed = []
    g_min, g_max, nsim, Idiff_range, gss_range = plotting_ranges
    for i in range(nsim):
        frequency.append(controller[i].metrics["frequency"])
        wavefrequency.append(controller[i].metrics["wavefrequency"])
        forward_speed.append(controller[i].metrics["fspeed_cycle"])
        #print(f"Controller {i} - Frequency: {frequency[i]}, Wave Frequency: {wavefrequency[i]}, Forward Speed: {forward_speed[i]}")
    
    all_metrics = defaultdict(dict)

    for k, ctrl in enumerate(controller):
        for metric, value in ctrl.metrics.items():
            all_metrics[metric][k] = value
    #print(all_metrics)
    # Extracting data from the controller metrics
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    freq_values = np.array([controller[i].metrics['frequency'] for i in range(len(controller))])
    amp_values = np.array([controller[i].metrics['amp'] for i in range(len(controller))])
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    lspeed_values = np.array([controller[i].metrics['lspeed_cycle'] for i in range(len(controller))])
    curvature_values = np.array([controller[i].metrics['curvature'] for i in range(len(controller))])
    torque_values = np.array([controller[i].metrics['torque'] for i in range(len(controller))])
    ptcc_values = np.array([controller[i].metrics['ptcc'] for i in range(len(controller))])

    #print('Len of arrays: ',len(curvature_values))

    parameter_pairs = [(Idiff, gss) for g, gss in enumerate(gss_range) for idiff, Idiff in enumerate(Idiff_range)]
    #print(parameter_pairs)

    curvature_matrix = np.reshape(curvature_values, (nsim, nsim)).T
    lspeed_matrix = np.reshape(lspeed_values, (nsim, nsim)).T

    curvature_df = pd.DataFrame(curvature_matrix, index=np.round(Idiff_range, decimals=2), columns= np.round(gss_range, decimals=2))
    # Set the color tone for the heatmap
    cmap_above = plt.cm.Greens  # Color tone for values above threshold (ptcc >= 1.5)
    cmap_below = plt.cm.Reds   # Color tone for values below threshold (ptcc < 1.5)
    #threshold = 1.5
    # Define a custom colormap
    colors_above = cmap_above(np.linspace(0, 1, 128))
    colors_below = cmap_below(np.linspace(1, 0, 128))
    colors_merged = np.vstack((colors_below, colors_above))
    cmap_custom = ListedColormap(colors_merged)
    #plot heatmap using seaborn
    cmap_gen = generate_color_gradient(nsim*nsim,map='jet')
    sn_ax = sns.heatmap(curvature_df,annot=False,cmap=cmap_gen,square=True)
    #make it sqaure
    
    sn_ax.invert_yaxis()
    #make xticks vertical
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #change the font size
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('W stretch')
    plt.ylabel('I_diff')
    plt.title('2D parameter search: Curvature')
    #save the heatmap
    plt.savefig(plot_path+'_curvature_Idiff_vs_gss.png', dpi=500)
    plt.show()

    lspeed_matrix = np.reshape(lspeed_values, (nsim, nsim)).T
    lspeed_df = pd.DataFrame(lspeed_matrix, index=np.round(Idiff_range, decimals=2), columns= np.round(gss_range, decimals=2))
    #plot heatmap using seaborn
    cmap_gen = generate_color_gradient(nsim*nsim,map='hot')
    sn_ax = sns.heatmap(lspeed_df,annot=False,cmap=cmap_gen,square=True)
    #make it sqaure
    sn_ax.invert_yaxis()
    #make xticks vertical
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #change the font size
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('W stretch')
    plt.ylabel('I_diff')
    plt.title('2D parameter search: Lateral Speed')
    #save the heatmap
    plt.savefig(plot_path+'lspeed_Idiff_vs_gss.png', dpi=500)
    plt.show()

    wavefrequency_matrix = np.reshape(wavefrequency_values, (nsim, nsim)).T
    wavefrequency_df = pd.DataFrame(wavefrequency_matrix, index=np.round(Idiff_range, decimals=2), columns= np.round(gss_range, decimals=2))
    #plot heatmap using seaborn
    cmap_gen = generate_color_gradient(nsim*nsim,map='jet')
    sn_ax = sns.heatmap(wavefrequency_df,annot=False,cmap=cmap_gen,square=True)
    #make it sqaure
    sn_ax.invert_yaxis()

    #make xticks vertical
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #change the font size
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('W stretch')
    plt.ylabel('I_diff')
    plt.title('2D parameter search: Wave Frequency')
    #save the heatmap
    plt.savefig(plot_path+'wavefrequency_Idiff_vs_gss.png', dpi=500)
    plt.show()

    frequency_matrix = np.reshape(freq_values, (nsim, nsim)).T
    frequency_df = pd.DataFrame(frequency_matrix, index=np.round(Idiff_range, decimals=2), columns= np.round(gss_range, decimals=2))
    #plot heatmap using seaborn
    cmap_gen = generate_color_gradient(nsim*nsim,map='jet')
    sn_ax = sns.heatmap(frequency_df,annot=False,cmap=cmap_gen,square=True)
    #make it sqaure
    sn_ax.invert_yaxis()
    #make xticks vertical
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #change the font size
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('W stretch')
    plt.ylabel('I_diff')
    plt.title('2D parameter search: Frequency')
    #save the heatmap
    plt.savefig(plot_path+'frequency_Idiff_vs_gss.png', dpi=500)
    plt.show()



    '''
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

    name_figure = "7_vary_gss_metrics.png"
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    plt.savefig(os.path.join(folder_path, name_figure))
    plt.show()
    '''
    idx, best_speed = max(enumerate(forward_speed), key=lambda x: x[1]) # Find the index and value of the best speed
    return idx, best_speed
def vary_gss_line(parameters, plotting_ranges, folder_path="figures/", plot_path='/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project 2/figures/exercise7/'):
    os.makedirs(plot_path, exist_ok=True)
    controller = run_multiple(parameters)
    
    g_min, g_max, nsim, Idiff_range, gss_range = plotting_ranges
    
    # Extracting data from the controller metrics
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    ptcc_values = np.array([controller[i].metrics['ptcc'] for i in range(len(controller))])
    frequency_values = np.array([controller[i].metrics['frequency'] for i in range(len(controller))])
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    curvature_values = np.array([controller[i].metrics['curvature'] for i in range(len(controller))])
    lateral_speed_values = np.array([controller[i].metrics['lspeed_cycle'] for i in range(len(controller))])
    # Create subplots
    fig, ax = plt.subplots(2, 1,figsize=(20, 16))
    fig2, ax2 = plt.subplots(2, 1,figsize=(20, 16))

    for i, gss in enumerate(gss_range):
        fcycle_speed_values_gss = []
        ptcc_values_gss = []
        frequency_values_gss = []
        wavefrequency_values_gss = []
        curvature_values_gss = []
        lateral_speed_values_gss = []
        for j, I in enumerate(Idiff_range):
            idx = i * len(Idiff_range) + j
            fcycle_speed_values_gss.append(fspeed_values[idx])
            ptcc_values_gss.append(ptcc_values[idx])
            frequency_values_gss.append(frequency_values[idx])
            wavefrequency_values_gss.append(wavefrequency_values[idx])
            curvature_values_gss.append(curvature_values[idx])
            lateral_speed_values_gss.append(lateral_speed_values[idx])
        
        # Plot forward speed for the current gss
        ax[0].plot(Idiff_range, ptcc_values_gss, label=f"gss={gss}")
        ax[1].plot(Idiff_range, fcycle_speed_values_gss, label=f"gss={gss}")

        ax2[0].plot(Idiff_range, frequency_values_gss, label=f"gss={gss}")
        ax2[1].plot(Idiff_range, wavefrequency_values_gss, label=f"gss={gss}")


    #ax[0].set_xlabel('I_diff')
    ax[0].set_ylabel('ptcc')
    #ax[0].set_title('ptcc vs I_diff for different gss values')
    ax[0].legend(fontsize='small')
    ax[0].grid(True)

    ax[1].set_xlabel('I')
    ax[1].set_ylabel('Forward speed (m/s)')
    #ax[1].set_title('Forward speed vs I_diff for different gss values')
    #ax[1].legend(fontsize='small')
    ax[1].grid(True)

    fig.savefig(plot_path + 'fspeed_ptcc_vs_I.png')  # Save the figure

    #ax2[0].set_xlabel('I diff')
    ax2[0].set_ylabel('Frequency (Hz)')
    #ax[0].set_title('ptcc vs I for different gss values')
    ax2[0].legend(fontsize='small')
    ax2[0].grid(True)

    ax2[1].set_xlabel('I')
    ax2[1].set_ylabel('Wave Frequency (-)')
    #axs[1].set_title('Forward speed vs I for different gss values')
    #axs[1].legend()
    ax2[1].grid(True)
    fig2.savefig(plot_path + 'frequency_wavefrequency_vs_I.png')  # Save the figure
    plt.show()
def exercise7_Idiff_plot_metrics():
    g_min = 0
    g_max = 15
    nsim = 20
    Idiff_range = np.linspace(0, 10, nsim)
    gss_range = np.linspace(g_min, g_max, 5)
    # create subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig_cur, axs_cur = plt.subplots(2, 1, figsize=(20, 20))
    for i, gss in enumerate(gss_range):
        ptcc_values_gss = []
        fcycle_speed_values_gss = []
        curvatures = []
        wavefrequencies = []
        frequencies = []
        lspeed_values = []
        for j, Idiff in enumerate(Idiff_range):
            parameters = SimulationParameters(
                simulation_i=gss * nsim,
                n_iterations=5001,
                log_path="",
                compute_metrics=3,
                return_network=True,
                w_stretch=gss,
                Idiff=Idiff,
                print_metrics=False
            )
            controller = run_single(parameters)
            ptcc_values_gss.append(controller.metrics['ptcc'])
            fcycle_speed_values_gss.append(controller.metrics['fspeed_cycle'])
            curvatures.append(controller.metrics['curvature'])
            wavefrequencies.append(controller.metrics['wavefrequency'])
            frequencies.append(controller.metrics['frequency'])
            lspeed_values.append(controller.metrics['lspeed_cycle'])

            
        # Plot both ptcc and fcycle_speed for the current gss
        axs[0].plot(Idiff_range, frequencies, label=f"gss={gss}")
        axs[1].plot(Idiff_range, wavefrequencies, label=f"gss={gss}")
        axs_cur[0].plot(Idiff_range, curvatures, label=f"gss={gss}")
        axs_cur[1].plot(Idiff_range, lspeed_values, label=f"gss={gss}")

    axs[0].set_xlabel('I diff')
    axs[0].set_ylabel('Frequency (Hz)')
    #axs[0].set_title('ptcc vs I for different gss values')
    #axs[0].legend(fontsize='small')
    axs[0].grid(True)

    axs[1].set_xlabel('I diff')
    axs[1].set_ylabel('Wave Frequency (-)')
    #axs[1].set_title('Forward speed vs I for different gss values')
    #axs[1].legend()
    axs[1].grid(True)
    
    axs_cur[0].set_xlabel('I diff')
    axs_cur[0].set_ylabel('Curvature')
    #axs[1].set_title('Forward speed vs I for different gss values')
    axs_cur[0].legend(fontsize='small')
    axs_cur[0].grid(True)

    axs_cur[1].set_xlabel('I diff')
    axs_cur[1].set_ylabel('Lateral speed (m/s)')
    #axs[1].set_title('Forward speed vs I for different gss values')
    axs[1].legend(fontsize='small')
    axs_cur[1].grid(True)


    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project 2/figures/exercise7/'
    #save figures
    fig.savefig( plot_path + 'wf_freq_vs_I.png')  # Save the figure
    fig_cur.savefig(plot_path + 'curvature_and_lspeed_vs_I.png')  # Save the figure
    plt.show()



def exercise7_I_plot_metrics():
    g_min = 0
    g_max = 2
    nsim = 20
    I_range = np.linspace(0, 30, nsim)
    gss_range = np.linspace(g_min, g_max, 10)
    # create subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for i, gss in enumerate(gss_range):
        ptcc_values_gss = []
        fcycle_speed_values_gss = []
        for j, I in enumerate(I_range):
            parameters = SimulationParameters(
                simulation_i=gss * nsim,
                n_iterations=5001,
                log_path="",
                compute_metrics=3,
                return_network=True,
                w_stretch=gss,
                I=I,
                print_metrics=False
            )
            controller = run_single(parameters)
            ptcc_values_gss.append(controller.metrics['ptcc'])
            fcycle_speed_values_gss.append(controller.metrics['fspeed_cycle'])
            curvatures = controller.metrics['curvature']
            wavefrequencies = controller.metrics['wavefrequency']
            frequencies = controller.metrics['frequency']
            
        # Plot both ptcc and fcycle_speed for the current gss
        axs[0].plot(I_range, ptcc_values_gss, label=f"gss={gss}")

        axs[1].plot(I_range, fcycle_speed_values_gss, label=f"gss={gss}")

    axs[0].set_xlabel('I')
    axs[0].set_ylabel('ptcc')
    #axs[0].set_title('ptcc vs I for different gss values')
    axs[0].legend(fontsize='small')
    axs[0].grid(True)

    axs[1].set_xlabel('I')
    axs[1].set_ylabel('Forward speed (m/s)')
    #axs[1].set_title('Forward speed vs I for different gss values')
    #axs[1].legend()
    axs[1].grid(True)
    

    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project 2/figures/exercise7/'
    plt.savefig(plot_path + 'ptcc_and_fcycle_speed_vs_I.png')  # Save the figure
    plt.show()



if __name__ == '__main__':
    #exercise7()
    exercise7_I()
    #exercise7_I_plot_metrics()
    #exercise7_Idiff_plot_metrics()

