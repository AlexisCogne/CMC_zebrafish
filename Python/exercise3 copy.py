from util.run_closed_loop import run_single, run_multiple, run_multiple2d
from simulation_parameters import SimulationParameters
from plotting_common import plot_time_histories
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
from plotting_common import *
from collections import defaultdict
import pandas as pd
import seaborn as sns


def calculate_energy(joints_data, joint_angles, times):
    '''
    Function to calculate energy used in each simulation
    Obtain the power by multiplying the torques by the angular velocity at each joint. And then finally integrate the power. 
    Energy ->  sum the energies for all joints

    Args:
    joints_data = control.joints
    joint_angles = control.joints_positions
    times = control.times

    ouput:
    energy of simulation
    
    '''
    dt = times[1] - times[0]
    angular_velocities = np.diff(joint_angles, axis=0) / dt
    power = [joints_data[:-1,:, i] * angular_velocities for i in range(joints_data.shape[2])]
    energy = np.sum(np.abs(power)) * dt

    return energy


def exercise3():

    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")

   
    S_min = 0
    S_max = 300
    nsim = 100
    #steepness_values = np.linspace(S_min, S_max, nsim)

    log_path_base = './logs/exercise3/'
    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project1_figures/exercise3/range0_300/'
    os.makedirs(log_path_base, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    log_path = os.path.join(log_path_base, f'steepness_')
    os.makedirs(log_path, exist_ok=True)

    # Define simulation parameters
    pars = [SimulationParameters(
        simulation_i=k*nsim,
        n_iterations=10001,
        controller="sine",
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        headless=True,
        print_metrics=False,
        steep=steepness,
        gain="trapezoid",
        A = 0.17917,
        wavefrequency =0.084
    )
    for k, steepness in enumerate(np.linspace(S_min, S_max, nsim))]

    # Run the simulation
    controller = run_multiple(pars, num_process=16)

    #Printing the results
    best_speed = -np.inf
    lowest_torque = np.inf
    lowest_lateral_speed = np.inf
    ptcc_max = -np.inf
    
    sim_range = nsim

    for k in range(sim_range):
        if controller[k].metrics["fspeed_cycle"] > best_speed:
            best_speed = controller[k].metrics["fspeed_cycle"]
            best_speed_idx = k
        if controller[k].metrics["torque"] < lowest_torque:
            lowest_torque = controller[k].metrics["torque"]
            lowest_torque_idx = k
        if controller[k].metrics["lspeed_cycle"] < lowest_lateral_speed:
            lowest_lateral_speed = controller[k].metrics["lspeed_cycle"]
            lowest_lateral_speed_idx = k
        if controller[k].metrics["ptcc"] > ptcc_max:
            ptcc_max = controller[k].metrics["ptcc"]
            ptcc_max_idx = k
        
        
        #Printing purposes
         
        print('---------------------------------')
        print("-------- Simulation #", k , "---------" )
        print('---------------------------------')
        #The 0.5 in the parameters is because the amplitude is twice of the A parameter we use as input for the model
        print(f'Parameters: Amp={np.round(0.5*controller[k].metrics["amp"],5)}, wavefrequency={np.round(controller[k].metrics["wavefrequency"],5)}')
        print("---- Computed metrics ----")
        print("Forward speed: ", "{:.2e}".format(controller[k].metrics["fspeed_cycle"]))
        print("Torque consumption: ", "{:.2e}".format(controller[k].metrics["torque"]))
        print("Lateral speed: ", "{:.2e}".format(controller[k].metrics["lspeed_cycle"]))
        print("ptcc: ", "{:.2e}".format(controller[k].metrics["ptcc"]))
    
    #Printing the best speed and lowest torque as well as their indixes
    steep = [steepness for k, steepness in enumerate(np.linspace(S_min, S_max, nsim))]
    print(" ")
    print("-----------------------")
    print("Performance summary")
    print("-----------------------")
    print(" ")

    print("best speed achieved: ", "{:.2e}".format(best_speed))
    print(f'best speed parameters: Amp={np.round(0.5 * controller[best_speed_idx].metrics["amp"],5)}, Wavefrequency = {np.round(controller[best_speed_idx].metrics["wavefrequency"],5)},  Steepness = {np.round(steep[best_speed_idx],5)}')
    print("lowest torque achieved: ", "{:.2e}".format(lowest_torque))
    print(f'lowest torque parameters: Amp={np.round(0.5 * controller[lowest_torque_idx].metrics["amp"],5)}, Wavefrequency = {np.round(controller[lowest_torque_idx].metrics["wavefrequency"],5)}')
    print("lowest lateral speed achieved: ", "{:.2e}".format(lowest_lateral_speed))
    print(f'lowest lateral speed parameters: Amp={np.round(0.5 * controller[lowest_lateral_speed_idx].metrics["amp"],5)}, Wavefrequency = {np.round(controller[lowest_lateral_speed_idx].metrics["wavefrequency"],5)}')
    print("ptcc max achieved: ", "{:.2e}".format(ptcc_max))
    print(f'ptcc max parameters: Amp={np.round(0.5 * controller[ptcc_max_idx].metrics["amp"],5)}, Wavefrequency = {np.round(controller[ptcc_max_idx].metrics["wavefrequency"],5)}')
    


    all_metrics = defaultdict(dict)
    for k, ctrl in enumerate(controller):
        for metric, value in ctrl.metrics.items():
            all_metrics[metric][k] = value
    #print(all_metrics)
            
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    amp_values = np.array([controller[i].metrics['amp'] for i in range(len(controller))])
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    lspeed_values = np.array([controller[i].metrics['lspeed_cycle'] for i in range(len(controller))])
    torque_values = np.array([controller[i].metrics['torque'] for i in range(len(controller))])
    ptcc_values = np.array([controller[i].metrics['ptcc'] for i in range(len(controller))])
    eff = fspeed_values-3*torque_values

    # Plot the results
    
    df = pd.DataFrame({'Steepness': steep, 'Wavefrequency': wavefrequency_values, 'Amplitude': amp_values, 'Speed': fspeed_values, 'lspeed': lspeed_values, 'torque': torque_values, 'ptcc': ptcc_values, 'efficiency': eff})
    print(df)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='Speed', ax=axes)
    #axes.set_title('Speed vs Steepness')
    axes.set_ylabel('Speed')

    plt.tight_layout()
    fig.savefig(plot_path+'Speed_vs_Steepness.png', dpi=500)
    plt.show()
    
    #save the figure with high resolution to plot path
   

    #print("Controller.joints_data: ", controller.joints)
    #print("Controller Keys: ", controller.__dict__.keys())
    #print("Controller Metrics: ", controller.joints_active_torques)
    #print("Controller metrics: ", controller[0].metrics.keys())
    energy = [calculate_energy(control.joints, control.joints_positions, control.times) for control in controller]
    #print("Energy: ", energy)
    # print("Energy lenght: ", len(energy))
    # print("Steepness lenght: ", len(steep))

    # Plot the energy vs steepness
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(x=steep, y=energy, ax=axes)
    #axes.set_title('Energy vs Steepness')
    axes.set_xlabel('Steepness')
    axes.set_ylabel('Energy')
    plt.tight_layout()
    plt.show()
    #save the figure with high resolution to plot path
    fig.savefig(plot_path+'Energy_vs_Steepness.png', dpi=500)

    fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='torque', ax=axes2)
    #axes2.set_title('Torque vs Steepness')
    axes2.set_ylabel('Torque')
    plt.tight_layout()
    fig2.savefig(plot_path+'Steepness_vs_toque', dpi=500)

    fig3, axes3 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='lspeed', ax=axes3)
    #axes3.set_title('Lateral Speed vs Steepness')
    axes3.set_ylabel('Lateral Speed')
    plt.tight_layout()
    fig3.savefig(plot_path+'Steepness_vs_lspeed', dpi=500)

    fig4, axes4 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='ptcc', ax=axes4)
    #axes4.set_title('PTCC vs Steepness')
    axes4.set_ylabel('PTCC')
    plt.tight_layout()
    fig4.savefig(plot_path+'Steepness_vs_ptcc', dpi=500)

    fig5, axes5 = plt.subplots(1, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='efficiency', ax=axes5)
    #axes4.set_title('PTCC vs Steepness')
    axes4.set_ylabel('efficiency')
    plt.tight_layout()
    fig4.savefig(plot_path+'Steepness_vs_eff', dpi=500)


def exercise3_single():

    # Plot left muscle activation for the specified joint
    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")

   
    # S_min = 1
    # S_max = 800
    # nsim = 100
    #steepness_values = np.linspace(S_min, S_max, nsim)

    # index we want to plot (e.g., joint 3)
    #joint_index = 3  

    # for steepness in steepness_values:

    log_path_base = './logs/exercise3/'
    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project1_figures/'
    os.makedirs(log_path_base, exist_ok=True)

    log_path = os.path.join(log_path_base, f'steepness_')
    os.makedirs(log_path, exist_ok=True)


    pars_single = SimulationParameters(
            n_iterations=10001,
            controller="sine",  
            log_path=log_path,
            compute_metrics=3,
            return_network=True,
            headless = False,
            A=1,
            epsilon=1.25,
            frequency=3,
            steep =400 ,
            gain= "trapezoid"                  #change depending on the desired behavior:  "sinusoidal", "squared", "trapezoid"
        )
    pylog.info("Running the simulation")
    controller = run_single(pars_single)

    # 3.2 Plotting the head trajectory 
    name_figure = "3_trapezoid_head_trajectory.png"
    folder_path = "figures/"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("head trajectory")
    plot_trajectory(controller, color = "red", save = True, path = file_path, xlabel=r"$\mathbf{x [m]}$", ylabel=r"$\mathbf{y [m]}$", title= "Head trajectory in the (x,y) plane")

def exercise3_shape_comparison():
    # This functions compares the trapezoidal shape of the muscle activation for different steepness values

    pylog.info("Implement exercise 3 for different steepnesses values")
    log_path = './logs/exercise3search/'
    os.makedirs(log_path, exist_ok=True)

    # Define the range of steepness values to simulate
    S_min = 0
    S_max = 15
    nsim = 16

    # index of the joint we want to plot (e.g., joint 3) for the left muscle activation
    joint_index = 3  

    # Define simulation parameters for the steepness comparison
    pars_list = [
        SimulationParameters(
            n_iterations=10001, 
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            A=0.258,
            epsilon=0.168,
            frequency=3,
            steep =steepness,
            gain = "trapezoid",
            headless=True,
            print_metrics=True,
            return_network = True
        )
        for j, steepness in enumerate(np.linspace(S_min, S_max, nsim))
    ]
    left_muscle_activation = np.zeros((10001, nsim))
    right_muscle_activation = np.zeros((10001,nsim))
    

    #Running the simulation
    controller = run_multiple(pars_list)
    left_idx = controller[0].muscle_l
    right_idx = controller[0].muscle_r
    
    for i in range(nsim):
        left_muscle_activation[:,i] = controller[i].state[:, controller[i].muscle_l[joint_index]]
        right_muscle_activation[:,i] = controller[i].state[:, controller[i].muscle_r[joint_index]]

    # Plotting the left muscle activation for the specified chosen joint over different steepnesses
    name_figure = "3_left_activations.png"
    folder_path = "figures/"
    file_path = os.path.join(folder_path, name_figure)
    plot_time_histories(
        controller[-1].times[3000:4000],
        left_muscle_activation[3000:4000],
        savepath = file_path,
        title = "Waveform comparisons for different steepness values",
        labels = [f"Steepness {steepness}" for j, steepness in enumerate(np.linspace(S_min, S_max, nsim))]
    )
    # Plotting the left and right muscle activation over time
    name_figure = "3_trapezoid_left_right_activations.png"
    folder_path = "figures/"
    file_path = os.path.join(folder_path, name_figure)
    plt.figure("left_right_activations")
    plot_left_right(
        controller[-1].times[3000:6000],
        controller[-1].state[3000:6000,:],
        left_idx,
        right_idx,
        cm="jet",
        offset=0.1,
        save = True,
        file_path = file_path)

if __name__ == '__main__':
    exercise3()
    #exercise3_single()
    #exercise3_shape_comparison()