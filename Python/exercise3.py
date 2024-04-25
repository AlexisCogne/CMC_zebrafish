from util.run_closed_loop import run_single, run_multiple, run_multiple2d
from simulation_parameters import SimulationParameters
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

    # index we want to plot (e.g., joint 3)
    joint_index = 3  

    # for steepness in steepness_values:

    log_path_base = './logs/exercise3/'
    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project1_figures/exercise3/range0_300/'
    os.makedirs(log_path_base, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # fig.suptitle(f'Comparison of Different Steepness Values for Joint {joint_index}')

    
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
    #controller = run_single(pars)
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
    # print('fspeed_values: ',fspeed_values)
    # print('wavefrequency_values: ',wavefrequency_values)
    # print('amp_values: ',amp_values)

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
    
    left_idx = controller.muscle_l
    right_idx = controller.muscle_r
    
    # example plot using plot_left_right
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig = plt.figure("left and right muscle activations")
    plot_left_right(
        controller.times,
        controller.state,
        left_idx,
        right_idx,
        cm="green",
        offset=0.1)
    
    fig.savefig(plot_path+'left_right.png', dpi=500)
    plt.show()
    fig2 = plt.figure("trajectory")
    plt.title("Trajectory")
    
    plot_trajectory(controller)

    plt.show()
    #save the figure with high resolution to plot path
    fig2.savefig(plot_path+'trajectory.png', dpi=500)


if __name__ == '__main__':
    exercise3()
