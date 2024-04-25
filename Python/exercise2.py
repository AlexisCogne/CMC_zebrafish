
from util.run_closed_loop import run_multiple, run_multiple2d
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
    power = joints_data[:-1, :] * angular_velocities
    energy = np.sum(np.abs(power)) * dt

    return energy

def exercise2():

    pylog.info("Ex 2")
    pylog.info("Implement exercise 2")
    log_path = './logs/exercise2/'
    plot_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/project1_figures/exercise2/sinusoidal/25_04/2/'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    nsim = 100
    amp_min = 0.1
    amp_max = 2
    wavefreq_min = 0
    wavefreq_max = 2

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    
    pars_list1 = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=10001, #making it short but should increase to at least 30'000
            log_path="",
            video_record=False,
            compute_metrics=3,
            A=amp,
            epsilon=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True,
            steep = 7,
            gain="trapezoid"
        )
        for i, amp in enumerate(np.linspace(amp_min, amp_max, nsim))
        for j, wavefrequency in enumerate(np.linspace(wavefreq_min, wavefreq_max, nsim))
    ]
    

    pars_list_amp = [
        SimulationParameters(
            simulation_i=i*nsim,
            n_iterations=10001, #making it short but should increase to at least 30'000
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            A=amp,
            #epsilon=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
        )
        for i, amp in enumerate(np.linspace(amp_min, amp_max, nsim))]

    
    pars_list_wf = [
        SimulationParameters(
            simulation_i=j*nsim,
            n_iterations=10001, #making it short but should increase to at least 30'000
            log_path= "", #log_path,
            video_record=False,
            compute_metrics=3,
            #A=amp,
            epsilon=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
        )
        for j, wavefrequency in enumerate(np.linspace(wavefreq_min, wavefreq_max, nsim))]
    
    pars_list = pars_list_wf
    controller = run_multiple(pars_list, num_process=10)
    #controller = run_multiple2d(pars_list_amp,pars_list_wf, num_process=10)

    #Printing the results
    best_speed = -np.inf
    lowest_torque = np.inf
    lowest_lateral_speed = np.inf
    ptcc_max = -np.inf
    if pars_list == pars_list1:
        sim_range =nsim**2
    else:
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
    print(" ")
    print("-----------------------")
    print("Performance summary")
    print("-----------------------")
    print(" ")

    print("best speed achieved: ", "{:.2e}".format(best_speed))
    print(f'best speed parameters: Amp={np.round(0.5 * controller[best_speed_idx].metrics["amp"],5)}, Wavefrequency = {np.round(controller[best_speed_idx].metrics["wavefrequency"],5)}')
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
    # Extracting data from the controller metrics
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    amp_values = np.array([controller[i].metrics['amp'] for i in range(len(controller))])
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    lspeed_values = np.array([controller[i].metrics['lspeed_cycle'] for i in range(len(controller))])
    torque_values = np.array([controller[i].metrics['torque'] for i in range(len(controller))])
    ptcc_values = np.array([controller[i].metrics['ptcc'] for i in range(len(controller))])

    #print('fspeed_values: ',fspeed_values)

    if pars_list == pars_list1:
        #make pairs of the values
        pairs_array = np.array([[amp_values[i], wavefrequency_values[i]] for i in range(len(amp_values))])
        #print('pairs: ',pairs_array)
        fspeed_matrix = np.reshape(fspeed_values, (nsim, nsim)).T
        
        amp_pd_arr = [amp for i, amp in enumerate(np.linspace(amp_min, amp_max, nsim))]
        wf_pd_arr = [wavefrequency for j, wavefrequency in enumerate(np.linspace(wavefreq_min, wavefreq_max, nsim))]
        #make data frame where wavefrequency is the index and amp is the columns
        #print('wave_round: ', np.unique(np.round(wavefrequency_values, decimals=2)))
        results_2 = pd.DataFrame(fspeed_matrix, index=np.round(wf_pd_arr, decimals=2), columns= np.round(amp_pd_arr, decimals=2))

        #plot heatmap using seaborn
        sn_ax = sns.heatmap(results_2,annot=False,cmap='nipy_spectral',square=True)
        #make it sqaure
        
        sn_ax.invert_yaxis()
        #make xticks vertical
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        #change the font size
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel('Amplitude')
        plt.ylabel('Wavefrequency')
        #plt.title('Heatmap of Forward Speed')
        #save the heatmap
        plt.savefig(plot_path+'Heatmap_Speed.png', dpi=500)
        plt.show()
        
        #Heatmap of torque
        torque_matrix = np.reshape(torque_values, (nsim, nsim)).T
        results_3 = pd.DataFrame(torque_matrix, index=np.round(wf_pd_arr, decimals=2), columns= np.round(amp_pd_arr, decimals=2))
        sn_ax = sns.heatmap(results_3,annot=False,cmap='nipy_spectral',square=True)
        sn_ax.invert_yaxis()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Amplitude')
        plt.ylabel('Wavefrequency')
        #plt.title('Heatmap of Torque')
        #save the heatmap
        plt.savefig(plot_path+'Heatmap_Torque.png', dpi=500)
        plt.show()

        #Heatmap of lateral speed
        lspeed_matrix = np.reshape(lspeed_values, (nsim, nsim)).T
        results_4 = pd.DataFrame(lspeed_matrix, index=np.round(wf_pd_arr, decimals=2), columns= np.round(amp_pd_arr, decimals=2))
        sn_ax = sns.heatmap(results_4,annot=False,cmap='nipy_spectral',square=True)
        sn_ax.invert_yaxis()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Amplitude')
        plt.ylabel('Wavefrequency')
        #plt.title('Heatmap of Lateral Speed')
        #save the heatmap
        plt.savefig(plot_path+'Heatmap_Lateral_Speed.png', dpi=500)
        plt.show()

        #Heatmap of ptcc
        ptcc_matrix = np.reshape(ptcc_values, (nsim, nsim)).T
        results_5 = pd.DataFrame(ptcc_matrix, index=np.round(wf_pd_arr, decimals=2), columns= np.round(amp_pd_arr, decimals=2))
        sn_ax = sns.heatmap(results_5,annot=False,cmap='nipy_spectral',square=True)
        sn_ax.invert_yaxis()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Amplitude')
        plt.ylabel('Wavefrequency')
        #plt.title('Heatmap of PTCC')
        #save the heatmap
        plt.savefig(plot_path+'Heatmap_PTCC.png', dpi=500)
        plt.show()


    else:
        #Plot speed vs single parameter
        df = pd.DataFrame({'Wavefrequency': wavefrequency_values, 'Amplitude': amp_values/2, 'Speed': fspeed_values})
        print(df)
        
        fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10))
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=df, x='Wavefrequency', y='Speed', ax=axes1)
        #axes1.set_title('Speed vs Wavefrequency')
        axes1.set_ylabel('Speed')
        fig1.savefig(plot_path+'Wavefrequency_vs_Speed.png', dpi=500)

        
        fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10))
        
        sns.lineplot(data=df, x='Amplitude', y='Speed', ax=axes2)
        sns.set_theme(style="whitegrid")
        #axes2.set_title('Amplitude vs Speed')
        axes2.set_ylabel('Speed')
        fig2.savefig(plot_path+'Wavefrequency_vs_Speed.png', dpi=500)

    
    # plt.tight_layout()
    # plt.show()

    



if __name__ == '__main__':
    exercise2()

