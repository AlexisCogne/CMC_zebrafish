
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



def exercise2():

    pylog.info("Ex 2")
    pylog.info("Implement exercise 2")
    log_path = './logs/exercise2/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 100
    amp_min = 0.1
    amp_max = 2
    wavefreq_min = 0
    wavefreq_max = 3

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    
    pars_list1 = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=10001, #making it short but should increase to at least 30'000
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            A=amp,
            epsilon=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
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
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            #A=amp,
            epsilon=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
        )
        for j, wavefrequency in enumerate(np.linspace(wavefreq_min, wavefreq_max, nsim))]
    
    pars_list = pars_list_amp
    controller = run_multiple(pars_list, num_process=10)

    #Printing the results
    best_speed = -np.inf
    lowest_torque = np.inf
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
        
        #Printing purposes
        print('---------------------------------')
        print("-------- Simulation #", k , "---------" )
        print('---------------------------------')
        #The 0.5 in the parameters is because the amplitude is twice of the A parameter we use as input for the model
        print(f'Parameters: Amp={np.round(0.5*controller[k].metrics["amp"],5)}, wavefrequency={np.round(controller[k].metrics["wavefrequency"],5)}')
        print("---- Computed metrics ----")
        print("Forward speed: ", "{:.2e}".format(controller[k].metrics["fspeed_cycle"]))
        print("Torque consumption: ", "{:.2e}".format(controller[k].metrics["torque"]))
    
    #Printing the best speed and lowest torque as well as their indixes
    print(" ")
    print("-----------------------")
    print("Performance summary")
    print("-----------------------")
    print(" ")

    print("best speed achieved: ", "{:.2e}".format(best_speed))
    print(f'best speed parameters: Amp={np.round(0.5 * controller[best_speed_idx].metrics["amp"],5)}, Wawefrequency = {np.round(controller[best_speed_idx].metrics["wavefrequency"],5)}')
    print("lowest torque achieved: ", "{:.2e}".format(lowest_torque))
    print(f'lowest torque parameters: Amp={np.round(0.5 * controller[lowest_torque_idx].metrics["amp"],5)}, Wawefrequency = {np.round(controller[lowest_torque_idx].metrics["wavefrequency"],5)}')
    '''
    # 2D heatmaps to show the resultsd
    data = np.concatenate([[[controller[i].metrics['amp'], controller[i].metrics['wavefrequency']] for i in range(j*nsim, (j+1)*nsim)] for j in range(nsim)])
    print("Data: ",data)
    extent = [amp_min*2, amp_max*2, wavefreq_min, wavefreq_max]
    results = [[controller[i].metrics['fspeed_cycle'] for i in range(j*nsim, (j+1)*nsim)] for j in range(nsim)]
    print('Results: ',results)
    # plt.imshow(results, interpolation='nearest',data=data, extent=extent, aspect='auto') 
    # # Add colorbar 
    # plt.colorbar() 
    # plt.xlabel('Amplitude')
    # plt.ylabel('wavefrequency')
    # plt.title("Heatmap with color bar") 
    # plt.show() 
    '''
    
    
    #Below, code from Max on April 16th, left it here in case
    
    # fspeed_array = np.array([[controller[i].metrics['fspeed_PCA'], controller[i].metrics['wavefrequency'], controller[i].metrics['amp']] for i in range(len(controller))])
    # fspeed_labels = ['fspeed_PCA', 'wavefrequency', 'amp']
    # print(fspeed_array.shape)
    # print(fspeed_labels[0],'        ', fspeed_labels[1],'        ', fspeed_labels[2])
    # print(fspeed_array)
    # plt.figure("2d plot")
    # plot_2d(fspeed_array,labels=fspeed_labels,cmap='nipy_spectral')
    # plt.show()
    #key,items = dict.items()
    
    # all_metrics = dict()
    # for k in range(len(controller)):
    #     for metric in controller[k].metrics:
    #         if metric not in all_metrics:
    #             all_metrics[metric] = dict()
    #         all_metrics[metric][k] = controller[k].metrics[metric]

    all_metrics = defaultdict(dict)

    for k, ctrl in enumerate(controller):
        for metric, value in ctrl.metrics.items():
            all_metrics[metric][k] = value
    #print(all_metrics)
    # Extracting data from the controller metrics
    # Extracting data from the controller metrics
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    amp_values = np.array([controller[i].metrics['amp'] for i in range(len(controller))])
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    #print('fspeed_values: ',fspeed_values)

    if pars_list == pars_list1:
        #make pairs of the values
        pairs_array = np.array([[amp_values[i], wavefrequency_values[i]] for i in range(len(amp_values))])
        #print('pairs: ',pairs_array)
        fspeed_matrix = np.reshape(fspeed_values, (nsim, nsim)).T
        #print('fspeed_matrix: ',fspeed_matrix)
        
        #
        amp_pd_arr = [amp for i, amp in enumerate(np.linspace(amp_min, amp_max, nsim))]
        wf_pd_arr = [wavefrequency for j, wavefrequency in enumerate(np.linspace(wavefreq_min, wavefreq_max, nsim))]
        #make data frame where wavefrequency is the index and amp is the columns
        #print('wave_round: ', np.unique(np.round(wavefrequency_values, decimals=2)))
        results_2 = pd.DataFrame(fspeed_matrix, index=np.round(wf_pd_arr, decimals=2), columns= np.round(amp_pd_arr, decimals=2))

        #print('results_2: ',results_2)

        #plot heatmap using seaborn
        sn_ax = sns.heatmap(results_2,annot=False,cmap='nipy_spectral')
        #sns.color_palette("Spectral", as_cmap=True)
        sn_ax.invert_yaxis()
        #make xticks vertical
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        #change the font size
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.xlabel('Amplitude')
        plt.ylabel('Wavefrequency')
        plt.title('Heatmap of Forward Speed')
        plt.show()
    else:
        #Plot speed vs single parameter
        df = pd.DataFrame({'Wavefrequency': wavefrequency_values, 'Amplitude': amp_values, 'Speed': fspeed_values})
        print(df)
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        sns.lineplot(data=df, x='Wavefrequency', y='Speed', ax=axes[0])
        axes[0].set_title('Speed vs Wavefrequency')
        axes[0].set_ylabel('Speed')
        sns.lineplot(data=df, x='Amplitude', y='Speed', ax=axes[1])
        axes[1].set_title('Amplitude vs Speed')
        axes[1].set_ylabel('Speed')

    
    plt.tight_layout()
    plt.show()

    
    #print(all_metrics['frequency'].values())
    


if __name__ == '__main__':
    exercise2()

