
from util.run_closed_loop import run_multiple, run_multiple2d
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog
from plotting_common import *


def exercise2():

    pylog.info("Ex 2")
    pylog.info("Implement exercise 2")
    log_path = './logs/exercise2/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 4
    amp_min = 0.1
    amp_max = 2
    wavefreq_min = 0.1
    wavefreq_max = 3

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list1 = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=15001,
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
    controller = run_multiple(pars_list1, num_process=16)
    best_speed = -np.inf
    lowest_torque = np.inf
    
    for k in range(nsim**2):
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
    #Below, code from Max on April 16th, left it here in case
    """
    fspeed_array = np.array([[controller[i].metrics['fspeed_PCA'], controller[i].metrics['wavefrequency'], controller[i].metrics['amp']] for i in range(len(controller))])
    fspeed_labels = ['fspeed_PCA', 'wavefrequency', 'amp']
    # print(fspeed_array.shape)
    # print(fspeed_labels[0],'        ', fspeed_labels[1],'        ', fspeed_labels[2])
    # print(fspeed_array)
    plt.figure("2d plot")
    plot_2d(fspeed_array,labels=fspeed_labels,cmap='nipy_spectral')
    plt.show()
        
    
    all_metrics = dict()
    for k in range(len(controller)):
        for metric in controller[k].metrics:
            if metric not in all_metrics:
                all_metrics[metric] = dict()
            all_metrics[metric][k] = controller[k].metrics[metric]
    print(all_metrics)
    print(all_metrics['frequency'][3])
    """


if __name__ == '__main__':
    exercise2()

