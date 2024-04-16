
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

    nsim = 2

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list1 = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            A=amp,
            freq=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
        )
        for i, amp in enumerate(np.linspace(0.1, 2, nsim))
        for j, wavefrequency in enumerate(np.linspace(0.1, 3, nsim))
    ]
    pars_list2 = [
        SimulationParameters(
            simulation_i=j*nsim+i,
            controller="sine",
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            A=amp,
            freq=wavefrequency,
            headless=True,
            print_metrics=False,
            return_network = True
        )
        for j, wavefrequency in enumerate(np.linspace(0, 2, nsim))
        for i, amp in enumerate(np.linspace(0, 2, nsim))
    ]

    controller = run_multiple(pars_list1, num_process=16)

    #store all metrics in a dictionary of dictionaries grouped by metric and then by run
    all_metrics = dict()
    for k in range(len(controller)):
        for metric in controller[k].metrics:
            if metric not in all_metrics:
                all_metrics[metric] = dict()
            all_metrics[metric][k] = controller[k].metrics[metric]
    print(all_metrics)
    print(all_metrics['frequency'][3])





    #print()
    # for k in range(len(controller)):
        # print(controller[k].metrics)
        # print('---------------------------------')
    # make array n by 3 corresponding to x,y,z. For controller.metrics['fspeed_pca'], controller.metrics['wavefrequency'], controller.metrics['amp']
    # fspeed_array = np.array([[controller[i].metrics['fspeed_PCA'], controller[i].metrics['wavefrequency'], controller[i].metrics['amp']] for i in range(len(controller))])
    # fspeed_labels = ['fspeed_PCA', 'wavefrequency', 'amp']
    # print(fspeed_array.shape)
    # print(fspeed_labels[0],'        ', fspeed_labels[1],'        ', fspeed_labels[2])
    # print(fspeed_array)
    # plot using plot_2d
    #plt.figure("2d plot")
    #plot_2d(fspeed_array,labels=fspeed_labels,cmap='nipy_spectral')
    #plt.show()

    


if __name__ == '__main__':
    exercise2()

