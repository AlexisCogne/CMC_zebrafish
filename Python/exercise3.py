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



def exercise3():
    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")

   
    S_min = 1
    S_max = 1000
    nsim = 10
    #steepness_values = np.linspace(S_min, S_max, nsim)

    # index we want to plot (e.g., joint 3)
    joint_index = 3  

    # for steepness in steepness_values:

    log_path_base = './logs/exercise3/'
    os.makedirs(log_path_base, exist_ok=True)

    
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
        gain="trapezoid"
    )
    for k, steepness in enumerate(np.linspace(S_min, S_max, nsim))]

    # Run the simulation
    #controller = run_single(pars)
    controller = run_multiple(pars, num_process=16)
    all_metrics = defaultdict(dict)

    for k, ctrl in enumerate(controller):
        for metric, value in ctrl.metrics.items():
            all_metrics[metric][k] = value
    #print(all_metrics)
    wavefrequency_values = np.array([controller[i].metrics['wavefrequency'] for i in range(len(controller))])
    amp_values = np.array([controller[i].metrics['amp'] for i in range(len(controller))])
    fspeed_values = np.array([controller[i].metrics['fspeed_cycle'] for i in range(len(controller))])
    print('fspeed_values: ',fspeed_values)
    print('wavefrequency_values: ',wavefrequency_values)
    print('amp_values: ',amp_values)
    # Plot the results
    steep = [steepness for k, steepness in enumerate(np.linspace(S_min, S_max, nsim))]
    df = pd.DataFrame({'Steepness': steep, 'Wavefrequency': wavefrequency_values, 'Amplitude': amp_values, 'Speed': fspeed_values})
    print(df)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    sns.lineplot(data=df, x='Steepness', y='Speed', ax=axes[0])
    axes[0].set_title('Speed vs Steepness')
    axes[0].set_ylabel('Speed')
    sns.lineplot(data=df, x='Steepness', y='Wavefrequency', ax=axes[1])
    axes[1].set_title('Wavefrequency vs Steepness')
    axes[1].set_ylabel('Wavefrequency')
    sns.lineplot(data=df, x='Steepness', y='Amplitude', ax=axes[2])
    axes[2].set_title('Amplitude vs Steepness')
    axes[2].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


    # left_muscle_activation = controller.state[:, controller.muscle_l[joint_index]]
    # right_muscle_activation = controller.state[:, controller.muscle_r[joint_index]]
    #print(controller.metrics)
    # Plot left muscle activation for the specified joint
    # axes[0].plot(controller.times, left_muscle_activation, label=f'Steepness {steepness:.1f}')
    # axes[0].set_title(f'Left Muscle Activation (Joint {joint_index})')
    # axes[0].set_ylabel('Activation')
    # axes[0].legend()

    # # Plot right muscle activation for the specified joint
    # axes[1].plot(controller.times, right_muscle_activation, label=f'Steepness {steepness:.1f}')
    # axes[1].set_title(f'Right Muscle Activation (Joint {joint_index})')
    # axes[1].set_ylabel('Activation')
    # axes[1].legend()

   
    # axes[1].set_xlabel('Time (s)')


    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    exercise3()