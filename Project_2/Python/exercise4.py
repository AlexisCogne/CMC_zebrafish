

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt
from plotting_common import plot_metric_vs_parameter

def collect_metrics(controllers, metric_key):
    """
    Collect the specified metric from a list of controllers.

    Parameters:
    - controllers: List of controllers
    - metric_key: Key of the metric to collect

    Returns:
    - List or numpy array of collected metric values
    """
    metric_values = [controller.metrics[metric_key] for controller in controllers if metric_key in controller.metrics]

    return metric_values
def find_range_above_threshold(parameter_values, metric_values, threshold):
    """
    Find the range of parameter values where the metric is above the threshold.

    Parameters:
    - parameter_values: Values of the parameter
    - metric_values: Values of the metric
    - threshold: Threshold value

    Returns:
    - Tuple containing the start and end values of the range
    """
    start_index = None
    end_index = None
    for i, metric_value in enumerate(metric_values):
        if metric_value > threshold:
            if start_index is None:
                start_index = i
            end_index = i
        else:
            if start_index is not None:
                break
    if start_index is None:
        return None, None
    else:
        return start_index, end_index, parameter_values[start_index], parameter_values[end_index]





def exercise4():

    pylog.info("Ex 4")
    pylog.info("Implement exercise 4")
    log_path = './logs/exercise4/'
    save_path = '/Users/maxgrobbelaar/Documents/EPFL_Spring_2024/Computational Motor control/Project 2/figures/'

    os.makedirs(log_path, exist_ok=True)
    
    nsim = 20

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    I_arr = np.linspace(0.05, 30, nsim)
    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim,
            n_iterations=10001,
            log_path="",
            video_record=False,
            compute_metrics=3,
            I=I,
            #b=b,
            headless=True,
            print_metrics=False,
            return_network=True,
        )
        for i, I in enumerate(I_arr)
        #for j, b in enumerate(np.linspace(8, 12, nsim))
    ]

    controllers = run_multiple(pars_list, num_process=8)
    #print("Controller keys: ",controllers[0].metrics.keys())
    # Call collect_metrics with the list of controllers and the desired metric key
    ptcc_vals = collect_metrics(controllers, 'ptcc')

    # Call the function with I_arr and metrics list, and specify the threshold
    start_index, end_index, start_range, end_range = find_range_above_threshold(I_arr, ptcc_vals, 1.5)

    # Print the range of I values where the metric is above 1.5
    if start_range is not None and end_range is not None:
        print(f"Range of I values where the metric is above 1.5: [{start_range}, {end_range}]")
    else:
        print("No range found where the metric is above 1.5")

    print([start_index, end_index])

    # Plot the collected metric values against I_arr
    plt.figure(figsize=(12, 10))
    plot_metric_vs_parameter(I_arr, ptcc_vals, 'I', 'ptcc', save_path=save_path + 'ptcc_vs_I.png')

    
    wavefrequency_vals = collect_metrics(controllers, 'wavefrequency')
    plot_metric_vs_parameter(I_arr, wavefrequency_vals, 'I', 'Wave Frequency', range_values=[start_index,end_index], save_path=save_path + 'wavefrequency_vs_I.png',color='r')

    frequency_vals = collect_metrics(controllers, 'frequency')
    plot_metric_vs_parameter(I_arr, frequency_vals, 'I', 'Frequency', range_values=[start_index,end_index], save_path=save_path + 'frequency_vs_I.png',color='b')
    
    plt.figure(figsize=(12, 10))
    fspeed_vals = collect_metrics(controllers, 'fspeed_cycle')
    plot_metric_vs_parameter(I_arr, fspeed_vals, 'I', 'fspeed', save_path=save_path + 'speed_vs_I.png',color='g')
    '''
    
    start_range, end_range = 26.3, 26.6
    nsim_r = 30
    I_arr_range = np.linspace(start_range, end_range, nsim_r)
    pars_list_range = [
        SimulationParameters(
            simulation_i=i*nsim_r,
            n_iterations=10001,
            log_path="",
            video_record=False,
            compute_metrics=3,
            I=I,
            #b=b,
            headless=True,
            print_metrics=False,
            return_network=True,
        )
        for i, I in enumerate(I_arr_range)
        #for j, b in enumerate(np.linspace(8, 12, nsim))
    ]

    controllers_range = run_multiple(pars_list_range, num_process=8)
    # Extract the frequency and wavefrequency metrics for the range
    frequency_values = [controller.metrics['frequency'] for controller in controllers_range]
    print("Frequency values: ", frequency_values)
    wavefrequency_values = [controller.metrics['wavefrequency'] for controller in controllers_range]
    print("Wavefrequency values: ", wavefrequency_values)
    # Plot the frequency against I_arr
    plot_metric_vs_parameter(I_arr_range, frequency_values, 'I', 'Frequency', save_path + 'frequency_vs_I.png')

    # Plot the wavefrequency against I_arr
    plot_metric_vs_parameter(I_arr_range, wavefrequency_values, 'I', 'Wave Frequency', save_path + 'wavefrequency_vs_I.png')
'''


if __name__ == '__main__':
    exercise4()