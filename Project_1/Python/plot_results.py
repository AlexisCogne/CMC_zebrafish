"""Plot results"""

import farms_pylog as pylog
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from util.rw import load_object
from plotting_common import plot_2d, save_figures, plot_left_right, plot_trajectory, plot_time_histories, plot_time_histories_multiple_windows
import numpy as np
import matplotlib
import os
matplotlib.rc('font', **{"size": 20})


def plot_exercise_multiple(n_simulations, logdir):
    """
    Example showing how to load a simulation file and use the plot2d function
    """
    pylog.info(
        "Example showing how to load the simulation file and use the plot2d function")
    fspeeds = np.zeros([n_simulations, 3])
    for i in range(n_simulations):
        # load controller
        controller = load_object(logdir+"controller"+str(i))
        fspeeds[i] = [
            controller.pars.amp,
            controller.pars.wavefrequency,
            np.mean(controller.metrics["fspeed_cycle"])
        ]

    plt.figure('exercise_multiple', figsize=[10, 10])
    plot_2d(
        fspeeds,
        ['Amp', 'wavefrequency', 'Forward Speed [m/s]'],
        cmap='nipy_spectral'
    )


def main(plot=True):
    """Main"""

    pylog.info(
        "Here is an example to show how you can load a single simulation and which data you can load")
    controller = load_object("logs/example_single/controller0")

    # neural data
    state = controller.state
    metrics = controller.metrics

    # mechanical data
    links_positions = controller.links_positions  # the link positions
    links_velocities = controller.links_velocities  # the link velocities
    joints_active_torques = controller.joints_active_torques  # the joint active torques
    joints_velocities = controller.joints_velocities  # the joint velocities
    joints_positions = controller.joints_positions  # the joint positions


if __name__ == '__main__':
    main(plot=True)

