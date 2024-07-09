
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
from plotting_common import  plot_trajectory, plot_left_right,plot_metric_vs_parameter
from exercise4 import collect_metrics
import numpy as np
import farms_pylog as pylog
import os
import matplotlib.pyplot as plt




def exercise5_performance(**kwargs):
    """
    This functions runs the simulation for default parameters, allowing to visualize the swimming behavior and to test its performance.
    """
    log_path = './logs/exercise5/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    all_pars = SimulationParameters(
        n_iterations=5001,
        log_path="",
        compute_metrics=3,
        return_network=True,
        **kwargs
    )

    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )
    #First of all we observe the fish swimming

    pylog.info("Plotting the result")

    left_idx = controller.muscle_l
    right_idx = controller.muscle_r

    # example plot using plot_trajectory
    plt.figure("trajectory_single")
    plot_trajectory(controller)

    #Printing the performance associated to the default parameters
    print(" ")
    print("-----------------------")
    print("Performance summary")
    print("-----------------------")
    print(" ")

    speed = controller.metrics["fspeed_cycle"]
    torque = controller.metrics["torque"]
    lateral_speed = controller.metrics["lspeed_cycle"]
    ptcc = controller.metrics["ptcc"]
    
    print("best speed achieved: ", "{:.2e}".format(speed))
    print("lowest torque achieved: ", "{:.2e}".format(torque))
    print("lowest lateral speed achieved: ", "{:.2e}".format(lateral_speed))
    print("ptcc max achieved: ", "{:.2e}".format(ptcc))
    #
    print(f'Parameters: frequency={np.round(0.5*controller.metrics["frequency"],5)}, Amp={np.round(0.5*controller.metrics["amp"],5)}, wavefrequency={np.round(controller.metrics["wavefrequency"],5)}')




def exercise5_turning_performance(**kwargs):
    """
    Function used to vary I_diff and measure the associated curvature and lateral speed.
    """
    log_path = './logs/exercise5/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    nsim = 8

    # Define the range of Idiff values to test
    Idiff_range = np.linspace(0, 4, nsim)
    
    par_list = [
        SimulationParameters(
            simulation_i=idiff*nsim,  
            n_iterations=3000,  
            log_path=log_path,
            video_record=False,  
            compute_metrics=3,  
            Idiff=Idiff,  
            headless=True,  
            print_metrics=False,
            return_network=True, 
            **kwargs
            )
        for idiff, Idiff in enumerate(Idiff_range)
    ]
    
    # Plot the interesting metric values against Idiff

    controllers = run_multiple(par_list, num_process=8)
    #lateral speed cycle
    lspeed_values = [controller.metrics["lspeed_cycle"] for controller in controllers]
    print("Lateral speed cycle values: ", lspeed_values)
    plot_metric_vs_parameter(Idiff_range, lspeed_values, 'Idiff', "lspeed_cycle")
    #lateral speed PCA
    lspeed_values_PCA = [controller.metrics["lspeed_PCA"] for controller in controllers]
    print("Lateral speed PCA values: ", lspeed_values_PCA)
    plot_metric_vs_parameter(Idiff_range, lspeed_values_PCA, 'Idiff', "lspeed_PCA")
    #curvature
    curv_values = [controller.metrics['curvature'] for controller in controllers]
    print("Curvature values: ", curv_values)
    plot_metric_vs_parameter(Idiff_range, curv_values, 'Idiff', "curvature")

    return Idiff_range, lspeed_values, lspeed_values_PCA, curv_values



def exercise5_additional_performance(**kwargs):
    """
    Function used to vary I_diff and measure the associated curvature and lateral speed.
    """
    log_path = './logs/exercise5/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    nsim = 8

    # Define the range of Idiff values to test
    Idiff_range = np.linspace(0, 4, nsim)
    
    par_list = [
        SimulationParameters(
            simulation_i=idiff*nsim,  
            n_iterations=3000,  
            log_path=log_path,
            video_record=False,  
            compute_metrics=3,  
            Idiff=Idiff,  
            headless=True,  
            print_metrics=False,
            return_network=True, 
            **kwargs
            )
        for idiff, Idiff in enumerate(Idiff_range)
    ]
    
    # Plot the interesting metric values against Idiff

    controllers = run_multiple(par_list, num_process=8)
    #lateral speed cycle
    lspeed_values = [controller.metrics["lspeed_cycle"] for controller in controllers]
    print("Lateral speed cycle values: ", lspeed_values)
    plot_metric_vs_parameter(Idiff_range, lspeed_values, 'Idiff', "lspeed_cycle")
    #lateral speed PCA
    lspeed_values_PCA = [controller.metrics["lspeed_PCA"] for controller in controllers]
    print("Lateral speed PCA values: ", lspeed_values_PCA)
    plot_metric_vs_parameter(Idiff_range, lspeed_values_PCA, 'Idiff', "lspeed_PCA")
    #curvature
    curv_values = [controller.metrics['curvature'] for controller in controllers]
    print("Curvature values: ", curv_values)
    plot_metric_vs_parameter(Idiff_range, curv_values, 'Idiff', "curvature")
    #plt.ylabel()
    #ptcc
    ptcc_values = [controller.metrics['ptcc'] for controller in controllers]
    print("PTCC values: ", ptcc_values)
    plot_metric_vs_parameter(Idiff_range, ptcc_values, 'Idiff', "ptcc")
    #wavefrequency
    w_values = [controller.metrics['wavefrequency'] for controller in controllers]
    print("Wavefrequency values: ", w_values)
    plot_metric_vs_parameter(Idiff_range, w_values, 'Idiff', "wavefrequency")
    #energy
    e_values = [calculate_energy(controller.joints, controller.joints_positions, controller.times) for controller in controllers]
    print("Energy: ", e_values)
    plot_metric_vs_parameter(Idiff_range, e_values, 'Idiff', "Rotational Energy")

    return Idiff_range, lspeed_values, lspeed_values_PCA, curv_values, ptcc_values, w_values, e_values



def exercise5_radius_check(fig, Idiffs, **kwargs):
    """
    Function used to plot the turning performance using the curvature.
    """
    pylog.info("Ex 5: Turning Performance Analysis")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    # Define parameters for the simulation runs
    par_list = [
        SimulationParameters(
            simulation_i=idiff * len(Idiffs),
            n_iterations=5001,
            log_path=log_path,
            compute_metrics=3,
            Idiff=Idiff,
            headless=True,
            print_metrics=False,
            return_network=True,
            **kwargs
        )
        for idiff, Idiff in enumerate(Idiffs)
    ]

    # Run the simulations
    controllers = run_multiple(par_list, num_process=4)

    # Extract curvature metric
    curv_values = [controller.metrics['curvature'] for controller in controllers]
    
    #Compute the turning radius from the curvature metric
    for i, controller in enumerate(controllers):
        curvature = curv_values[i]
        turning_radius = compute_turning_radius(curvature)

        plt.figure(f"Trajectory_Idiff_{Idiffs[i]}")
        title_text = (f'Turning radius VS center of mass trajectory\n'
                      f'Idiff={Idiffs[i]:.2f}, '
                      f'Curvature ={{curvature:.2f}}, '
                      f'Turning Radius ={{turning_radius:.2f}}').format(
                      curvature=curvature, turning_radius=turning_radius)
        #plot the mean (center of mass of the trajectory)
        plot_center_trajectory(controller, label=f'Head trajectory',title=title_text)
        #plot_trajectory(controller, label=f'Idiff={Idiffs[i]}',
        #                title=f'Turning radius VS center of mass trajectory\nIdiff={Idiffs[i]}, Curvature(--)={curvature:.3f}, Turning Radius (--)={turning_radius:.3f}')

        # Plot the turning radius as a circle
        if turning_radius != float('inf'):
            plot_turning_radius(controller, turning_radius)      #plot the turning radius obtained from the curvature computation 
        
        plt.legend(loc='lower right')
        plt.grid(True)

    plt.show()
    return curv_values




def plot_center_trajectory(controller, label=None, color=None, save=False, xlabel='x [m]', ylabel="y [m]", title="", path=None, sim_fraction=1):
        link_positions = np.array(controller.links_positions)[:, :, :2]  # Extract x and y coordinates
        n_steps = link_positions.shape[0]
        n_steps_considered = round(n_steps * sim_fraction)

        # Compute center of mass trajectory and plot it
        center_of_mass_trajectory = np.mean(link_positions[-n_steps_considered:, :, :], axis=1)

        """Plot center of mass trajectory"""
        plt.plot(center_of_mass_trajectory[:-1, 0], center_of_mass_trajectory[:-1, 1], label=label, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.axis('equal')
        plt.grid(True)
        if save:
            plt.savefig(path)



def compute_turning_radius(curvature):
    """
    Function computing the turning radius from the curvature.
    """
    if curvature != 0:
        turning_radius = 1 / curvature
    else:
        turning_radius = float('inf')  # Straight line, infinite turning radius
    return np.abs(turning_radius)


def plot_turning_radius(controller, turning_radius):
    """
    Function plotting the turning radius on the trajectory plot with respect to the center of mass.
    """
    head_positions = np.array(controller.links_positions)[:, 0, :]
    center_of_mass = np.mean(head_positions[:, :2], axis=0)

    circle = plt.Circle(center_of_mass, turning_radius, color='r', fill=False, linestyle='--', label='Turning Radius [m]')
    plt.gca().add_patch(circle)
    
    
    



def plot_swimming_behavior(specific_Idiff):
    """Plots the swimming trajectory and the CPG activations based on the chosen Idiff. """

    pylog.info("Implement exercise 5 swimming")
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    pars = SimulationParameters(
        n_iterations=5001,
        log_path=log_path,
        compute_metrics=3,
        return_network=True,
        Idiff=specific_Idiff,
        headless=False)
    
    pylog.info("Running the simulation")
    controller= run_single(pars)

    pylog.info("Plotting the result")

    left_idx = controller.muscle_l
    right_idx = controller.muscle_r    
    left_CPG = controller.all_v_left
    right_CPG = controller.all_v_right

    # example plot using plot_trajectory
    plt.figure("trajectory_single")
    plot_trajectory(controller)

    #Plot CPG activities
    plt.figure('CPG_activities_single')
    plot_left_right(
        controller.times,
        controller.state,
        left_CPG,
        right_CPG,
        cm="jet",
        offset=0.1)
    
    #Plot MC activities
    plt.figure('muscle_activities_single')
    plot_left_right(
        controller.times,
        controller.state,
        left_idx,
        right_idx,
        cm="jet",
        offset=0.1)
    


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


if __name__ == '__main__':

    #-----run the simulation for default parameters, allowing to visualize the swimming behavior and to test its performance
    exercise5_performance(headless=False)

    #-----test different Idiffs and obtain the performance: lateral speeds and curvatures associated
    exercise5_turning_performance()

    #-----test different Idiffs and obtain additional (not explicitly requested) performance: ptcc, wavefrequency and rotational energy
    exercise5_additional_performance()

    #-----check that turning radius match curvature expected
    figname="ex5"
    fig = plt.figure(figname)
    Idiffs= np.linspace(0, 4, 10)
    exercise5_radius_check(fig, Idiffs)

    #-----plot swimming trajectory and neuron activities
    plot_swimming_behavior(specific_Idiff=4)

    plt.show()