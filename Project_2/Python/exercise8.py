
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm


def exercise8():

    pylog.info("Ex 8")
    pylog.info("Implement exercise 8")
    log_path = './logs/exercise8/'
    os.makedirs(log_path, exist_ok=True)

    # 2D Parameters search
    sig_min, sig_max, n_sim_sig = 0.0, 30, 31
    sigma_range = np.linspace(sig_min, sig_max, n_sim_sig)

    w_str_min, w_str_max, n_sim_w_str = 0.0, 10, 11
    w_str_range = np.linspace(w_str_min, w_str_max, n_sim_w_str)

    pars_list = [SimulationParameters(
    n_iterations=5001, 
    log_path=log_path,
    compute_metrics=3,
    return_network=True,
    method = 'noise',
    w_stretch = w_str,
    noise_sigma = sig,
    )
    for i, w_str in enumerate(w_str_range) # Parent loop changing every n_sim_sig
    for j, sig in enumerate(sigma_range) # Child loop restarting every n_sim_sig
    ]

    # Running the 2D parameters search
    feedback_investigation(pars_list, w_str_range, sigma_range)


def feedback_investigation(pars_list, w_str_range, sigma_range):
    
    controllers = run_multiple(pars_list, num_process=4)
    fspeed_values = [controller.metrics["fspeed_PCA"] for controller in controllers]
    ptcc_values = [controller.metrics["ptcc"] for controller in controllers]

    # Convert the ptcc_values to a 2D array
    ptcc_array = np.array(ptcc_values).reshape(len(w_str_range), len(sigma_range))

    # Set the color tone for the heatmap
    cmap_above = plt.cm.Greens  # Color tone for values above threshold (ptcc >= 1.5)
    cmap_below = plt.cm.Reds   # Color tone for values below threshold (ptcc < 1.5)
    threshold = 1.5

    # Set the color tone for the heatmap
    cmap_above = plt.cm.Greens  # Color tone for values above threshold (ptcc >= 1.5)
    cmap_below = plt.cm.Reds   # Color tone for values below threshold (ptcc < 1.5)
    threshold = 1.5

    # Define a custom colormap
    colors_above = cmap_above(np.linspace(0, 1, 128))
    colors_below = cmap_below(np.linspace(1, 0, 128))
    colors_merged = np.vstack((colors_below, colors_above))
    cmap_custom = ListedColormap(colors_merged)

    # Normalize the data with a TwoSlopeNorm to account for the threshold
    vmin, vmax = 0, np.nanmax(ptcc_array) #np.nanmin(ptcc_array), np.nanmax(ptcc_array)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)

    # Plot the heatmap using the custom colormap and normalization
    plt.figure()
    plt.imshow(ptcc_array, cmap=cmap_custom, norm=norm, origin='lower')

    # Set the colorbar with the appropriate normalization and labels
    cb = plt.colorbar(label='Peak-to-Through Correlation Coefficients [-]')
    cb.set_ticks([vmin, threshold, vmax])
    cb.set_ticklabels([str(0), str(threshold), str(round(vmax,1))])

    # Plot the heatmap for PTCC values
    folder_path = "figures/"
    name_figure = "8_ptcc_heatmap.png"
    
    plt.xlabel('Sigma [-]')
    plt.ylabel('W Stretch [-]')
    plt.xticks(range(0, len(sigma_range), 2), sigma_range[0::2], rotation=45, ha='right')
    plt.yticks(range(0, len(w_str_range), 2), w_str_range[0::2])
    plt.title('2D Parameter Search: PTCC Values', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, name_figure))
    plt.show()

    # Plot the heatmap for Fspeed values
    fspeed_array = np.array(fspeed_values).reshape(len(w_str_range), len(sigma_range))
    name_figure = "8_fspeed_heatmap.png"
    plt.figure()
    plt.imshow(fspeed_array, cmap='hot', origin='lower')
    plt.colorbar(label='Forward Speed PCA [m/s]')
    plt.xlabel('Sigma [-]')
    plt.ylabel('W Stretch [-]')
    plt.xticks(range(0, len(sigma_range), 2), sigma_range[0::2], rotation=45, ha='right')
    plt.yticks(range(0, len(w_str_range), 2), w_str_range[0::2])
    plt.title('2D Parameter Search: fspeed PCA Values', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, name_figure))
    plt.show()



if __name__ == '__main__':
    exercise8()

