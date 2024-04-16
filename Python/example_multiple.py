
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog


def exercise_multiple():

    log_path = './logs/example_multiple/'
    os.makedirs(log_path, exist_ok=True)

    nsim = 2

    pylog.info(
        "Running multiple simulations in parallel from a list of SimulationParameters")
    pars_list = [
        SimulationParameters(
            simulation_i=i*nsim+j,
            n_iterations=3001,
            log_path=log_path,
            video_record=False,
            compute_metrics=3,
            amp=amp,
            wavefrequency=wavefrequency,
            headless=True,
            print_metrics=True,
            return_network = True
        )
        for i, amp in enumerate(np.linspace(0.05, 2, nsim))
        for j, wavefrequency in enumerate(np.linspace(0., 0.1, nsim))
    ]

    controller = run_multiple(pars_list, num_process=16)
    for k in range(len(controller)):
        print(controller[k].metrics)
        print('---------------------------------')

    # make array n by 3 corresponding to x,y,z. For controller.metrics['fspeed_pca'], controller.metrics['wavefrequency'], controller.metrics['amp']
    fspeed_array = np.array([[controller[i].metrics['fspeed_PCA'], controller[i].metrics['wavefrequency'], controller[i].metrics['amp']] for i in range(len(controller))])
    fspeed_labels = ['fspeed_PCA', 'wavefrequency', 'amp']
    print(fspeed_array.shape)
    print(fspeed_labels[0],'      ', fspeed_labels[1],'      ', fspeed_labels[2])
    print(fspeed_array)
if __name__ == '__main__':
    exercise_multiple()

