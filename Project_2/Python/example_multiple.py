
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
            compute_metrics=2,
            I=I,
            b=b,
            headless=True,
            print_metrics=False
        )
        for i, I in enumerate(np.linspace(0.05, 3, nsim))
        for j, b in enumerate(np.linspace(8, 12, nsim))
    ]

    run_multiple(pars_list, num_process=8)


if __name__ == '__main__':
    exercise_multiple()

