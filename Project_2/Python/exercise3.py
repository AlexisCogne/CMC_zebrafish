
from util.run_open_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog


def exercise3():

    pylog.info("Ex 3")
    pylog.info("Implement exercise 3")
    log_path = './logs/exercise3/'
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':
    exercise3()

