
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
import os
import numpy as np
import farms_pylog as pylog


def exercise1():

    pylog.info("Ex 1")
    pylog.info("Implement exercise 1")
    log_path = './logs/exercise1/'
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':
    exercise1()

