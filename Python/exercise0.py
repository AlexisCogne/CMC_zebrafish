
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog


def exercise0():

    pylog.info("Ex 0")
    pylog.info("Implement exercise 0")
    log_path = './logs/exercise0/'
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':
    exercise0()

