

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple
import numpy as np
import farms_pylog as pylog
import os


def exercise7():

    pylog.info("Ex 7")
    pylog.info("Implement exercise 7")
    os.makedirs('./logs/exercise7', exist_ok=True)


def main(ind=0, w_stretch=0):

    log_path = './logs/exercise7/w_stretch'+str(ind)+'/'
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':
    exercise7()

