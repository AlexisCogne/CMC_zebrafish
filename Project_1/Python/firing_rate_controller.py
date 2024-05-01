"""Network controller"""

import numpy as np
from scipy.integrate import ode
from scipy.sparse import csr_array


class FiringRateController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

