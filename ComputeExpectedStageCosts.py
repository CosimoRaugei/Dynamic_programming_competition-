"""ComputeExpectedStageCosts.py

Template to compute the expected stage cost.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch
Authors: Marius Baumann, Antonio Terpin

--
ETH Zurich
Institute for Dynamic Systems and Control
--
"""

import numpy as np
from Const import Const

def compute_expected_stage_cost(C: Const) -> np.array:


    base_cost = -1.0

    costs = {
        C.U_no_flap: base_cost,
        C.U_weak: base_cost + C.lam_weak,
        C.U_strong: base_cost + C.lam_strong,
    }
    Q = np.zeros((C.K, C.L))
    for l, u in enumerate(C.input_space):
        Q[:, l] = costs.get(u, base_cost)
        
    return Q
    

