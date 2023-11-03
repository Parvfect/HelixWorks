

import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import ParityCheckMatrix
from scipy.linalg import null_space
import sympy as sympy
from misc_tests import *


def coupon_collector_channel(arr, R):
    return [arr[random.randint(0, len(arr) - 1)] for i in range(R)]

# GF(5)
# dv dc 2 4
# k n 3 6


# 4C2 symbols - ignoring (2,3) to make it GF(5)
# Would be also 5: (2,3) which is being ignored
symbols = {0:(0,1), 1:(0,2), 2:(0,3), 3:(1,2), 4:(1,3)}

dv, dc, k, n = 2, 4, 3, 6

input_arr = [0, 1, 0]

# Initialize the parity check matrix and tanner graphs
PM = ParityCheckMatrix(dv, dc, k, n, ffdim=5)
Harr = PM.get_H_arr()

graph = TannerGraph(dv, dc, k, n, ffdim=5)
graph.establish_connections(Harr)

H = PM.createHMatrix(Harr=Harr)
G = PM.get_generator_matrix()
