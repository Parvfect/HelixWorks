

import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r
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
print("Harr: \n", Harr)

graph = TannerGraph(dv, dc, k, n, ffdim=5)
graph.establish_connections(Harr)

H = PM.createHMatrix(Harr=Harr)
print("H: \n", H)
G = r.parity_to_generator(H, ffdim=5)
print("G: \n", G)

print(np.dot(G, H.T) % 5)

if np.any(np.dot(G, H.T) % 5 != 0):
    print("Matrices are not valid, aborting simulation")
    exit()

input_arr = [np.random.randint(0,5) for i in range(k)]

print(input_arr)
C = np.dot(input_arr, G) % 5

if np.any(np.dot(C, H.T) % 5 != 0):
    print("Codeword is not valid, aborting simulation")
    exit()

print(C)

read_length = 5

output_motifs = []
for i in C:
    print(symbols[i])
    output_motifs.append(i:coupon_collector(symbols[i], read_length))

print(output_motifs)