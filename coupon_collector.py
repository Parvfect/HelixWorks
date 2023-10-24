

import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import createHMatrix
from scipy.linalg import null_space
import sympy as sympy

def get_symbols(motifs, k):
    """ Implement n choose k and get all the symbols from the motifs """

    n = len(motifs)
    symbols = []


def coupon_collector_channel(arr, R):

    output_arr = []
    for i in range(R):
        output_arr.append(arr[random.randint(0, len(arr) - 1)])
    return output_arr

def htog():
    H = createHMatrix(3,6, 10, 20)
    # Reduce H to row echelon form
    print(H)
    H = sympy.Matrix(H, pivots=False, normalize_last=False)
    print(H.rref())


# 8C4 Model
n = 8
k = 4
#motifs = np.arange(n)
#symbols = get_symbols(motifs, k)
#symbol_indices = np.arange(len(symbols))
# Should we make a dictionary of symbols and indices?
#t = TannerGraph(3,6, 100, 200)

# Choice of Symbols - gotta have to think about the pipeline - look at the example for it to make sense


htog()

"""
for i in range(R):
    choose_symbols_based_on_H()
    pass_symbols_through_coupon_collector_channel()
    decode()
"""

