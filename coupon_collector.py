

import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import createHMatrix
from scipy.linalg import null_space

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
    H = createHMatrix(3,6, 4, 8)
    Gs = null_space(H.T)
    
    G = Gs[:, 0:4]
    print(np.dot(G[0],(H.T)))


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

