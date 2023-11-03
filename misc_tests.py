
import numpy as np


def G_H_test(G, H, ffdim=2):
    """ Checks if the generator matrix and parity check matrix are compatible"""
    
    if np.any(np.dot(G, H.T) % ffdim):
        print("G and H are not compatible")
        exit()