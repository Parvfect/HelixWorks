
from graph import TannerGraph
from bec import generate_input_arr, generate_erasures
import numpy as np
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r


def get_code_words(input_arr, G, ffield=2):
    """ Converts the input array to Code Words for given Generator Matrix """
    # Ensure the product dimensions are aligned
    return np.dot(input_arr, G) % ffield

def bec_channel_simulation(dv, dc, k, n, ffield=2):
    """ Simulates complete Channel Simulation for Binary Erasure Channel including encoding """
    
    # Initialize ParityCheckMatrix class
    ParityMatrix = ParityCheckMatrix(dv, dc, k, n, ffield)

    # Get the Connections for the Variable Nodes
    Harr = ParityMatrix.get_H_arr()

    # Initialize Tanner Graph and establish it's connections
    graph = TannerGraph(dv, dc, k, n)
    graph.establish_connections(Harr)

    # Create Parity Matrix using generated Harr
    H = ParityMatrix.createHMatrix(Harr)

    H_rref = r.get_reduced_row_echleon_form_H(H)
    H_st, switches = r.switch_columns(H_rref, r.check_standard_form_variance(H_rref))
    
    G = r.standard_H_to_G(H_st, switches=switches)

    # The orientation is proving to be annoying - I don't know what I've done but I need G.H^T = 0 to get it right

    # Get the Code words by encoding an input array (here - generated randomly)
    input_arr = generate_input_arr(k)
    C = get_code_words(input_arr, G, ffield).astype(int)

    print(C)
    # Use Tanner Graph to Decode and get Frame Error Rate Curve for the Channel
    graph.frame_error_rate(input_arr=C, plot=True)

    # Let's try decoding for the simplest case - let's not do fer


if __name__ == "__main__":
    dv, dc, k, n = 3, 6, 5, 10
    ffield = 2
    bec_channel_simulation(dv, dc, k, n)