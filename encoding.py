
from graph import TannerGraph
from Hmatrixbaby import ParityCheckMatrix
from bec import generate_input_arr, generate_erasures
import numpy as np


dv, dc, k, n = 3, 6, 500, 1000
ffield = 2

def get_code_words(input_arr, G, ffield=2):
    """ Converts the input array to Code Words for given Generator Matrix """
    # Ensure the product dimensions are aligned
    return np.dot(input_arr, G) % ffield

def bec_channel_simulation(dv, dc, k, n, ffield=2)
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

    # Get Generator Matrix for the corresponding Parity Matrix
    G = ParityMatrix.get_G_from_H(H)

    # Get the Code words by encoding an input array (here - generated randomly)
    C = get_code_words(generate_input_arr(k), G, ffield)

    # Simulate Passing Code through Channel
    C_post_channel = generate_erasures(C, 0.1)
    
    # Use Tanner Graph to Decode and get Frame Error Rate Curve for the Channel
    graph.assign_values(C_post_channel)
    graph.frame_error_rate(input_arr=C_post_channel, plot=True)



