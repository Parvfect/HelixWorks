
from graph import TannerGraph
from bec import generate_input_arr, generate_erasures
import numpy as np
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r
import matplotlib.pyplot as plt

def get_code_words(input_arr, G, ffield=2):
    """ Converts the input array to Code Words for given Generator Matrix """
    # Ensure the product dimensions are aligned
    return np.dot(input_arr, G) % ffield

def basic_encoding_procedure(dv, dc, k, n, ffield=2):
    """ Keeping a function of the basic encoding procedure for reference """
    
    # Initialize ParityCheckMatrix class
    ParityMatrix = ParityCheckMatrix(dv, dc, k, n, ffield)

    # Get the Connections for the Variable Nodes
    Harr = ParityMatrix.get_H_arr()

    # Initialize Tanner Graph and establish it's connections using the generated Harr
    graph = TannerGraph(dv, dc, k, n)
    graph.establish_connections(Harr)

    # Create Parity Matrix using generated Harr
    H = ParityMatrix.createHMatrix(Harr)

    H_rref = r.get_reduced_row_echleon_form_H(H)
    H_st, switches = r.switch_columns(H_rref, r.check_standard_form_variance(H_rref))
    
    G = r.standard_H_to_G(H_st, switches=switches)

    # Checking if it is a valid generator matrix
    if np.any(np.dot(G, H.T) % 2):
        print("Matrices not valid - aborting simulation")
        return

    # Get the Code words by encoding an input array (here - generated randomly)
    input_arr = generate_input_arr(k)
    C = get_code_words(input_arr, G, ffield).astype(int)

    # Checking if it is a valid codeword
    if np.any(np.dot(C, H.T) % 2):
        print("Codeword is not valid - aborting simulation")
        return

    # Use Tanner Graph to Decode and get Frame Error Rate Curve for the Channel
    graph.frame_error_rate(input_arr=C, plot=True, establish_connections=False, label="Input")
    graph.frame_error_rate(plot=True, establish_connections=False, label="All zero")
    plt.legend()
    plt.title("Comparision of BEC decoder on random input vs all zero input")
    plt.show()

if __name__ == "__main__":
    dv, dc, k, n = 3, 6, 10, 20
    ffield = 2
    basic_encoding_procedure(dv, dc, k, n, ffield=ffield)
    