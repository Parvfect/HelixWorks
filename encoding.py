
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

    # Repeating till we get a 1 in the first index that we erase to check decoding
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

    # Checking if it is a valid generator matrix
    if np.any(np.dot(G, H.T) % 2):
        print("Matrices not valid - aborting simulation")
        return

    # Get the Code words by encoding an input array (here - generated randomly)
    input_arr = generate_input_arr(k)
    #input_arr = np.zeros(k)
    C = get_code_words(input_arr, G, ffield).astype(int)

    # Checking if it is a valid codeword
    if np.any(np.dot(C, H.T) % 2):
        print("Codeword is not valid - aborting simulation")
        return

    # Use Tanner Graph to Decode and get Frame Error Rate Curve for the Channel
    graph.frame_error_rate(input_arr=C, plot=True, establish_connections=False, label="Input")
    graph.frame_error_rate(plot=True, establish_connections=False, label="All zero")
    

def testing(dv, dc, k, n):
    """ Simulates complete Channel Simulation for Binary Erasure Channel including encoding """
    

    # Get the Connections for the Variable Nodes
    Harr = np.array([0, 11,  7,  3,  5,  2,  4,  9,  6,  8,  1, 10])
    print(Harr)

    # Initialize Tanner Graph and establish it's connections
    graph = TannerGraph(dv, dc, k, n)
    graph.establish_connections(Harr)

    H = np.array([[1, 1, 1, 0, 0, 1], [0, 1, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1]])

    print(H)

    #H_rref = r.get_reduced_row_echleon_form_H(H)
    #H_st, switches = r.switch_columns(H_rref, r.check_standard_form_variance(H_rref))
    #G = r.standard_H_to_G(H_st, switches=switches)

    G = np.array([[1, 1, 0, 1, 0, 0], [1 , 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0]])
    print(G)
    print(np.dot(G, H.T) % 2)


    if np.any(np.dot(G, H.T) % 2):
        print("Matrices not valid - aborting simulation")
        return

    # The orientation is proving to be annoying - I don't know what I've done but I need G.H^T = 0 to get it right

    # Get the Code words by encoding an input array (here - generated randomly)
    input_arr = generate_input_arr(k)
    C = get_code_words(input_arr, G).astype(int)
    print(C)
    #C = np.array([0,0,1,1,1,0]) 

    if np.any(np.dot(C, H.T) % 2):
            print("Codeword is not valid - aborting simulation")
            return
    
    frame_error_rate  = graph.frame_error_rate(input_arr=C, plot=True, establish_connections=False, label="Input")
    print(frame_error_rate)
    print(graph.Harr)
    

def bec_channel_simulation(dv, dc, k, n, ffield=2):
    """ Simulates complete Channel Simulation for Binary Erasure Channel including encoding """
    
    # Initialize ParityCheckMatrix class
    ParityMatrix = ParityCheckMatrix(dv, dc, k, n, ffield)

    while True:
        # Repeating till we get a 1 in the first index that we erase to check decoding
        # Get the Connections for the Variable Nodes
        Harr = ParityMatrix.get_H_arr()

        # Initialize Tanner Graph and establish it's connections
        graph = TannerGraph(dv, dc, k, n)
        graph.establish_connections(Harr)

        # Create Parity Matrix using generated Harr
        H = ParityMatrix.createHMatrix(Harr)

        print(H)

        H_rref = r.get_reduced_row_echleon_form_H(H)
        H_st, switches = r.switch_columns(H_rref, r.check_standard_form_variance(H_rref))
        
        G = r.standard_H_to_G(H_st, switches=switches)

        # Checking if it is a valid generator matrix
        if np.any(np.dot(G, H.T) % 2):
            print("Matrices not valid - aborting simulation")
            return

        # The orientation is proving to be annoying - I don't know what I've done but I need G.H^T = 0 to get it right

        # Get the Code words by encoding an input array (here - generated randomly)
        input_arr = generate_input_arr(k)
        #input_arr = np.zeros(k)
        C = get_code_words(input_arr, G, ffield).astype(int)
        #print(np.dot(C, H.T % 2))
        # Checking if it is a valid codeword
        print(C)
        
        if np.any(np.dot(C, H.T) % 2):
            print("Codeword is not valid - aborting simulation")
            return
        
        frame_error_rate  = graph.frame_error_rate(input_arr=C, plot=True, establish_connections=False, label="Input")
        print(frame_error_rate)
        graph.frame_error_rate(plot=True, establish_connections=False, label="All zero")
        plt.legend()
        plt.title("Comparision of BEC decoder on random input vs all zero input")
        plt.show()
        break
        

    # Let's try decoding for the simplest case - let's not do fer


if __name__ == "__main__":
    dv, dc, k, n = 3, 6, 100, 200
    ffield = 2
    bec_channel_simulation(dv, dc, k, n, ffield=ffield)
    #testing(2, 4, 3, 6)
    #testing(3, 6, 10, 20)