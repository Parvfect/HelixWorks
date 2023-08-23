# /frame_error_rate.py - Functions for calculating the FER's

from bec import generate_erasures
from ldpc import parity_matrix_permuter, dumb_decoder
import numpy as np
import matplotlib.pyplot as plt

def fer_n(rate=0.5):
    """ Obtaining fer for growing n for fixed rate """
    n = 20
    input_arr = np.zeros(n)
    output_arr = generate_erasures(input_arr, 0.2)
    H = parity_matrix_permuter(20,40)
    print(dumb_decoder(output_arr, H))


def permuation_fer(k=3, n=6):
    """ Get the frame error rate for variations of H for a fixed (k, n) """

    # Initalize input array of zeros
    input_arr = np.zeros(n)
    iterations = 50000

    erasure_probabilities = np.arange(0,1,0.025)

    for t in range(1):
        H = parity_matrix_permuter(k,n)
        frame_error_rate = []

        for i in erasure_probabilities:

            # Generate erasures in input array
            counter = 0

            for j in range(iterations):
                
                output_arr = generate_erasures(input_arr, i)
                # Get average error rate for 20 runs
                if dumb_decoder(output_arr, H).tolist()[0] == np.zeros(n).tolist():
                    counter+=1
            
            error_rate = (iterations - counter)/100
            frame_error_rate.append(error_rate)
        
        plt.plot(erasure_probabilities, frame_error_rate)
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for (3,6) code")
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    plt.xlim(0.2, 0.8)
    plt.show()

fer_n()