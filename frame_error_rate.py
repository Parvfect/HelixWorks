# /frame_error_rate.py - Functions for calculating the FER's

from bec import generate_erasures
from ldpc import parity_matrix_permuter, dumb_decoder
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fer_n(rate=0.5):
    """ Obtaining fer for growing n for fixed rate """
    n = 20
    input_arr = np.zeros(n)
    
    iterations = 500
    n = np.arange(20, 40, 5)   
    erasure_probabilities = np.arange(0,1,0.025)

    for t in tqdm(n):
        H = parity_matrix_permuter(t,2*t)
        frame_error_rate = []
        input_arr = np.zeros(2*t)

        for i in tqdm(erasure_probabilities):

            # Generate erasures in input array
            counter = 0

            for j in range(iterations):
                
                output_arr = generate_erasures(input_arr, i)
                # Get average error rate for 20 runs
                if dumb_decoder(output_arr, H).tolist()[0] == np.zeros(2*t).tolist():
                    counter+=1
            
            error_rate = (iterations - counter)/100
            frame_error_rate.append(error_rate)
        
        plt.plot(erasure_probabilities, frame_error_rate, label="n = " + str(t))
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for varying n")
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    plt.legend()
    plt.show()

def permuation_fer(k=10, n=20):
    """ Get the frame error rate for variations of H for a fixed (k, n) """

    # Initalize input array of zeros
    input_arr = np.zeros(n)
    iterations = 500

    erasure_probabilities = np.arange(0,1,0.025)

    for t in tqdm(range(10)):
        H = parity_matrix_permuter(k,n)
        H = H
        frame_error_rate = []

        for i in tqdm(erasure_probabilities):

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
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for ({},{}) code".format(k,n))
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    plt.xlim(0.2,0.9)
    plt.show()

#fer_n()
permuation_fer()