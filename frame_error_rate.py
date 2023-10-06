# /frame_error_rate.py - Functions for calculating the FER's

"""
Change in curve for varying n
Change in curve for different H matrices
Density Evolution
"""

from bec import generate_erasures
from ldpc import parity_matrix_permuter, dumb_decoder
from Hmatrixbaby import createHMatrix, getH
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from graph import TannerGraph

def fer_n_graph(rate=0.5, iterations=20):
    """ Obtaining fer for growing n for fixed rate """
    
    # Number of Check Nodes
    n = [500]   
    # Erasure Probabilities
    erasure_probabilities = np.arange(0,1,0.05)

    # Iterate through the different sizes of Tanner Graphs
    for t in tqdm(n):

        # Generate Tanner Graph and establish it's connections
        tanner = TannerGraph(3, 6, t, 2*t)
        tanner.establish_connections()
        frame_error_rate = []
        input_arr = np.zeros(2*t)

        for i in tqdm(erasure_probabilities):

            counter = 0
            for j in range(iterations):
                
                # Assigning values to Variable Nodes after generating erasures in zero array
                tanner.assign_values(generate_erasures(input_arr, i))

                # Getting the average error rates for iteration runs
                if tanner.bec_decoder() == input_arr:
                    counter += 1
            
            # Calculate Error Rate and append to list
            error_rate = (iterations - counter)/100
            frame_error_rate.append(error_rate)
        
        # Plotting the FER for each n
        plt.plot(erasure_probabilities, frame_error_rate, label="n = " + str(t))
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for varying n")
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    # Displaying final figure
    plt.legend()
    plt.show()

def fer_n(rate=0.5):
    """ Obtaining fer for growing n for fixed rate """

    iterations = 50
    n = [500]   
    erasure_probabilities = np.arange(0,1,0.025)

    for t in tqdm(n):

        # Create Parity Check Matrice for each n
        H = createHMatrix(3, 6, t,2*t)
        frame_error_rate = []
        input_arr = np.zeros(2*t)

        for i in tqdm(erasure_probabilities):

            # Generate erasures in input array
            counter = 0
            for j in range(iterations):
                
                output_arr = generate_erasures(input_arr, i)
                # Get average error rate for 20 runs
                if (dumb_decoder(output_arr, H) == input_arr).all():
                    counter+=1

            # Calculate Error Rate and append to list
            error_rate = (iterations - counter)/100
            frame_error_rate.append(error_rate)
        
        # Plotting the FER for each n
        plt.plot(erasure_probabilities, frame_error_rate, label="n = " + str(t))
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for varying n")
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    # Displaying final figure
    plt.legend()
    plt.show()


def permuation_fer(k=500, n=1000, iterations=20):
    """ Get the frame error rate for variations of H for a fixed (k, n) """

    # Initalize input array of zeros
    input_arr = np.zeros(n)

    # Arrange ERasure Probabilities
    erasure_probabilities = np.arange(0,1,0.025)

    for t in tqdm(range(20)):
        # Create Parity Check Matrice for each n
        H = createHMatrix(3, 6, 500, 1000)
        frame_error_rate = []

        for i in tqdm(erasure_probabilities):

            # Generate erasures in input array
            counter = 0

            for j in range(iterations):
                
                output_arr = generate_erasures(input_arr, i)
                # Get average error rate for 20 runs
                if (dumb_decoder(output_arr, H) == np.zeros(n)).all():
                    counter+=1
            
            error_rate = (iterations - counter)/100
            frame_error_rate.append(error_rate)
        
        # Plotting the FER for each n
        plt.plot(erasure_probabilities, frame_error_rate)
        plt.title("Frame Error Rate as a Function of Erasure Probabilities for ({},{}) code".format(k,n))
        plt.ylabel("Frame Error Rate")
        plt.xlabel("Erasure Probability")

    plt.show()


def ensemble_average():


#fer_n_graph()
permuation_fer()

"""
Instead of using that old method, just generate all the Tanner Graphs and individually get the FER for them and see where  the bottleneck lies using a profiler
"""