
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from bec import generate_input_arr, generate_erasures
from tqdm import tqdm

def generate_tanner_graph(parity_check_matrix):
    """ Creates Tanner Graph for Parity Check Matrix - needs to be tested"""
    G = nx.Graph()

    rows = parity_check_matrix.shape[0]
    cols = parity_check_matrix.shape[1]

    # For each row add a check node
    for i in range(rows):
        G.add_node(i, bipartite=0)

    # For each column add a variable node
    for i in range(cols):
        G.add_node(i + rows, bipartite=1)

    
    for i in range(rows):
        for j in range(cols):
            if parity_check_matrix[i][j] == 1:
                G.add_edge(i, j+rows, weight=1)
    # Add all the variable nodes and then add check nodes as edge
   
   
    nx.draw(G, with_labels=True)
    plt.show()
    return G


def dumb_decoder(output_arr, H, max_iterations=100):
    """ Implementation of a SPA specific to the Binary Erasure Channel """

    # The column and row order needs some work - not sure if it is right 
    # Using results without complete understanding

    iterations = 0
    H = np.matrix(H)
    output_arr = np.matrix(output_arr)

    # Iterate through the rows of the parity check matrix
    for i in range(H.shape[1]): # -1 because of the transpose
        
        # Intiialise counter and sum
        counter = 0
        sum_row = 0

        for j in range(H.shape[0]):
            
            # If None value in output arr
            if output_arr.item(0,j):
                # If Parity check matrix value is also one
                if H.item(j,i) == 1:
                    counter += 1
            else:
                sum_row = sum_row + output_arr.item(0,j)*H.item(j,i)
        
        if counter == 1:
            # Find the position of erasure and replace it with the sum modulo 2 of the rest
            output_arr[0, np.argwhere(np.isnan(output_arr))] = sum_row % 2
        
        iterations += 1
        
        # If there are no erasures left or max iterations reached return the output array
        if not np.isnan(output_arr.A1).any() or iterations == max_iterations:
            return output_arr

    return output_arr


def test_decoder(H):
    """ Create different permutations of a (3,6) erased code to see where it does not decode """
    assert dumb_decoder(np.matrix([np.nan, 0, 0, 0, 0, np.nan]), H).tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert dumb_decoder(np.matrix([np.nan, 0, 0, 0, 0, 0]), H).tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert dumb_decoder(np.matrix([0, 0, 0, 0, 0, np.nan]), H).tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert dumb_decoder(np.matrix([0, 0, 0, 0, 0, 0]), H).tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert dumb_decoder(np.matrix([np.nan, 0, 0, np.nan, 0, 0]), H).tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print("Tests passed!")


def parity_matrix_permuter(k, n, dv, dc, max_iterations=100000):
    """ Given a specific input and output length, reuturns all the permutations of an H matrix combination 
    k - Input Length
    n - Codeword Length
    Rate = k/n
    """

    arr = np.array(np.concatenate((np.ones(dc), np.zeros(n - dc))))
    arr = random.permutation(arr)
    iterations = 0
    temp_space = np.zeros((n, n-k))

    while True:
        
        counter = 0
        for i in range(0, (n-k)):
            temp_space[:,i] = random.permutation(arr) 
        
        for i in temp_space:
            if not np.count_nonzero(i==1) == dv:
                counter += 1
                break

        if counter == 0:
            break

        iterations += 1

    return temp_space

def test_parity_matrix(H, dv, dc, k, n):
    """ Verify that the parity matrix is valid """

    assert H.shape == (n, n-k)
    assert np.count_nonzero(H) == dv*(n) == dc*(n-k)
    assert np.count_nonzero(H.sum(axis=1) == dv) == n
    assert np.count_nonzero(H.sum(axis=0) == dc) == n-k
    print("Tests passed! Parity Matrix is Valid! \n{}".format(H))
    


H = np.matrix([
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1]
]) 





"""
for i in range(100):
    t = generate_erasures(np.zeros(6), 0.97)
    print(dumb_decoder(t, H))
"""