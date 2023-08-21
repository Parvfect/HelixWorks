
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from bec import generate_input_arr, generate_erasures


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

def decode_matrix(pcm, max_iterations):
    """ 
    Uses SPA for the Binary Erasure Channel to decode the output 
    Returns False if Decoding unsuccesful (may be smarter to just return the sub optimal decoded)
    """

    copy_matrix = pcm.clone()
    flag = False

    while iterations < max_iterations: # There should be an and condition for theoretics 

        # Row Decoding
        for i in pcm:
            if i.count_nans() == 1: # Should be generalised based on larger cases
                erased_bit_location = find_erased_bit(i)
                pcm[i][j] = get_sum(i) % 2
        
        # Column Decoding
        for j in pcm:
            if i.count_nans() == 1: # Should be generalised based on larger cases
                erased_bit_location = find_erased_bit(i)
                pcm[i][j] = get_sum(i) % 2
        
        if copy_matrix.nans() == 0:
            return copy_matrix
        
    return False

def create_parity_check_matrix(i,j):
    """ Using BEC create redundnacy code using erasure probability to generate partiy check matrix"""
    pass



H = np.matrix([
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1]
]) 

# Input array is a zero vector
input_arr = np.zeros(6)

def dumb_decoder(output_arr, H, max_iterations=100):
    """ Implementation of a SPA specific to the Binary Erasure Channel """

    iterations = 0
    H = np.matrix(H).T
    output_arr = np.matrix(output_arr)

    # Iterate through the rows of the parity check matrix
    for i in range(H.shape[1]): # -1 because of the transpose
        
        # Intiialise counter and sum
        counter = 0
        sum_row = 0

        for j in range(H.shape[0]):
            
            if not output_arr.item(0,j):
                if H.item(j,i) == 1:
                    counter += 1
            else:
                sum_row = sum_row + output_arr.item(0,j)*H.item(j,i)
        
        if counter == 1:
            # Find the position of erasure and replace it with the sum modulo 2 of the rest
            output_arr[np.argwhere(np.isnan(output_arr))] = sum_row % 2
        
        iterations += 1
        
        # If there are no erasures left or max iterations reached return the output array
        if not np.isnan(output_arr.A1).any() or iterations == max_iterations:
            return output_arr

    return False

print(dumb_decoder(np.matrix([0, 0, 0, 0, 0, np.nan]), H))