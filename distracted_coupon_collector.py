
import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r
from scipy.linalg import null_space
import sympy as sympy
from itertools import combinations
from pstats import Stats
import re
from cProfile import Profile
from tqdm import tqdm
import matplotlib.pyplot as plt

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """

    symbols = list(combinations(np.arange(n_motifs), picks))
    return {i:symbols[i] for i in range(len(symbols))}

def distracted_coupon_collector_channel(arr, R, P, n_motifs):
    
    reads = []
    for i in range(R):
        if random.random() > P:
            reads.append(arr[random.randint(0, len(arr) - 1)])
        else:
            reads.append(random.randint(0, n_motifs - 1))    
    return reads

def simulate_reads(C, read_length, symbols, P, n_motifs):
    """ Simulates the reads from the coupon collector channel """
    
    reads = []
    # Simulate one read
    for i in C:
        read = distracted_coupon_collector_channel(symbols[i], read_length, P, n_motifs)
        reads.append(read)

    # Make reads a set
    return reads


def get_possible_symbols(reads, symbol_arr):
     # Convert to possible symbols

    reads = [set(i) for i in reads]
    possible_symbols = []

    for i in reads:
        read_poss = []
        if tuple(i) in symbol_arr:
            read_poss.append(symbol_arr.index(tuple(i)))
            possible_symbols.append(read_poss)
        else: 
            # Get closest matches
            for j in symbol_arr:
                if list(i)[0] in j:
                    read_poss.append(symbol_arr.index(j))
            possible_symbols.append(read_poss)
    
    return possible_symbols


def read_symbols(C, read_length, symbols, P, n_motifs):
    symbol_arr = list(symbols.values())
    reads = simulate_reads(C, read_length, symbols, P, n_motifs)
    return get_possible_symbols(reads, symbol_arr)


def run_singular_decoding(symbols, read_length, P, n_motifs):
    
    symbol_arr = list(symbols.values())
    reads = simulate_reads(C, read_length, symbol_arr, P, n_motifs)

    print("The Reads are:")
    print(reads)
    print()

    # Convert to possible symbols
    
    possible_symbols = read_symbols(C, read_length, symbols, P, n_motifs)
    #possible_symbols = get_possible_symbols(reads, symbol_arr)

    print("The Symbol Possibilites based on the reads are:")
    print(possible_symbols)
    print()

    # Assigning values to Variable Nodes
    graph.assign_values(possible_symbols)

    print("Decoded Values are")
    decoded_values = graph.coupon_collector_decoding()
    print(decoded_values)

    # Check if it is a homogenous array - if not then decoding is unsuccessful
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            print("Decoding successful")
    else:
        print("Decoding unsuccessful")


def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim):
    
    symbols = choose_symbols(n_motifs, n_picks)
    symbol_keys = list(symbols.keys())

    symbols.pop(69)
    symbols.pop(68)
    symbols.pop(67)

    PM = ParityCheckMatrix(dv, dc, k, n, ffdim=ffdim)
    Harr = PM.get_H_arr()

    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)
    H = PM.createHMatrix(Harr=Harr)
    G = r.parity_to_generator(H, ffdim=ffdim)


    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]

    # Encode the input array
    C = np.dot(input_arr, G) % ffdim

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    return graph, C, symbols


if __name__ == "__main__":
    
    n_motifs, n_picks = 8, 4
    dv, dc, k, n, ffdim = 3, 6, 50, 100, 67
    #run_singular_decoding(4)
    graph, C, symbols = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim)
    run_singular_decoding(symbols, 5, 0.1, n_motifs)

