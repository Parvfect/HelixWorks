




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

def coupon_collector_channel(arr, R):
    return [arr[random.randint(0, len(arr) - 1)] for i in range(R)]

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

def simulate_reads(C, read_length, symbols):
    """ Simulates the reads from the coupon collector channel """
    
    reads = []
    # Simulate one read
    for i in C:
        read = coupon_collector_channel(symbols[i], read_length)
        reads.append(read)

    # Make reads a set
    return reads

def read_symbols(C, read_length, symbols):
    symbol_arr = list(symbols.values())
    reads = simulate_reads(C, read_length, symbols)
    return get_possible_symbols(reads, symbol_arr)


def run_singular_decoding(read_length):
    
    reads = simulate_reads(C, read_length, symbol_arr)

    print("The Reads are:")
    print(reads)
    print()

    # Convert to possible symbols
    
    possible_symbols = read_symbols(C, read_length, symbols)
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

    symbols.pop(69)
    symbols.pop(68)
    symbols.pop(67)

    symbol_arr = list(symbols.values())
    symbol_keys = list(symbols.keys())


    Harr = r.get_H_arr(dc, dv, k, n)

    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)
    H = r.get_H_Matrix(dc, dv, k, n, Harr)
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


def frame_error_rate(graph, C, symbols, iterations=10, uncoded=False, bec_decode=False, label=None):
    """ Returns the frame error rate curve - for same H, same G, same C"""
    read_lengths = np.arange(6, 20)
    frame_error_rate = []

    for i in tqdm(read_lengths):
        counter = 0
        for j in range(iterations):
            # Assigning values to Variable Nodes after generating erasures in zero array
            symbols_read = read_symbols(C, i, symbols)
            if not uncoded:
                graph.assign_values(read_symbols(C, i, symbols))
                if bec_decode:
                    decoded_values = graph.coupon_collector_erasure_decoder()
                else:
                    decoded_values = graph.coupon_collector_decoding()
            else:
                decoded_values = symbols_read
            # Getting the average error rates for iteration runs
  
            if sum([len(i) for i in decoded_values]) == len(decoded_values):
                if np.all(np.array(decoded_values).T[0] == C):
                    counter += 1
            """
            # Adaptive Iterator
            if prev_error - ((iterations - counter)/iterations) < 0.001:
                break

            prev_error = (iterations - counter)/iterations
            """

        # Calculate Error Rate and append to list
        error_rate = (iterations - counter)/iterations
        frame_error_rate.append(error_rate)
    
    
    plt.plot(read_lengths, frame_error_rate, 'o')
    plt.plot(read_lengths, frame_error_rate, label=label)
    plt.title("Frame Error Rate for CC for {}-{}  {}-{} for 8C4 Symbols".format(k, n, dv, dc))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    # Displaying final figure
    #plt.legend()
    plt.xlim(6,20)
    plt.ylim(0,1)
    #plt.show()

    return frame_error_rate


with Profile() as prof:
    # GF(67)
    # dv dc 3 6
    # k n 50 100
    # 8C4


    n_motifs, n_picks = 8, 4
    dv, dc, k, n, ffdim = 3, 6, 100, 200, 67
    #run_singular_decoding(4)
    graph, C, symbols = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim)
    
    
    
    graph, C, symbols = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim)
    print(frame_error_rate(graph, C, symbols, iterations=100, label='100-150'))
    print(frame_error_rate(graph, C, symbols, iterations=100, bec_decode=True, label='100-150 bec'))


    

    """
    k, n = 500, 1000
    graph, C, symbols = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim)
    print(frame_error_rate(graph, C, symbols, iterations=100, label='100-200'))
    print(frame_error_rate(graph, C, symbols, iterations=100, bec_decode=True, label='100-200 bec'))
    """
    

    #k, n = 250, 500
    #graph, C, symbols = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim)
    #print(frame_error_rate(graph, C, symbols, iterations=100, label='50-100'))

    plt.xticks(np.arange(6, 20, 1))
    plt.grid()
    plt.legend()
    plt.show()

    (
        Stats(prof)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_stats(10)
    )


# Let us store H and G for a 100 - 200 code and get a figure for that - after optimizing iterations