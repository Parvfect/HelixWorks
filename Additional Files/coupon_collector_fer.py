


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



# GF(67)
# dv dc 3 6
# k n 50 100
# 8C4


# 4C2 symbols - ignoring (2,3) to make it GF(5)
# Would be also 5: (2,3) which is being ignored

symbols = choose_symbols(8, 4)

# Hard coding symbols to pop for now

for i in range(3):
    symbols.pop(len(symbols)-i-1)
symbol_arr = list(symbols.values())
symbol_keys = list(symbols.keys())

dv, dc, k, n, read_length = 3, 6, 20, 40, 7

input_arr = [0, 1, 0]

# Print simulation parameters
print()
print("Simulation Parameters: ")
print("dv:{} , dc:{} , k:{} , n:{} ".format(dv, dc, k, n))   
print("8C4 symbols - randomly picking 3 to ignore to make it GF(67)")
print("Symbols are: ", symbols)
print("Read Length: ", read_length)
print("\n")

# Initialize the parity check matrix and tanner graphs
PM = ParityCheckMatrix(dv, dc, k, n, ffdim=5)
Harr = PM.get_H_arr()
graph = TannerGraph(dv, dc, k, n, ffdim=5)
graph.establish_connections(Harr)
print("Harr")
print(Harr)
print()
print("Check Node Connections : \n")
print(graph.get_connections())
print()

H = PM.createHMatrix(Harr=Harr)
#H_shuffle = np.array([[random.randint(1,4) if i == 1 else i for i in j] for j in H])
print("H: \n", H)
print()

G = r.parity_to_generator(H, ffdim=5)
print("G: \n", G)
print()

# Check if G and H are valid
if np.any(np.dot(G, H.T) % 5 != 0):
    print("Matrices are not valid, aborting simulation")
    exit()

# Generate a random input array
input_arr = [random.choice(symbol_keys) for i in range(k)]
print("Input Array \n", input_arr)
print()

# Encode the input array
C = np.dot(input_arr, G) % 5

# Check if codeword is valid
if np.any(np.dot(C, H.T) % 5 != 0):
    print("Codeword is not valid, aborting simulation")
    exit()

print("Codeword: \n", C)
print()


reads = simulate_reads(C, read_length, symbol_arr)

print("The Reads are:")
print(reads)
print()

possible_symbols = read_symbols(C, read_length, symbols)
#possible_symbols = get_possible_symbols(reads, symbol_arr)

print("The Symbol Possibilites based on the reads are:")
print(possible_symbols)
print()



# Assigning values to Variable Nodes
graph.assign_values(possible_symbols)

print("Decoded Values are")
#decoded_values = graph.coupon_collector_decoding()
print(decoded_values)

# Check if it is a homogenous array - if not then decoding is unsuccessful
if sum([len(i) for i in decoded_values]) == len(decoded_values):
    if np.all(np.array(decoded_values).T[0] == C):
        print("Decoding successful")
else:
    print("Decoding unsuccessful")
"""

# Gotta store H
# Gotta store G

def frame_error_rate(graph, C, symbols, iterations=10):
    """ Returns the frame error rate curve - for same H, same G, same C"""
    read_lengths = np.arange(7, 20)
    frame_error_rate = []

    for i in tqdm(read_lengths):
        counter = 0
        for j in range(iterations):
            # Assigning values to Variable Nodes after generating erasures in zero array
            graph.assign_values(read_symbols(C, i, symbols))
            decoded_values = graph.coupon_collector_decoding()
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
    
    
    plt.plot(read_lengths, frame_error_rate)
    plt.title("Frame Error Rate for CC for {}-{}  {}-{} for 8C4 Symbols".format(k, n, dv, dc))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    # Displaying final figure
    #plt.legend()
    plt.ylim(0,1)
    plt.show()

    return frame_error_rate



with Profile() as prof:
    print(frame_error_rate(graph, C, symbols, iterations=1))

    (
        Stats(prof)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_stats(10)
    )


# Let us store H and G for a 100 - 200 code and get a figure for that - after optimizing iterations