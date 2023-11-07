


import random
import numpy as np
from graph import TannerGraph
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r
from scipy.linalg import null_space
import sympy as sympy
from itertools import combinations

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
   

# GF(67)
# dv dc 3 6
# k n 50 100
# 8C4


# 4C2 symbols - ignoring (2,3) to make it GF(5)
# Would be also 5: (2,3) which is being ignored

symbols = choose_symbols(8, 4)

for i in range(8c4 - closest_prime_number):
    some_dict.pop( random.choice(some_dict.keys()) ) 

symbols.pop()
symbol_arr = symbols.values()

dv, dc, k, n, read_length = 3, 6, 100, 200, 6

input_arr = [0, 1, 0]

# Print simulation parameters
print()
print("Simulation Parameters: ")
print("dv:{} , dc:{} , k:{} , n:{} ".format(dv, dc, k, n))   
print("4C2 symbols - ignoring (2,3) to make it GF(5)")
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
input_arr = [np.random.randint(0,5) for i in range(k)]
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

read_length = 5

reads = []
# Simulate one read
for i in C:
    read = coupon_collector_channel(symbols[i], read_length)
    reads.append(read)

# Make reads a set
reads = [set(i) for i in reads]
print("The Reads are:")
print(reads)
print()

# Convert to possible symbols
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

print("The Symbol Possibilites based on the reads are:")
print(possible_symbols)
print()


# Assigning values to Variable Nodes
graph.assign_values(possible_symbols)

print("Decoded Values are")
decoded_values = np.array(graph.coupon_collector_decoding().T[0])
print(decoded_values)

if np.all(decoded_values == C):
    print("Decoding successful")
