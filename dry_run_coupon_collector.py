
import numpy as np
from graph import TannerGraph
import random


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


symbols = {0:(0,1), 1:(0,2), 2:(0,3), 3:(1,2), 4:(1,3)}
symbol_arr = [(0,1), (0,2), (0,3), (1,2), (1,3)]
Harr = [11, 0, 3, 8, 5, 2, 6, 1, 4, 10, 7, 9]
g = TannerGraph(2, 4, 3, 6, ffdim=5)
g.establish_connections(Harr)
H = np.array([[1,1,1,1,0,0], [0,0,1,1,1,1], [1,1,0,0,1,1]])
G = np.array([[0,0,0,0,1,4], [4,1,0,0,0,0], [0,0,1,4,0,0]])

if np.any(np.dot(G, H.T) % 5 != 0):
    print("Matrices are not valid, aborting simulation")
    exit()

input_arr = [2, 3, 4]
C = np.dot(input_arr, G) % 5
print("Codeword: \n", C)
print()

if np.any(np.dot(C, H.T) % 5 != 0):
    print("Codeword is not valid, aborting simulation")
    exit()

reads = simulate_reads(C, 5, symbols)
possible_symbols = get_possible_symbols(reads, symbol_arr)

# Hardcoding Possible Symbols 
possible_symbols = [[0, 1, 2], [1, 3], [4], [1], [2], [3]]

# It should decode as [2, 3, 4, 1, 2, 3]

g.assign_values(possible_symbols)

print(g.coupon_collector_decoding())