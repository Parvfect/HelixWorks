
import numpy as np
from graph import TannerGraph
from row_echleon import get_H_arr, get_H_Matrix, parity_to_generator
from itertools import combinations

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """

    symbols = list(combinations(np.arange(1, n_motifs+1), picks))
    return {i:symbols[i] for i in range(len(symbols))}

def save_parameters():
    dv, dc = 3, 9
    n = 168
    k = 112
    ffdim = 67

    Harr = get_H_arr(dc, dv, k, n)
    H = get_H_Matrix(dc, dv, k, n, Harr)
    G = parity_to_generator(H, ffdim=ffdim)

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    np.save("Harr.npy", Harr)
    np.save("H.npy", H)
    np.save("G.npy", G)
    
    print("Saved files - Harr, H, G")
    
def load_parameters():
    """ Reads stored Harr, H, G, Tanner Graph and verifies they are correct """
   
    dv, dc = 3, 9
    n = 168
    k = 112
    ffdim = 67

    Harr = np.load("Harr.npy")
    H = np.load("H.npy")
    G = np.load("G.npy")

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    graph = TannerGraph(dv, dc, k, n, ffdim)
    graph.establish_connections(Harr)

    return Harr, H, G, graph

def create_mask(C, channel_input_symbols):
    """ 
    Creates the mask to convert from FF67 to FF70 by mod 70 addition 
    ----
    Codeword + Mask % 70 = Channel Input
    """
    assert len(C) == len(channel_input_symbols)
    return [(channel_input_symbols[i] - C[i]) % 70 for i in range(len(C))]

def get_symbol_index(symbols, symbol):

    for i in symbols:
        if set(i) == set(symbol):
            return symbols.index(i)
            
def generate_symbol_possibilites(channel_output, symbols, motifs, n_picks):
    """ 
    Generates the Symbol Possibilites with the Motif Combinations in output. Works in FF70
    """
    
    symbol_possibilities = []
    for i in channel_output:

        # Will only work for the Coupon Collector Channel
        motifs_encountered = i
        motifs_not_encountered = set(motifs) - set(motifs_encountered)
        
        read_symbol_possibilities = []

        # For the case of distraction
        if len(motifs_encountered) > n_picks:
            return symbols

        if len(motifs_encountered) == n_picks:
            read_symbol_possibilities = [get_symbol_index(symbols, motifs_encountered)]
        
        else:
            
            remaining_motif_combinations = [set(i) for i in combinations(motifs_not_encountered, n_picks - len(motifs_encountered))]
            
            for i in remaining_motif_combinations:
                possibe_motifs = motifs_encountered.union(i)
                symbols = [set(i) for i in symbols]
                if possibe_motifs in symbols:
                    read_symbol_possibilities.append(get_symbol_index(symbols, motifs_encountered.union(i)))
        
        symbol_possibilities.append(read_symbol_possibilities)
    
    return symbol_possibilities

def invert_mask(mask, masked_symbol_possibilites_ff70):
    """ Inverts the mask and returns unmasked symbol possff70 """
    assert len(mask) == len(masked_symbol_possibilites_ff70) == 168
    return [[(j + mask[i]) % 70 for j in masked_symbol_possibilites_ff70[i]]for i in range(168)]

def filter_symbols(symbol_possibilites_ff70):
    """ Gets rid of all symbols that are greater than FF(67) """
    return [[i for i in j if i < 67]for j in symbol_possibilities]
