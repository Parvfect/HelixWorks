
import random
import numpy as np
from graph import TannerGraph
from tanner import VariableTannerGraph
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
from protograph_interface import get_Harr
import sys

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]

def coupon_collector_channel(arr, R):
    return [arr[random.randint(0, len(arr) - 1)] for i in range(R)]

def get_symbol_index(symbols, symbol):

    for i in symbols:
        if set(i) == set(symbol):
            return symbols.index(i)

def get_possible_symbols(reads, symbols, motifs, n_picks):
    
    reads = [set(i) for i in reads]
    symbol_possibilities = []
    for i in reads:

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
 
def simulate_reads(C, read_length, symbols):
    """ Simulates the reads from the coupon collector channel """
    return [coupon_collector_channel(symbols[i], read_length) for i in C]

def read_symbols(C, read_length, symbols, motifs, picks):
    return get_possible_symbols(simulate_reads(C, read_length, symbols), symbols, motifs, picks)

def display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C):

    print("The number of motifs are {}".format(n_motifs))
    print("The number of picks are {}".format(n_picks))
    print("The dv is {}".format(dv))
    print("The dc is {}".format(dc))
    print("The k is {}".format(k))
    print("The n is {}".format(n))
    print("GF{}".format(ffdim))
    print("The Motifs are \n{}\n".format(motifs))
    print("The Symbols are \n{}\n".format(symbols))
    print("The Harr is \n{}\n".format(Harr))
    print("The Parity Matrice is \n{}\n".format(H))
    print("The Generator Matrix is \n{}\n".format(G))
    print("The Codeword is \n{}\n".format(C))
    return

def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=True):
    """ Returns the parameters for the simulation """

    motifs = np.arange(1, n_motifs+1)
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)

    Harr = r.get_H_arr(dc, dv, k, n)

    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    H = r.get_H_Matrix(dc, dv, k, n, Harr)
    G = r.parity_to_generator(H, ffdim=ffdim)
    
    assert not np.any(np.dot(G, H.T) % ffdim != 0):
    
    input_arr = [random.choice(symbol_keys) for i in range(k)]
    C = np.dot(input_arr, G) % ffdim
    
    assert not np.any(np.dot(C, H.T) % ffdim != 0):
    
    if display:
        display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C)

    return graph, C, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    motifs = np.arange(1, n_motifs+1)
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)
    graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)

    if Harr is None and H is None and G is None:
        Harr, dc, dv, k, n = get_Harr()   
        graph.establish_connections(Harr)
        H = r.get_H_matrix_sclpdc(dc, dv, k, n, Harr)
        G = r.parity_to_generator(H, ffdim=ffdim)

    assert not np.any(np.dot(G, H.T) % ffdim != 0):

    input_arr = [random.choice(symbol_keys) for i in range(k)]
    C = np.dot(input_arr, G) % ffdim

    assert not np.any(np.dot(C, H.T) % ffdim != 0):

    if display:
        display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C)

    return graph, C, symbols, motifs

def run_singular_decoding(graph, C, read_length, symbols, motifs, n_picks):
    
    reads = simulate_reads(C, read_length, symbols)    
    possible_symbols = read_symbols(C, read_length, symbols, motifs, n_picks)
    graph.assign_values(possible_symbols)
    decoded_values = graph.coupon_collector_decoding()
    
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            print("Decoding successful")
    else:
        print("Decoding unsuccessful")


def frame_error_rate(graph, C, symbols, motifs, n_picks, iterations=10, uncoded=False, bec_decode=False, label=None):
    """ Returns the frame error rate curve - for same H, same G, same C"""
    
    read_lengths = np.arange(2, 8)
    frame_error_rate = []

    for i in tqdm(read_lengths):
        counter = 0
        for j in tqdm(range(iterations)):
            symbols_read = read_symbols(C, i, symbols, motifs, n_picks)
            if not uncoded:
                graph.assign_values(read_symbols(C, i, symbols, motifs, n_picks))
                if bec_decode:
                    decoded_values = graph.coupon_collector_erasure_decoder()
                else:
                    decoded_values = graph.coupon_collector_decoding()
            else:
                decoded_values = symbols_read
            
            if sum([len(i) for i in decoded_values]) == len(decoded_values):
                if np.all(np.array(decoded_values).T[0] == C):
                    counter += 1
        
        error_rate = (iterations - counter)/iterations
        frame_error_rate.append(error_rate)
    
    plt.plot(read_lengths, frame_error_rate, 'o')
    plt.plot(read_lengths, frame_error_rate, label=label)
    plt.title("Frame Error Rate for CC for {}-{}  {}-{} for 8C4 Symbols".format(k, n, dv, dc))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    plt.xlim(1,19)
    plt.ylim(0,1)

    return frame_error_rate

