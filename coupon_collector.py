
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
from protograph_interface import get_Harr_sc_ldpc, get_dv_dc
import sys
from load_saved_codes import get_saved_code

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """

    # Reference Motif Address starts from 1 not 0
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
            
            # The symbol possibilites are the motifs that are encountered in combination with the motifs that are not encountered

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
    
    reads = []
    # Simulate one read
    for i in C:
        read = coupon_collector_channel(symbols[i], read_length)
        reads.append(read)

    # Make reads a set
    return reads

def read_symbols(C, read_length, symbols, motifs, picks):
    reads = simulate_reads(C, read_length, symbols)
    return get_possible_symbols(reads, symbols, motifs, picks)


def display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C, ffdim):

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

def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)

    graph = TannerGraph(dv, dc, k, n, ffdim=ffdim)

    if Harr is None:
        Harr = r.get_H_arr(dc, dv, k, n)
        H = r.get_H_Matrix(dc, dv, k, n, Harr)
        G = r.parity_to_generator(H, ffdim=ffdim)

    graph.establish_connections(Harr)


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

    if display:
        display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C, ffdim)

    return graph, C, symbols, motifs

def get_parameters_sc_ldpc(n_motifs, n_picks, dv, dc, k, n, ffdim, display=True, Harr=None, H=None, G=None):
    """ Returns the parameters for the simulation """

    # Starting adresses from 1
    motifs = np.arange(1, n_motifs+1)
    
    symbols = choose_symbols(n_motifs, n_picks)
    
    symbols.pop(-1)
    symbols.pop(-2)
    symbols.pop(-3)
    
    symbol_keys = np.arange(0, ffdim)
    
    if Harr is None:
        Harr, dv, dc, k, n = get_Harr_sc_ldpc(dv, dc, k, n)   
    else:
        dv, dc = get_dv_dc(dv, dc, k, n, Harr)
    
    graph = VariableTannerGraph(dv, dc, k, n, ffdim=ffdim)
    graph.establish_connections(Harr)

    if H is None and G is None:
        H = r.get_H_matrix_sclpdc(dc, dv, k, n, Harr)
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

    if display:
        display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C)

    return graph, C, symbols, motifs



def run_singular_decoding(graph, C, read_length, symbols, motifs, n_picks):
    
    reads = simulate_reads(C, read_length, symbols)

    # Convert to possible symbols
    possible_symbols = read_symbols(C, read_length, symbols, motifs, n_picks)
    #possible_symbols = get_possible_symbols(reads, symbol_arr)

    # Assigning values to Variable Nodes
    graph.assign_values(possible_symbols)

    decoded_values = graph.coupon_collector_decoding()
 
    # Check if it is a homogenous array - if not then decoding is unsuccessful
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            print("Decoding successful")
            return np.array(decoded_values).T[0]
    else:
        print("Decoding unsuccessful")
        return None


def frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, iterations=50, uncoded=False, bec_decode=False, label=None, code_class=""):
    """ Returns the frame error rate curve - for same H, same G, same C"""
    read_lengths = np.arange(2, 12)
    frame_error_rate = []

    for i in tqdm(read_lengths):
        counter = 0
        for j in tqdm(range(iterations)):
            # Assigning values to Variable Nodes after generating erasures in zero array
            symbols_read = read_symbols(C, i, symbols, motifs, n_picks)
            if not uncoded:
                graph.assign_values(read_symbols(C, i, symbols, motifs, n_picks))
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

        error_rate = (iterations - counter)/iterations
        frame_error_rate.append(error_rate)
    
    
    plt.plot(read_lengths, frame_error_rate, 'o')
    plt.plot(read_lengths, frame_error_rate, label=label)
    plt.title("Frame Error Rate for CC for {}{}-{}  {}-{} for 8C4 Symbols".format(code_class, k, n, dv, dc))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    # Displaying final figure
    plt.xlim(1,19)
    plt.ylim(0,1)

    return frame_error_rate

def run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, code_class="", iterations=50, bec_decoder=False, uncoded=False, saved_code=False, singular_decoding=True):

    Harr, H, G = None, None, None

    if saved_code:
        Harr, H, G = get_saved_code(dv, dc, k, n, L, M, code_class=code_class)
    
    if code_class == "sc_":
        graph, C, symbols, motifs = get_parameters_sc_ldpc(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, Harr=Harr, H=H, G=G)
    else:
        graph, C, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, Harr =Harr, H=H, G=G)
    
    if singular_decoding:
        run_singular_decoding(graph, C, 8, symbols, motifs, n_picks)
    
    print(frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, iterations=iterations, label=f'CC Decoder', code_class=code_class))
    
    if bec_decoder:
        print(frame_error_rate(graph, C, symbols, motifs, n_picks, iterations=100, bec_decode=True, label='BEC Decoder'))
    
    if uncoded:
        print(frame_error_rate(graph, C, symbols, motifs, n_picks, iterations=100, uncoded=True, label='Uncoded'))
    
    plt.xticks(np.arange(1, 19, 1))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    with Profile() as prof:
        n_motifs, n_picks = 8, 4
        dv, dc, ffdim = 3, 9, 67
        k, n = 612, 1020
        L, M = 10, 102
        read_length = 6
        run_fer(n_motifs, n_picks, dv, dc, k, n, L, M, ffdim, code_class="sc_", saved_code=True)
    (
        Stats(prof)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_stats(10)
    )

    