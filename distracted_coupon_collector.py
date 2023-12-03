
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

def get_symbol_index(symbols, symbol):
    """ Get Symbol index for each symbol """
    for i in symbols:
        if set(i) == set(symbol):
            return symbols.index(i)

def display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C, ffdim):
    """Displays Parameters of Simulation """

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

def choose_symbols(n_motifs, picks):
    """Creates Symbol Array as a combination of Motifs
    
    Args: 
        n_motifs (int): Total Number of Motifs
        picks (int): Number of Motifs per Symbol
    Returns: 
        symbols (list): List of all the Symbols as motif combinations
    """

    # Reference Motif Address starts from 1 not 0
    return [list(i) for i in (combinations(np.arange(1, n_motifs+1), picks))]


def distracted_coupon_collector_channel(symbol, R, P, n_motifs):
    """Model of the Distracted Coupon Collector Channel. Flips a coin, if the probability is within interference, randomly attach a motif from the set of all motifs. Otherwise randomly select from the set of motifs for the passed symbol
    
    Args: 
        symbol (list) : List of motifs as a symbol
        R (int): Read Length
        P (float): Probability of Interference 
        n_motifs (int): Number of motifs in Total
    
    Returns: 
        reads (list) : List of Reads for the Symbol
    """

    reads = []
    for i in range(R):
        if random.random() > P:
            reads.append(random.choice(symbol))
        else:
            reads.append(random.randint(1, n_motifs))    
    return reads

def simulate_reads(C, read_length, symbols, P):
    """Simulates the reads from the coupon collector channel 
   
    Args:
        C (list): Length N list Codeword of Symbols
        read_length (int): Read Length for the Simulated Read
        symbols(list): List of Symbols (Motif Combinations)
        P (Float): Probability of Interference
    
    Returns:
        reads (list) : [length of Codeword, read_length] list of all the reads
    """
    reads = []
    # Simulate one read
    for i in C:
        read = distracted_coupon_collector_channel(symbols[i], read_length, P, 70)
        reads.append(read)

    return reads

def get_possible_symbols(reads, symbols, motifs, n_picks):
    """Given the reads, generates Symbol Possiblities by utilising the motifs encountered and comparing it to the remaining motifs. If Intersection takes place, and the motifs encountered are greater than the n_picks, returns all the symbols as a possibility. 

    Args: 
        reads (list): [length of Codeword, read_length] list of all the reads
        symbols (list): List of all the symbols as motif combinations
        motifs (list): The list of all the motifs
        n_picks (int): Number of Picks from the Total Motifs 
    
    Returns: 
        symbol_possibilities (list): [length of codeword, x no of. possible symbols] list of all the symbol possibilites for the codeword
    """
    
    symbol_possibilities = []
    for i in reads:

        motifs_encountered = set(i)
        motifs_not_encountered = set(motifs) - set(motifs_encountered)
        
        read_symbol_possibilities = []

        if len(motifs_encountered) > n_picks:
            read_symbol_possibilities = list(np.arange(0,67)) # Should be ffdim not 67
        
        elif len(motifs_encountered) == n_picks:
            read_symbol_possibilities = get_symbol_index(symbols, motifs_encountered) 
            
            # In case interference causes an Illegal Symbol
            if read_symbol_possibilities is None:
                read_symbol_possibilities = list(np.arange(0,67))
            else:
                read_symbol_possibilities = [read_symbol_possibilities]
        else:
            remaining_motif_combinations = [set(i) for i in combinations(motifs_not_encountered, n_picks - len(motifs_encountered))]
            
            for i in remaining_motif_combinations:
                possibe_motifs = motifs_encountered.union(i)
                symbols = [set(i) for i in symbols]
                if possibe_motifs in symbols:
                    read_symbol_possibilities.append(get_symbol_index(symbols, motifs_encountered.union(i)))

        symbol_possibilities.append(read_symbol_possibilities)
    
    return symbol_possibilities

def read_symbols(C, read_length, symbols, motifs, picks, P):
    """Mainframe for Passing Codeword through Channel, Simulating Reads and then generating Symbol Possibilities. Could be generalised by then passing channel as well 

    Args:
        C (list) : Length N Codeword of Symbols
        read_length (int): Read length
        symbols (list): List of all the Symbols as Motif Combinations
        motifs (list): List of all the motifs
        picks (int): Number of Motifs per Symbol
        P (float): Probability of Interference
    Returns: 
        symbol_possibilities (list): [length of codeword, x no of. possible symbols] list of all the symbol possibilites for the codeword
    """

    reads = simulate_reads(C, read_length, symbols, P)
    return get_possible_symbols(reads, symbols, motifs, picks)


def get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim,  Harr=None, H=None, G=None, display=True,):
    """Returns the parameters required for a simulation
    
    Args: 
        n_motifs (int): Total number of Motifs
        n_picks (int): Number of Motifs Per Symbol
        dv (int): Number of Connections Per Variable Node
        dc (int): Number of Connections Per Check Node
        k (int): Length of Input
        n (int): Length of Codeword
        ffdim (int): Finite Field Dimension (Prime Number)
    
    Optional Args:
        Harr (array): (n-k)*n Array of Variable Node Connections
        H (array): (n-k, n) Parity Check Matrix
        G (array): (k, n) Generator Matrix
        display (boolean): Display the Simulation Parameters
    
    Returns: 
        graph (TannerGraph): Connected Tanner Graph
        C (list): List of all the Codewords
        symbols (list): List of all the Symbols as a combination of Motifs
        motifs (list): List of all the motifs
    """

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

def get_parameters_sc_ldpc(n_motifs, n_picks, dv, dc, k, n, ffdim, Harr=None, H=None, G=None, display=True):
    """Returns the parameters required for a simulation
    
    Args: 
        n_motifs (int): Total number of Motifs
        n_picks (int): Number of Motifs Per Symbol
        dv (int): Number of Connections Per Variable Node
        dc (int): Number of Connections Per Check Node
        k (int): Length of Input
        n (int): Length of Codeword
        ffdim (int): Finite Field Dimension (Prime Number)
    
    Optional Args:
        Harr (array): (n-k)*n Array of Variable Node Connections
        H (array): (n-k, n) Parity Check Matrix
        G (array): (k, n) Generator Matrix
        display (boolean): Display the Simulation Parameters
    
    Returns: 
        graph (TannerGraph): Connected Tanner Graph
        C (list): List of all the Codewords
        symbols (list): List of all the Symbols as a combination of Motifs
        motifs (list): List of all the motifs
    """

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


def run_singular_decoding(graph, C, read_length, symbols, motifs, n_picks, P):
    """Runs a Singular Decoding on a Connected Tanner Graph for a Simulation with a specific read length

    Args:
        graph (TannerGraph): Connected Tanner Graph
        C (list): N Length Symbol Codeword
        read_length (int): Read Length Per Symbol
        symbols (list): List of all the Symbols as Motif Combinations
        motifs (list): List of all the Motifs
        n_picks (int): Number of Motifs Per Symbol
        P (float): Probability of Interference
    
    Returns: 
        Boolean Variable as for result of Decoding Operation
    """
    

    possible_symbols = read_symbols(C, read_length, symbols, motifs, n_picks, P)
    
    #possible_symbols = get_possible_symbols(reads, symbol_arr)

    # Assigning values to Variable Nodes
    graph.assign_values(possible_symbols)

    decoded_values = graph.coupon_collector_decoding()
 
    # Check if it is a homogenous array - if not then decoding is unsuccessful
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            print("Decoding successful")
            return True
    else:
        print("Decoding unsuccessful")
        return False


def frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, P, iterations=50, read_lengths=np.arange(2,30), uncoded=False, bec_decode=False, label=None, code_class="", ):
    """Calculates, Plots and Returns the Frame Error Rate for a given Tanner Graph over the Read lengths
    
    Args:
        k (int): Input Length
        n (int): Codeword Length
        dv (int): Number of Variable Node Connections
        dc (int): Number of Check Node Connections
        graph (TannerGraph): Connected Tanner Graph
        C (list): N Length Codeword of Symbols
        symbols (list): List of all the Symbols as Motif Combinations
        motifs(list): List of all the Motifs 
        n_picks(int): Number of Motifs Per Symbol
        P (float): Probability of Interference
    
    Optional Args:
        iterations (int): Number of Iterations Per Read Length [50]
        read_lengths (list): List of all the Read Lengths [2,12]
        uncoded (Boolean): Run without encoding [False]
        bec_decode (Boolean): Run using the Binary Erasure Decoder [False]
        label (str): Label for the Graphs [None]
        code_class (str): Specifier for sc codes [""]
    
    Returns:
        frame_error_rate (list): List of FERs for the different read lengths
    """
    frame_error_rate = []

    for i in tqdm(read_lengths):
        counter = 0
        for j in tqdm(range(iterations)):
            # Assigning values to Variable Nodes after generating erasures in zero array
            symbols_read = read_symbols(C, i, symbols, motifs, n_picks, P)
            if not uncoded:
                graph.assign_values(read_symbols(C, i, symbols, motifs, n_picks, P))
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
    plt.title("Frame Error Rate for DCC for {}{}-{}  {}-{} for P={}".format(code_class, k, n, dv, dc, P))
    plt.ylabel("Frame Error Rate")
    plt.xlabel("Read Length")

    # Displaying final figure
    plt.xlim(1,19)
    plt.ylim(0,1)

    return frame_error_rate

def run_simulation(n_motifs, n_picks, k, n, dv, dc, ffdim, P, L=0, M=0, code_class="", iterations=10, uncoded=False, bec_decode=False, saved_code=False, singular_decoding=True):
    """Runs the Simulation - plotting FER for a given set of parameters

    Args:
        n_motifs (int): Total number of Motifs
        n_picks (int): Number of Motifs per Symbol
        k (int): Input Length
        n (int): Codeword Length
        dv (int): Number of Variable Node Connections
        dc (int): Number of Check Node Connections
        ffdim (int): Finite Field Dimension (Prime Number)
        P (float): Probability of Interference
    
    Optional Args:
        L (int): sc-ldpc parameter [0]
        M (int): sc-ldpc parameter [0]
        code_class (str): Specifier for sc-codes [""]
        iterations (int): Number of Iterations Per Read Length [50]
        read_lengths (list): List of all the Read Lengths [2,12]
        uncoded (Boolean): Run without encoding [False]
        bec_decode (Boolean): Run using the Binary Erasure Decoder [False]
        saved_code (Boolean): Load a Saved Code instead of Generating (must exist in database)
        singular_decoding (Boolean): Run a Singular Decoding Operation before FER Simulation
    """

    Harr, H, G = None, None, None

    if saved_code:
        Harr, H, G = get_saved_code(dv, dc, k, n, L, M, code_class=code_class)
    
    if code_class == "sc_":
        graph, C, symbols, motifs = get_parameters_sc_ldpc(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, Harr=Harr, H=H, G=G)
    else:
        graph, C, symbols, motifs = get_parameters(n_motifs, n_picks, dv, dc, k, n, ffdim, display=False, Harr =Harr, H=H, G=G)
    
    if singular_decoding:
        run_singular_decoding(graph, C, 8, symbols, motifs, n_picks, P)
    
    print(frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, P, iterations=iterations, label=f'CC Decoder', code_class=code_class))
    
    if bec_decode:
        print(frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, P, iterations=iterations, label='BEC Decoder', code_class=code_class, bec_decode=True))
    
    if uncoded:
        print(frame_error_rate(k, n, dv, dc, graph, C, symbols, motifs, n_picks, P, iterations=iterations, label='Uncoded', code_class=code_class, uncoded=True))
    
    plt.xticks(np.arange(1, 19, 1))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    n_motifs, n_picks = 8, 4
    dv, dc, ffdim = 3, 9, 67
    k, n = 852, 1278
    L, M = 0, 0
    read_length = 6
    P = 0.01
    run_simulation(n_motifs, n_picks, k, n, dv, dc, ffdim, P, saved_code=True)