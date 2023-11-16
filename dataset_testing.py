
import numpy as np
from graph import TannerGraph
from row_echleon import get_H_arr, get_H_Matrix, parity_to_generator

def choose_symbols(n_motifs, picks):
    """ Returns Symbol Dictionary given the motifs and the number of picks """

    symbols = list(combinations(np.arange(n_motifs), picks))
    return {i:symbols[i] for i in range(len(symbols))}

def save_parameters():
    dv, dc = 3, 9
    n = 168
    k = 112

    Harr = get_Harr(dv, dc, k, n, ffddim=ffdim)
    H = get_H_Matrix(Harr, ffdim=ffdim)
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
    n-k = 56

    Harr = np.load("Harr.npy")
    H = np.load("H.npy")
    G = np.load("G.npy")

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    graph = TannerGraph(dv, dc, n, k, ffdim)
    graph.establish_connections(Harr)

    return Harr, H, G, graph

def create_mask(C, channel_input_symbols):
    """ 
    Creates the mask to convert from FF67 to FF70 by mod 70 addition 
    ----
    Codeword + Mask % 70 = Channel Input
    """
    assert len(C) == len(channel_input_symbols) == 168
    return [(channel_input_symbols[i] - C[i]) % 70 for i in range(168)]

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

def encoder(channel_input):
    """ 
    Takes in the Channel Input (FF70) and returns the mask, the codeword and the Tanner Graph. Utilises a stored Harr, H and graph which is stored as a file 
    ------
    Input 
    (21,8,4) Motifs
    Output
    params to be fed into the decoder
    (symbols, motifs, channel_input_symbols, mask, Harr, H, G, graph, Codeword)
    """
    
    # What does the Motif Combination Channel Input look like in shape?
    assert len(channel_input) == ()
    dv, dc = 3, 9
    n = 168
    k = 112
    n-k = 56
    ffdim = 67
    n_motifs, n_picks = 8, 4

    symbols = choose_symbols(n_motifs, n_picks)
    motifs = np.arange(1, 9)
    symbol_keys = list(symbols.keys())
    
    # Converting the Motif Combinations as Channel Input into their Symbol Forms
    channel_input_symbols = [symbols.index(i) for i in channel_input]

    symbols.pop(69)
    symbols.pop(68)
    symbols.pop(67)

    Harr, H, G, graph = read_parameters()

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = [random.choice(symbol_keys) for i in range(k)]
    C = np.dot(input_arr, G) % ffdim
    
    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()
    
    mask = create_mask(C, channel_input_symbols)

    return (symbols, motifs, channel_input_symbols, mask, Harr, H, G, graph, C)


def decoder(channel_output, params):
    """ 
    Decoder for Dataset Testing. Takes in the Channel Output and params returned from Encoder and returns True or False based on whether the decoding was successful or not.
    ------
    Input 
    (21,8,motifs_encountered) channel output, params (symbols, motifs, channel_input_symbols, mask, Harr, H, G, graph, C)
    Output
    True or False 
    """

    # Unpack the Parameters from the Encoder
    symbols, motifs, channel_input_symbols, mask, Harr, H, G, graph, C = params

    masked_symbol_possibilities_ff70 = generate_symbol_possibilites(channel_output, symbols, motifs, n_picks=4)

    symbol_possibilites_ff70 = invert_mask(masked_symbol_possibilities_ff70)

    symbol_possibilities = filter_symbols(symbol_possibilites_ff70)

    assert len(symbol_possibilities) == len(self.vns)
    graph.assign_values(symbol_possibilities)

    decoded_values = graph.coupon_collector_decoding()
    
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            return True
    
    return False

