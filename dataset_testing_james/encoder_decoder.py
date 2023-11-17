
import numpy as np
from dependencies import *

def encoder(channel_input):
    """ 
    Takes in the Channel Input (FF70) and returns the mask, the codeword and the Tanner Graph. Utilises a stored Harr, H and graph which is stored as a file 
    ------
    Input 
    (21,8,4) Motifs
    Output
    params to be fed into the decoder
    (channel_input_symbols, mask, Harr, H, G, graph, Codeword)
    """
    
    # What does the Motif Combination Channel Input look like in shape?
    assert len(channel_input) == 21 and len(channel_input[0]) == 8 and len(channel_input[0][0]) == 4
    dv, dc = 3, 9
    n = 168
    k = 112
    ffdim = 67
    n_motifs, n_picks = 8, 4

    symbols = choose_symbols(n_motifs, n_picks)
    motifs = np.arange(1, 9)
    
    # Converting the Motif Combinations as Channel Input into their Symbol Forms
    channel_input_symbols = [[symbols.index(i) for i in j]for j in channel_input]
    # Flattening the symbol array
    channel_input_symbols = [item for sublist in channel_input_symbols for item in sublist]

    Harr, H, G, graph = load_parameters()

    if np.any(np.dot(G, H.T) % ffdim != 0):
        print("Matrices are not valid, aborting simulation")
        exit()

    input_arr = create_random_input_arr(k)
    C = np.dot(input_arr, G) % ffdim
    
    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()
    
    mask = create_mask(C, channel_input_symbols)

    return (channel_input_symbols, mask, Harr, H, G, graph, C)


def decoder(channel_output, params):
    """ 
    Decoder for Dataset Testing. Takes in the Channel Output and params returned from Encoder and returns True or False based on whether the decoding was successful or not.
    ------
    Input 
    (21,8,motifs_encountered) channel output, params (channel_input_symbols, mask, Harr, H, G, graph, C)
    Output
    True or False 
    """

    # Unpack the Parameters from the Encoder
    channel_input_symbols, mask, Harr, H, G, graph, C = params

    # Flattening one layer down 
    channel_output = [item for sublist in channel_output for item in sublist]

    symbols = choose_symbols(8, 4)
    motifs = np.arange(1, 9)
    masked_symbol_possibilities_ff70 = generate_symbol_possibilites(channel_output, symbols, motifs, n_picks=4)

    symbol_possibilites_ff70 = invert_mask(masked_symbol_possibilities_ff70, mask)

    symbol_possibilities = filter_symbols(symbol_possibilites_ff70)

    assert len(symbol_possibilities) == len(graph.vns)
    graph.assign_values(symbol_possibilities)

    decoded_values = graph.coupon_collector_decoding()
 
    if sum([len(i) for i in decoded_values]) == len(decoded_values):
        if np.all(np.array(decoded_values).T[0] == C):
            return True
    
    return False

