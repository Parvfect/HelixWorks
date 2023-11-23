
from dependencies import convert_base_67, create_mask, create_csv_file, fix_combinations_shape, get_codeword, read_combinations_from_csv, convert_dimensions_coding_type, generate_symbol_possibilites, filter_symbols, invert_mask
from coupon_collector import choose_symbols, simulate_reads, run_singular_decoding, coupon_collector_channel
from graph import TannerGraph
import numpy as np
import pandas as pd
import sys

n_motifs, n_picks = 8, 4
dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
read_length = 8

def encoder():

    # Convert Image to Base 67 after padding
    input_arr, b = convert_base_67()

    # Load Matrices and verify they are correct
    G = np.load("G.npy")
    H = np.load("H.npy")
    assert not np.any(np.matmul(G, H.T) % ffdim != 0)

    # Choose Symbols for base 67 and 70
    base_67_symbols = choose_symbols(n_motifs, n_picks)
    base_70_symbols = choose_symbols(n_motifs, n_picks)
    base_67_symbol_keys = np.arange(0, ffdim)
    base_70_symbol_keys = np.arange(0, 70)

    # Pop Three symbols for Base 67
    base_67_symbols.pop()
    base_67_symbols.pop()
    base_67_symbols.pop()

    # Split Input Array into 8 segments
    len_division = int(len(input_arr)/8)
    input_arrs = [input_arr[i: i+ len_division] for i in range(0, len(input_arr), len_division)]

    # Get codeword, mask, channel input, encoded combination for each input array
    base_67_codewords = []
    masks = []
    base70_codewords = []
    encoded_combinations = []

    for i in range(len(input_arrs)):
            
        base_67_codewords.append(get_codeword(input_arrs[i], G, ffdim))
        mask, base70_codeword = create_mask(base_67_codewords[i])
        masks.append(mask)
        base70_codewords.append(base70_codeword)
        encoded_combinations.append([base_70_symbols[i] for i in base70_codeword])

    # Convert Encoded Combination to right size and Pad for 1278-1280 x 8
    output_arrs = fix_combinations_shape(encoded_combinations, base_70_symbols)
    
    # Write the Encoded Combinations to Encoded.csv
    create_csv_file(output_arrs, filename='encoded.csv')

    # Pass and save Relevant params
    return base_67_codewords, masks, base70_codewords

def simulate_channel():
    """ 
    Simulates the Channel - reads from encoded.csv - passes it through the channel and writes the result to channel_output.csv
    """
    encoded_symbols = read_combinations_from_csv("encoded.csv")

    # Base 70 Symbols - might want to store this instead
    base_70_symbols = choose_symbols(n_motifs, n_picks)

    reads = []
    # Simulate Reads and convert to set and write to Channel Output.csv
    for i in encoded_symbols:
        for j in i:
            reads.append(coupon_collector_channel(j, read_length))

    # Write the reads to the csv
    create_csv_file(reads, filename='channel_output.csv')

def decoder(base_67_codewords, masks, base70_codewords):
    # Read Channel Output and put it into a numpy array
    output_symbols_unpacked = read_combinations_from_csv("encoded.csv")

    # Remove the Padded Zeros and repack into coding dimensions
    output_symbols_base_70 = convert_dimensions_coding_type(output_symbols_unpacked)

    Harr = np.load("Harr.npy")
    graph = TannerGraph(dv,dc,k,n,ffdim)

    codewords_base_67 = []

    for i in range(len(output_symbols_base_70)):

        # Generate Symbol Possibilities
        symbol_possibilities_ff70 = generate_symbol_possibilites(output_symbols_base_70[i], choose_symbols(n_motifs, n_picks), np.arange(1, 9), 4) 

        # Invert Mask
        symbol_possibilities_ff67 = invert_mask(symbol_possibilities_ff70, masks[i])

        # Filter Symbols
        symbols_ff67 = filter_symbols(symbol_possibilities_ff67)

        # Assign Values
        assert len(symbols_ff67) == len(graph.vns)
        graph.assign_values(symbols_ff67)

        # Decode 
        decoded_vals = np.array(graph.coupon_collector_decoding()).T[0]

        # Append to Codeword Array
        codewords_base_67.append(decoded_vals)

    # Check if Decoded Values are the Same as the Base 67 Codeword
    assert codewords_base_67 == base_67_codewords

    # Convert Back to Input Array and Check (Get Unchanged Column Indices from G)

    # Convert Back to the File and Check

    pass

base_67_codewords, masks, base70_codewords = encoder()
simulate_channel()
decoder(base_67_codewords, masks, base70_codewords)