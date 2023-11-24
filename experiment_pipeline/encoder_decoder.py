
from dependencies import convert_base_67, create_mask, create_csv_file, fix_combinations_shape, get_codeword, read_combinations_from_csv, convert_dimensions_coding_type, generate_symbol_possibilites, filter_symbols, invert_mask, find_I_columns
from coupon_collector import choose_symbols, simulate_reads, run_singular_decoding, coupon_collector_channel
from graph import TannerGraph
import numpy as np
import pandas as pd
import sys
import filecmp
import os

n_motifs, n_picks = 8, 4
dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
read_length = 8
padding_zeros = 217

def encoder():

    # Convert Image to Base 67 after padding
    input_arr, b, padding_zeros = convert_base_67()

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
    base_70_codewords = []
    encoded_combinations = []

    # Initializing seed for the masks
    rng = np.random.default_rng(seed=42)

    for i in range(len(input_arrs)):
        
        base_67_codeword = get_codeword(input_arrs[i], G, ffdim)
        base_67_codewords.append(base_67_codeword)
        mask = create_mask(rng, len(base_67_codeword))
        base_70_codeword = [(base_67_codeword[i] + mask[i]) % 70 for i in range(len(mask))]
        base_70_codewords.append(base_70_codeword)
        encoded_combinations.append([base_70_symbols[i] for i in base_70_codeword])

    # Convert Encoded Combination to right size and Pad for 1278-1280 x 8
    output_arrs = fix_combinations_shape(encoded_combinations, base_70_symbols)
    
    # Write the Encoded Combinations to Encoded.csv
    create_csv_file(output_arrs, filename='encoded.csv')

    # Pass and save Relevant params
    return padding_zeros

def simulate_channel(read_length=8):
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
            read = coupon_collector_channel(j, read_length)
            reads.append(list(set(read)))

    len_division = 8
    read_arr = [reads[i: i+ len_division] for i in range(0, len(reads), len_division)]
    create_csv_file(read_arr, filename='channel_output.csv')

def decoder():
    # Read Channel Output and put it into a numpy array
    output_symbols_unpacked = read_combinations_from_csv("channel_output.csv")

    # Remove the Padded Zeros and repack into coding dimensions
    output_symbols_base_70 = convert_dimensions_coding_type(output_symbols_unpacked)

    Harr = np.load("Harr.npy")
    G = np.load("G.npy")
    graph = TannerGraph(dv,dc,k,n,ffdim)
    graph.establish_connections(Harr)

    codewords_base_67 = []
    rng = np.random.default_rng(seed=42)

    for i in range(len(output_symbols_base_70)):

        # Generate Symbol Possibilities
        symbol_possibilities_ff70 = generate_symbol_possibilites(output_symbols_base_70[i], choose_symbols(n_motifs, n_picks), np.arange(1, 9), 4) 

        # Get the Mask
        mask = create_mask(rng, len(symbol_possibilities_ff70))

        # Invert Mask
        symbol_possibilities_ff67 = invert_mask(symbol_possibilities_ff70, mask)

        # Filter Symbols
        symbols_ff67 = filter_symbols(symbol_possibilities_ff67)

        # Assign Values
        assert len(symbols_ff67) == len(graph.vns)
        graph.assign_values(symbols_ff67)

        # Decode 

        decoded_values = graph.coupon_collector_decoding()
        
        if sum([len(i) for i in decoded_values]) == len(symbols_ff67):
            decoded_vals = np.array(decoded_values).T[0]
        else:
            return False

        # Append to Codeword Array
        codewords_base_67.append(decoded_vals)

    #assert np.all([(codewords_base_67[i] == base_67_codewords[i]).all() for i in range(len(codewords_base_67))])
    #print("Codewords were decoded succesfully")

    # Convert Back to Input Array and Check (Get Unchanged Column Indices from G)
    column_indices = find_I_columns(G)
    recovered_input = np.array([[i[int(j)] for j in column_indices]for i in codewords_base_67]).flatten()
    
    #assert (recovered_input == input_arr).all()
    #print("Input was recovered succesfully")

    # Convert Back to the File and Check
    len_recovered_input = len(recovered_input) - padding_zeros
    recovered_arr = recovered_input[:len_recovered_input]
    
    recovered_vals = recovered_arr.astype(int)
    b2 = 0
    for pw in range(len(recovered_vals)):
        b2 += int(67**pw) * int(recovered_vals[pw])

    # Length parameter is the byte size of the input file
    byte_length = os.path.getsize("input_txt.txt")
    byte_array = b2.to_bytes(byte_length,'big') 

    with open("output.txt", "wb") as file_handle:
        file_handle.write(byte_array)

    # Check the files are identical
    return filecmp.cmp('input_txt.txt', 'output.txt')



