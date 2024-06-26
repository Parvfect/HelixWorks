
import numpy as np
from coupon_collector import choose_symbols, display_parameters, run_singular_decoding, frame_error_rate, simulate_reads
from graph import TannerGraph
import matplotlib.pyplot as plt
import sys
import filecmp
import csv
import random
import pandas as pd
from itertools import combinations

# So we convert a binary file to the number system 
# file of 1280 bits out of which we encode the first 1278 bits

def convert_base_67(filename="input_txt.txt", bits=None):
    """ 
    Converts Image to Base 67 and pads it to the right size. Returns representations at symbol level - 1278 x 8 
    """

    a = 67 ** (852 * 8 - 1)   # Largest Base 67 Number

    c = 2**a.bit_length()
    # The Maximum integer we can get through the bytes
    b = 2**a.bit_length() # I don't understand what b is
    print(f"The number of bits to encode is {b.bit_length()}\n")
    
    with open(filename, "rb") as file_handle:  
        b = int.from_bytes(bytearray(file_handle.read()), byteorder='big')
        print(bytearray(file_handle.read()))
    

    #b = int.from_bytes(bytearray(file_handle.read()), byteorder='big')
    quot = b
    lst = []
    while quot > 0:
        quot, rem = divmod(quot, 67)
        lst.append(rem)
    
    print(len(lst))
    # Pads the zeros according to the leftover size
    padding_zeros = 852*8 - len(lst)
    print(f"The Number of Padding Zeros are {padding_zeros}")
    [lst.append(0) for i in range(padding_zeros)]
    print(len(lst))
    return lst, b, padding_zeros

def read_combinations_from_csv(filename):

    """
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for sequence in spamreader:
            print(sequence)
        
    sys.exit()
    """
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
    encoded_symbols = df.to_numpy()
    assert encoded_symbols.shape == (1280, 8)

    # Converting the motifs to numpy array
    for i in range(encoded_symbols.shape[0]):
        for j in range(encoded_symbols.shape[1]):
            t = encoded_symbols[i,j]
            t = np.fromstring(t.replace("[", "").replace("]", ""), sep=',', dtype=int)
            encoded_symbols[i,j] = t
    
    return encoded_symbols

def convert_dimensions_coding_type(combinations):
    """ 
    Converts from 1280 x 8 x n to
    8 x 1280 x n
    """

    # Unpacking one dimension
    combinations = [num for sublist in combinations for num in sublist]
    
    # Removing the padded zeros
    for i in range(16):
        combinations.pop()

    # Repacking into the required format
    len_division = int(len(combinations)/8)
    combination_arr = [combinations[i:i+len_division] for i in range(0, len(combinations), len_division)]

    return combination_arr

def convert_base_2(base_67_bits):
    b2 = 0
    for pw in range(base_67_bits):
        b2 += 67 ** pw * lst[pw]
    
    byte_array = int.to_bytes(b2)
    return b2

def get_codeword(input_arr, G, ffdim):
    
    # Encode the input array
    C = np.dot(input_arr, G) % ffdim
    return C

def decode(C, G, H, graph, read_length, symbols, motifs, n_picks):

    # Check if codeword is valid
    if np.any(np.dot(C, H.T) % ffdim != 0):
        print("Codeword is not valid, aborting simulation")
        exit()

    #display_parameters(n_motifs, n_picks, dv, dc, k, n, motifs, symbols, Harr, H, G, C, ffdim)
    if np.any(np.dot(G, H.T) % ffdim != 0):
            print("Matrices are not valid, aborting simulation")
            exit()

    decoded_vals = run_singular_decoding(graph, C, read_length, symbols, motifs, n_picks)
    return decoded_vals, combinations

def find_I_columns(G):
    """ Returns column indices corresponding to I after permuting from standard form G to G"""

    G = np.array(G)
    k = G.shape[0]
    I = np.eye(k)
    I_indices = np.zeros(k)
    G_transpose = G.T

    for i in range(k):
        for j in range(len(G_transpose)):
            if (G_transpose[j]==I[i]).all():
                I_indices[i] = j # Column indice
                break
        
    return I_indices

def create_csv_file(output_arr, filename):
    with open(filename, 'w', newline="") as f:
        f.write('"Payload1","Payload2","Payload3","Payload4","Payload5","Payload6","Payload7","Payload8"\n')
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        for i in range(len(output_arr)):
            new_line = [f"{i}" for i in output_arr[i]] 
            writer.writerow(new_line)

def create_mask(rng, mask_length):
    """ 
    Creates the mask to convert from FF67 to FF70 by mod 70 addition to a psuedo random string
    Takes in the rng seed so we can save our masks
    ----
    Codeword + Mask % 70 = Channel Input
    """
    return [rng.integers(0,70) for i in range(mask_length)]

def invert_mask(masked_symbol_possibilites_ff70, mask):
    """ Inverts the mask and returns unmasked symbol possff70 """
    assert len(mask) == len(masked_symbol_possibilites_ff70)
    return [[(j - mask[i]) % 70 for j in masked_symbol_possibilites_ff70[i]]for i in range(len(mask))]

def fix_combinations_shape(combinations, symbols):
    # Unpacking
    encoded_combinations_list = [combination for encoded_combination in combinations for combination in encoded_combination]
    # Add 16 Oligos of random symbols that we don't transmit over
    for i in range(16):
        encoded_combinations_list.append(symbols[0])
    len_division = 8
    # Chop up into 8 input arrays
    output_arrs = [encoded_combinations_list[i: i+ len_division] for i in range(0, len(encoded_combinations_list), len_division)]
    return output_arrs

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
        if i is None or len(i) > n_picks:
            read_symbol_possibilities = symbols
            break

        # Will only work for the Coupon Collector Channel
        motifs_encountered = set(i)
        motifs_not_encountered = set(motifs) - set(motifs_encountered)
        
        read_symbol_possibilities = []

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

def filter_symbols(symbol_possibilites_ff70):
    """ Gets rid of all symbols that are greater than FF(67) """
    filtered_symbols = [[i for i in j if i < 67]for j in symbol_possibilites_ff70]

    # If symbol poss is empty, replace with all the symbols for the decoder
    for i in range(len(filtered_symbols)):
        if len(filtered_symbols[i]) == 0:
            filtered_symbols[i] = list(np.arange(67))

    return filtered_symbols
    