
import numpy as np
from coupon_collector import choose_symbols, display_parameters, run_singular_decoding, frame_error_rate, simulate_reads
from graph import TannerGraph
from load_saved_codes import get_saved_code
import matplotlib.pyplot as plt
import sys
import filecmp
import csv
import random
import pandas as pd

# So we convert a binary file to the number system 
# file of 1280 bits out of which we encode the first 1278 bits

def convert_base_67(filename="input_txt.txt", bits=None):

    a = 67 ** (852 * 8 - 1)   # Largest Base 67 Number

    c = 2**a.bit_length()
    # The Maximum integer we can get through the bytes
    b = 2**a.bit_length() # I don't understand what b is
    print(f"The number of bits to encode is {b.bit_length()}\n")
    
    with open(filename, "rb") as file_handle:  
        b = int.from_bytes(bytearray(file_handle.read()), byteorder='big')
    

    #b = int.from_bytes(bytearray(file_handle.read()), byteorder='big')
    quot = b
    lst = []
    while quot > 0:
        quot, rem = divmod(quot, 67)
        lst.append(rem)
    
    # Pads the zeros according to the leftover size
    padding_zeros = 852*8 - len(lst)
    print(f"The Number of Padding Zeros are {padding_zeros}")
    [lst.append(0) for i in range(padding_zeros)]
    print(len(lst))
    return lst, b


def convert_base_2(base_67_bits):
    b2 = 0
    for pw in range(base_67_bits):
        b2 += 67 ** pw * lst[pw]
    
    byte_array = int.to_bytes(b2)
    return b2

def get_codeword(input_arr):
    
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
    header = ["“Payload1“", "“Payload2“","“Payload3“","“Payload4“","“Payload5“","“Payload6“","“Payload7“","“Payload8“"]
    with open(filename, 'w', newline="") as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)
        for i in range(len(output_arr)):
            new_line = [f'“{i}“' for i in output_arr[i]] 
            writer.writerow(new_line)

def create_mask(C):
    """ 
    Creates the mask to convert from FF67 to FF70 by mod 70 addition to a psuedo random string
    ----
    Codeword + Mask % 70 = Channel Input
    """
    channel_input_symbols = [random.randint(0,69) for i in range(len(C))]
    return [(channel_input_symbols[i] - C[i]) % 70 for i in range(len(C))], channel_input_symbols

def invert_mask(masked_symbol_possibilites_ff70, mask):
    """ Inverts the mask and returns unmasked symbol possff70 """
    assert len(mask) == len(masked_symbol_possibilites_ff70)
    return [[(j + mask[i]) % 70 for j in masked_symbol_possibilites_ff70[i]]for i in range(len(mask))]

def fix_combinations_shape(combinations):
    # Unpacking
    encoded_combinations_list = [combination for encoded_combination in combinations for combination in encoded_combination]
    # Add 16 Oligos of random symbols that we don't transmit over
    for i in range(16):
        encoded_combinations_list.append(symbols[0])
    len_division = 8
    # Chop up into 8 input arrays
    output_arrs = [encoded_combinations_list[i: i+ len_division] for i in range(0, len(encoded_combinations_list), len_division)]
    return output_arrs

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

input_arr, b = convert_base_67()

len_division = int(len(input_arr)/8)
# Chop up into 8 input arrays
input_arrs = [input_arr[i: i+ len_division] for i in range(0, len(input_arr), len_division)]


# Save the column indices
Harr, H, G = get_saved_code(3,9,852,1278)
column_indices = find_I_columns(G)
input_vals = []
combining_stuff = []
n_motifs, n_picks = 8, 4
dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
read_length = 8
#run_singular_decoding(4)
graph = TannerGraph(dv, dc, k, n, ffdim)
graph.establish_connections(Harr)
motifs = np.arange(1, n_motifs+1)

symbols = choose_symbols(n_motifs, n_picks)
base_70_symbols = choose_symbols(n_motifs, n_picks)

symbols.pop()
symbols.pop()
symbols.pop()

symbol_keys = np.arange(0, ffdim)
encoded_combinations = []
C = []
codewords = []
masks = []
base70_codewords = []

for i in range(len(input_arrs)):
        
    C.append(get_codeword(input_arrs[i]))
    mask, base70_codeword = create_mask(C[i])
    masks.append(mask)
    base70_codewords.append(base70_codeword)
    encoded_combinations.append([base_70_symbols[i] for i in base70_codeword])

output_arrs = fix_combinations_shape(encoded_combinations)
create_csv_file(output_arrs, filename='encoded.csv')

# Now we simulate reads on the base 70 codewords and write to the decoded file
reads = []

for i in base70_codewords:
    read = simulate_reads(i, read_length, base_70_symbols)
    read = [list(set(i)) for i in read]
    reads.append(read)

reads = fix_combinations_shape(reads)
create_csv_file(reads, filename='channel_output.csv')



# Now we read back what we get from the channel
df = pd.read_csv("channel_output.csv", header=0,encoding = "ISO-8859-1")
channel_output = df.to_numpy()
assert channel_output.shape == (1280,8)

# We have to remove the padded zeros, flatten one layer down and rearrange into networks of 8
# We then run the decoder pipeline for each input array and put it together
# Need some better organization



sys.exit()
decoded_values, combinations = run_decoding(i)
recovered_input = [decoded_values[int(j)] for j in column_indices]
combining_stuff.append(combinations)
input_vals.append(recovered_input)




input_vals = np.array(input_vals).flatten()

input_vals = input_vals[:6441]
input_vals = input_vals.astype(int)
b2 = 0
for pw in range(len(input_vals)):
    b2 += int(67**pw) * int(input_vals[pw])

print(b == b2)


byte_array = b2.to_bytes(4884, 'big') 
with open("output.txt", "wb") as file_handle:
    file_handle.write(byte_array)

# Check the files are identical
print(filecmp.cmp('input_txt.txt', 'output.txt'))


