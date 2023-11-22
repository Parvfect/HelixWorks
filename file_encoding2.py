
import numpy as np
from coupon_collector import choose_symbols, display_parameters, run_singular_decoding, frame_error_rate
from graph import TannerGraph
from load_saved_codes import get_saved_code
import matplotlib.pyplot as plt
import sys
import filecmp
import csv

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
    
    padding_zeros = 852*8 - len(lst)

    [lst.append(0) for i in range(padding_zeros)]
    print(len(lst))
    return lst, b

def pad_input():
    return

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


def create_encoded_file(output_arr):
    header = ["Payload1", "Payload2","Payload3","Payload4","Payload5","Payload6","Payload7","Payload8"]
    with open('foo.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)
        for i in range(len(output_arr)):
            new_line = output_arr[i]
            writer.writerow(new_line)
        

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

symbols.pop()
symbols.pop()
symbols.pop()

symbol_keys = np.arange(0, ffdim)
encoded_combinations = []

for i in input_arrs:
        
    C = get_codeword(i)
    encoded_combinations.append([symbols[i] for i in C])

# Unpacking
encoded_combinations_list = [combination for encoded_combination in encoded_combinations for combination in encoded_combination]

# Add 16 Oligos of random symbols that we don't transmit over

# pad with 16 0 symbols
for i in range(16):
    encoded_combinations_list.append(symbols[0])
encoded_combinations_list = np.array(encoded_combinations_list)
print(encoded_combinations_list.shape)

len_division = 8
# Chop up into 8 input arrays
output_arrs = [encoded_combinations_list[i: i+ len_division] for i in range(0, len(encoded_combinations_list), len_division)]
print(np.array(output_arrs).shape)

create_encoded_file(output_arrs)



# Then we write to encoded.csv





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


