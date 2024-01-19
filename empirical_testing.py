

# Reading from files empirical_encoded and empirical_channel_output, first for the 99.74 decoded consensus. 
# Attempting to decode. There are subsitutions, so first attempting the coupon collector decoding and then with the QSPA decoder. Will need to hard code likelihoods for it however. 

import pandas as pd
import galois
import row_echleon as r
import numpy as np
from itertools import combinations
from experiment_pipeline.dependencies import create_mask, invert_mask, find_I_columns
import ast
from distracted_coupon_collector import choose_symbols
from qspa_decoder_interface import get_symbol_likelihood
from tanner import VariableTannerGraph
from tqdm import tqdm
import os
import filecmp

np_arr_filename = 'symbol_likelihoods_collection_99.74consensus.npy'
#decoded_filename = r'C:\Users\Parv\Doc\HelixWorks\code\data\E1C01-01-1280\OAS\T1-DC-99.74\EIC01-01-1280-T1_decoded_consensus.tsv'
decoded_filename = r"C:\Users\Parv\Doc\HelixWorks\code\data\E1C01-01-1280\OAS\T1-DC-99.74\EIC01-01-1280-T1_decoded_consensus.tsv"
encoded_filename = r"C:\Users\Parv\Doc\HelixWorks\code\data\E1C01-01-1280\OAS\T1-DC-99.74\EIC01-01-1280-T1_encoded.tsv"
symbols = choose_symbols(8,4)

def read_payloads_from_file(filename):

    df = pd.read_csv(filename, sep="\t")
    # Getting purely the payload columns
    payloads = df.drop(['ONT_Barcode', 'HW_Address'], axis=1)
    payloads_arr = payloads.to_numpy()
    payloads_arr = np.array([[ast.literal_eval(j) for j in i] for i in payloads_arr])
    payloads_arr = payloads_arr.reshape(8,1280,4) # 8 Codewords of length 1280 from
    return payloads_arr

def unmask_likelihood_arr(likelihood_arr, mask):
    """Reorders element from likelihood array based on the mask and then removes the last three and normalizes

    Args: 
        likelihood_arr (list(float)): Probabiilty Array for Symbol
        mask (list(int)): Base 70 Mask that is added to codeword prior to transmission
    Returns:
        unmasked_likelihood_arr (list(float)): Reordered based on the mask, reduced to base 67 and renormalized likelihood array
    """

    unmasked_likelihood_arr = np.zeros(70)

    for i in range(70):
        try:
            unmasked_likelihood_arr[(i-mask) % 70] = likelihood_arr[i]
        except:
            print((i-mask) %70)
            print(len(likelihood_arr))
            print(i)
    unmasked_likelihood_arr = list(unmasked_likelihood_arr)

    unmasked_likelihood_arr.pop()
    unmasked_likelihood_arr.pop()
    unmasked_likelihood_arr.pop()

    norm_factor = sum(unmasked_likelihood_arr)
    unmasked_likelihood_arr = [i/norm_factor for i in unmasked_likelihood_arr]
    return unmasked_likelihood_arr

def get_symbol_likelihood_arr(filename):
    """Reads the output file, converts to motifs encountered assuming all are encountered initially and then adjusting based on symbols observed and then to  8 x 1280 x 67 Symbol Likelihoods to be fed to QSPA Decoder for preloaded G and Harr. Also saves the array to np_savefilepath

        Args:
            filename(str) : Filepath of channel output file
        Returns:
            symbol_likelihood_arr (np.array) : 8 x 1280 x 67 Symbol Likelihood Array
    """
    channel_output_payloads = read_payloads_from_file(filename)

    channel_output_payloads = channel_output_payloads.reshape(10240, 4)
    channel_output_payloads = channel_output_payloads[:-16]
    len_division = 1278

    channel_output_payloads = channel_output_payloads.reshape(8,1278,4)
    #channel_output_payloads = np.array([channel_output_payloads[i:i+len_division] for i in range(0, len(channel_output_payloads), len_division)])

    
    #motif_occurance_base_arr = [1,1,1,1,1,1,1,1] # We want to assume that each motif was seen once at least - let's see if this works better
    motif_occurance_base_arr = [0,0,0,0,0,0,0,0] # We want to assume that each motif was seen once at least
    
    #channel_output_payloads = channel_output_payloads.reshape()

    # 8 codewords of Length 1280
    codeword_len = channel_output_payloads.shape[1]
    num_codewords = channel_output_payloads.shape[0]

    missing_motif_count = 0
    symbol_likelihoods_collection = [] # 8 x 1280 x 67 - 8 Cycles, 1280 - len 67 symbol likelihood arrays
    rng = np.random.default_rng(seed=42)

    # For each cycle - For each payload - convert motifs to symbol likelihood arrays to feed into QSPA Decoder
    for i in tqdm(range(num_codewords)):
        
        symbol_likelihoods = []
        mask = create_mask(rng, 1278) # that's  how we divided it
        
        for j in tqdm(range(codeword_len)): # Ignore last two since they are padded
            
            payload_motifs = channel_output_payloads[i,j]
            
            motif_occurences = motif_occurance_base_arr.copy()
            
            for motif in payload_motifs:
                
                if motif == 0:
                    missing_motif_count += 1
                    #print(f"Missing Motif observed - count = {missing_motif_count}")
                    continue
                
                motif_occurences[motif-1] += 1
            
            payload_symbol_likelihood_arr = get_symbol_likelihood(4, motif_occurences, P=0.02, pop=False)

            unmasked_payload_symbol_likelihood_arr = unmask_likelihood_arr(payload_symbol_likelihood_arr, mask[j])
            
            symbol_likelihoods.append(unmasked_payload_symbol_likelihood_arr)
        symbol_likelihoods_collection.append(symbol_likelihoods)


    symbol_likelihoods_collection = np.array(symbol_likelihoods_collection)

    np.save(np_arr_filename, symbol_likelihoods_collection)

    return symbol_likelihoods_collection

def decode(symbol_likelihood_arrs, encoded_symbols):

    dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
    Harr = np.load("Harr_empirical.npy")
    graph = VariableTannerGraph(3, 9, 852, 1278)
    graph.establish_connections(Harr)

    GF = galois.GF(ffdim)
    GFH = GF(np.array(r.get_H_Matrix(dv, dc, k, n, Harr), dtype=int))

    rng = np.random.default_rng(seed=42)
    decoded_arrs = []

    for i in tqdm(range(len(symbol_likelihood_arrs))):
        symbol_likelihood_arr = symbol_likelihood_arrs[i][:1278] # Since last two are padded zeros to get to right size
        assert len(symbol_likelihood_arr) == len(graph.vns) == 1278
        graph.assign_values(symbol_likelihood_arr)
        #z = [np.argmax(i) for i in symbol_likelihood_arr]
        z = graph.qspa_decoding(GFH, GF, max_iterations=20)
        z = list(z)
        z = [int(k) for k in z]
        #z = [int(i) for i in z]
        mask = create_mask(rng, 1278)
        print(f"Decoded unmasked - \n{z[:10]} \n")
        print(f"Encoded masked - \n{encoded_symbols[i][:10]} \n")
        print(f'Mask - \n{mask[:10]}\n')

        if z == encoded_symbols[i][:1278]:
            print(f"Cycle {i} is Decoded Succesfully")
        else:
            print(f"Cycle {i} failed to decode")
        
        decoded_arrs.append(z)
    
    return decoded_arrs

symbol_likelihood_arrs = get_symbol_likelihood_arr(decoded_filename)
#symbol_likelihood_arrs = np.load(np_arr_filename)

encoded_payloads = read_payloads_from_file(encoded_filename)
encoded_payloads = encoded_payloads.reshape(10240, 4)
encoded_payloads = encoded_payloads[:-16]
encoded_payloads = encoded_payloads.reshape(8,1278,4)
    
codeword_arrs = []
num_codewords = 8
codeword_len = 1278

rng = np.random.default_rng(seed=42)


# Unmaskining initial codeword to check
for i in range(num_codewords):
    codeword = []
    mask = create_mask(rng, 1278)

    for j in range(codeword_len):
        symbol = symbols.index(list(encoded_payloads[i][j]))
        codeword.append(symbols.index(list(encoded_payloads[i][j])))
    codeword = [(codeword[i] - mask[i]) % 70 for i in range(len(codeword))]
    codeword_arrs.append(codeword)


G = np.load("G_empirical.npy")
padding_zeros = 217
final_codewords_arr = decode(symbol_likelihood_arrs, codeword_arrs)
column_indices = find_I_columns(G)
recovered_input = np.array([[i[int(j)] for j in column_indices]for i in final_codewords_arr]).flatten()
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

print(filecmp.cmp("input_txt.txt", "output.txt"))
