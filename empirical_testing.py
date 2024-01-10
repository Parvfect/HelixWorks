

# Reading from files empirical_encoded and empirical_channel_output, first for the 99.74 decoded consensus. 
# Attempting to decode. There are subsitutions, so first attempting the coupon collector decoding and then with the QSPA decoder. Will need to hard code likelihoods for it however. 

import pandas as pd
import galois
import row_echleon as r
import numpy as np
from itertools import combinations
from experiment_pipeline.dependencies import create_mask
import ast
from distracted_coupon_collector import choose_symbols
from qspa_decoder_interface import get_symbol_likelihood
from tanner import VariableTannerGraph
from tqdm import tqdm

np_arr_filename = 'symbol_likelihoods_collection_99.74consensus.npy'
decoded_filename = r'C:\Users\Parv\Doc\HelixWorks\code\data\E1C01-01-1280\OAS\T1-DC-99.74\EIC01-01-1280-T1_decoded_consensus.tsv'
encoded_filename = r'C:\Users\Parv\Doc\HelixWorks\code\data\E1C01-01-1280\OAS\T1-DC-99.74\EIC01-01-1280-T1_encoded.tsv'
rng = np.random.default_rng(seed=42) # Mask seed

def read_payloads_from_file(filename):

    df = pd.read_csv(filename, sep="\t")
    # Getting purely the payload columns
    payloads = df.drop(['ONT_Barcode', 'HW_Address'], axis=1)
    payloads_arr = payloads.to_numpy()
    return np.array([[ast.literal_eval(j) for j in i] for i in payloads_arr])

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
    
    #motif_occurance_base_arr = [1,1,1,1,1,1,1,1] # We want to assume that each motif was seen once at least - let's see if this works better
    motif_occurance_base_arr = [0,0,0,0,0,0,0,0] # We want to assume that each motif was seen once at least
    
    #channel_output_payloads = channel_output_payloads.reshape()

    # 1280 payloads x 8 cycles x 4 motifs
    num_payloads = channel_output_payloads.shape[0]
    num_cycles = channel_output_payloads.shape[1]

    missing_motif_count = 0
    symbol_likelihoods_collection = [] # 8 x 1280 x 67 - 8 Cycles, 1280 - len 67 symbol likelihood arrays

    # For each cycle - For each payload - convert motifs to symbol likelihood arrays to feed into QSPA Decoder
    for i in range(num_cycles):
        
        symbol_likelihoods = []
        mask = create_mask(rng, 1278) # that's  how we divided it
        print(mask[:5])
        
        for j in range(num_payloads-2): # Ignore last two since they are padded
            
            payload_motifs = channel_output_payloads[j,i]
            
            motif_occurences = motif_occurance_base_arr.copy()
            
            for motif in payload_motifs:
                
                if motif == 0:
                    missing_motif_count += 1
                    print(f"Missing Motif observed - count = {missing_motif_count}")
                    continue
                
                motif_occurences[motif-1] += 10
            
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

    for i in tqdm(range(len(symbol_likelihood_arrs))):
        symbol_likelihood_arr = symbol_likelihood_arrs[i][:1278] # Since last two are padded zeros to get to right size
        assert len(symbol_likelihood_arr) == len(graph.vns) == 1278
        graph.assign_values(symbol_likelihood_arr)
        z = [np.argmax(i) for i in symbol_likelihood_arr]
        #z = graph.qspa_decoding(GFH, GF)
        z = list(z)
        z = [int(k) for k in z]
        print(z[:5])
        print(encoded_symbols[i][:5])
        if z == encoded_symbols[i][:1278]:
            print(f"Cycle {i} is Decoded Succesfully")
        else:
            print(f"Cycle {i} failed to decode")


symbol_likelihood_arrs = get_symbol_likelihood_arr(encoded_filename)
#symbol_likelihood_arrs = np.load(np_arr_filename)
print(symbol_likelihood_arrs.shape)

encoded_payloads = read_payloads_from_file(encoded_filename)
codeword_arrs = []
num_cycles = 8
num_payloads = 1280

symbols = choose_symbols(8,4)

for i in range(num_cycles):
    codeword = []
    for j in range(num_payloads):
        codeword.append(symbols.index(list(encoded_payloads[j][i])))
    codeword_arrs.append(codeword)

decode(symbol_likelihood_arrs, codeword_arrs)