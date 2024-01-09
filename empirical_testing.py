

# Reading from files empirical_encoded and empirical_channel_output, first for the 99.74 decoded consensus. 
# Attempting to decode. There are subsitutions, so first attempting the coupon collector decoding and then with the QSPA decoder. Will need to hard code likelihoods for it however. 

import pandas as pd
import gaolis
import row_echleon as r
import numpy as np
from itertools import combinations
import ast
from distracted_coupon_collector import choose_symbols
from qspa_decoder_interface import get_symbol_likelihood
from tanner import VariableTannerGraph

np_arr_filename = 'symbol_likelihoods_collection_99.74consensus.npy'

def read_payloads_from_file(filename):

    df = pd.read_csv(filename, sep="\t")
    # Getting purely the payload columns
    payloads = df.drop(['ONT_Barcode', 'HW_Address'], axis=1)
    payloads_arr = payloads.to_numpy()
    return np.array([[ast.literal_eval(j) for j in i] for i in payloads_arr])


def convert_to_symbols(encoded_payloads_arr):
    symbols = choose_symbols(8,4)
    return np.array([[symbols.index(j) for j in i] for i in encoded_payloads_arr])


"""
encoded_symbols = convert_to_symbols(read_payloads_from_file("empirical_encoded.tsv"))

encoded_symbols = encoded_symbols.reshape(1280*8, 1)
encoded_symbols_flattened = np.array([i[0] for i in encoded_symbols])
"""


def get_symbol_likelihood_arr(filename):
    """Reads the output file, converts to motifs encountered assuming all are encountered initially and then adjusting based on symbols observed and then to  8 x 1280 x 67 Symbol Likelihoods to be fed to QSPA Decoder for preloaded G and Harr. Also saves the array to np_savefilepath

        Args:
            filename(str) : Filepath of channel output file
        Returns:
            symbol_likelihood_arr (np.array) : 8 x 1280 x 67 Symbol Likelihood Array
    """
    channel_output_payloads = read_payloads_from_file("empirical_channel_output.tsv")
    # Flattening array - assuming it reshapes nicely

    #motif_occurance_base_arr = [1,1,1,1,1,1,1,1] # We want to assume that each motif was seen once at least - let's see if this works better
    motif_occurance_base_arr = [0,0,0,0,0,0,0,0] # We want to assume that each motif was seen once at least
    

    # 1280 payloads x 8 cycles x 4 motifs
    num_payloads = channel_output_payloads.shape[0]
    num_cycles = channel_output_payloads.shape[1]

    missing_motif_count = 0
    symbol_likelihoods_collection = [] # 8 x 1280 x 67 - 8 Cycles, 1280 - len 67 symbol likelihood arrays

    # For each cycle - For each payload - convert motifs to symbol likelihood arrays to feed into QSPA Decoder
    for i in range(num_cycles):
        symbol_likelihoods = []
        for j in range(num_payloads):
            payload_motifs = channel_output_payloads[j,i]
            motif_occurences = motif_occurance_base_arr.copy()
            symbol = invert_mask(get_symbol(payload_motifs)) # To be implemented - invert the mask and filter the symbols as per experiment pipeline
            for motif in payload_motifs:
                if motif == 0:
                    missing_motif_count += 1
                    print(f"Missing Motif observed - count = {missing_motif_count}")
                    continue
                motif_occurences[motif-1] += 1
            payload_symbol_likelihood_arr = get_symbol_likelihood(4, motif_occurences, P=0.2)
            symbol_likelihoods.append(payload_symbol_likelihood_arr)
        symbol_likelihoods_collection.append(symbol_likelihoods)


    symbol_likelihoods_collection = np.array(symbol_likelihoods_collection)

    np.save(np_arr_filename, symbol_likelihoods_collection)

    return symbol_likelihoods_collection

symbol_likelihood_arrs = get_symbol_likelihood_arr("empirical_channel_output.tsv")
symbol_likelihood_arrs = np.load(np_arr_filename)
print(symbol_likelihood_arrs.shape)


def decode(symbol_likelihood_arrs, encoded_symbols):
    
    assert symbol_likelihood_arrs.shape == (8, 1280, 67)

    dv, dc, k, n, ffdim = 3, 9, 852, 1278, 67
    Harr = np.load("Harr_empirical.npy")
    graph = VariableTannerGraph(3, 9, 852, 1278)
    graph.establish_connections(Harr)

    GF = galois.GF(ffdim)
    GFH = GF(np.array(r.get_H_Matrix(dv, dc, k, n, Harr), dtype=int))

    for i in range(symbol_likelihood_arrs):
        symbol_likelihood_arr = symbol_likelihood_arrs[i][:1278] # Since last two are padded zeros to get to right size
        assert len(symbol_likelihood_arr) == len(graph.vns) == 1278
        graph.assign_values(symbol_likelihood_arr)
        z = graph.decode(GFH, GF)
        assert (z == encoded_symbols[i]).all()
        print(f"Cycle {i} is Decoded Succesfully")



