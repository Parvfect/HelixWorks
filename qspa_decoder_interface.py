
from qspa_conv import hw_likelihoods, QSPADecoder
import utils
import galois
import row_echleon as r
import numpy as np
from tqdm import tqdm
#from graph import TannerGraph
from tanner import VariableTannerGraph
import random
import matplotlib.pyplot as plt
from distracted_coupon_collector import distracted_coupon_collector_channel, choose_symbols

def test_symbol_likelihood():
    n_picks = 4
    motif_occurences = [2, 1, 2, 1, 0, 0, 0, 1]
    P = 0.02
    get_symbol_likelihood(n_picks, motif_occurences, P)


def get_symbol_likelihood(n_picks, motif_occurences, P):
    """Generates Likelihood Array for a Symbol after it's passed through the channel, using the number of times each motif is encountered

        Args:
            n_picks (int): Number of motifs per symbol
            motif_occurences (array) (n_motifs,): Array of Occurence of Each Motif Encountered [0,0,1,1,2,3,0] 
            P (float): Interference Probability
        Returns:
            likelihoods: array (n_motifs choose k_motifs, ) - Normalized likelihood for each symbol (in lexicographical order).
    """

    # Getting the Likelihoods from Alberto's Likelihood Generator
    likelihoods = hw_likelihoods(n_picks, motif_occurences, P)

    # Popping the last three likelihoods to make the symbols match
    likelihoods.pop()
    likelihoods.pop()
    likelihoods.pop()

    # Re-normalising
    norm_factor = 1/sum(likelihoods)
    likelihoods = [norm_factor*i for i in likelihoods]

    # Precision - summing up to 0.9999999999999997
    assert sum(likelihoods) >= 0.99 and sum(likelihoods) < 1.01

    return likelihoods

def simulate_reads(C, symbols, read_length, P, n_motifs, n_picks):
    """Simulates reads using the QSPA Decoder
        Args:
            C (list) (n,): Codeword
            read_length (int): Read Length
            P (Float): Interference Probability
            n_motifs (int): Number of Motifs in Total
            n_picks (int): Number of Motifs Per Symbol
        Returns: 
            reads (list) : [length of Codeword, no. of symbols] list of all the reads as likelihoods
    """

    likelihood_arr = []
    for i in C:
        motif_occurences = np.zeros(n_motifs)
        reads = distracted_coupon_collector_channel(symbols[i], read_length, P, n_motifs)

        # Collecting Motifs Encountered
        for i in reads:
            motif_occurences[i-1] += 1

        symbol_likelihoods = get_symbol_likelihood(n_picks, motif_occurences, P)
        likelihood_arr.append(symbol_likelihoods)

    return likelihood_arr

def simulate(Harr, GFH, GFK, symbols, P, n_code, k, read_length=10, max_iter=10, cc=False):

    ffdim = 67
    n_motifs, n_picks = 8, 4
    dv, dc = 3, 9
    
    m_checks = GFH.shape[0]

    GF = galois.GF(ffdim)

    input_arr = [random.randint(0,66) for i in range(k)]
    C = np.matmul(GF(input_arr), GFK)

    symbol_likelihoods_arr = np.array(simulate_reads(C, symbols, read_length, P, n_motifs, n_picks))
    
    assert symbol_likelihoods_arr.shape == (n_code, ffdim)
    
    if cc:
        graph = VariableTannerGraph(dv, dc,k, n_code, ffdim=ffdim)
        graph.establish_connections(Harr)
        graph.assign_values(symbol_likelihoods_arr)
        #z = graph.coupon_collector_decoding()
        z = graph.qspa_decoding(GFH, GF)
    else:
        decoder = QSPADecoder(n_code, m_checks, GF, GFH)
        # Will have to replace that max Iter with the break condition that we had before
        z = decoder.decode(symbol_likelihoods_arr, max_iter=max_iter)
        
    return np.array_equal(C, z)


def fer(P, n_code, k, iterations=10, read_lengths=np.arange(8,24), max_iter=10, cc_decoding=False, label=None):
    
    ffdim = 67
    n_motifs, n_picks = 8, 4
    dv, dc = 3, 9
    
    
    Harr = r.get_H_arr(dv, dc, k, n_code)
    H = np.array(r.get_H_Matrix(dv, dc, k, n_code, Harr), dtype=int)
    GF = galois.GF(ffdim)
    GFH = GF(H) # * GF(np.random.choice(GF.elements[1:], size=H.shape))
    GFK = GFH.null_space()
    symbols = choose_symbols(n_motifs, n_picks)
    

    fers = []
    for i in tqdm(read_lengths):
        counter=0
        for j in tqdm(range(iterations)):
            if cc_decoding:
                flag = simulate(Harr, GFH, GFK, symbols, P, n_code, k, i, max_iter, cc=True)
            else:
                flag = simulate(Harr, GFH, GFK, symbols, P, n_code, k, i, max_iter)
            
            if flag:
                counter += 1
        fers.append((iterations - counter)/iterations)
    
    print(fers)
    plt.plot(read_lengths, fers, label=label)
    plt.title(f"FER for DCC Decoder for P={P} with codeword {n_code} and rate {n_code/k}")
    plt.xlabel("Read Lengths")
    plt.ylabel("FER")


P = 0.02
n_code, k = 150, 100
iterations = 5
read_lengths = np.arange(5, 12)
max_iter=20
#fer(P, n_code, k, iterations, read_lengths, max_iter)
fer(P, n_code, k, iterations, read_lengths, max_iter, cc_decoding=True, label="CC")
plt.legend()
plt.show()
