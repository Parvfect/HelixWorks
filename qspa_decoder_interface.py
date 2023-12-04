

# Probability of likelihoods instead of input array
# Parity Check Matrix of Gaolis Field form - which should be simple enough

# Look at the notebook notes, write the simple functions, see how well the decoder performs for a small code first on the dcc
# Will need to compare it to the cc so a cc case plotted alongside it would be handy
# Use Larger codes if fast enough at Decoding
# If not, adapt the code to my Tanner structure


def __init__(self, n, m, GF, GFH):
        self.n = n      # length of codeword
        self.m = m      # number of parity-check constraints
        self.GF = GF    # Galois field
        self.GFH = GFH  # parity-check matrix (with elements in GF)

        self.nonzero_cols, self.nonzero_rows = self.index_nonzero()

        # Store additive inverse for each element in GF
        self.idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])

decoder = QSPADecoder(n_code, m_checks, GF, GFH)
    z = decoder.decode(P, max_iter=10)

P: array (n, GF.order)
            Initial likelihoods for each symbol in received codeword.
max_iter: int
    Maximum number of iterations that decoder should run for.
