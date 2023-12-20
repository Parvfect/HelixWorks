

import numpy as np
import matplotlib.pyplot as plt

read_lengths = np.arange(8, 24)
dcc_fers = []

dcc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cc = [0.9, 0.7, 0.6, 0.55, 0.5, 0.56, 0.59, 0.56, 0.69, 0.73, 0.86, 0.84, 0.94, 0.95, 0.99, 0.99]

plt.plot(read_lengths, dcc, label="Prob Decoder")

plt.plot(read_lengths, cc, label="CC Decoder")
plt.xlabel("Read Lengths")
plt.ylabel("FER")
plt.legend()
plt.title("Decoding over DCC (100-150) with P=0.2")
plt.show()