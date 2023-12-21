

import numpy as np
import matplotlib.pyplot as plt

read_lengths = np.arange(5, 20)
dcc_fers = []
dcc = [1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


plt.plot(read_lengths, dcc, label="Prob Decoder")

plt.plot(read_lengths, cc, label="CC Decoder")
plt.xlabel("Read Lengths")
plt.ylabel("FER")
plt.legend()
plt.title("Decoding over DCC (100-150) with P=0.2")
plt.show()