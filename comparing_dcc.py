

import numpy as np
import matplotlib.pyplot as plt

read_lengths = np.arange(5, 20)
dcc_fers = []
dcc = [1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cc = [1.0, 1.0, 0.982, 0.895, 0.75, 0.648, 0.578, 0.538, 0.521, 0.529, 0.605, 0.683, 0.772, 0.826, 0.881]

plt.plot(read_lengths, dcc, label="Prob Decoder")

plt.plot(read_lengths, cc, label="CC Decoder")
plt.grid()
plt.xticks(read_lengths)
plt.xlabel("Read Lengths")
plt.ylabel("FER")
plt.legend()
plt.title("Decoding over DCC (100-150) with P=0.02")
plt.show()