

import matplotlib.pyplot as plt
import numpy as np

fers = [1.0, 1.0, 1.0, 1.0, 1.0, 0.13020833333333334, 0.08928571428571429, 0.034129692832764506, 0.022747952684258416, 0.010513036164844407, 0.006824075337791729]
rl = np.arange(1,12)

plt.plot(rl, fers)
plt.title("2010 size SCLDPC FER 10^-3 Error Rate")
plt.xlabel("Read Length")
plt.ylabel("Frame Error Rate")
plt.show()