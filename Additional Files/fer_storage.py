
import matplotlib.pyplot as plt
import numpy as np

reads = np.arange(6,20)

uncoded = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9995, 0.997, 0.978, 0.954, 0.8905, 0.806]
c1020 = [0.97, 0.85, 0.47, 0.22, 0.09, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
c1020bec = [0.99, 0.95, 0.78, 0.49, 0.32, 0.1, 0.04, 0.02, 0.01, 0.0, 0.02, 0.0, 0.01, 0.01]

c50100 = [1.0, 0.88, 0.22, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
c50100bec = [1.0, 0.98, 0.72, 0.17, 0.06, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

c100200 = [1.0, 0.91, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
c100200bec = [1.0, 0.99, 0.4, 0.06, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0]

c500k = [1.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
c500kbec = [1.0, 1.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Without bec plots
plt.plot(reads, uncoded, label="Uncoded 100")
plt.plot(reads, c1020, label="10-20")
plt.plot(reads, c50100, label="50-100")
plt.plot(reads, c100200, label="100-200")
plt.plot(reads, c500k, label="500-1000")
plt.legend()
plt.xticks(np.arange(6, 20, 1))
plt.xlim(6,19)
plt.ylim(0,1)
plt.grid()    
plt.xlabel("Read Length")
plt.ylabel("Frame Error Rate")
plt.title("Frame Error Rate for CC for 8C4 Symbols")
plt.show()


# Uncoded - bec - our decoded (for the 50 - 100 blocklength)
plt.plot(reads, c50100, label="50-100")
plt.plot(reads, c50100bec,  label="50-100 bec")
plt.plot(reads, uncoded, label="Uncoded 100")
plt.title("Comparing performance of BEC and CC Decoding for 50-100 blocklength")
plt.xticks(np.arange(6, 20, 1))
plt.grid()  
plt.xlim(6,19)
plt.ylim(0,1)  
plt.legend()
plt.xlabel("Read Length")
plt.ylabel("Frame Error Rate")
plt.show()

