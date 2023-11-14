

import numpy as np


# Load a saved code

H = np.load(f"C:\Users\Parv\Doc\HelixWorks\code\codes\dvdckn36100200\2a0c2c86-d507-43fb-bcd4-43ad715a071a\H.npy")
G = np.load(f"C:\Users\Parv\Doc\HelixWorks\code\codes\dvdckn36100200\2a0c2c86-d507-43fb-bcd4-43ad715a071a\G.npy")

if np.any(np.matmul(H, G.T) != 0):
    print("H and G are not orthogonal")