
from Hmatrixbaby import ParityCheckMatrix
import row_echleon as r
import uuid
import os
import numpy as np

dv, dc, k, n, ffdim = 3, 6, 1000, 2000, 67
PM = ParityCheckMatrix(dv, dc, k, n, ffdim=ffdim)
Harr = PM.get_H_arr()
H = PM.createHMatrix(Harr=Harr)
G = r.parity_to_generator(H, ffdim=ffdim)

if np.any(np.dot(G, H.T) % ffdim != 0):
    print("Matrices are not valid, aborting simulation")
    exit()

unique_filename = str(uuid.uuid4())

filename = "codes/dv_dc_k_n_ffdim={}_{}_{}_{}_{}/{}".format(dv, dc, k, n, ffdim, unique_filename)

# Create the directory if it does not exist
if not os.path.exists(filename):
    os.makedirs(filename)

# Save the Harr, H and G matrices
np.save(filename + "/Harr", Harr)
np.save(filename + "/H", H)
np.save(filename + "/G", G)

print("Saved the matrices to {}".format(filename))