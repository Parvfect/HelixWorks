

import numpy as np
import os
import random


def get_saved_code(dv, dc, k, n, ffdim=67):
    supplementary_path = r"C:\Users\Parv\Doc\HelixWorks\code\codes"
    code_path = "dv_dc_k_n_ffdim={}_{}_{}_{}_{}".format(dv, dc, k, n, ffdim)
    new_path = os.path.join(supplementary_path, code_path)
    unique_id = random.choice(os.listdir(new_path))
    final_path = os.path.join(new_path, unique_id)

    Harr = np.load(os.path.join(final_path, "Harr.npy"))
    H = np.load(os.path.join(final_path, "H.npy"))
    G = np.load(os.path.join(final_path, "G.npy"))

    assert not np.any(np.matmul(G, H.T) %ffdim != 0)

    return Harr, H, G

