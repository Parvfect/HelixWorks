

import numpy as np
import os
import random


def get_saved_code(dv, dc, k, n, L, M, ffdim=67, code_class=""):

    
    supplementary_path = r"C:\Users\Parv\Doc\HelixWorks\code\codes"

    if code_class == "sc_":
        code_path = f"{code_class}dv_dc_k_n_L_M_ffdim={dv}_{dc}_{k}_{n}_{L}_{M}_{ffdim}"
    else:
        code_path = f"{code_class}dv_dc_k_n_ffdim={dv}_{dc}_{k}_{n}_{ffdim}"
    new_path = os.path.join(supplementary_path, code_path)
    unique_id = random.choice(os.listdir(new_path))
    final_path = os.path.join(new_path, unique_id)

    Harr = np.load(os.path.join(final_path, "Harr.npy"))
    H = np.load(os.path.join(final_path, "H.npy"))
    G = np.load(os.path.join(final_path, "G.npy"))

    assert not np.any(np.matmul(G, H.T) %ffdim != 0)

    return Harr, H, G

