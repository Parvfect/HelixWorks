

import sc_ldpc_protograph
import numpy as np


def get_vns(dv, dc, M, L):
    # dv - VN degree
    # dc - CN degree
    # L - number of segments
    # M - number of VNs per segment
    #
    # Total number of VNs: L * M
    # Total number of CNs: (L + dv - 1) * dv / dc * M

    # (Note: we used to have something more like M=1000, L=50, but we can try smaller codes first)

    cns_per_pos = int(dv / dc * M)

    # VN connections are generated one segment at a time
    # 'vns' is an array of VN connections for each consecutive segment
    # 'vns[i]' contains an array of CN indices that VN is connected to 
    Harr = []
    for vn_position in range(L):   
        seed = vn_position * cns_per_pos
        vns = seed + sc_ldpc_protograph.gen_slots_from_position(dv, dc, M)
        # ... fill in the next M VNs in the H matrix
        # ... (you can also directly add these vns to your Tanner graph)
        Harr.append(vns)
    
    return np.array(Harr)

def get_Harr():
    """ Interface to get all the cns the vns are connected to """

    dv = 3
    dc = 6
    M = 6
    L = 1
    Harr = get_vns(dv, dc, M, L)
    n = L*M
    cns_len = int((L + dv - 1) * dv / dc * M)
    k = int(n - ((L + dv - 1) * dv / dc * M))
    dcs = np.zeros(cns_len, dtype=int)
    dvs = [3 for i in range(L * M)] 

    for i in Harr:
        #for j in i:
        dcs[i] += 1

    Harr = Harr.flatten()

    # Will also need to get dc's and dv's to initialize the Tanner graph - which can be obtained from the array graph itself.

    return Harr, dcs, dvs, k, n


if __name__ == '__main__':
    get_Harr()