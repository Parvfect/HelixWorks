

import sc_ldpc_protograph


def get_vns(k, n):
    # dv - VN degree
    # dc - CN degree
    # L - number of segments
    # M - number of VNs per segment
    #
    # Total number of VNs: L * M
    # Total number of CNs: (L + dv - 1) * dv / dc * M

    # Example values:
    dv = 3
    dc = 6


    M = 20
    L = 2
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
        print(vns)
        Harr.append(vns)
    
    return Harr

def get_Harr(k, n):
    """ Interface to get all the cns the vns are connected to """

    Harr = get_vns(k, n)
    print(Harr)
    # Will also need to get dc's and dv's to initialize the Tanner graph - which can be obtained from the array graph itself.


if __name__ == '__main__':
    get_Harr(1, 1)