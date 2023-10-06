
import matplotlib.pyplot as plt
import numpy as np

def density_evolution_bec(erasure_proability, d_v, d_c, max_iterations=1000, plot=False):
    """ Get the density evolution of a LDPC code """

    E = 0
    Earr = []
    E_o = erasure_proability
    Earr.append(erasure_proability)
    for i in range (max_iterations):
        E = erasure_proability * (1 - (1 - E_o)**(d_c - 1))**(d_v - 1)
        Earr.append(E)
        E_o = E

    if plot:
        plt.plot(0, erasure_proability, marker='o', color='gray')
        plt.plot(Earr, '--')
    
    return Earr

def plot_density_evolution():
    density_evolution_bec(0.3, 3, 6, plot=True)
    density_evolution_bec(0.4296, 3, 6, plot=True)
    density_evolution_bec(0.5, 3, 6, plot=True)
    density_evolution_bec(0.6, 3, 6, plot=True)
    density_evolution_bec(0.7, 3, 6, plot=True)
    plt.xlim(0, 100)
    plt.ylim(0, 1)  
    plt.title("Density Evolution for (6,3) LDPC Code")
    plt.ylabel("Erasure Probability")
    plt.xlabel("Iterations")
    plt.show()


def threshold_binary_search(dc, dv):
    """ Utilise Density Evolution to find the threshold of a LDPC code using binary search"""

    ll, ul = 0, 1
    while True:
        mid = (ul + ll)/2
        end_point = density_evolution_bec(mid, dc, dv)[-1]

        if end_point > 0.001:
            ul = mid
        elif end_point == 0:
            ll = mid
        else:
            print("Threshold: ", mid)
            return mid
        

threshold_binary_search(3,6)
