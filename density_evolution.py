
import matplotlib.pyplot as plt
import numpy as np

# Seems like some form of Runge Kutta Expansion is the way to go

#E_l = E_o (1 -  (1 - E_{l-1})^{d_c - 1})^{d_v - 1}


def density_evolution(erasure_proability, d_c, d_v, max_iterations=1000):
    """ Get the density evolution of a LDPC code """

    E = 0
    Earr = []
    E_o = erasure_proability
    Earr.append(erasure_proability)
    for i in range (max_iterations):
        E = erasure_proability * (1 - (1 - E_o)**(d_c - 1))**(d_v - 1)
        print(E)
        Earr.append(E)
        E_o = E

    plt.plot(Earr, '--')
    
density_evolution(0.3, 3, 6)
density_evolution(0.4296, 3, 6)
density_evolution(0.5, 3, 6)
density_evolution(0.6, 3, 6)
density_evolution(0.7, 3, 6)
plt.xlim(0, 100)
plt.ylim(0, 1)  
plt.show()


