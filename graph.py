
import numpy as np
from Hmatrixbaby import createHMatrix, getH

class CheckNode:

    def __init__(self, dc, identifier)
        self.value = 0
        self.links = np.zeros(dc)
        self.identifier = identifier
    
class VariableNode:
    def __init__(self, dv, identifier):
        self.value = 0
        self.links = np.zeros(dv)
        self.identifier = identifier

class TannerGraph:

    def __init__(self, dv, dc, k, n):
        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
        
    def establish_connections(self):
        """ Establishes connections between variable nodes and check nodes """
        Harr = getH(self.dv, self.dc, self.k, self.n)

        for i in Harr:
            # Want to have head structure - so I don't need to establish connections twice
            