

import numpy as np
from Hmatrixbaby import ParityCheckMatrix
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
import time
import row_echleon as r
from bec import generate_erasures
from tqdm import tqdm
from cProfile import Profile
from density_evolution import threshold_binary_search
from pstats import Stats
import re
import sys

def permuter(arr, ffield, vn_value):

    possibilities = set(arr[0])
    new_possibilities = set()
    for i in range(1, len(arr)):
        for k in possibilities:
            for j in arr[i]:
                new_possibilities.add((j+k) % ffield)
                if len(new_possibilities) == ffield:
                    return vn_value
        possibilities = new_possibilities 
        new_possibilities = set()
    
    return {(-p)%ffield for p in possibilities}


def conv_circ(u, v):
    """Perform circular convolution between u and v over GF using FFT."""
    return np.real(np.fft.ifft(np.fft.fft(u) * np.fft.fft(v)))

def perform_convolutions(arr_pd):
    """ Combines all the Probability distributions within the array using the Convolution operator
    
    Args:
        arr_pd (arr): Array of Discrete Probability Distributions
    
    Returns:
        conv_pd (arr): Combined Probability Distributions after taking convolution over all of the pdf
    """

    pdf = arr_pd[0]

    for i in arr_pd[1:]:
        pdf = conv_circ(pdf, i)

    return pdf


class Node:

    def __init__(self, no_connections, identifier):
        self.value = 0
        self.links = np.zeros(no_connections, dtype=int)
        self.identifier = identifier

    def add_link(self, node):
        """ Adds a link to the node. Throws an error if the node is full """
        
        # Check if node is full
        #if np.all(self.links):
         #   raise ValueError("Node is full")

        # Add to empty link 
        for (i,j) in enumerate(self.links):
            if not j:
                self.links[i] = node.identifier
                break
        
        return self.links
    
    def get_links(self):
        return self.links

    def replace_link(self, node, index):
        """ Replaces a link with another link """
        self.links[index] = node
        return self.links
    
class CheckNode(Node):

    def __init__(self, dc, identifier):
        super().__init__(dc, identifier)
    
class VariableNode(Node):
    def __init__(self, dv, identifier):
        super().__init__(dv, identifier)


class Link(Node):
    def __init__(self, cn, vn, value):
        self.cn = cn
        self.vn = vn
        self.value = value

class VariableTannerGraph:
    """ Initializes empty, on establishing connections creates H and forms links """

    def __init__(self, dv, dc, k, n, ffdim=2):
        
        # Check if connections are non-uniform
        if type(dv) == list:
            assert len(dv) == n and len(dc) == n-k
            self.vns = [VariableNode(dv[i], i) for i in range(n)]
            self.cns = [CheckNode(dc[i], i) for i in range(n-k)]
            self.dv = dv
            self.dc = dc
        else:
            self.vns = [VariableNode(dv, i) for i in range(n)]
            self.cns = [CheckNode(dc, i) for i in range(n-k)]
            self.dv = [dv for i in range(n)]
            self.dc = [dc for i in range(n-k)]

        # For the singular case - it remains as an integer, but for the Changing Case it goes to a list, need to make sure that does not break everything
        self.k = k
        self.n = n
        self.ffdim = ffdim
        self.links = {}

    def add_link(self, cn_index, vn_index, link_value):
        """ Adds a link to the links data structure """
        self.links[(cn_index, vn_index)] = link_value
    
    def update_link_weight(self, cn_index, vn_index, link_value):
        """ Updates Link weight """
        self.add_link(cn_index, vn_index, link_value)
    
    def get_link_weight(self, cn_index, vn_index):
        """ Get Link Weight """
        return self.links[(cn_index, vn_index)]
    
    def update_within_link_weight(self, cn_index, vn_index, val_index, new_value):
        self.links[(cn_index, vn_index)][val_index] = new_value

    def get_vn_value(self, vn_index):
        return self.vns[vn_index].value

    def get_cn_value(self, cn_index):
        return self.cns[cn_index].value

    def establish_connections(self, Harr=None):
        """ Establishes connections between variable nodes and check nodes """
        
        # In case Harr is sent as a parameter
        if Harr is None:
            # If we are creating, assuming it's not scldpc - really needs some unification here champ
            self.Harr = r.get_H_arr(self.dv[0], self.dc[0], self.k, self.n)
        else:
            self.Harr = np.array(Harr)

        # Our Harr is implementation is different - will need to be considered when adapting - assuming that this is the check nodes they are connected to
        Harr = self.Harr

        # Divide Harr into dv parts  
        # But dv is a list in the case of the changing case
        # All the dvs are the same for this case
        dv = self.dv[0]

        if len(np.unique(self.dc)) == 1:
            Harr = Harr // self.dc[0]
        
        Harr = [Harr[i:i+dv] for i in range(0, len(Harr), dv)]

        # Checking for spatially coupled
        
        
        # Establish connections
        for (i,j) in enumerate(Harr):
            for k in j:
                self.vns[i].add_link(self.cns[k])
                self.cns[k].add_link(self.vns[i])
                self.add_link(k, i, 0)

    def get_connections(self):
        """ Returns the connections in the Tanner Graph """
        return [(i.identifier, j) for i in self.cns for j in i.links]

    def get_cn_link_values(self, cn):
        """ Returns the values of the link weights for the cn as an array"""
        vals = []
        for i in cn.links:
            vals.append(self.get_link_weight(self.cn.identifier, i))

        return vals

    def visualise(self):
        """ Visualise Tanner Graph """
        G = nx.Graph()

        rows = len(self.cns)
        cols = len(self.vns)

        # For each row add a check node
        for i in range(rows):
            G.add_node(i, bipartite=0)

        # For each column add a variable node
        for i in range(cols):
            G.add_node(i + rows, bipartite=1)
        
        # Utilise the links to add edges
        for (i,j) in enumerate(self.cns):
            for k in j.links:
                G.add_edge(i, k + rows, weight=1)
    
    
        nx.draw(G, with_labels=True)
        plt.show()
        return G

    def assign_values(self, arr):   

        assert len(arr) == len(self.vns) 

        for i in range(len(arr)):
            self.vns[i].value = arr[i]

    def get_max_prob_codeword(self):
        """Returns the most possible Codeword using the probability likelihoods established in the VN's

        Returns:
            codeword (arr): n length codeword with symbols
        """
        z = np.zeros(self.n)
        for j in range(self.n):
            probs = 1 * P[j, :]
            for a in range(self.GF.order):
                for i in idxs:
                    probs[a] *= self.get_link_weight(i, j)[a]
            z[j] = np.argmax(probs)
        z = self.GF(z.astype(int))
        return z


    def validate_codeword(self, H, GF, max_prob_codeword):
        """ Checks if the most probable codeword is valid as a termination condition of qspa decoding """
        return not np.matmul(H, max_prob_codeword).any()

    def remove_from_array(self, vals, current_value):
        """ Removes current value from vals"""

        new_vals = []
        for i in range(len(vals)):
            if np.array_equal(vals[i], current_value):
                continue
            new_vals.append(vals[i])
        return new_vals 

    def qspa_decoding(self, H, GF, max_iterations=10):
        
        # Additive inverse of GF Field
        idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])
        
        # Initial likelihoods
        P = [i.value for i in self.vns]

        for i in range(max_iterations):
            # VN Update
            for i in self.cns:
                link_weights = self.get_cn_link_values(i)
                for j in i.links:
                    vals = link_weights.copy()
                    current_value = self.get_link_weight(i.identifier, j)
                    vals = self.remove_from_array(vals, current_value)        
                    pdf = perform_convolutions(vals)
                    self.update_link_weight(i,j,pdf[idx_shuffle])
                
            max_prob_codeword = get_max_prob_codeword()
            
            # Check if we have got a valid codeword
            if self.validate_codeword(H, GF, max_prob_codeword):
                return max_prob_codeword

            # CN Update
            for a in range(GF.order):
                for j in self.vns:
                    vn_index = j.identifier
                    idxs = self.nonzero_rows[j]
                    for i in j.links:
                        # Initial Likelihoods
                        self.update_within_link_weight(i, vn_index, a, P[j,a])

                        for t in j.links[t!=i]:
                            link_weight = self.get_link_weight(i, vn_index)
                            alternate_link_weight = self.get_link_weight(t, vn_index)
                            link_weight[a] *= alternate_link_weight[a]
                            self.update_within_link_weight(i, j.identifier, a, link_weight)
                            
                        # Normalization
                        val = self.get_link_weight(i,vn_value)
                        norm_factor = sum(val)
                        normalized_value = [i/norm_factor for i in val]
                        self.update_link_weight(i,j, normalized_value)
            
            if iterations > max_iterations:
                return max_prob_codeword