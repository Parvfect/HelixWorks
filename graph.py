
import numpy as np
from Hmatrixbaby import ParityCheckMatrix
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
import time
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

class TannerGraph:
    """ Initializes empty, on establishing connections creates H and forms links """

    def __init__(self, dv, dc, k, n, ffdim=2):

        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
        self.ffdim = ffdim

    def establish_connections(self, Harr=None):
        """ Establishes connections between variable nodes and check nodes """
        
        # In case Harr is sent as a parameter
        if Harr is None:
            self.Harr = ParityCheckMatrix(self.dv, self.dc, self.k, self.n).get_H_arr()
        else:
            self.Harr = np.array(Harr)
        
        Harr = self.Harr//self.dc

        # Divide Harr into dv parts
        Harr = [Harr[i:i+self.dv] for i in range(0, len(Harr), self.dv)]

        # Establish connections
        for (i,j) in enumerate(Harr):
            for k in j:
                self.vns[i].add_link(self.cns[k])
                self.cns[k].add_link(self.vns[i])


    def get_connections(self):
        """ Returns the connections in the Tanner Graph """
        return [(i.identifier, j) for i in self.cns for j in i.links]

    def get_cn_link_values(self, cn):
        """ Returns the values of the connected vns for the cn as a dd array"""
        vals = []
        for i in cn.links:
            vals.append(self.vns[i].value)

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

    def bec_decode(self, max_iterations=100):
        """ Assuming VNs have been initialized with values, perform BEC decoding """

        filled_vns = sum([1 for i in self.vns if not np.isnan(i.value)])
        resolved_vns = 0

        for iteration in range(max_iterations):
            
            # For each check node  
            for (i,j) in enumerate(self.cns):
                # See all connection VN values
                erasure_counter = 0
                # Counting number of erasures
                for k in j.links:
                    if np.isnan(self.vns[int(k)].value):
                        erasure_counter += 1
                
                # If Erasure counter is equal to 1, fill erasure
                if erasure_counter == 1:
                    sum_links = 0
                    erasure_index = 0
                    
                    for k in j.links:
                        # Collect all values in an array
                        if np.isnan(self.vns[int(k)].value):
                            #sum_links += self.vns[int(k)].value
                            erasure_index = k
                        else:
                            #erasure_index = k
                            sum_links += self.vns[int(k)].value
                            
                    
                    # Replace erasure with sum modulo 2
                    self.vns[int(erasure_index)].value = sum_links % 2
                    resolved_vns+=1
            
            # Check in every iteration - otherwise it will be too slow
            if filled_vns+resolved_vns == self.n:
                return np.array([i.value for i in self.vns])


        return np.array([i.value for i in self.vns])
    
    def coupon_collector_erasure_decoder(self, max_iterations=100):
        """ Belief Propagation decoding for the general case (currently only works for BEC) )"""

        unresolved_vns = sum([1 for i in self.vns if len(i.value) > 1 ])
        resolved_vns = 0
        prev_resolved_vns = 0
        
        for iteration in range(max_iterations):
            
            for i in self.cns:
                for j in i.links:
                    sum_vns = 0
                    uncertainty_check = False
                    
                    for k in i.links:
                        if k != j:
                            if not type(self.vns[k].value) == int:
                                if len(self.vns[k].value) > 1:
                                    uncertainty_check = True
                                    break
                            if type(self.vns[k].value) == int:
                                sum_vns += self.vns[k].value    
                            else:
                                sum_vns += self.vns[k].value[0]
                    
                    if uncertainty_check:
                        continue
                    
                    if len(self.vns[k].value) > 1:
                        resolved_vns += 1    
                    
                    self.vns[j].value = [-sum_vns % self.ffdim]  

                if unresolved_vns == resolved_vns:
                    return np.array([i.value for i in self.vns])
            
            if prev_resolved_vns == resolved_vns:
                    return [i.value for i in self.vns]
            prev_resolved_vns = resolved_vns
            
        return np.array([i.value for i in self.vns])


    def coupon_collector_decoding(self, max_iterations=10000):
        """ Decodes for the case of symbol possiblities for each variable node 
            utilising Belief Propagation - may be worth doing for BEC as well 
        """
        
        unresolved_vns = sum([1 for i in self.vns if len(i.value) > 1 ])
        resolved_vns = 0
        total_possibilites = sum([len(i.value) for i in self.vns])
        
        while True:
            # Iterating through all the check nodes
            for i in self.cns:
                
                vn_vals = self.get_cn_link_values(i)
                
                for j in i.links:
                
                    vals = vn_vals.copy()
                    current_value = self.vns[j].value
                    vals = self.remove_from_array(vals, current_value)

                    possibilites = permuter(vals, self.ffdim, current_value)
                    new_values = set(current_value).intersection(set(possibilites))
                    self.vns[j].value = list(new_values)
                    
                    """
                    if len(new_values) < len(current_value) and len(possibilites) > 1:
                        print("I reached here")
                    """
                    if len(current_value) > 1 and len(new_values) == 1:
                        resolved_vns += 1
                    
                decoded_values = [i.value for i in self.vns]

                if unresolved_vns ==  resolved_vns and sum([len(i) for i in decoded_values]) == len(decoded_values):
                    return np.array([i.value for i in self.vns])
            
            if sum([len(i.value) for i in self.vns]) == total_possibilites:
                return [i.value for i in self.vns]
            
            total_possibilites = sum([len(i.value) for i in self.vns])
            
            prev_resolved_vns = resolved_vns   
        
        return [i.value for i in self.vns]

    def get_max_prob_codeword(self):
        """Returns the most possible Codeword using the probability likelihoods established in the VN's

        Returns:
            codeword (arr): n length codeword with symbols
        """

        codeword = np.zeros(len(self.vns))
        for i in range(len(self.vns)):
            vn_value = self.vns[i].value
            max_prob_symbol = list(vn_value).index(max(vn_value))
            codeword[i] = max_prob_symbol
        
        return codeword

    def validate_codeword(self, H, GF, max_prob_codeword):
        """ Checks if the most probable codeword is valid as a termination condition of qspa decoding """
        return not np.matmul(H, GF(max_prob_codeword.astype(int))).any()

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
                vn_vals = self.get_cn_link_values(i)
                for j in i.links:
                    vals = vn_vals.copy()
                    current_value = self.vns[j].value
                    vals = self.remove_from_array(vals, current_value)        
                    pdf = perform_convolutions(vals)
                    self.vns[j].value = pdf[idx_shuffle]
                
            # Check for max prob codeword and parity

            # CN Update
            for a in range(GF.order):
                for j in self.vns:
                    idxs = self.nonzero_rows[j]
                    for i in idxs:
                        # Initial Liklihoods
                        Q[i, j, a] = 1 * P[j, a]

                        # Don't understand this step - has to do with CN update
                        for t in idxs[idxs != i]:
                            Q[i, j, a] *= S[t, j, a]

                        # Normalization
                        Q[i, j, :] /= sum(Q[i, j, :])
            return Q

                
            # Break condition check - could make it a post VN check
            max_prob_codeword = self.get_max_prob_codeword()
            if self.validate_codeword(H, GF, max_prob_codeword):
                return max_prob_codeword

        return self.get_max_prob_codeword()