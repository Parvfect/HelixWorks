

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
        
        # Check if connections are non-uniform
        if type(dv) == list:
            assert len(dv) == n and len(dc) == n-k
            self.vns = [VariableNode(dv[i], i) for i in range(n)]
            self.cns = [CheckNode(dc[i], i) for i in range(n-k)]

        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]

        # For the singular case - it remains as an integer, but for the Changing Case it goes to a list, need to make sure that does not break everything
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
        total_possibilites = 0
        
        while True:
            # Iterating through all the check nodes
            for i in self.cns:
                
                vn_vals = self.get_cn_link_values(i)
                
                for j in i.links:
                
                    vals = vn_vals.copy()
                    current_value = self.vns[j].value
                    vals.remove(current_value)
                    
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
            

            # Need to confirm the break condition is right
            # If we haven't increased certainty of any of the VNs as compared to the previous iteration, we break
            total_possibilites = sum(len(i.value) for i in self.vns)
            if sum(len(i.value) for i in self.vns) == total_possibilites:
                return [i.value for i in self.vns]

            #if prev_resolved_vns == resolved_vns:
            #       return [i.value for i in self.vns]
            
            total_possibilites = sum(len(i.value) for i in self.vns)
            
            prev_resolved_vns = resolved_vns
                
        
        return [i.value for i in self.vns]


    def frame_error_rate(self, input_arr=None, iterations=50, plot=False, ensemble=False, establish_connections=True, label=None):
        """ Get the FER for the Tanner Graph """

        erasure_probabilities = np.arange(0,1,0.05)
        frame_error_rate = []
        
        # Creating an all zero vector for input in case no input is passed
        if input_arr is None:
            input_arr = np.zeros(self.n)
        
        if establish_connections:
            self.establish_connections()

        for i in tqdm(erasure_probabilities):
            counter = 0
            prev_error = 5
            for j in range(iterations):
                
                if ensemble:
                    self.establish_connections()

                # Assigning values to Variable Nodes after generating erasures in zero array
                self.assign_values(generate_erasures(input_arr, i))

                # Getting the average error rates for iteration runs
                if np.all(self.bec_decode() == input_arr):
                    counter += 1    

            # Calculate Error Rate and append to list
            error_rate = (iterations - counter)/iterations
            frame_error_rate.append(error_rate)
        
        if plot:
            plt.plot(erasure_probabilities, frame_error_rate, label = "({},{}) {}".format(self.k, self.n, label))
            plt.title("Frame Error Rate for BEC for {}-{}  {}-{} LDPC Code".format(self.k, self.n, self.dv, self.dc))
            plt.ylabel("Frame Error Rate")
            plt.xlabel("Erasure Probability")

            # Displaying final figure
            plt.legend()
            plt.ylim(0,1)
            plt.show()

        return frame_error_rate


def test():
     with Profile() as profile:
        dv, dc, k, n = 3, 6, 1000,
        2000
        t = TannerGraph(dv, dc, k, n)
        t.frame_error_rate(plot=True, ensemble=False)
        
        # Get the Threshold
        threshold = threshold_binary_search(dv, dc)
        plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")

        
        plt.show()
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats("cumtime")
            .print_stats(10)
        )

if __name__ == "__main__":
    test_arr = [[0], [3], [2], [0], [3], [2], [2, 4], [2, 4], [0, 1, 2], [0]]
