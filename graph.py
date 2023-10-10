
import numpy as np
from Hmatrixbaby import createHMatrix, getH
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

# Graph Implementation - similar to Adjacency List
# Need to speeden up the decoding for the Graphs

class Node:

    def __init__(self, no_connections, identifier):
        self.value = 0
        self.links = np.zeros(no_connections)
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

    def __init__(self, dv, dc, k, n):
        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
    
    def establish_connections(self):
        """ Establishes connections between variable nodes and check nodes """
        self.Harr = getH(self.dv, self.dc, self.k, self.n)
        Harr = self.Harr//self.dc

        # Divide Harr into dv parts
        Harr = [Harr[i:i+self.dv] for i in range(0, len(Harr), self.dv)]

        # Establish connections
        for (i,j) in enumerate(Harr):
            for k in j:
                self.vns[i].add_link(self.cns[k])
                self.cns[k].add_link(self.vns[i])

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

    def bec_decode(self, max_iterations=100):
        """ Assuming VNs have been initialized with values, perform BEC decoding """

        filled_vns = sum([1 for i in self.vns if not np.isnan(i.value)])
        resolved_vns = 0
        resolved_vns_temp = 0
    
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
                        if self.vns[int(k)].value:
                            erasure_index = k
                        else:
                            sum_links += self.vns[int(k)].value
                    
                    # Replace erasure with sum modulo 2
                    self.vns[int(erasure_index)].value = sum_links % 2
                    resolved_vns+=1
            
            if resolved_vns == resolved_vns_temp or filled_vns+resolved_vns == self.n:
                break

            resolved_vns_temp = resolved_vns

        return np.array([i.value for i in self.vns])

    def assign_values(self, arr):   

        assert len(arr) == len(self.vns) 

        for i in range(len(arr)):
            self.vns[i].value = arr[i]

    def frame_error_rate(self, iterations=50, plot=False, ensemble=True):
        """ Get the FER for the Tanner Graph """

        erasure_probabilities = np.arange(0,1,0.05)
        frame_error_rate = []
        input_arr = np.zeros(self.n)
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
                
                """
                # Adaptive Iterator
                if prev_error - ((iterations - counter)/iterations) < 0.001:
                    break

                prev_error = (iterations - counter)/iterations
                """

            # Calculate Error Rate and append to list
            error_rate = (iterations - counter)/iterations
            frame_error_rate.append(error_rate)
        
        if plot:
            plt.plot(erasure_probabilities, frame_error_rate, label = "({},{})".format(self.k, self.n))
            plt.title("Frame Error Rate for BEC for {}-{}  {}-{} LDPC Code".format(self.k, self.n, self.dv, self.dc))
            plt.ylabel("Frame Error Rate")
            plt.xlabel("Erasure Probability")

            # Displaying final figure
            plt.legend()
            #plt.ylim(0,1)

        return frame_error_rate



with Profile() as profile:
    t = TannerGraph(3, 6, 100, 200)
    t.frame_error_rate(plot=True, ensemble=False)
    t = TannerGraph(3, 6, 500, 1000)
    t.frame_error_rate(plot=True, ensemble=False)
    t = TannerGraph(3, 6, 1000, 2000)
    t.frame_error_rate(plot=True, ensemble=False)
    t = TannerGraph(3, 6, 2000, 4000)
    t.frame_error_rate(plot=True, ensemble=False)
    t = TannerGraph(3, 6, 4000, 8000)
    t.frame_error_rate(plot=True, ensemble=False)
    t = TannerGraph(3, 6, 8000, 16000)
    t.frame_error_rate(plot=True, ensemble=False)
    
    # Get the Threshold
    threshold = threshold_binary_search(self.dv, self.dc)
    plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")

    
    plt.show()
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_stats(10)
    )