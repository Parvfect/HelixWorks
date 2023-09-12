
import numpy as np
from Hmatrixbaby import createHMatrix, getH
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt


class Node:

    def __init__(self, no_connections, identifier):
        self.value = 0
        self.links = np.zeros(no_connections)
        self.identifier = identifier

    def add_link(self, node):
        """ Adds a link to the node. Throws an error if the node is full """
        
        # Check if node is full
        if np.all(self.links):
            raise ValueError("Node is full")

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

    def __init__(self, dv, dc, k, n):
        self.vns = [VariableNode(dv, i) for i in range(n)]
        self.cns = [CheckNode(dc, i) for i in range(n-k)]
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
    
    def establish_connections(self):
        """ Establishes connections between variable nodes and check nodes """
        Harr = getH(self.dv, self.dc, self.k, self.n)%4
        
        # Divide Harr into dv parts
        Harr = np.split(Harr, self.n//self.dv)

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
                G.add_edge(i, k.identifier + rows, weight=1)
    
    
        nx.draw(G, with_labels=True)
        plt.show()
        return G

t = TannerGraph(2, 4, 3, 6)
t.establish_connections()