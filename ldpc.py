
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def generate_tanner_graph(parity_check_matrix):
    """ Creates Tanner Graph for Parity Check Matrix - needs to be tested"""
    G = nx.Graph()

    rows = parity_check_matrix.shape[0]
    cols = parity_check_matrix.shape[1]

    # For each row add a check node
    for i in range(rows):
        G.add_node(i, bipartite=0)

    # For each column add a variable node
    for i in range(cols):
        G.add_node(i + rows, bipartite=1)

    
    for i in range(rows):
        for j in range(cols):
            if parity_check_matrix[i][j] == 1:
                G.add_edge(i, j+rows, weight=1)
    # Add all the variable nodes and then add check nodes as edge
   
   
    nx.draw(G, with_labels=True)
    plt.show()
    return G

def create_parity_check_matrix(i,j):
    """ Using BEC create redundnacy code using erasure probability to generate partiy check matrix"""
    pass


