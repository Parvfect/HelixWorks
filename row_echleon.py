
# Lovely will need to write my own method

import sympy as sympy 
import numpy as np
from Hmatrixbaby import ParityCheckMatrix
# Assuming the condition to form a H matrix reduces a row echleon H to standard form


def get_reduced_row_echleon_form_H(H, ffdim=2):
    """ Returns the reduced row echleon form of H """
    H_rref = sympy.Matrix(H).rref()[0]
    # Convert to finite field dimension
    for i in range(H_rref.shape[0]):
        for j in range(H_rref.shape[1]):
            H_rref[i,j] = H_rref[i,j]%ffdim
     
    # Convert to Integer Matrix
    return np.array(H_rref).astype(int)

def check_standard_form_variance(H):
    """ Checks if the H matrix is in standard form and returns columns that need changing """

    n = H.shape[1]
    k = n - H.shape[0]
    shape = H.shape
    I_dim = n-k 
    I = np.eye(I_dim)
    columns_to_change = {}
    rows = shape[1]
    
    # Check if the last I_dim columns are I
    if np.all(H[:,k:n] == I):
        return None
    else:
        # Find the columns that need changing
        for i in range(k, n):
            if not np.all(H[:,i] == I[:,i-k]):
                columns_to_change[i] = I[:,i-k]
    
    return columns_to_change

def switch_columns(H, columns_to_change):
    """ Finds and makes the necessary column switches to convert H to standard form """
 
    n = H.shape[1]
    k = n - H.shape[0]
    column_positions = list(columns_to_change.keys())
    changes_made = []
    switches = []
    I = np.eye(n-k)
    
    for i in column_positions:
        for j in range(n):
            if np.all(H[:,j] == I[:, i-k]):
                if j in changes_made:
                    continue
                changes_made.append(i)
                switches.append((i,j))
                t = H[:,i].copy()
                H[:,i] = H[:,j]
                H[:,j] = t
                break

    if not columns_to_change:
        print("Cannot convert to Standard Form")

    return H, switches

def standard_H_to_G(H, ffdim=2, switches = None):
    """ Inverts the standard H matrix to get the Generator Matrix"""
    n = H.shape[1]
    k = n - H.shape[0]
    P = H[:,0:k]
    G = np.hstack((np.eye(k), P.T))
    
    # Since switches made forward, need to reverse list to undo
    switches = list(reversed(switches))
    if switches: 
        for i in switches:
            t = G[:,i[0]].copy()
            G[:,i[0]] = G[:,i[1]]
            G[:,i[1]] = t

    return G

def display_results(dv=3, dc=6, k=5, n=10):

    print("dv = {}\n dc = {}\n k = {}\n n = {}".format(dv, dc, k, n))
    H = ParityCheckMatrix(dv, dc, k, n).createHMatrix()
    print("Initial Parity Matrix\n")
    print(H)
    print()

    H_rref = get_reduced_row_echleon_form_H(H)
    H_st, switches = switch_columns(H_rref, check_standard_form_variance(H_rref))
    print("Standard Form of H\n")
    print(H_st)
    print()

    print("Generator Matrix\n")
    G = standard_H_to_G(H_st, switches=switches)
    print(G)
    print()

    print("G.H^T\n")
    print(np.dot(G, H.T) % 2)


if __name__ == "__main__":
    display_results()
