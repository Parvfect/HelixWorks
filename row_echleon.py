
# Lovely will need to write my own method

import sympy as sympy 
import numpy as np
from Hmatrixbaby import createHMatrix
# Assuming the condition to form a H matrix reduces a row echleon H to standard form


def get_reduced_row_echleon_form_H(H, ffdim=2):
    """ Returns the reduced row echleon form of H """
    H_rref = sympy.Matrix(H.T).rref()[0]

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

def find_columns_to_change(H, columns_to_change):
    """ Finds column switches to be made to get the H matrix in standard form """
    
    print(H)
    print(columns_to_change)
    
    n = H.shape[1]
    k = n - H.shape[0]
    switches = []
    # Columns will be independent in I don't need to mark them
    for i in columns_to_change.keys():
        for j in range(n):
            if np.all(H[:,j] == columns_to_change[i]):
                switches.append((i,j))
                continue

    if len(switches) != len(columns_to_change):
        print("Cannot convert to Standard Form")

    return switches

def make_column_switches(H, switches):
    """ Switches the columns of H to get it in standard form """
    for i in switches:
        t = H[:,i[0]]
        H[:,i[0]] = H[:,i[1]]
        H[:,i[1]] = t
    return H

def invert_standard_H(H, binary=True):
    """ Inverts the standard H matrix to get the Generator Matrix"""
    pass

H = createHMatrix(3, 6, 5, 10)
H_rref = get_reduced_row_echleon_form_H(H)
print(find_columns_to_change(H_rref, check_standard_form_variance(H_rref)))
#row_echleon(H.T)



