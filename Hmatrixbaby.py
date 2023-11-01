
import numpy as np
import sympy as sympy 
from pstats import Stats
import re
from cProfile import Profile

def len_unique_elements(arr):
    return len(set(arr))

class ParityCheckMatrix:
    """ Class for creating a random parity check matrix for given parameters """

    def __init__(self, dv, dc, k, n, ffdim=2):

        assert dv*(n) == dc*(n-k), "Invalid Parity Check Matrix Dimensions"
        self.dv = dv
        self.dc = dc
        self.k = k
        self.n = n
        self.ffdim = ffdim
    
    def get_H_arr(self):
        """ Gets Tanner Graph Connections for each Variable Node """

        # Initialize an array of size dv*n and fill it with numbers for each variable node's connections
        arr = np.arange(0, self.dv*self.n)
        flag = 0

        while True:
            flag = 0

            # Generate a random permutation of the array
            arr = np.random.permutation(arr)

            # Checking if each check node is connected to dv variable nodes
            t = [arr[i:i+self.dv] for i in range(0, len(arr), self.dv)]
            for i in t:
                i = i//self.dc
                if len_unique_elements(i) != self.dv:
                    flag +=1
            
            # Break if all check nodes are connected to dv variable nodes
            if flag == 0:
                break
        
        return arr

    def createHMatrix(self, Harr=None):
        """ Creates the H matrix from the Variable Node Connections of the Tanner Graph """
        
        if Harr is None:
            # Getting the Variable Node Connections
            Harr = self.get_H_arr()
        
        # Initialize H matrix
        H = np.zeros((self.n, self.n-self.k))

        # Fill H matrix where Variable Node is connected to Check Node
        for (i,j) in enumerate(Harr):
            H[i//self.dv, j//self.dc] = 1

        return H.T

    def get_reduced_row_echleon_form(self, H=None):
        """ Returns the reduced row echleon form of H """

        if H is None:
            H = self.createHMatrix()
        
        # Get the reduced row echleon form of H
        H_rref = np.array(sympy.Matrix(H.T).rref()[0])

        # Convert to finite field dimension
        for i in range(H_rref.shape[0]):
            for j in range(H_rref.shape[1]):
                H_rref[i,j] = H_rref[i,j] % self.ffdim
        
        # Convert to Integer Matrix
        return H_rref.astype(int)
    
    def get_standard_form(self, H_rref=None):
        """ Converts H to standard form from reduced row echleon form """

        def check_standard_form_variance(H):

            # Getting n-k dimension Identity Matrix
            I = np.eye(self.n - self.k)

            # Initializing columns to change
            columns_to_change = {}
            
            # Check if the last I_dim columns are I
            if np.all(H[:,self.k:self.n] == I):
                return None
            else:
                # Find the columns that need changing
                for i in range(self.k, self.n):
                    if not np.all(H[:,i] == I[:,i-self.k]):
                        columns_to_change[i] = I[:,i-self.k]
            
            return columns_to_change

        def switch_columns(H, columns_to_change):
            """ Finds and makes the necessary column switches to convert H to standard form """
            
            # Getting the positions of the columns to change
            column_positions = list(columns_to_change.keys())

            # Initializing variables
            changes_made, switches = [], []
            
            # Getting n-k dimension Identity Matrix
            I = np.eye(self.n-self.k)
            
            # Finding the columns that need to be swapped and swapping them, while fixing the changes made
            for i in column_positions:

                # Iterate through the columns of H
                for j in range(self.n):

                    # Check if the column is equal to the Identity Matrix column we are looking for
                    if np.all(H[:,j] == I[:, i-self.k]):

                        # If the column is after k and is fixed, continue
                        if j in changes_made:
                            continue

                        # Otherwise, switch the columns and add the switch to the list
                        changes_made.append(i)
                        switches.append((i,j))

                        # Switch the columns
                        t = H[:,i].copy()
                        H[:,i] = H[:,j]
                        H[:,j] = t
                        
                        break

            if not columns_to_change:
                print("Cannot convert to Standard Form")

            return H, switches

        # Check if H has been passed as a parameter
        if H_rref is None:
            H_rref = self.get_reduced_row_echleon_form()

        return switch_columns(H_rref, check_standard_form_variance(H_rref))

    def get_generator_matrix(self, H_standard=None, switches=None):
        """ Inverts the standard H matrix to get the Generator Matrix"""

        if H_standard is None:
            H_standard, switches = self.get_standard_form()

        # Getting the k-n dimensions
        n = H_standard.shape[1]
        k = n - H_standard.shape[0]

        # Getting the P matrix
        P = H_standard[:,0:k]

        # Getting the standard form of the G matrix
        G = np.hstack((np.eye(k), P.T))

        # Performing the column switches to retreive the original G matrix
        if switches: 
            switches = list(reversed(switches))
            for i in switches:
                t = G[:,i[0]].copy()
                G[:,i[0]] = G[:,i[1]]
                G[:,i[1]] = t
            
        return G

    def get_G_from_H(self, H):
        """ Create a Generator Matrix for a given H - should it be outside the class ? """
        H_rref = self.get_reduced_row_echleon_form(H)
        H_standard, switches = self.get_standard_form(H)
        return self.get_generator_matrix(H_standard, switches)


def generatorProfiling(dv, dc, k, n):
    with Profile() as profile:
        dv, dc, k, n = 3, 6, 100, 200
        H = ParityCheckMatrix(dv, dc, k, n)
        H.get_generator_matrix()
        # Everytime it generates a new one, might be much smarter to have a self method
        
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats("cumtime")
            .print_stats(10)
        )
        

        # Don't really need sympy beyond the rref method

if __name__ == "__main__":
    generatorProfiling(3, 6, 10, 20)