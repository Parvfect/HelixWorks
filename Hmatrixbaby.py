
import numpy as np


def len_unique_elements(arr):
    return len(set(arr))

def getH(dv, dc, k, n):
    assert dv*(n) == dc*(n-k)

    arr = np.arange(0, dv*n)
    flag = 0

    while True:
        flag = 0
        arr = np.random.permutation(arr)
    
        t = [arr[i:i+dv] for i in range(0, len(arr), dv)]
        # For each part check if it it connected to a unique check node
        # If not, permute the part

        
        for i in t:
            i = i//dc
            #if len(np.unique(i)) != dv:
            if len_unique_elements(i) != dv:
                flag +=1
        
        if flag == 0:
            break
        
    return arr

def createHMatrix(dv, dc, k, n):

    Harr = getH(dv, dc, k, n)
    # Create H matrix from Harr
    H = np.zeros((n, n-k))
    for (i,j) in enumerate(Harr):
        H[i//dv, j//dc] = 1
    return H



# Convert H to Matrix form and feed into methods to see results