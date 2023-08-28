

def column_permuter(H, dv, dc, k, n):
    """ Randomly Add a 1 to the columns that don't have a 1 """
        
        for i in range(n-k):
            if np.count_nonzero(H[:,i] == 1) == dv:
                continue
            else:
                H[random.randint(0, n-1), i] = 1
    
        return H

def row_permuter(H, dv, dc, k, n):
    """ Randomly Add a 1 to the rows that don't have a 1 """
        
        for i in range(n):
            if np.count_nonzero(H[i,:] == 1) == dc:
                continue
            else:
                H[i, random.randint(0, n-k-1)] = 1
    
        return H

def getH(k,n, dv, dc):
    
    assert dv*(n) == dc*(n-k)

    H = np.zeros((n,n-k))
    counter = 0 

    while True:

        if np.count_nonzero(H) == dv*(n) == dc*(n-k):
            break

        if counter % 2 == 0:
            H = column_permuter(H, dv, dc, k, n)
        else:
            H = row_permuter(H, dv, dc, k, n)

    return H

print(getH(7, 14, 3, 6))