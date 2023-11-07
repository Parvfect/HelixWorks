
import numpy as np
import time 

def G_H_test(G, H, ffdim=2):
    """ Checks if the generator matrix and parity check matrix are compatible"""
    
    if np.any(np.dot(G, H.T) % ffdim):
        print("G and H are not compatible")
        exit()


def permuter(arr, ffield):
    """ Assuming input of multi dim array, returns all permutations """

    possibilites = set()

    def helper(arr, sum):
        
        if not arr:
            possibilites.add(-(sum % ffield) % ffield)
        else:
            for i in arr[0]:
                helper(arr[1:], sum + i)
            return
        return
    
    helper(arr, 0)
    return possibilites
        

def random_picker_tester():
    """ Picks a random element from an array """

    arr = [np.arange(4) for i in range(6)]
    permutations = []
    picks = 4
    
    starttime = time.time()
    # 6 layers deep will probs be max
    for i in range(picks):
        for j in range(picks):
            for k in range(picks):
                for t in range(picks):
                    for s in range(picks):
                        for l in range(picks):
                            permutations.append([arr[0][i],arr[1][j],arr[2][k],arr[3][t], arr[4][s], arr[5][l]])
    stoptime = time.time()
    print("Time taken for for loop ", stoptime - starttime)

    random_permutations = set()
    length_permutations = len(permutations)

    starttime = time.time()
    while len(random_permutations) < length_permutations:
        random_permutations.add(tuple(np.random.choice(arr[i]) for i in range(6)))
    stoptime = time.time()
    print("Time taken for random method ", stoptime - starttime)
    

permuter()