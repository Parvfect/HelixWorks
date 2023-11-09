
import numpy as np


def permuter(arr, ffield):

    possibilities = set(arr[0])
    new_possibilities = set()
    counter = 1
    for i in range(1, len(arr)):
        for k in possibilities:
            for j in arr[i]:
                new_possibilities.add(-((j + k)%ffield)%ffield)
        possibilities = new_possibilities
        new_possibilities = set()

    return possibilities

        
        

print(permuter([[1,2,3], [4,5,6], [7,8,9]]))