
# Lovely will need to write my own method

import sympy as sympy 
import numpy as np

"""
Pivot the matrix
Find the pivot, the first non-zero entry in the first column of the matrix.
Interchange rows, moving the pivot row to the first row.
Multiply each element in the pivot row by the inverse of the pivot, so the pivot equals 1.
Add multiples of the pivot row to each of the lower rows, so every element in the pivot column of the lower rows equals 0.
"""

def row_echleon(arr):
    # Find the pivot - first non-zero entry in the first column of the matrix.
    # Interchange rows
    # Multiply each element in the pivot row by the inverse of the pivot
    # Add multiples of the pivot row to each of the lower rows so every element in the pivot column of the lower rows equals 0
    # Repeat until all rows have been pivoted

    pivot, pivot_index = 0, 0

    for i in range(20):
        for i in range(arr.shape[1]):
            if arr[0,i] != 0:
                pivot = arr[0,i]
                pivot_index = i
                break

        # Interchange rows
        arr[0], arr[pivot_index] = arr[pivot_index], arr[0]

        # Multiply each element in the pivot row by the inverse of the pivot
        if pivot != 1:
            for i in range(len(arr[0])):
                arr[0][i] = arr[0][i] / pivot

        # Add multiples of the pivot row to each of the lower rows so every element in the pivot column of the lower rows equals 0
        for i in range(1, len(arr)):
            if arr[i][pivot_index] != 0:
                multiple = arr[i][pivot_index]
                for j in range(len(arr[i])):
                    arr[i][j] = arr[i][j] - (multiple * arr[0][j])
       
    print(arr)

t = np.random.rand(3,3)
print(t)
row_echleon(t)

