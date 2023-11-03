
import math
import numpy as np
import random

def generate_input_arr(length_arr):
    """ Generates an array of bits of length length_arr """
    return [random.randint(0, 1) for i in range(length_arr)]

def generate_erasures(input_arr, probability):  
    """ Simulates Binary Erasure Channel by replacing bits with erasures with probability probability"""    
    
    #one_erasure = list(input_arr)
    #one_erasure[0] = np.nan
    #return one_erasure

    return [i if random.random() > probability else np.nan for i in input_arr]

