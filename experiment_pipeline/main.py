
from encoder_decoder import encoder, simulate_channel, decoder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def frame_error_rate(iterations=5, plot=True):

    read_lengths = np.arange(1,12)
    frame_error_rate = []

    for i in tqdm(read_lengths):
        correct_reads = 0
        for j in tqdm(range(iterations)):
            simulate_channel(i)
            if decoder():
                correct_reads += 1
        frame_error_rate.append((iterations - correct_reads)/iterations)
    
    if plot:
        plt.plot(read_lengths, frame_error_rate, 'o')
        plt.plot(read_lengths, frame_error_rate)
        plt.xlim(read_lengths[0], read_lengths[-1])
        plt.xticks(np.arange(read_lengths[0], read_lengths[-1]))
        plt.ylim(0,1)
        plt.title("Frame Error Rate for Decoded File")
        plt.xlabel("Read Length")
        plt.ylabel("Frame Error Rate")
        plt.show()

def fixed_read_length(read_length=9, iterations=1000):

    correct_reads = 0
    for i in tqdm(range(iterations)):
        simulate_channel(read_length)
        if decoder():
            correct_reads += 1
    
    print(f"Error Rate for {iterations} iterations for {read_length} read length is {(iterations - correct_reads)/iterations}")

fixed_read_length()

