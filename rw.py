
# Need to add an adaptive iteration 
# Get the previous results and under a specific variance? 
# Max Iterations to 100 until it converges
# How do we know it converges?
# 

def frame_error_rate(self, iterations=20, plot=False):
        """ Get the FER for the Tanner Graph """

        erasure_probabilities = np.arange(0,1,0.05)
        frame_error_rate = []
        input_arr = np.zeros(self.n)

        for i in tqdm(erasure_probabilities):
            counter = 0
            for j in range(iterations):
                
                self.establish_connections()
                # Assigning values to Variable Nodes after generating erasures in zero array
                self.assign_values(generate_erasures(input_arr, i))

                # Getting the average error rates for iteration runs
                if np.all(self.bec_decode() == input_arr):
                    counter += 1
            
            # Calculate Error Rate and append to list
            error_rate = (iterations - counter)/iterations
            frame_error_rate.append(error_rate)
    