
from tanner import VariableTannerGraph, conv_circ
import numpy as np

class TannerQSPA(VariableTannerGraph):

    def __init__(self, dv, dc, k, n, ffdim=2):
        super().__init__(dv,dc,k,n,ffdim)

        self.vn_links = {}
        self.cn_links = {}

    def decode(self, symbols_likelihood_arr, H, GF, max_iterations=50):

        self.GF = GF
                
        # Additive inverse of GF Field
        self.idx_shuffle = np.array([
            (GF.order - a) % GF.order for a in range(GF.order)
        ])
        
        # Setting the VN Links with the initial symbol likelihoods
        self.initialize_vn_links(self.P)

        # Initilizing the CN Links
        self.initialize_cn_links()

        prev_max_prob_codeword = self.get_max_prob_codeword(self.P, GF)

        iterations = 0

        #for i in range(max_iterations):
        while(True):
            
            self.cn_update_qspa(copy_links)

            max_prob_codeword = self.get_max_prob_codeword(self.P, GF)

            parity = not np.matmul(H, max_prob_codeword).any()
            if parity:
                print("Decoding converges")
                return max_prob_codeword

            self.vn_update_qspa()

            if np.array_equal(max_prob_codeword, prev_max_prob_codeword) or iterations > max_iterations:
                break
            
            prev_max_prob_codeword = max_prob_codeword

            iterations+=1
            print(f"Iteration {iterations}")

        print("Decoding does not converge")
        return max_prob_codeword
    
    def get_max_prob_codeword(self, P, GF):
        """Calculates the most possible Codeword using the probability likelihoods established in the VN's and influenced by the initial probability likelihoods.

        Returns:
            codeword (arr): n length most probable codeword with symbols
        """
        z = np.zeros(self.n)
        for j in self.vns:
            vn_index = j.identifier
            probs = 1 * P[vn_index]
            for a in range(GF.order):
                for i in j.links:
                    probs[a] *= self.get_cn_link_weight(i, vn_index)[a]
            z[vn_index] = np.argmax(probs) 
        z = GF(z.astype(int))
        return z

    def initialize_vn_links(self, P):
        """ Sets all the links from a VN to the VN initial likelihood array """
        for i in self.vns:
            vn_index = i.identifier
            for j in i.links:
                self.update_link_weight(j, vn_index, 1*P[vn_index])

    def cn_update_qspa(self):
        """ CN Update for the QSPA Decoder. For each CN, performs convolutions for individual VN's as per the remaining links and updates the individual link values after finishing each link. Repeats for all the CN's """
        
        #Update CN links using VN Links

        for i in self.cns:
            cn_index = i.identifier
            vns = i.links
            new_pdfs = []
            for j in vns:
                conv_indices = [idx for idx in vns if idx != j]
                pdf = conv_circ(self.get_vn_link_weight(cn_index, conv_indices[0]), self.get_vn_link_weight(cn_index, conv_indices[1]))
                for indice in conv_indices[2:]:
                    pdf = conv_circ(pdf, self.get_vn_link_weight(cn_index, indice))
                #new_pdfs.append(pdf[self.idx_shuffle])
                self.update_cn_link_weight(i,j,pdf[self.idx_shuffle]) 

    def vn_update_qspa(self):
        """ Updates the CN as per the QSPA Decoding. Conditional Probability of a Symbol being favoured yadayada """

        # Use the CN links to update the VN links by taking the favoured probabilities

        copy_links = self.links.copy()
        for a in range(self.GF.order):
            for j in self.vns:
                vn_index = j.identifier
                for i in j.links:
                    copy_links[(i, vn_index)][a] = self.P[vn_index][a]
                    for t in j.links[j.links!=i]:
                        copy_links[(i,vn_index)][a] *= self.get_link_weight(t, vn_index)[a]

                    sum_copy_links = np.einsum('i->', copy_links[i, vn_index]) # Seems to be twice as fast or smth
                    #sum_copy_links = np.sum(copy_links[i, vn_index])
                    #sum_copy_links = sum(copy_links[i, vn_index])
                    copy_links[i, vn_index] = copy_links[i, vn_index]/sum_copy_links
                    
        self.links = copy_links
        


