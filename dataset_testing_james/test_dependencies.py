
from dependencies import *
from encoder_decoder import encoder, decoder
import pickle as pickle

def test_choose_symbols():
    assert len(choose_symbols(8,4)) == 70
    print("Choose Symbols Test Passed Succesfully!")

def test_save_parameters():
    save_parameters()
    print("Save Parameters Test Passed Succesfully!")

def test_load_parameters():
    Harr, H, G, graph = load_parameters()
    assert len(graph.vns) == G.shape[1] == H.shape[1]
    print("Load Parameters Test Passed Succesfully!")
    pass

def test_create_mask():
    assert create_mask([2,69,5], [3,4,5]) == [69,65,0]
    print("Create Mask Test Passed Succesfully!")
    pass

def test_generate_symbol_possibilites():
    symbols = choose_symbols(8,4)
    motifs = np.arange(1,9)
    transmission = [symbols[3], symbols[4], symbols[67], symbols[69]]
    assert generate_symbol_possibilites(transmission, symbols, motifs, 4) == [[3], [4], [67], [69]]
    print("Generate Symbol Test Passed Successfully!")
    pass

def test_invert_mask():
    assert invert_mask([[2],[69],[5]], [69,65,0]) == [[1],[64],[5]]
    print("Invert mask Test Passed Succesfully!")
    pass

def test_filter_symbols():
    assert filter_symbols([[3],[4],[5], [69, 66, 65], [71]]) == [[3], [4], [5], [66, 65], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]]
    print("Filter symbols test passed succesfully!")
    pass

def test_encoder():
    channel_input = pickle.load(open('sample_input/channel_input.p', 'rb'))
    channel_input_symbols, mask, Harr, H, G, graph, C = encoder(channel_input)
    assert len(C) == len(mask) == len(channel_input_symbols)
    print("Encoder Tests Passed")

def test_pipeline():
    channel_input = pickle.load(open('sample_input/channel_input.p', 'rb'))
    channel_output = pickle.load(open('sample_input/channel_output_1000_without_substitution.p', 'rb'))
    params = encoder(channel_input)
    assert decoder(channel_output, params) == True
    print("Decoder Tests Passed")

def conduct_all_tests():
    test_choose_symbols()
    test_save_parameters()
    test_load_parameters()
    test_create_mask()
    test_generate_symbol_possibilites()
    test_invert_mask()
    test_filter_symbols()
    test_encoder()
    test_pipeline()

if __name__ == "__main__":
    conduct_all_tests()
    print("All Tests Passed Succesfully!")
