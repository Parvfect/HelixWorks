
from dependencies import *

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
    assert create_mask([3,4,5], [2,69,5]) == [69,65,0]
    print("Create Mask Test Passed Succesfully!")
    pass

def test_generate_symbol_possibilites():
    symbols = choose_symbols(8,4)
    motifs = np.arange(1,9)
    transmission = [symbols[3], symbols[4], symbols[67], symbols[69]]
    assert generate_symbol_possibilites(transmission, symbols, motifs, 4) == [[3], [4], [67], [68]]
    pass

def test_invert_mask():
    pass

def test_filter_symbols():
    pass

def conduct_all_tests():
    #test_choose_symbols()
    #test_save_parameters()
    #test_load_parameters()
    #test_create_mask()
    test_generate_symbol_possibilites()
    test_invert_mask()
    test_filter_symbols()

if __name__ == "__main__":
    conduct_all_tests()
    print("All Tests Passed Succesfully!")
