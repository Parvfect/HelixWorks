
from encoder_decoder import encoder, decoder
import pickle
from test_dependencies import conduct_all_tests

def run_pipeline(channel_input, channel_output):
    params = encoder(channel_input)
    return decoder(channel_output, params)

conduct_all_tests()
print("All Tests Passed Succesfully")
channel_input = pickle.load(open('sample_input/channel_input.p', 'rb')) # Replace Channel Input
channel_output = pickle.load(open('sample_input/channel_output_1000_without_substitution.p', 'rb')) # Replace Channel Output
print(run_pipeline(channel_input, channel_output))