
from encoder_decoder import encoder, decoder
import pickle

def run_pipeline(channel_input, channel_output):
    params = encoder(channel_input)
    return decoder(channel_output, params)

channel_input = pickle.load(open('sample_input/channel_input.p', 'rb'))
channel_output = pickle.load(open('sample_input/channel_output_1000_without_substitution.p', 'rb'))
print(run_pipeline(channel_input, channel_output))