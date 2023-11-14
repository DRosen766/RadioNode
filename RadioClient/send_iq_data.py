# Generate and Send IQ Data to Radio Server

from http import client
import requests
import csv
import time
import numpy as np
import matplotlib.pyplot as pyplot
import torch
from torch import Tensor
from tqdm import tqdm
import pandas as pd
# Datagen Imports
from cloudd_rf.iqdata_gen import iqdata_gen
from cloudd_rf.metadata_gen import metadata_gen
from cloudd_rf.imagedata_gen import imagedata_gen
# Dataset Parameters
rand_seed = 10                                            # Seed for the random number generator for repeatability (note: script must use all of the same generation parameter bounds and values).
num_training_examples = 4                                     # Number of different radio frequency spectrum examples to be created for the dataset.
num_testing_examples = 1                                      # Number of different radio frequency spectrum examples to be created for the dataset.
max_sigs = 1
# Spectrum Parameters
obs_int = 1024                                              # Observation length of the spectrum for each example.
max_trials = 10000                                          # How many tries the generator will attempt to fit the maximum number of signals in the spectrum (note: if allow_collision=True, this parameter doesn't do anything).
bandwidth_bounds = [0.25, 0.5]                              # Bandwidth min and max of any created signal with possible range (0.0, 1.0).
cent_freq_bounds = [-0.25, 0.25]                            # Center Frequency min and max of any created signal with possible range [-0.5, 0.5].
start_bounds = [0, int(0.25*obs_int)]                       # Starting sample min and max of any created signal with possible range [0, obs_int].
duration_bounds = [int(0.5*obs_int), int(0.75*obs_int)]     # Duration (in samples) min and max of any created signal with possible range (0, obs_int].
snr_bounds = [5, 20]                                        # SNR (in dB) min and max of any created signal.
sig_types = [['2-ASK',  ['ask', 2]],                        # Potential signal types of any created signal.
             ['4-ASK',  ['ask', 4]],
             ['8-ASK',  ['ask', 8]],
             ['BPSK',   ['psk', 2]], 
             ['QPSK',   ['psk', 4]],
             ['8-PSK',  ['psk', 8]],
             ['16-QAM', ['qam', 16]],
             ['64-QAM', ['qam', 64]],
             ['Constant Tone', ['constant']],
             ['P-FMCW', ['p_fmcw']],
             ['N-FMCW', ['n_fmcw']]]

allow_collisions = False                                    # True: Signals can be overlapped in time and/or frequency. False: No overlap in signals but may not generate max_sigs.

# Image Parameters
image_width = 1000                                          # Image width (in pixels).
image_height = 500                                          # Image height (in pixels).
fft_size = 256                                              # FFT size used to generate the spectrogram image.
overlap = 255                                               # FFT overlap used to generate the spectrogram image.

# Initialize Metadata File
metadata_file_name = 'metadata.csv'
# fid = open(metadata_file_name, 'w', encoding='UTF8', newline='')
# writer = csv.writer(fid)
# header = ['file_name', 'Center Frequency', 'Bandwidth', 'Start Time (samples)', 'Stop Time (samples)', 'SNR', 'Signal Type']
# writer.writerow(header)

# Initalize Generators
rng = np.random.default_rng(rand_seed)

meta_gen = metadata_gen(obs_int, rng)
iq_gen = iqdata_gen(obs_int, rng)
im_gen = imagedata_gen(image_width, image_height, fft_size, overlap)
label_map = {"['2-ASK', ['ask', 2]]" : 0,
             "['4-ASK', ['ask', 4]]" : 1,
             "['8-ASK', ['ask', 8]]" : 2,
             "['BPSK', ['psk', 2]]" : 3, 
             "['QPSK', ['psk', 4]]" : 4,
             "['8-PSK', ['psk', 8]]" : 5,
             "['16-QAM', ['qam', 16]]" : 6,
             "['64-QAM', ['qam', 64]]" : 7,
             "['Constant Tone', ['constant']]" : 8,
             "['P-FMCW', ['p_fmcw']]" : 9,
             "['N-FMCW', ['n_fmcw']]" : 10}

# Init http
# ip_addr = "18.191.171.241:5000"
ip_addr = "http://127.0.0.1:5000"
metadata_file_name = 'metadata.csv'
fid = open(metadata_file_name, 'w', encoding='UTF8', newline='')
writer = csv.writer(fid)



training_examples = []
training_sig_types = []
training_snr = []
testing_examples = []
testing_sig_types = []
testing_snr = []

# Create Dataset
radioConnection = requests.Session()
for k in tqdm(range(num_training_examples + num_testing_examples)):
    start_time = time.time()
    burst_metadata = meta_gen.gen_metadata(max_sigs, max_trials, bandwidth_bounds, cent_freq_bounds, start_bounds, duration_bounds, snr_bounds, sig_types, allow_collisions)
    iq_data, burst_metadata = iq_gen.gen_iq(burst_metadata)
    iq_data_clone = np.copy(iq_data)
    iq_data = np.concatenate((np.real(iq_data),np.imag(iq_data_clone)))
    
    radioConnection.put("{}{}".format(ip_addr, "/send_iq"), params={"k":k}, data=bytearray(iq_data))

    iqdata_file_name = 'server_data/iqdata/example_' + str(k+1) + '.dat'
    for metadata in burst_metadata:
        # extract and send params
        params = {"metadata_file_name" : metadata_file_name,
                "iqdata_file_name":iqdata_file_name, 
                  "cent_freq" : metadata.cent_freq, 
                  "bandwidth" : metadata.bandwidth, 
                  "snr" : metadata.snr, 
                  "sig_type" : metadata.sig_type}
        radioConnection.put("{}{}".format(ip_addr, "/send_metadata"), params= params)

# after all metadata has been created, load it to s3 with endpoint
radioConnection.post("{}{}".format(ip_addr, "/upload_metadata"), params= {"metadata_file_name":metadata_file_name})


