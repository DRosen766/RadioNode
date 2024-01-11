# Generate and Send Test IQ Data to S3 Bucket

import csv
import numpy as np
from tqdm import tqdm
import boto3
import json
import os

# Datagen Imports
from cloudd_rf.iqdata_gen import iqdata_gen
from cloudd_rf.metadata_gen import metadata_gen
from cloudd_rf.imagedata_gen import imagedata_gen

# Dataset Parameters
rand_seed = 10                                            # Seed for the random number generator for repeatability (note: script must use all of the same generation parameter bounds and values).
num_testing_examples = 200                                     # Number of different radio frequency spectrum examples to be created for the dataset.
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
metadata_file_name = 'test_metadata.csv'
fid = open(f"test_data/{metadata_file_name}", 'w', encoding='UTF8', newline='')
writer = csv.writer(fid)
# header = ['file_name', 'Center Frequency', 'Bandwidth', 'Start Time (samples)', 'Stop Time (samples)', 'SNR', 'Signal Type']
# writer.writerow(header)

# Initalize Generators
rng = np.random.default_rng(rand_seed)
meta_gen = metadata_gen(obs_int, rng)
iq_gen = iqdata_gen(obs_int, rng)
im_gen = imagedata_gen(image_width, image_height, fft_size, overlap)

# connect to bucket
bucket = boto3.resource("s3").Bucket("test-radio-bucket-766318")
bucket.download_file("label_map.json", "label_map.json")
label_map = json.load(open("label_map.json"))

# iterate through number of testing examples
for k in tqdm(range(num_testing_examples)):
    
    # create metadata
    burst_metadata = meta_gen.gen_metadata(max_sigs, max_trials, bandwidth_bounds, cent_freq_bounds, start_bounds, duration_bounds, snr_bounds, sig_types, allow_collisions)
    # generate iq data
    iq_data, burst_metadata = iq_gen.gen_iq(burst_metadata)
    iq_data_clone = np.copy(iq_data)
    iq_data = np.concatenate((np.real(iq_data),np.imag(iq_data_clone)))
    iqdata_file_name = 'example_' + str(k) + '.dat'
    iq_data.tofile(f"test_data/iqdata/temp_{iqdata_file_name}")
    bucket.upload_file(f"test_data/iqdata/temp_{iqdata_file_name}", f"test/iqdata/{iqdata_file_name}")
    # delete local iqdata file
    os.remove(f"test_data/iqdata/temp_{iqdata_file_name}")

    for metadata in burst_metadata:
        # extract and send params
        metadata_info = {"metadata_file_name" : metadata_file_name,
                "iqdata_file_name":iqdata_file_name, 
                  "cent_freq" : metadata.cent_freq, 
                  "bandwidth" : metadata.bandwidth, 
                  "snr" : metadata.snr, 
                  "sig_type" : str(metadata.sig_type)}
        writer.writerow(list(metadata_info.values()))

# write metadata file to bucket
bucket.upload_file(f"test_data/{metadata_file_name}", f"test/{metadata_file_name}")



