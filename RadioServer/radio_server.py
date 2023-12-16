from flask import Flask, request
from werkzeug.utils import secure_filename
import boto3
from sagemaker.estimator import Estimator
import array
import numpy as np
import csv

app = Flask(__name__)

iq_data_sample = []

@app.put("/send_iq")
def send_iq():
    k = int(request.args.get("k"))
    iqdata_file_name = str(request.args.get("iqdata_file_name"))

    # extract iq data from request
    iq_data = request.get_data()
    iq_data = np.array(array.array('f', iq_data))

    # move iq_data to file
    iq_data.tofile(iqdata_file_name)

    # upload file to s3 bucket
    boto3.resource('s3').Bucket('radio-bucket-766318').upload_file(iqdata_file_name, iqdata_file_name)
    return "created {}".format(iqdata_file_name)

@app.put("/send_metadata")
def send_metadata():
    # extract params
    metadata_file_name = str(request.args.get("metadata_file_name"))
    iqdata_file_name = str(request.args.get("iqdata_file_name"))
    cent_freq = float(request.args.get("cent_freq"))
    bandwidth = float(request.args.get("bandwidth"))
    snr = float(request.args.get("snr"))
    sig_type = str(request.args.get("sig_type"))
    data = [iqdata_file_name, cent_freq, bandwidth, snr, sig_type]
    # write to local metadata file
    metadata_file_name = './RadioServer/{}'.format(metadata_file_name)
    fid = open(metadata_file_name, 'a', encoding='UTF8', newline='')
    writer = csv.writer(fid)
    writer.writerow(data)
    return "Success"


@app.post("/upload_metadata")
def upload_metadata():
    metadata_file_name = str(request.args.get("metadata_file_name"))
    boto3.resource('s3').Bucket('radio-bucket-766318').upload_file("./RadioServer/{}".format(metadata_file_name), "server_data/{}".format(metadata_file_name))
    return "created {}".format(metadata_file_name)

@app.post("/train_model")
def train_model():
    pass