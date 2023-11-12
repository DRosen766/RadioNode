from flask import Flask, request
from werkzeug.utils import secure_filename
import boto3
import array
import numpy as np
app = Flask(__name__)

iq_data_sample = []

@app.put("/send_iq")
def append_iq():
    k = int(request.args.get("k"))

    # extract iq data from request
    iq_data = request.get_data()
    iq_data = np.array(array.array('f', iq_data))

    # move iq_data to file
    iqdata_file_name = 'server_data/iqdata/example_' + str(k+1) + '.dat'
    iq_data.tofile(iqdata_file_name)

    # upload file to s3 bucket
    boto3.resource('s3').Bucket('radio-bucket-766318').upload_file(iqdata_file_name, iqdata_file_name)
    return "created {}".format(iqdata_file_name)