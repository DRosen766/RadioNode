from flask import Flask, request
import boto3
import array
import numpy as np
import csv
import os

app = Flask(__name__)

@app.route('/invocations', methods=['POST'])
def predict():
    # print(os.environ["SM_MODEL_DIR"])
    print("hello world")
    return "hello world"