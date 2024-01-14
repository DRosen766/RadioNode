from flask import Flask, request
import boto3
import array
import numpy as np
import csv
import os
import flask

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    # """Determine if the container is working and healthy. In this sample container, we declare
    # it healthy if we can load the model successfully."""
    # health = ClassificationService.get_model() is not None  

    # status = 200 if health else 404
    print("Pinged!")
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def predict():
    print(os.environ["SM_MODEL_DIR"])
    print("hello world")
    return "hello world"