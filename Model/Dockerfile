FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y nginx

# install the SageMaker Training Toolkit 
RUN python3 -m pip install sagemaker-training
RUN python3 -m pip install sagemaker
RUN python3 -m pip install pandas
RUN python3 -m pip install matplotlib
RUN python3 -m pip install numpy
RUN python3 -m pip install tqdm
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install seaborn
RUN python3 -m pip install torchvision
RUN python3 -m pip install boto3
RUN python3 -m pip install --default-timeout=900 torch
RUN python3 -m pip install flask
RUN python3 -m pip install gevent
RUN python3 -m pip install gunicorn
RUN pip install fastai



# copy the training script inside the container
RUN chmod 755 /opt/
COPY /opt/ml/code/model.py /opt/ml/code/model.py
COPY /opt/ml/code/serve /opt/ml/code/serve
COPY /opt/ml/code/predictor_server.py /opt/ml/code/predictor_server.py
RUN chmod 755 /opt/ml/code/serve
COPY /opt/ml/code/train_model.py /opt/ml/code/train_model.py

# add testing files to container
COPY tests/test_predictor.py test/test_predictor.py

# create directories for data
WORKDIR /opt/ml/
RUN mkdir train_data
RUN mkdir train_data/iqdata
RUN mkdir test_data
RUN mkdir test_data/iqdata
WORKDIR /

# add server directory and python directory to path
ENV PATH="/usr/bin/python3:/opt/ml/code:${PATH}"
# define train_model.py as the script entry point
ENV SAGEMAKER_PROGRAM  /opt/ml/code/train_model.py
ENV SM_TRAINING_DATA_BUCKET test-radio-bucket-766318
