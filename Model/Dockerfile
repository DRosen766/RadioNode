FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

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


# copy the training script inside the container
COPY /opt/ml/code/model.py /opt/ml/code/model.py
# COPY /opt/ml/code/sandbox.py /opt/ml/code/sandbox.py
COPY /opt/ml/code/train_model.py /opt/ml/code/train_model.py
WORKDIR /opt/ml/
RUN mkdir train_data
RUN mkdir train_data/iqdata
RUN mkdir test_data
RUN mkdir test_data/iqdata
WORKDIR /opt/ml/code/
# define train_model.py as the script entry point
ENV SAGEMAKER_PROGRAM  /opt/ml/code/train_model.py

ENV SM_TRAINING_DATA_BUCKET test-radio-bucket-766318