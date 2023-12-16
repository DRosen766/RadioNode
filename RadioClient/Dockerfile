FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# install the IOT sdk 
RUN python3 -m pip install sagemaker-training
RUN python3 -m pip install sagemaker
RUN python3 -m pip install pandas
RUN python3 -m pip install matplotlib
RUN python3 -m pip install numpy
RUN python3 -m pip install tqdm
RUN python3 -m pip install --default-timeout=900 torch

# copy the training script inside the container
COPY train_test.py /opt/ml/code/train_test.py

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train_test.py