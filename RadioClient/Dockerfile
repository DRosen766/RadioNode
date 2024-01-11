FROM ubuntu

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN git clone https://github.com/aws/aws-iot-device-sdk-python-v2.git
WORKDIR "/aws-iot-device-sdk-python-v2"
RUN python3 -m pip install .
WORKDIR "/"
RUN python3 -m pip install awscrt
RUN python3 -m pip install numpy

ARG AWS_IOT_ENDPOINT
ARG AWS_IOT_CA_FILE
ARG AWS_IOT_CERT
ARG AWS_IOT_KEY

RUN mkdir /certs
COPY /certs /certs

COPY iot_client.py iot_client.py

# define train.py as the script entry point
ENV TOPIC "iq_data"
ENV AWS_IOT_CA_FILE $AWS_IOT_CA_FILE
ENV AWS_IOT_CERT $AWS_IOT_CERT
ENV AWS_IOT_KEY $AWS_IOT_KEY
ENV AWS_IOT_ENDPOINT $AWS_IOT_ENDPOINT
RUN python3 iot_client.py