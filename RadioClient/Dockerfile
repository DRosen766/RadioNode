FROM ubuntu

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

ARG AWS_IOT_KEY

COPY temp.py temp.py

# define train.py as the script entry point
ENV TOPIC "test/topic"
ENV AWS_IOT_CA_FILE $AWS_IOT_CA_FILE
# ENV AWS_IOT_KEY $AWS_IOT_KEY
ENV AWS_IOT_KEY $AWS_IOT_KEY
ENV AWS_IOT_ENDPOINT $AWS_IOT_ENDPOINT
RUN echo 
RUN python3 temp.py > out.txt