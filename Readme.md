# Radio Client
## define environment variables
- AWS_IOT_CA_FILE: name of CA file (ex. Amazon-root-CA-1.pem)
- AWS_IOT_KEY: name of key file (ex. private.pem.key)
- AWS_IOT_CERT: name of key file (ex. device.pem.crt)
- AWS_IOT_ENDPOINT: your iot endpoint

## add folder with certifications to RadioClient
#### Tree should look like:
    |RadioNode
      --RadioClient
        --certs
        
## build and run docker container: 
- `cd RadioClient`
- `docker build -t client_container . --build-arg AWS_IOT_CA_FILE="$env:AWS_IOT_CA_FILE" --build-arg AWS_IOT_KEY="$env:AWS_IOT_KEY" --build-arg AWS_IOT_CERT="$env:AWS_IOT_CERT" --build-arg AWS_IOT_ENDPOINT="$env:AWS_IOT_ENDPOINT"`
- `docker run -it client_container`


# Model
## Testing
- example for sending example training data to s3 bucket: `tests/send_iq_train_data.py`
- example for sending example testing data to s3 bucket: `tests/send_iq_test_data.py`
- example model for testing: `tests/example_model.py`
- example for training example model on local machine: `tests/train_model_local.py`
- example for training containerized model with aws sagemaker estimator: `tests/train_model_local.py`

## Containerizing Model
- from `Model/` run `docker build -t model_container .` 
- - NOTE: Ensure docker daemon is running
- to run: `docker run -p 127.0.0.1:5000:5000 --mount src=$env:USERPROFILE/.aws,target="/root/.aws",type=bind -it model_container`