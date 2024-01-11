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
- `./exec_docker.sh`