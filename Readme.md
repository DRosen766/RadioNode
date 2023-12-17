# Radio Client
## define environment variables
- AWS_IOT_CA_FILE: path of CA file (ex. %USERPROFILE%\certs\Amazon-root-CA-1.pem)
- AWS_IOT_KEY: path of key file (ex. %USERPROFILE%\certs\private.pem.key)
- AWS_IOT_CERT: path of key file (ex. %USERPROFILE%\certs\device.pem.crt)
- AWS_IOT_ENDPOINT: your iot endpoint

## build and run docker container: 
- `cd RadioClient`
- `./exec_docker.sh`