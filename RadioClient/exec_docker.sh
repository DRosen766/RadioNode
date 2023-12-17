docker build -t client_container . --build-arg AWS_IOT_CA_FILE="$env:AWS_IOT_CA_FILE" --build-arg AWS_IOT_KEY="$env:AWS_IOT_KEY" --build-arg AWS_IOT_CERT="$env:AWS_IOT_CERT" --build-arg AWS_IOT_ENDPOINT="$env:AWS_IOT_ENDPOINT"
docker run -it client_container

