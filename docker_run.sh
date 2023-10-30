docker build -t digit:v1 -f docker/Dockerfile .
docker run -v ./models:/digit/models digit:v1