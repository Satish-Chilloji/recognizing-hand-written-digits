docker build -t base -f docker/Dockerfile .
docker build -t digits -f Dockerfile .
docker run -it digits
docker tag digits:latest satimlops23.azurecr.io/base:latest
docker push satimlops23.azurecr.io/base:latest
docker tag digits:latest satimlops23.azurecr.io/digits:latest
docker push satimlops23.azurecr.io/digits:latest



