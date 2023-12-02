docker build -t base -f docker/DependencyDockerfile .
docker build -t digits -f Dockerfile .
docker run -it digits
az acr build --file docker/Dockerfile --registry satimlops23 --image base .
az acr build --file Dockerfile --registry satimlops23 --image digits .

# docker tag digits:latest satimlops23.azurecr.io/digits:latest
# docker push satimlops23.azurecr.io/digits:latest