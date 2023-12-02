# recognizing-hand-written-digits
This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Build Docker
docker build -t digit:v1 -f docker/Dockerfile .
# Run Docker
docker run -it -p 8000:5000 digit:v1


# Azure Login
az login --use-device