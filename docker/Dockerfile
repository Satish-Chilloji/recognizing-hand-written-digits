#FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digit/
RUN apt-get update
#RUN apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /digit/requirements.txt
# COPY requirements.txt /requirements.txt
#RUN ["cd", "/digit"]
WORKDIR /digit
VOLUME /digit/models
CMD ["python","digit_recognition.py"]