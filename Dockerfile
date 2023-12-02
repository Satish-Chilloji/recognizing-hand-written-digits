#FROM ubuntu:23.10
FROM python:3.9.17
WORKDIR /digit
COPY . /digit/
#RUN apt-get update
#RUN apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /digit/requirements.txt
#RUN pip install -r /digit/requirements.txt 
# COPY requirements.txt /requirements.txt
#RUN ["cd", "/digit"]
#ENV FLASK_APP=digit/api/assignment4.py
#CMD ["python","-m","flask","run","--host=0.0.0.0"]
ENV FLASK_APP=api/assignment4.py
CMD ["python", "api/assignment4.py"]
#CMD ["python","-m","flask","run"]
# VOLUME /digit/models
# CMD ["python","digit_recognition.py"]