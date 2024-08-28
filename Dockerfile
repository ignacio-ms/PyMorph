#FROM ubuntu:22.04
#FROM python:3.10
FROM python:3.11
LABEL authors="imarcoss"

#RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip install numpy
RUN pip install -r requirements.txt

RUN mkdir /app/output