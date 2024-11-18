#FROM ubuntu:22.04
#FROM python:3.10
FROM python:3.11
LABEL authors="imarcoss"
LABEL org.opencontainers.image.source=https://github.com/ignacio-ms/ml_cheatsheet

#RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

mkdir /app/libs
COPY requirements.txt .
COPY libs libs

ENV LD_LIBRARY_PATH=/app/libs:$LD_LIBRARY_PATH

RUN pip install numpy
RUN pip install -r requirements.txt

RUN mkdir /app/output