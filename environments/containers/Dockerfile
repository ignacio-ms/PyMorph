FROM ubuntu:22.04
LABEL authors="imarcoss"

RUN apt-get update && apt-get install -y  \
    build-essential \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    libpython3.10 \
    libpython3.10-dev \
    git \
    && apt-get clean

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install numpy

COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]