FROM ubuntu:22.04
#FROM python:3.11
LABEL authors="imarcoss"
#LABEL org.opencontainers.image.source=https://github.com/ignacio-ms/ht_morphogenesis

RUN apt-get update && apt-get install -y  \
    build-essential \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    libpython3.10 \
    libpython3.10-dev \
    cmake \
    git \
    patchelf \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#RUN mkdir /app/opt/ /app/opt/libs /app/opt/base /app/opt/SurfaceMapComputation

#COPY libs /app/opt/libs
#COPY base /app/opt/base
#COPY opt/SurfaceMapComputation /app/opt/SurfaceMapComputation

#ENV LD_LIBRARY_PATH=/app/opt/libs:/app/opt/base:$LD_LIBRARY_PATH
#ENV PATH=/app/opt/SurfaceMapComputation:$PATH

#RUN chmod +x /app/opt/SurfaceMapComputation/* \
#    && chmod +x /app/opt/libs/* \
#    && chmod +x /app/opt/base/*

RUN pip install --upgrade pip \
    && pip install numpy \
    && pip install pyradiomics \
    && pip install pypardiso \
    && pip install pyglet

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /app/output

ENTRYPOINT ["python3"]