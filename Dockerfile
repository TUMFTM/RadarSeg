# get base image
FROM tensorflow/tensorflow:2.2.0-gpu

# set maintainer
MAINTAINER felix fent <felix.fent@tum.de>

# install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    vim \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# upgrade pre-existing pip packages
RUN pip install --upgrade --no-cache-dir \
    pip \
    numpy==1.18.5 \
    protobuf==3.12.2 \
 && rm -rf ~/.cache/pip/*

# install additional pip packages
RUN pip install --no-cache-dir \
    pandas==1.0.4 \
    scipy==1.4.1 \
    matplotlib==3.2.1 \
    pytz==2019.3 \
    python-dateutil==2.8.1 \
    kiwisolver==1.2.0 \
    pyparsing==2.4.7 \
    cycler==0.10.0 \
    seaborn==0.10.0 \
    opencv-python==4.2.0.34 \
    typeguard==2.7.1 \
    progressbar2==3.50.1 \
    configobj==5.0.6 \
 && rm -rf ~/.cache/pip/*

# install the nuscenes-devkit and its requirements
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    pyquaternion==0.9.5 \
    torch==1.5.0 \
    scikit_learn==0.22.2 \
    Pillow==6.2.1 \
    nuscenes-devkit==1.0.8 \
 && rm -rf ~/.cache/pip/*
