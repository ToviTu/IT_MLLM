# Start from a PyTorch image with CUDA
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
# Flash-attn work around @ https://github.com/Dao-AILab/flash-attention/issues/509
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Set the working directory in the container
WORKDIR /app
RUN mkdir /dataset/
RUN mkdir /models/

COPY . .

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    wget \
    curl \
    dnsutils \
    nano \
    zip \
    unzip \
    git \
    s3cmd \
    ffmpeg \
    screen \
    fonts-freefont-ttf \
    inotify-tools \
    parallel \
    pciutils \
    ncdu \
    libbz2-dev \
    gettext \
    apt-transport-https \
    gnupg2 \
    time \
    openssl \
    redis-tools \
    ca-certificates \
    hdf5-tools\
    vim

# Add the official NVIDIA repository for CUDA Toolkit packages
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
#    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
#    && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#    && dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#    && cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
#    && apt-get update \
#    && apt-get -y install cuda

#RUN rm cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install any dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install -e ".[train]"
RUN pip install flash-attn --no-build-isolation

# Install lm-eval
RUN pip install git+https://github.com/ToviTu/lm-evaluation-harness.git@main
# Install lmm-eval
RUN pip install git+https://github.com/ToviTu/lmms-eval.git@llava_plain
