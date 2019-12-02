FROM nvidia/cuda:10.2-devel-ubuntu16.04
# NOTE: build this image from dir above Dockerfile location
# USE BASH
SHELL ["/bin/bash", "-c"]

## modify below
ARG username=pvnet
ARG groupid=1000
ARG userid=1000
## end modify

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         vim \
         ca-certificates \
         python-qt4 \
         libjpeg-dev \
         zip \
         unzip \
         sudo \
         libpng-dev \
         libeigen3-dev \
         libgoogle-glog-dev=0.3.4-0.1 \
         libsuitesparse-dev=1:4.4.6-1 \
         libatlas-base-dev=3.10.2-9 &&\
     rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV PYTHON_VERSION=3.6.7

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

ENV PATH=$PATH:/opt/conda/bin/

RUN adduser --disabled-password --gecos '' $username \
    && adduser $username sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $username

# clone the repo and build ceres from source
RUN cd /home/$username \
    && git clone https://github.com/jwdinius/pvnet.git \
    && cd pvnet \
    && chmod a+rwx build_ceres_for_docker.sh \
    && ./build_ceres_for_docker.sh /home/$username

# Create Enviroment
RUN cd /home/$username/pvnet \
    && conda env create -f environment.yml

# follow remainder of steps in README
RUN source activate pvnet \
    && cd /home/$username/pvnet/lib/ransac_voting_gpu_layer \
    && python setup.py build_ext --inplace
RUN source activate pvnet \
    && cd /home/$username/pvnet/lib/utils/extend_utils \
    && python build_extend_utils_cffi_for_docker.py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$username/pvnet/lib/utils/extend_utils/lib

WORKDIR /home/$username
