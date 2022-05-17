# import from dockerhub of cuda base
FROM nvidia/cuda:11.4.0-runtime-centos7

# Maintainer label
LABEL maintainer="Raffaele Gerosa <rgerosa@ucsd.edu>"

# Copy the Miniconda3 file that needs to be periodically updated
RUN curl -LJO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
RUN set -x chmod Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

# Set conda environment 
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Update conda
RUN conda update -n base -c defaults conda

# Install mamba
RUN conda install -y mamba -c conda-forge

# Create the environment for weaver: first thing take out the torch cu113 specifics cause are not in the base pip package
COPY weaver-environment.yaml .
RUN sed -i "s/- torch==1.11.0+cu113//g"  weaver-environment.yaml
RUN sed -i "s/- torchaudio==0.11.0+cu113//g"  weaver-environment.yaml
RUN sed -i "s/- torchvision==0.12.0+cu113//g"  weaver-environment.yaml
RUN conda env create -f weaver-environment.yaml

# Initialize conda bash config
RUN conda init bash

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "weaver", "/bin/bash", "-c"]

# Install torch by hand
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
