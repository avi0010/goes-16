# Use the official Ubuntu image as the base
FROM ubuntu:latest

# Set environment variables for non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common &&\
    apt-get install -y python3 python3-pip libgdal-dev gdal-bin postgresql postgresql-contrib

# Set the locale (optional but recommended for PostgreSQL)
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN mkdir /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install GDAL==$(gdal-config --version)