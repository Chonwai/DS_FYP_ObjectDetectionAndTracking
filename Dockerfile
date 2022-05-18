FROM python:3.7
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# FROM yssong67/cuda-opencv:112-810-440-1804
ENV DEBIAN_FRONTEND noninteractive

COPY requirements.txt .

RUN apt-get update -y

RUN apt-get -y install python3-pip

RUN pip install opencv-python-headless
RUN pip install -r requirements.txt
RUN pip install pyparsing==2.4.2

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV YOLO_ROOT /person_hat
RUN mkdir $YOLO_ROOT
WORKDIR /person_hat

COPY . .