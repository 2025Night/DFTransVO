FROM docker.1ms.run/colmap/colmap:latest
MAINTAINER Paul-Edouard Sarlin
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ARG PYTHON_VERSION=3.8
RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION} && \
    apt-get install -y python3.8-distutils && \
    apt-get install -y git
RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install notebook
