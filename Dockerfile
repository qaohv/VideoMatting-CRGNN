FROM nvcr.io/nvidia/pytorch:21.07-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	ffmpeg \
	libsm6 \
	libxext6 \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python==4.5.3.56

RUN mkdir /opt/app
WORKDIR /opt/app

COPY checkpoint /opt/app/checkpoint
COPY models /opt/app/models
COPY pretrained /opt/app/pretrained
COPY test.py /opt/app/

RUN cd models/archs/dcn && python3 setup.py develop
