# FROM nvcr.io/nvidia/pytorch:23.03-py3
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y python3 python3-pip git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0

WORKDIR /service

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install gradio

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
