FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg \
    git \
    python3.10 \
    python3-pip

WORKDIR /Biped-Locomotion-Simulation-Only-Walking-Control

COPY requirements.txt /Biped-Locomotion-Simulation-Only-Walking-Control
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /Biped-Locomotion-Simulation-Only-Walking-Control
ENTRYPOINT ["python3", "scripts/run_dcm_walk.py", "--headless"]