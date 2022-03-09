FROM ubuntu:latest

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN ln -fs /usr/share/zoneinfo/Europe/Athens /etc/localtime

# install system packages
RUN apt -y update
RUN apt -y upgrade

# install extras
RUN apt -y install wget git tmux vim netcat htop unzip nano curl

# install build tools
RUN apt -y install make cmake g++ build-essential

# install nvidia build tools
RUN apt -y install nvidia-cuda-dev nvidia-cuda-toolkit nvidia-utils-510

# install open mpi
RUN apt -y update
RUN apt -y install openmpi-bin

# mount volume
VOLUME ["/des"]


