# syntax=docker/dockerfile:1
FROM ubuntu:18.04
RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install python3-pip python3.6 bash vim valgrind gcc gdb cgdb python3-venv git htop -y
COPY ./* numc/
SHELL ["/bin/bash", "-c"]
RUN cd numc 
RUN python3.6 -m venv .venv 
RUN source .venv/bin/activate
RUN cd numc && pip3 install -r requirements.txt
CMD bash
