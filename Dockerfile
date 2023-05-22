FROM nvcr.io/nvidia/pytorch:22.08-py3 as nemo

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install ffmpeg -y \
    && python3 -m pip install --upgrade pip \
    && git clone https://github.com/NVIDIA/NeMo.git /workspace/nemo \
    && cd /workspace/nemo \
    && git checkout v1.11.0
    
RUN cd /workspace/nemo && pip install -e .[all] --ignore-installed llvmlite

FROM nemo as service

ARG DEBIAN_FRONTEND=noninteractive

COPY . /opt/asr
RUN python3 -m pip install -r /opt/asr/requirements.txt
WORKDIR /opt/asr

ENTRYPOINT [ "python3", "server.py" ]
