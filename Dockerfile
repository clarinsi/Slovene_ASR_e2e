FROM nvcr.io/nvidia/nemo:22.07 as nemo

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install ffmpeg -y

FROM nemo as service

ARG DEBIAN_FRONTEND=noninteractive

COPY . /opt/asr
RUN python3 -m pip install -r /opt/asr/requirements.txt
WORKDIR /opt/asr

ENTRYPOINT [ "python3", "server.py" ]
