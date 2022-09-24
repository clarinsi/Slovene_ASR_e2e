# ASR API

Implements the ASR API for NeMo Conformer CTC BPE E2E models. For more details about building such models, see the official [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html) and [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo).

The API provides two endpoints `/api/healthcheck`, to retrieve the service status, and `/api/transcribe` to request a transcription of an audio file. The only accepted format is WAV 16kHz, 16bit PCM, mono. The maximal accepted audio duration is 300s. Note that transcription of one 300s audio file on cpu will take advantage of all available cores, consume up to 16GB RAM and may take ~180s (on a system with 24 vCPU).

# Prerequisites

- docker >= 20.10.17
- docker compose >= 2.6.0
- NeMo model and `model.info`

# Model.info

The expected format for `model.info` is:
```yml
language_code: # dash saparated two-letter ISO 639-1 Langauge Code, lowercase, and ISO 3166 Country Code, uppercase, eg. sl-SI
domain: # model domain
version: # model version
info:
  build: # build time in YYYYMMDD-HHSS format
  am:
    framework: nemo:conformer:ctc:bpe
    ... # aditional info, optional
features: # optional
  ... # information about special features
  remap:
  ... # list of {src text}: {tgt text} remappings to be applied prior to returning the end result
```

The NeMo model file is expected in the same folder, named as `conformer_ctc_bpe.nemo`.

The Conformer CTC BPE E2E Automated Speech Recognition model developed as part of work package 2 of the Development of Slovene in a Digital Environment, RSDO, project (https://slovenscina.eu/govorne-tehnologije), can be downloaded from http://hdl.handle.net/11356/1737.

# Deployment

Run `docker compose up -d` to deploy on cpu or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` to run on gpu.

# Approximate memory consumption for cpu deployment

- 2GB RAM for service and model
- 16GB RAM per 300s request
