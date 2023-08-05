#!/bin/bash

# Check if model.info is not found
if ! [[ -f models/v2.0/model.info ]]; then
    mkdir -p models/
    # URL to the tar.zst file
    url="https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1737/sl-SI_GEN_nemo-2.0.tar.zst"

    # Download the tar.zst file
    wget "$url" -O model.tar.zst

    # Unzstd and extract the contents
    unzstd model.tar.zst
    tar -xf model.tar -C models/

    # Remove the downloaded tar.zst and extracted tar file
    rm model.tar.zst model.tar

    echo "Model downloaded and extracted successfully."
else
    echo "model.info file found. No need to download or extract."
fi

docker compose -f docker-compose.yml up
# -f docker-compose.gpu.yml up -d