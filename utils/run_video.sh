#!/bin/bash

input=$2

# Create directories
rm -rf frames

mkdir frames
docker exec -i tandem /bin/bash -c "cd tandem/tandem; rm -rf calib.txt; rm -rf data; rm -rf results; mkdir -p data; mkdir -p results"

# Extarct frames
ffmpeg -i $input ./frames/%06d.png

# Copy files
docker cp ./frames/. tandem:/home/work/tandem/tandem/data  # copy frames
docker cp calib.txt tandem:/home/work/tandem/tandem/  # copy cam calib file
rm -rf frames

docker exec -i tandem /bin/bash -c "cd tandem/tandem; build/bin/tandem_dataset preset=gui result_folder=results files=data calib=calib.txt mvsnet_folder=exported/tandem mode=1"
