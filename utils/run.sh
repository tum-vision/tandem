#!/bin/bash

input=$2
data=$4
calib=$6

# Create directories
rm -rf frames
mkdir frames
docker exec -i tandem /bin/bash -c "cd tandem/tandem; rm -rf data; rm-rf results; mkdir -p data; mkdir -p results"

# Extarct frames
if [ "$data" == "evo" ]
then
    ffmpeg -i $input ./frames/%06d.png
else
    ffmpeg -i $input ./frames/%06d.png
fi

docker cp ./frames/. tandem:/home/work/tandem/tandem/data  # copy frames
rm -rf frames

docker exec -i tandem /bin/bash -c "cd tandem/tandem; build/bin/tandem_dataset preset=gui result_folder=results files=data calib=calib/$calib mvsnet_folder=exported/tandem mode=1"
