#!/bin/bash

input=$2

# Create firectories
docker exec -i tandem /bin/bash -c "cd tandem/tandem; rm -rf calib.txt; rm -rf data; rm -rf results; mkdir -p data; mkdir -p results"

# Copy files
docker cp $input/. tandem:/home/work/tandem/tandem/data  # copy frames
docker cp calib.txt tandem:/home/work/tandem/tandem/  # copy cam calib file

#rm -rf $input

docker exec -i tandem /bin/bash -c "cd tandem/tandem; build/bin/tandem_dataset preset=gui result_folder=results files=data calib=calib.txt mvsnet_folder=exported/tandem mode=1 tracking=sparse"
