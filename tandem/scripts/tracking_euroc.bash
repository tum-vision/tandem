#!/bin/bash

if [[ -z "${EUROC_TANDEM_FORMAT}" ]]; then
  echo "Please export EUROC_TANDEM_FORMAT as described in the README.md"
  exit 1
fi

export mvsnet_folder=exported/tandem

for sequence in V1_01_easy V1_02_medium V2_01_easy V2_02_medium; do

  export scene=$EUROC_TANDEM_FORMAT/$sequence

  for run in 0 1 2 3 4; do
    export result_folder=results/tracking/dense/euroc/$sequence/$run
    rm -rf $result_folder
    mkdir -p $result_folder
    echo -e "\n\nRUNNING --- $result_folder"

    build/bin/tandem_dataset \
      preset=dataset \
      result_folder=$result_folder \
      files=$scene/images \
      calib=$scene/camera.txt \
      mvsnet_folder=$mvsnet_folder \
      exit_when_done=1 \
      mode=1 > $result_folder/out.txt
  done
done
