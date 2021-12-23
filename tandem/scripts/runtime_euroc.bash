#!/bin/bash

if [[ -z "${EUROC_TANDEM_FORMAT}" ]]; then
  echo "Please export EUROC_TANDEM_FORMAT as described in the README.md"
  exit 1
fi

sequence=V1_01_easy

export result_folder=results/runtime/dense/euroc/$sequence
rm -rf $result_folder
mkdir -p $result_folder

export scene=$EUROC_TANDEM_FORMAT/$sequence
export mvsnet_folder=exported/tandem

STARTTIME=$(date +%s)

build/bin/tandem_dataset \
  preset=runtime \
  result_folder=$result_folder \
  files=$scene/images \
  calib=$scene/camera.txt \
  mvsnet_folder=$mvsnet_folder \
  exit_when_done=1 \
  mode=1 > $result_folder/out.txt

ENDTIME=$(date +%s)

echo "The run took $(($ENDTIME - $STARTTIME)) seconds which includes warm-up and loading."
