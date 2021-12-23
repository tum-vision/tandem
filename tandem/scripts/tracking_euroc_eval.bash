#!/bin/bash

if [[ -z "${EUROC_TANDEM_FORMAT}" ]]; then
  echo "Please export EUROC_TANDEM_FORMAT as described in the README.md"
  exit 1
fi

if [[ -z "${TANDEM_PY2}" ]]; then
  echo "Please export TANDEM_PY2 as described in the README.md"
  exit 1
fi

py2=$TANDEM_PY2

for sequence in V1_01_easy V1_02_medium V2_01_easy V2_02_medium; do

  export scene=$EUROC_TANDEM_FORMAT/$sequence

  for run in 0 1 2 3 4; do
    export result_folder=results/tracking/dense/euroc/$sequence/$run
    echo "--- EVAL --- $sequence $run"

    if test -f "$result_folder/result.txt"; then
      scale=$($py2 tum_rgbd_eval_tools/align_se3.py $scene/groundtruth_tum_rgbd.txt $result_folder/result.txt)
      $py2 tum_rgbd_eval_tools/evaluate_ate.py \
        $scene/groundtruth_tum_rgbd.txt \
        $result_folder/result.txt \
        $scale \
        --plot $result_folder/evaluate_ate.png
    else
      echo "fail"
    fi

  done
done
