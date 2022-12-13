#!/bin/bash

# get parameter from system
user=`id -un`

# start sharing xhost
xhost +local:root

# incase if you need proxy
# uncomment and replace it with your own institution server:
# -e http_proxy=http://10.141.6.84:7890 \
# -e https_proxy=http://10.141.6.84:7890 \

# run docker
docker run --rm \
  --ipc=host \
  --gpus all \
  --privileged \
  -p 3751:22 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:$docker/.Xauthority \
  -v $HOME/work:/home/work/projects \
  -v /mnt/Data/Datasets/dm-vio:/mnt/Data/Datasets/dm-vio \
  -e XAUTHORITY=$home_folder/.Xauthority \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it chengran222/tumtandem