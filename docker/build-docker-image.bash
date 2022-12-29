#!/bin/bash

file_dir=`dirname $0`

# get parameter from system
user=`id -un`
group=`id -gn`
uid=`id -u`
gid=`id -g`

# --build-arg http_proxy=http://10.141.6.84:7890 \
# --build-arg https_proxy=http://10.141.6.84:7890 \

# build docker images
docker build -t pytholic/tandem -f dockerfile . \
    --build-arg USER=${user} \
    --build-arg UID=${uid} \
    --build-arg GROUP=${group} \
    --build-arg GID=${gid} \
