# Tandem Docker

This is the docker environment setup tutorial for `TUM Tandem`, it supports remote `OpenGL` display, thus you can render `Pangolin window` with remote ssh.

## Build Image

Run the following commands to build the image:

```shell
cd docker
./build-dokcer-image.bash
```

and run

```shell
./run-docker-container.bash
```

to run the container.

Please note that you can customize your own `ssh port` by modifying the `run-docker-container.bash` file:

```shell
docker run --rm \
  --name tandem \
  --ipc=host \
  --gpus all \
  --privileged \
  -p 3751:22 \         <======= please choose your own port
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
```

Remove the `--rm` flag in case you don't want to erase all the changes and data after you stop the `container`.

You can also pull the built image from `dockerhub.io`:

```shell
docker pull pytholic/tumtandem:tumtandem
```

## Run

Then in a new remote or local terminal, run the following command to connect the remote environment:

```shell
ssh -p 3751 work@xx.xx.xx.xx
# default password: work123
```

Or if you already `docker run` run once, you can simple do:

```shell
docker start <container name>
```

Add you camera calibration file in the specified [format](https://github.com/pytholic/tandem/blob/master/tandem/README.md).

Run the following comand in the terminal.
```shell
build/bin/tandem_dataset \
      preset=dataset \
      result_folder=path/to/save/results \
      files=path/to/scene/images \
      calib=path/to/scene/camera.txt \
      mvsnet_folder=exported/tandem \
      mode=1
```

### Bash script
Alternatively, you can use `utils/run.sh` script.

Put your `input` video and `calib.txt` in the same folder as `run.sh`. The nrun the following command in the terminal.

Set executable permission on the script.
```shell
chmod +x run.sh
```

```shell
./run.sh --input <input video name> --data <camera or data model>  

# "data" can be one of [evo, gopro, iphone, euroc, replica].
```


## Note
Incase you get some `X11` related error, rememebr that you need to set display inside your container same as your host display.
```shell
# In the host shell
echo $DISPLAY
:1

# In the container shell
echo $DISPLAY
:0

export DISPLAY=:1
```

Currently `cva-mvsnet` model supports `640x480` input size. If you want to input different size, follow instructions [here](https://github.com/pytholic/tandem/tree/master/cva_mvsnet) to export the model with desired dimensions. Update the path while running tandem accordingly.