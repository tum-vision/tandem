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
Alternatively, you can use bash scripts insice `/utils.

Put your `input` video and `calib.txt` in the same folder as `run_video.sh`. Then run the following command in the terminal.

Set executable permission on the script.
```shell
chmod +x run_video.sh
```

```shell
./run_video.sh --input <input video path> 
```

If your input are `images`, then use `run_image.sh`
```shell
./run_image.sh --input <path to images folder> 
```


## Note

### Display
In case you get some `X11` related error, rememebr that you need to set display inside your container same as your host display.
```shell
# In the host shell
echo $DISPLAY
:1

# In the container shell
echo $DISPLAY
:0

export DISPLAY=:1
```

If `docker start` is not working, run with `-ai` flag.
```shell
docker start -ai tandem
```
Run the following if the error is related to `X11` error.
```shell
xhost + local:
```
Ref -> https://stackoverflow.com/questions/73490184/sudo-nautilus-gives-authorization-required-but-no-authorization-protocol-specif

### Input size
Currently `cva-mvsnet` model supports `640x480` input size. If you want to input different size, follow instructions [here](https://github.com/pytholic/tandem/tree/master/cva_mvsnet) to export the model with desired dimensions. Update the path while running tandem accordingly.

### Nvidia container toolkit
In case you face some `GPU` supported related issues, rememebr that you need to install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your host.

```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```