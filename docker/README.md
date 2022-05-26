# Tandem Docker

This is the docker environment setup tutorial for `TUM Tandem`, it supports remote `OpenGL` display, thus you can render `Pangolin window` with remote ssh.

### Build Image

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
  --ipc=host \
  --gpus all \
  --privileged \
  -p 3751:22 \         <======= please choose your own port
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
```

Then in a new remote or local terminal, run the following command to connect the remote environment:

```shell
ssh -p 3751 work@xx.xx.xx.xx
# default password: work123
```

You can also pull the built image from `dockerhub.io`:

```shell
docker pull chengran222/tumtandem:origin
```