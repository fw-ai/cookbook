# Docker container for text model

We put together a container that includes dependencies required to run recipes operating on textual model.

## Prerequisites

If you start with a plain Linux host, you will need to install the required CUDA drivers and docker engine. Follow the instructions listed [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

By default, running docker commands requires sudo access. To remove this restriction, run the following commands:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## [Optional] Set up the environment

In many cases, it's convenient to use the host filesystem for certain types of data that should be preserved across container instances or even shared by multiple containers running concurrently.
For instance, keeping the source code on the host makes it easier to edit it through a remote IDE session. Storing the HuggingFace :hugs: cache on the host and sharing it across containers reduces the disk usage. Finally, model training logs and artifacts are also easier to preserve across container upgrades if they reside in the host filesystem.
Throughout the documentation and code, the default location of the code is `/workspace` and model artifacts are assumed to be stored under `/mnt/text`. You can easily override this location when running individual commands.

## Build and run the docker container

The docker container can be built by running the following command in this directory:
```
docker build -t fireworks_cb_text .
```
After the image built finishes, the container can be instantiated in interactive model with the following command:
```
docker run --privileged -it --gpus all -p 8888:8888 \
  --mount type=bind,source="/workspace",target="/workspace" \
  --mount type=bind,source="$HOME/.cache/huggingface",target="/root/.cache/huggingface" \
  --mount type=bind,source="$HOME/.ssh",target="/root/.ssh" \
  --mount type=bind,source="/mnt/text",target="/mnt/text" \
  --ipc=host --net=host --cap-add  SYS_NICE \
  fireworks_cb_text /bin/bash
```
Feel free to update the mount locations based on your preferences.

If loose connectivity or want to open up more sessions to a running container, first run the following command to identify the name of the running container name:
```
docker ps
```
Use the relevant container name (something like `epic_spence`) to log into it in interactive mode:
```
docker exec -it <container_name> bash
```

## Jupyter notebooks

You can run Jupyter notebooks inside the container.

Make sure that your docker container was started with the `-p 8888:8888` flag.
Start the notebook server with the following command `jupyter notebook -i 0.0.0.0`.
Follow the url printed by the server but replace the host with `localhost` (or whatever
is the name of the host where you run the container), e.g., `http://localhost:8888/?token=...`.

If you are connected to the host remotely using VSCode, you may want to set up port forwarding
for port 8888. The `PORTS` tab should be visible at the bottom of the editor window.
If it's not, go to `Terminal` -> `New Terminal`. `PORTS` should appear as one of the tabs.


## You are all set!

Now you can run the recipes inside the container that should have all the required dependencies. Have fun!
