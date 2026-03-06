```bash
docker build -t cookbook_${USER} .
docker run -it --privileged --gpus all -p 8888:8888 \
  --ipc=host --net=host --cap-add  SYS_NICE \
  --name ${USER}_dev \
  cookbook_${USER} /bin/bash
```
