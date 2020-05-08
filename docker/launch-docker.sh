# NOTE: run this from pvnet root
docker run -it --rm \
    -v $(pwd)/data:/home/pvnet/pvnet/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --name pvnet-nvidia-c \
    --net host \
    --privileged \
    --gpus all \
    pvnet-nvidia
