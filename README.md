# HRI_Project
```
# GUI 권한 허용
xhost +local:docker

# 실행
docker run -it --rm \
    --net=host \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/snd:/dev/snd \
    -e DISPLAY=$DISPLAY \
    hri_project

# main 실행
python3 hint_task.py
```
