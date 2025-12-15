# HRI_Project
```
# GUI 권한 허용
xhost +local:docker

# 실행
docker run -it --rm \
  --net=host \
  --privileged \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/aprl/Desktop/hri:/ros2_ws/ws \
  jsk_hri


# main 실행
python3 hint_task.py
```
