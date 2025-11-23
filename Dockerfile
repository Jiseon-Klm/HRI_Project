# 1. 베이스 이미지: ROS 2 Humble (Ubuntu 22.04 기반, AMD/ARM 자동 지원)
FROM ros:humble-ros-base-jammy

# 2. 필수 설정: 빌드 중 멈춤 방지 및 쉘 설정
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 3. 시스템 패키지 통합 설치 (RealSense + ReSpeaker + GUI + 오디오)
# 중복되는 패키지를 정리하고 한 번에 설치하여 이미지 크기를 줄입니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 공통 빌드 도구
    git \
    cmake \
    build-essential \
    pkg-config \
    curl \
    ca-certificates \
    gedit \
    # RealSense 의존성 (USB, 그래픽)
    libssl-dev \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    ffmpeg \
    libsndfile1 \
    udev \
    v4l-utils \
    # ReSpeaker/Audio 의존성
    python3-pip \
    python3-setuptools \
    alsa-utils \
    portaudio19-dev \
    libasound2 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. RealSense SDK 설치 (소스코드 빌드 & RSUSB 백엔드)
WORKDIR /usr/src
RUN git clone --depth 1 --branch v2.56.5 https://github.com/IntelRealSense/librealsense.git && \
    mkdir -p librealsense/build && \
    cd librealsense/build && \
    cmake .. \
    -DFORCE_RSUSB_BACKEND=true \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=true && \
    make -j$(nproc) && \
    make install && \
    # Udev 규칙 복사
    cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/ && \
    # 소스코드 정리 (이미지 다이어트)
    cd /usr/src && rm -rf librealsense

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Python 패키지 설치 (ReSpeaker + AI/Vision/Audio 라이브러리)
RUN pip3 install --no-cache-dir \
    pyaudio \
    pyusb \
    numpy \
    click \
    transformers \
    openai-whisper \
    opencv-python \
    mediapipe \
    sounddevice \
    google-genai \
    soundfile \
    pillow

# 6. ROS 2 워크스페이스 설정 및 ReSpeaker 패키지 빌드
WORKDIR /ros2_ws
# ReSpeaker ROS 2 패키지 클론
RUN git clone https://github.com/Jiseon-Klm/respeaker_ros2.git src/respeaker_ros2

# ROS 2 패키지 빌드 (환경설정 로드 후 빌드)
RUN source /opt/ros/humble/setup.bash && \
    colcon build --packages-select respeaker_ros2

# 7. 환경 변수 및 실행 설정
# 라이브러리 경로 및 그래픽 가속 호환성 설정
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV LIBGL_ALWAYS_SOFTWARE=1

# 터미널 접속 시 자동으로 ROS 및 워크스페이스 환경설정 로드
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# 기본 명령어
CMD ["bash"]
