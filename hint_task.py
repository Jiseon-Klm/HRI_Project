#!/usr/bin/env python3
"""
gesture_nav_agent.py

- 카메라(RealSense 등) + MediaPipe로 사람의 손짓을 읽어서
  gesture ∈ {"turn right", "turn left"} 로 유지
- ReSpeaker 마이크 입력은 "초기 한 번만" 받아서 STT로 사람의 instruction을 얻음
- 이때 사용자가 말하는 동안의 제스처를 초기 한 번만 캡처해서 고정함
- 이후에는 마이크는 더 이상 사용하지 않고,
  6초 단위로 현재 카메라 RGB 관측 + (초기 제스처, 초기 instruction)을
  Gemini 멀티모달 모델에 넣어서
  action_space = ["forward", "left", "right", "stop", "goal_signal"] 중
  next_action을 계속 업데이트해서 터미널에 출력.
"""

import threading
import time
import subprocess
import re
import os

import cv2
import mediapipe as mp
import numpy as np

from google import genai
from google.genai import types

from llm_stt_tts import STTProcessor
from config import (
    CAMERA_DEVICE_ID,   # 지금은 안 써도 되지만 호환용으로 남겨둠
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
)

# LLM이 최종적으로 선택해야 하는 액션 공간
# prompt.txt와 정확히 동일한 토큰을 사용해야 파싱이 잘 됨
ACTION_SPACE = ["forward", "left", "right", "stop", "goal_signal"]

# 프롬프트 템플릿 캐시용 전역 변수
PROMPT_TEMPLATE: str | None = None


def load_prompt_template(path: str = "prompt.txt") -> str:
    """
    외부 텍스트 파일에서 Gemini용 프롬프트 템플릿을 읽어온다.
    - prompt.txt 안에 있는 {spoken_text}, {gesture_str}는
      .replace()로 치환해서 사용한다.
    """
    global PROMPT_TEMPLATE
    if PROMPT_TEMPLATE is not None:
        return PROMPT_TEMPLATE

    if not os.path.exists(path):
        raise FileNotFoundError(f"Gemini 프롬프트 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    print(f"[INFO] Gemini 프롬프트 템플릿 로드: {path}")
    return PROMPT_TEMPLATE


# ==========================
# 0) RealSense 카메라 열기
# ==========================
def open_realsense_capture():
    """
    1) `v4l2-ctl --list-devices`에서 'RealSense'가 들어간 디바이스 블록을 찾고
    2) 그 블록 안의 /dev/videoX 후보들에 대해:
         - OpenCV(cv2.VideoCapture)로 dev를 열어서
         - 해상도 설정 후 frame을 한 번 읽어봄
       → 정상 프레임이 나오면 그 cap을 그대로 반환.

    반환: (cap, index, dev_path)
    - cap: 이미 열린 cv2.VideoCapture 객체 (release 하지 않고 돌려줌)
    - index: /dev/videoX 의 X
    - dev_path: "/dev/videoX" 문자열
    """
    print("[INFO] RealSense 카메라 자동 탐색: `v4l2-ctl --list-devices` 실행")

    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] `v4l2-ctl --list-devices` 실행 실패: {e}")

    lines = proc.stdout.splitlines()

    current_name = None
    current_devices = []
    devices_info = []  # [(name, ["/dev/video0", "/dev/video1", ...]), ...]

    for line in lines:
        if not line.strip():
            # 빈 줄 → 이전 블록 종료
            if current_name is not None:
                devices_info.append((current_name, list(current_devices)))
            current_name = None
            current_devices = []
            continue

        if not line.startswith("\t") and not line.startswith(" "):
            # 새 장치 이름 라인
            if current_name is not None:
                devices_info.append((current_name, list(current_devices)))
            current_name = line.strip().rstrip(":")
            current_devices = []
        else:
            # /dev/videoX 라인
            dev_path = line.strip()
            if dev_path.startswith("/dev/video"):
                current_devices.append(dev_path)

    # 마지막 블록 처리
    if current_name is not None:
        devices_info.append((current_name, list(current_devices)))

    # RealSense 관련 블록만 필터링
    rs_blocks = [(name, devs) for name, devs in devices_info if "realsense" in name.lower()]

    if not rs_blocks:
        raise RuntimeError(
            "[ERROR] `v4l2-ctl --list-devices` 결과에서 RealSense 장치를 찾지 못했습니다.\n"
            "RealSense가 /dev/video* 로 제대로 잡혀 있는지 확인하세요."
        )

    # RealSense 블록 내 모든 /dev/video* 후보 수집
    candidate_devs = []
    for name, devs in rs_blocks:
        print(f"[DEBUG] RealSense block: '{name}' -> {devs}")
        candidate_devs.extend(devs)

    if not candidate_devs:
        raise RuntimeError(
            "[ERROR] RealSense 장치는 있으나 /dev/video* 노드를 찾지 못했습니다."
        )

    # 각 후보 dev에 대해 실제로 OpenCV로 열고, 프레임을 읽어보며 검사
    for dev in candidate_devs:
        print(f"[INFO] RealSense 후보 노드 테스트: {dev}")
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print(f"[WARN] {dev} 를 열 수 없습니다 (isOpened() == False)")
            cap.release()
            continue

        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print(f"[WARN] {dev} 에서 유효한 프레임을 읽지 못했습니다.")
            cap.release()
            continue

        # 여기까지 왔으면 실제로 usable한 스트림이라고 판단
        m = re.search(r"/dev/video(\d+)", dev)
        if not m:
            print(f"[WARN] {dev} 의 인덱스를 파싱하지 못했습니다.")
            cap.release()
            continue

        idx = int(m.group(1))
        print(f"[INFO] 사용 가능한 RealSense 카메라 확정: device='{dev}', index={idx}")
        # cap은 열어둔 채로 반환
        return cap, idx, dev

    # 아무 dev도 통과하지 못한 경우
    raise RuntimeError(
        "[ERROR] RealSense /dev/video* 노드 중 어느 것도 OpenCV로 열거나 프레임을 읽을 수 없습니다.\n"
        "컨테이너에서 RealSense 권한/udev/--device 설정을 다시 확인해보세요."
    )


# ==========================
# 1) 카메라 + 제스처 쓰레드 (제스처 freeze 기능 + 디버깅 로그)
# ==========================
class GestureCamera(threading.Thread):
    """
    - 카메라 프레임에서 MediaPipe Hands로 손 검출
    - '오른쪽을 가리키는 손짓' ⇒ "turn right"
    - '왼쪽을 가리키는 손짓' ⇒ "turn left"
    - 그 외(수직/애매/손 없음)는 '제스처 없음' ⇒ gesture 갱신 안 함 (이전 값 유지)
    - gesture 변수에는 절대 'stop' 같은 값 안 넣음
    - main에서 한 번 freeze 요청이 오면, 그 이후로는 제스처를 더 이상 업데이트하지 않음
    """

    def __init__(self, cap, horizontal_ratio_threshold=1.3):
        """
        cap: open_realsense_capture() 에서 이미 '프레임까지' 확인하고 넘겨준
             cv2.VideoCapture 객체 (release 미호출 상태)
        horizontal_ratio_threshold:
            |dx| >= horizontal_ratio_threshold * |dy| 일 때만
            '수평에 가깝게 뻗은 손가락'으로 간주.
            값이 클수록 '옆으로 더 확실히 뻗은' 제스처만 인식.
        """
        super().__init__(daemon=True)
        self.cap = cap
        self.running = True

        self._lock = threading.Lock()
        self._latest_frame_bgr = None

        # gesture는 "turn right", "turn left" 또는 None
        self._gesture = None

        self.horizontal_ratio_threshold = horizontal_ratio_threshold

        # 제스처 freeze 플래그
        self._gesture_frozen = False

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        """가장 최근 프레임(BGR)을 복사해서 반환 (없으면 None)"""
        with self._lock:
            if self._latest_frame_bgr is None:
                return None
            return self._latest_frame_bgr.copy()

    def get_gesture(self):
        """
        현재까지 인식된(또는 freeze된) 제스처 반환.
        - "turn right"
        - "turn left"
        - None (아직 아무 제스처도 안정적으로 인식된 적 없음)
        """
        with self._lock:
            return self._gesture

    def freeze_gesture_once(self):
        """
        제스처를 한 번만 freeze.
        이후에는 _update_gesture에서 더 이상 값이 바뀌지 않음.
        """
        with self._lock:
            if not self._gesture_frozen:
                print(f"[GESTURE] freeze 시점 제스처: {self._gesture}")
                self._gesture_frozen = True
                print(f"[GESTURE] 초기 제스처 freeze: {self._gesture}")

    def _update_gesture(self, new_gesture):
        """
        gesture 값 갱신.
        - new_gesture가 None이면 무시
        - frozen 상태면 무시
        - 값이 바뀔 때만 로그 출력
        """
        if new_gesture is None:
            return

        with self._lock:
            if self._gesture_frozen:
                return
            if new_gesture != self._gesture:
                self._gesture = new_gesture
                print(f"[GESTURE] {self._gesture}")

    def _infer_gesture_from_hand(self, hand_landmarks):
        WRIST = 0
        INDEX_TIP = 8
    
        wrist = hand_landmarks.landmark[WRIST]
        tip = hand_landmarks.landmark[INDEX_TIP]
    
        dx = tip.x - wrist.x
        dy = tip.y - wrist.y  # y는 아래로 증가 (이미지 좌표)
    
        # 너무 짧으면 노이즈
        if (dx**2 + dy**2) < 1e-4:
            print("[DEBUG] 너무 짧은 벡터라 노이즈로 간주")
            return None
    
        ratio = abs(dx) / (abs(dy) + 1e-6)
        print(f"[DEBUG] dx={dx:.4f}, dy={dy:.4f}, |dx|/|dy|={ratio:.2f}")
    
        # *** 여기 문제났던 줄 ***
        if abs(dx) < self.horizontal_ratio_threshold * abs(dy):
            print("[DEBUG] 수평 성분이 부족해서 제스처로 인정 안 함")
            return None
    
        if dx > 0:
            print("[DEBUG] → turn right 후보")
            return "turn right"
        else:
            print("[DEBUG] → turn left 후보")
            return "turn left"
