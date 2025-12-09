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
# 0) RealSense 카메라 열기 관련 유틸
# ==========================

def is_rgb_stream_device(dev_path: str) -> bool:
    """
    주어진 /dev/videoX 노드가 '컬러(RGB) 스트림'을 제공하는지
    v4l2-ctl --list-formats-ext 결과를 기반으로 논리적으로 판별한다.

    - 픽셀 포맷이 GREY, Y8, Y16, Z16 등 '단일 채널/Depth' 위주이면 IR/Depth로 간주
    - 픽셀 포맷에 RGB, YUYV, MJPG, NV12 등 '컬러 포맷'이 하나라도 있으면 RGB 후보로 간주
    """
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--device", dev_path, "--list-formats-ext"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except Exception as e:
        print(f"[WARN] {dev_path}: v4l2-ctl --list-formats-ext 실패: {e}")
        return False

    out = proc.stdout.lower()

    # 1) 장치가 아예 "Video Capture" 타입을 지원하는지 확인 (webcam/카메라)
    if "type: video capture" not in out:
        print(f"[DEBUG] {dev_path}: 'Type: Video Capture' 포맷이 없음.")
        return False

    # 2) 모노/Depth로 확실한 포맷 키워드들
    mono_keywords = [
        "greyscale", "greyscale", "grey", "y8", "y10", "y12", "y16",
        "z16", "depth", "infrared", "ir",
    ]

    # 3) 컬러 포맷(멀티 채널)로 간주할 키워드들
    color_keywords = [
        "rgb", "bgr", "yuyv", "uyvy", "nv12", "nv21", "mjpg", "mjpeg", "yuv",
    ]

    # 컬러 포맷이 하나라도 있으면 RGB 후보
    has_color = any(kw in out for kw in color_keywords)

    if has_color:
        print(f"[DEBUG] {dev_path}: 컬러 포맷 감지 → RGB 후보")
        return True

    # 컬러는 없고, 모노/Depth 키워드만 있으면 IR/Depth로 간주
    if any(kw in out for kw in mono_keywords):
        print(f"[DEBUG] {dev_path}: 모노/Depth 포맷만 감지 → RGB 아님")
        return False

    # 여기까지 왔는데도 컬러/모노를 명확히 구분 못 하면 보수적으로 False
    print(f"[DEBUG] {dev_path}: 컬러/모노 판별 불가 → RGB 아님으로 처리")
    return False


def open_realsense_capture():
    """
    1) `v4l2-ctl --list-devices`에서 'RealSense'가 들어간 디바이스 블록을 찾고
    2) 그 블록 안의 /dev/videoX 후보들 중에서
         - v4l2-ctl --device /dev/videoX --list-formats-ext 결과를 기반으로
           '컬러(RGB) 스트림'을 제공하는 노드만 필터링
         - 그 RGB 후보 노드들에 대해 OpenCV(cv2.VideoCapture)로 열어서
           실제로 프레임이 나오는지 확인
       → 정상 RGB 프레임이 나오면 그 cap을 그대로 반환.

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
    rs_blocks = [
        (name, devs) for name, devs in devices_info
        if "realsense" in name.lower()
    ]

    if not rs_blocks:
        raise RuntimeError(
            "[ERROR] `v4l2-ctl --list-devices` 결과에서 RealSense 장치를 찾지 못했습니다.\n"
            "RealSense가 /dev/video* 로 제대로 잡혀 있는지 확인하세요."
        )

    # RealSense 블록 내 모든 /dev/video* 후보 수집
    all_candidate_devs = []
    for name, devs in rs_blocks:
        print(f"[DEBUG] RealSense block: '{name}' -> {devs}")
        all_candidate_devs.extend(devs)

    if not all_candidate_devs:
        raise RuntimeError(
            "[ERROR] RealSense 장치는 있으나 /dev/video* 노드를 찾지 못했습니다."
        )

    # 1차 필터: 'RGB/컬러 스트림'을 제공하는 노드만 남기기
    rgb_candidate_devs = []
    for dev in all_candidate_devs:
        if is_rgb_stream_device(dev):
            rgb_candidate_devs.append(dev)

    if not rgb_candidate_devs:
        raise RuntimeError(
            "[ERROR] RealSense /dev/video* 중 RGB(컬러) 포맷을 제공하는 노드를 찾지 못했습니다.\n"
            "v4l2-ctl --device /dev/videoX --list-formats-ext 출력으로 포맷 구성을 확인해보세요."
        )

    # 2차 필터: RGB 후보 노드들에 대해 실제로 프레임이 잘 나오는지 확인
    for dev in rgb_candidate_devs:
        print(f"[INFO] RealSense RGB 후보 노드 테스트: {dev}")
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

        # 여기까지 왔으면 실제로 usable한 'RGB 컬러 스트림'이라고 판단
        m = re.search(r"/dev/video(\d+)", dev)
        if not m:
            print(f"[WARN] {dev} 의 인덱스를 파싱하지 못했습니다.")
            cap.release()
            continue

        idx = int(m.group(1))
        print(f"[INFO] 사용 가능한 RealSense RGB 카메라 확정: device='{dev}', index={idx}")
        # cap은 열어둔 채로 반환
        return cap, idx, dev

    # RGB 후보들 중 어느 것도 실제로 프레임을 못 읽는 경우
    raise RuntimeError(
        "[ERROR] RealSense RGB /dev/video* 노드 중 어느 것도 OpenCV로 프레임을 읽을 수 없습니다.\n"
        "컨테이너/호스트의 권한, udev, --device 설정을 다시 확인해보세요."
    )


# ==========================
# 1) 카메라 + 제스처 쓰레드 (제스처 freeze 기능 추가)
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

    def __init__(self, cap, horizontal_ratio_threshold=1.3, debug=False):
        """
        cap: open_realsense_capture() 에서 이미 '프레임까지' 확인하고 넘겨준
             cv2.VideoCapture 객체 (release 미호출 상태)
        horizontal_ratio_threshold:
            |dx| >= horizontal_ratio_threshold * |dy| 일 때만
            '수평에 가깝게 뻗은 손가락'으로 간주.
            값이 클수록 '옆으로 더 확실히 뻗은' 제스처만 인식.
        debug:
            True면 프레임 정보/검출 결과를 좀 더 많이 로그로 찍고,
            첫 프레임을 파일로 저장해서 실제로 어떤 영상인지 확인할 수 있게 함.
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

        # === 디버그 관련 ===
        self.debug = debug
        self._first_frame_dumped = False
        self._frame_count = 0
        self._hand_detect_count = 0


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
        """
        단순 규칙 기반 제스처 해석 (좌/우만):

        - 손목(WRIST) → 검지 손끝(INDEX_TIP) 방향 벡터 (dx, dy) 사용
        - |dx|가 |dy|보다 충분히 크면 "수평에 가깝다"고 판단
            → dx > 0  : turn right
            → dx < 0  : turn left
        - 그 외 (수직에 가깝거나 애매한 경우) → 제스처 없음 (None)

        ※ 이미지 좌표 기준: x 오른쪽 증가, y 아래쪽 증가
        """
        WRIST = 0
        INDEX_TIP = 8

        wrist = hand_landmarks.landmark[WRIST]
        tip = hand_landmarks.landmark[INDEX_TIP]

        dx = tip.x - wrist.x
        dy = tip.y - wrist.y  # 아래로 증가

        # 너무 짧은 벡터는 노이즈로 간주
        if (dx ** 2 + dy ** 2) < 1e-4:
            return None

        # 수평 성분이 수직 성분보다 충분히 크지 않으면 → 제스처 없음
        if abs(dx) < self.horizontal_ratio_threshold * abs(dy):
            return None

        # 여기까지 왔으면 '옆으로 꽤 뻗은' 손가락이라고 간주
        if dx > 0:
            # 화면 오른쪽을 가리킴 ⇒ 로봇 에고 프레임 기준 "turn right"
            return "turn right"
        else:
            return "turn left"

    def run(self):
        if not self.cap.isOpened():
            print("[ERROR] GestureCamera: 전달받은 cap이 이미 닫혀 있습니다.")
            return

        print("[INFO] GestureCamera 쓰레드 시작")

        while self.running:
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                print("[WARN] 카메라 프레임 읽기 실패")
                time.sleep(0.05)
                continue

            self._frame_count += 1

            # === 디버그: 첫 프레임 한 번 저장해서 실제로 RGB인지 확인 ===
            if self.debug and not self._first_frame_dumped:
                try:
                    os.makedirs("/tmp/gesture_debug", exist_ok=True)
                    dump_path = "/tmp/gesture_debug/first_frame_bgr.jpg"
                    cv2.imwrite(dump_path, frame_bgr)
                    print(f"[DEBUG] GestureCamera: 첫 프레임을 {dump_path} 에 저장했습니다.")
                    print(f"[DEBUG] frame shape={frame_bgr.shape}, dtype={frame_bgr.dtype}, mean={frame_bgr.mean():.2f}")
                    self._first_frame_dumped = True
                except Exception as e:
                    print(f"[DEBUG] 첫 프레임 저장 실패: {e}")

            # 최신 프레임 저장 (freeze 여부와 관계 없이 계속 업데이트)
            with self._lock:
                self._latest_frame_bgr = frame_bgr

            # 제스처가 이미 freeze 되었더라도, 카메라 프레임은 계속 갱신해야 함
            if self._gesture_frozen:
                time.sleep(0.03)
                continue

            # MediaPipe Hands는 RGB 입력
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(frame_rgb)

            new_gesture = None  # 기본값: 이번 프레임에서는 새 제스처 없음

            if result.multi_hand_landmarks:
                self._hand_detect_count += 1
                if self.debug and self._hand_detect_count % 30 == 0:
                    # 대략 1초마다 한 번씩 손 검출 카운트 찍기
                    print(f"[DEBUG] MediaPipe: 손 검출 누적 {self._hand_detect_count} / 프레임 {self._frame_count}")

                hand_lms = result.multi_hand_landmarks[0]
                g = self._infer_gesture_from_hand(hand_lms)
                if g is not None:
                    new_gesture = g
            else:
                if self.debug and self._frame_count % 60 == 0:
                    # 2초마다 한 번씩 "손 못 찾음" 로그
                    print(f"[DEBUG] MediaPipe: 이 시점까지 손 검출 없음 (frame={self._frame_count})")

            # 제스처 갱신 (None이면 무시 / freeze면 무시)
            self._update_gesture(new_gesture)

            # 약 30fps
            time.sleep(0.03)

        self.cap.release()
        self._mp_hands.close()
        print("[INFO] GestureCamera 쓰레드 종료")


# ==========================
# 2) Gemini로 액션 결정 (초기 gesture/instruction 사용)
# ==========================

def _normalize_gesture_for_prompt(gesture: str | None) -> str:
    """
    제스처 문자열을 prompt.txt에서 기대하는
    {"left", "right", "forward", "none"} 중 하나로 변환.
    """
    if gesture is None:
        return "none"
    g = gesture.lower()
    if "right" in g:
        return "right"
    if "left" in g:
        return "left"
    if "forward" in g:
        return "forward"
    return "none"


def query_gemini_action(client, model_name, frame_bgr, gesture, spoken_text):
    """
    - frame_bgr: 최신 카메라 RGB 관측 (BGR 포맷)
    - gesture : "turn right" / "turn left" / None (초기 제스처)
    - spoken_text: STT로 변환된 사람 발화 내용 (초기 instruction)

    반환: (next_action, full_text)
      - next_action ∈ ACTION_SPACE
      - full_text  : Gemini가 생성한 전체 응답 문자열
    """
    if frame_bgr is None:
        print("[WARN] frame_bgr가 None → 안전하게 'stop' 반환")
        return "stop", "[LLM] no frame_bgr (returned 'stop')"

    ok, jpg = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        print("[ERROR] 이미지 JPEG 인코딩 실패 → 'stop' 반환")
        return "stop", "[LLM] jpeg encode failed (returned 'stop')"

    image_part = types.Part.from_bytes(
        data=jpg.tobytes(),
        mime_type="image/jpeg",
    )

    # gesture가 아직 한 번도 인식 안된 경우 → "none"으로
    gesture_str = _normalize_gesture_for_prompt(gesture)
    spoken_text = spoken_text or ""

    print(f"[DEBUG] Gemini spoken_text : {spoken_text}")
    print(f"[DEBUG] Gemini gesture_str : {gesture_str}")

    # system_instruction은 high-level role만 지정하고
    # 상세 정책은 prompt.txt에 포함되어 있음
    system_instruction = (
        "You are a navigation decision module for a mobile robot. "
        "You must follow the prompt instructions and finally choose ONE action "
        "from this action_space: forward, left, right, stop, goal_signal. "
        "In your answer, clearly state the final chosen action following the required format."
    )

    # --- 외부 텍스트 파일에서 템플릿 로드 ---
    template = load_prompt_template("prompt.txt")

    # prompt.txt 안의 플레이스홀더를 실제 값으로 치환
    prompt = (
        template
        .replace("{spoken_text}", spoken_text)
        .replace("{gesture_str}", gesture_str)
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            system_instruction,
            image_part,
            prompt,
        ],
    )

    # 전체 텍스트
    full_text = (response.text or "").strip()
    full_lower = full_text.lower()

    # next_action 파싱: 맨 앞 토큰 기준 우선
    for action in ACTION_SPACE:
        if full_lower.startswith(action):
            return action, full_text

    # 그게 안 되면, 포함 여부라도 체크
    for action in ACTION_SPACE:
        if action in full_lower:
            return action, full_text

    # 최후의 보루: stop
    return "stop", full_text


# ==========================
# 3) STT + 메인 루프 (초기 한 번만 STT / 제스처 freeze)
# ==========================
class DummyVolumeSignal:
    """Qt 없이 STTProcessor.listen_once를 쓰기 위한 더미 객체"""

    def emit(self, level, db):
        # 여기서는 볼륨 UI가 없으니 아무 것도 안 함
        pass


def main():
    # 0) RealSense 카메라 열기 (cap + index + dev_path)
    try:
        cap, rs_index, dev_path = open_realsense_capture()
    except RuntimeError as e:
        print(e)
        print("[FATAL] RealSense 카메라를 찾을 수 없어 종료합니다.")
        return

    # 1) 제스처 카메라 쓰레드 시작 (이미 열린 cap을 넘김)
    cam_thread = GestureCamera(cap=cap)
    cam_thread.start()

    # 2) STT 준비 (ReSpeaker 입력, 초기 한 번만 사용)
    stt = STTProcessor()

    # 3) Gemini 클라이언트 준비
    client = genai.Client(api_key=GEMINI_API_KEY)

    dummy_volume = DummyVolumeSignal()

    # ==========================
    # 3-1) 초기 음성 instruction + 제스처 한 번만 획득
    # ==========================
    spoken_text_initial = None
    initial_gesture = None

    try:
        while True:
            print("\n[MAIN] 초기 instruction 발화를 기다리는 중... (ReSpeaker로 말해줘)")
            audio_path = stt.listen_once(volume_level_changed=dummy_volume)

            if audio_path is None:
                print("[MAIN] 녹음된 오디오가 없음, 다시 대기")
                continue

            spoken_text = stt.transcribe(audio_file=audio_path).strip()
            if not spoken_text:
                print("[MAIN] STT 결과가 비어 있음, 다시 대기")
                continue

            # 여기서 STT가 성공한 시점의 제스처를 freeze
            cam_thread.freeze_gesture_once()
            initial_gesture = cam_thread.get_gesture()

            spoken_text_initial = spoken_text
            print(f"[INIT] 초기 instruction STT: {spoken_text_initial}")
            print(f"[INIT] 초기 제스처: {initial_gesture}")
            break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, 초기 STT 단계에서 종료.")
        cam_thread.stop()
        cam_thread.join()
        return

    print("\n[INFO] 초기 instruction + 제스처 확보 완료.")
    print("[INFO] 이후에는 마이크는 더 이상 사용하지 않고,")
    print("       6초마다 카메라 RGB + (초기 제스처, 초기 instruction)을 기반으로")
    print("       Gemini에게 다음 액션을 질의합니다.\n")

    # ==========================
    # 3-2) 6초 주기로 카메라 뷰 기반 next_action 업데이트
    # ==========================
    try:
        while True:
            frame_bgr = cam_thread.get_latest_frame()
            gesture_now = cam_thread.get_gesture()  # freeze 이후엔 항상 initial_gesture와 동일

            # next_action과 전체 LLM 응답을 모두 받음
            next_action, llm_text = query_gemini_action(
                client=client,
                model_name=GEMINI_MODEL_NAME,
                frame_bgr=frame_bgr,
                gesture=initial_gesture,
                spoken_text=spoken_text_initial,
            )

            print(f"[LOOP] 현재 제스처(실시간/동일): {gesture_now}")
            print(f"[RESULT] instruction    : {spoken_text_initial}")
            print(f"[RESULT] initial_gesture: {initial_gesture}")
            print(f"[RESULT] next_action    : {next_action}")
            print("[RESULT] full LLM reply --------------------------------")
            print(llm_text)
            print("--------------------------------------------------------")
            print("-" * 60)

            time.sleep(6.0)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, 종료 중...")

    finally:
        cam_thread.stop()
        cam_thread.join()
        print("[INFO] 종료 완료")


if __name__ == "__main__":
    main()
