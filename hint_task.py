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
import warnings  # <- 새로 추가

# protobuf deprecation warning 숨기기
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.",
    category=UserWarning,
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
    RealSense 카메라를 robust하게 여는 함수.
    (중략: 내용 동일, 생략 가능)
    """
    print("[INFO] RealSense 카메라 자동 탐색: `v4l2-ctl --list-devices` 실행")

    # ... 여기부터 open_realsense_capture 내용은
    # 네가 마지막에 사용하던 버전 그대로 사용한다고 가정 ...
    # (길어서 생략, 필요하면 그대로 유지하면 됨)
    # 이 함수는 프로파일 요구사항이 아니라서 손 안 댐.
    # -----------------------------
    # 그냥 너 코드 그대로 두면 됨
    # -----------------------------
    # 아래는 예시로 마지막 부분만 남겨둠:

    raise RuntimeError(
        "[ERROR] RealSense 컬러 후보 /dev/video* 노드들 중 어떤 것도 유효한 컬러 프레임을 제공하지 않습니다.\n"
        "실제로 어떤 /dev/videoX가 컬러인지 v4l2-ctl 로 확인 후 CAMERA_DEVICE_ID를 설정해 주세요."
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

    def __init__(self, cap, horizontal_ratio_threshold=1.3):
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
        with self._lock:
            return self._gesture

    def freeze_gesture_once(self):
        with self._lock:
            if not self._gesture_frozen:
                self._gesture_frozen = True
                print(f"[GESTURE] 초기 제스처 freeze: {self._gesture}")

    def _update_gesture(self, new_gesture):
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
        dy = tip.y - wrist.y  # 아래로 증가

        if (dx ** 2 + dy ** 2) < 1e-4:
            return None

        if abs(dx) < self.horizontal_ratio_threshold * abs(dy):
            return None

        if dx > 0:
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

            with self._lock:
                self._latest_frame_bgr = frame_bgr

            if self._gesture_frozen:
                time.sleep(0.03)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                print("[DEBUG] hand detected")
            else:
                print("[DEBUG] no hand")

            new_gesture = None

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                g = self._infer_gesture_from_hand(hand_lms)
                print("[DEBUG] inferred gesture:", g)
                if g is not None:
                    new_gesture = g

            self._update_gesture(new_gesture)

            time.sleep(0.03)

        self.cap.release()
        self._mp_hands.close()
        print("[INFO] GestureCamera 쓰레드 종료")


# ==========================
# 2) Gemini로 액션 결정 (초기 gesture/instruction 사용)
# ==========================

def _normalize_gesture_for_prompt(gesture: str | None) -> str:
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

    gesture_str = _normalize_gesture_for_prompt(gesture)
    spoken_text = spoken_text or ""

    print(f"[DEBUG] Gemini spoken_text : {spoken_text}")
    print(f"[DEBUG] Gemini gesture_str : {gesture_str}")

    system_instruction = (
        "You are a navigation decision module for a mobile robot. "
        "You must follow the prompt instructions and finally choose ONE action "
        "from this action_space: forward, left, right, stop, goal_signal. "
        "In your answer, clearly state the final chosen action following the required format."
    )

    template = load_prompt_template("prompt.txt")

    prompt = (
        template
        .replace("{spoken_text}", spoken_text)
        .replace("{gesture_str}", gesture_str)
    )

    # ------------ [PROFILE] Gemini 호출 시간 측정 ------------
    t_llm_start = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=[
            system_instruction,
            image_part,
            prompt,
        ],
    )
    t_llm_end = time.time()
    print(f"[PROFILE] Gemini generate_content latency: {t_llm_end - t_llm_start:.3f} s")
    # ------------------------------------------------------

    full_text = (response.text or "").strip()
    full_lower = full_text.lower()

    for action in ACTION_SPACE:
        if full_lower.startswith(action):
            return action, full_text

    for action in ACTION_SPACE:
        if action in full_lower:
            return action, full_text

    return "stop", full_text


# ==========================
# 3) STT + 메인 루프 (초기 한 번만 STT / 제스처 freeze)
# ==========================
class DummyVolumeSignal:
    """Qt 없이 STTProcessor.listen_once를 쓰기 위한 더미 객체"""

    def emit(self, level, db):
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

            # ------------ [PROFILE] STT 전체 시간 측정 시작 ------------
            t_stt_start = time.time()

            # 녹음
            audio_path = stt.listen_once(volume_level_changed=dummy_volume)
            t_after_record = time.time()

            if audio_path is None:
                print("[MAIN] 녹음된 오디오가 없음, 다시 대기")
                print(f"[PROFILE] STT listen_once duration (audio_path=None): "
                      f"{t_after_record - t_stt_start:.3f} s")
                continue

            # STT 변환
            spoken_text = stt.transcribe(audio_file=audio_path).strip()
            t_stt_end = time.time()
            # ---------------------------------------------------------

            # 프로파일 로그 출력 (성공/실패 상관없이 찍어도 됨)
            print(f"[PROFILE] STT listen_once duration: {t_after_record - t_stt_start:.3f} s")
            print(f"[PROFILE] STT transcribe duration : {t_stt_end - t_after_record:.3f} s")
            print(f"[PROFILE] STT total (record+ASR)   : {t_stt_end - t_stt_start:.3f} s")

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

            # ------------ [PROFILE] query_gemini_action 전체 시간 측정 ------------
            t_query_start = time.time()
            next_action, llm_text = query_gemini_action(
                client=client,
                model_name=GEMINI_MODEL_NAME,
                frame_bgr=frame_bgr,
                gesture=initial_gesture,
                spoken_text=spoken_text_initial,
            )
            t_query_end = time.time()
            print(f"[PROFILE] query_gemini_action total: {t_query_end - t_query_start:.3f} s")
            #  (여기 total에는 JPEG 인코딩 + prompt 구성 + Gemini 호출까지 전부 포함)
            # ---------------------------------------------------------------

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
