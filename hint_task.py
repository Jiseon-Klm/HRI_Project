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
  action_space = ["forward", "left", "right", "stop", "goal"] 중
  next_action을 계속 업데이트해서 터미널에 출력.
"""
import threading
import time
import subprocess
import re
import json
import os
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import base64

# from openai import OpenAI
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

from llm_stt_tts import STTProcessor, TTSProcessorPiper
from config import (
    CAMERA_DEVICE_ID,   # 지금은 안 써도 되지만 호환용으로 남겨둠
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
)
import warnings  # <- 새로 추가
LOCAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# protobuf deprecation warning 숨기기
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.",
    category=UserWarning,
)
# LLM이 최종적으로 선택해야 하는 액션 공간
# prompt.txt와 정확히 동일한 토큰을 사용해야 파싱이 잘 됨
ACTION_SPACE = ["forward", "left", "right", "stop", "goal"]

# 프롬프트 템플릿 캐시용 전역 변수
PROMPT_TEMPLATE: str | None = None
CHATGPT_MODEL_NAME = "gpt-5"  
LOCAL_QWEN_VLM_DIR = "./qwen_lora_merged"

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
class RealSenseColorCap:
    def __init__(self, width=640, height=480, fps=30, serial: str | None = None, timeout_ms=2000):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed (rs is None)")

        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)

        # ✅ 핵심: COLOR 스트림만 강제로 열기 (IR/Depth는 애초에 enable 안 함)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = self.pipeline.start(self.config)
        self._opened = True

        # warm-up (auto exposure settle)
        for _ in range(15):
            frames = self.pipeline.wait_for_frames(timeout_ms)
            _ = frames.get_color_frame()

        # 추가 안전 확인: 실제로 color stream이 열렸는지 확인
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        if color_stream.format() != rs.format.bgr8:
            # 이 케이스는 거의 없지만, 있으면 바로 fail 시켜서 문제를 빨리 드러내요
            raise RuntimeError(f"Color stream format is not BGR8: {color_stream.format()}")

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        try:
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
            color = frames.get_color_frame()
            if not color:
                return False, None
            frame_bgr = np.asanyarray(color.get_data())  # 이미 BGR8
            if frame_bgr is None or frame_bgr.size == 0:
                return False, None
            return True, frame_bgr
        except Exception:
            return False, None

    def release(self):
        if self._opened:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self._opened = False

def open_realsense_capture():
    """
    RealSense 카메라를 'pyrealsense2 COLOR 스트림'으로만 연다.
    - V4L2(/dev/videoX) fallback을 절대 하지 않는다. (IR 혼선 근본 차단)
    - 여러 대 연결되어 있어도 'RGB/Color 센서가 있는 디바이스'를 우선 선택한다.
    반환: (cap, index, dev_path)
      - cap      : RealSenseColorCap 객체
      - index    : 더미로 -1
      - dev_path : 식별 문자열
    """
    print("[INFO] RealSense 카메라 오픈: pyrealsense2 COLOR ONLY (NO V4L2 fallback)")

    if rs is None:
        raise RuntimeError("pyrealsense2 is not available (rs is None).")

    # 1) 연결된 디바이스 탐색
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("[FATAL] No RealSense device detected (rs.context().query_devices() == 0).")

    # 2) 'Color/RGB 센서'가 있는 디바이스를 우선 선택
    chosen_serial = None
    chosen_name = None

    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)

            # 센서 목록에서 Color/RGB 센서 여부 확인
            has_color = False
            for s in dev.query_sensors():
                try:
                    sname = s.get_info(rs.camera_info.name).lower()
                    # RealSense 드라이버/모델별로 표현이 조금 달라서 넉넉히 체크
                    if ("rgb" in sname) or ("color" in sname):
                        has_color = True
                        break
                except Exception:
                    continue

            if has_color:
                chosen_serial = serial
                chosen_name = name
                break

        except Exception:
            continue

    # Color 센서 있는 디바이스를 못 찾으면, 그래도 첫 디바이스라도 잡아보되
    # 이후 RealSenseColorCap에서 color stream enable 시 실패하면 바로 에러로 끝남.
    if chosen_serial is None:
        dev0 = devices[0]
        chosen_serial = dev0.get_info(rs.camera_info.serial_number)
        chosen_name = dev0.get_info(rs.camera_info.name)
        print(f"[WARN] Color/RGB sensor device not clearly identified. Fallback to first device: {chosen_name} ({chosen_serial})")
    else:
        print(f"[INFO] Selected RealSense device: {chosen_name} (serial={chosen_serial})")

    # 3) COLOR 스트림만 여는 캡처 생성 (실패하면 즉시 예외)
    cap = RealSenseColorCap(width=640, height=480, fps=30, serial=chosen_serial)

    # 4) 반환 형식은 기존 코드와 맞춤
    return cap, -1, f"realsense:color(bgr8):{chosen_serial}"

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
            return "right"
        else:
            return "left"

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
            if result.multi_hand_landmarks:
                print("[DEBUG] hand detected")
          
            new_gesture = None  # 기본값: 이번 프레임에서는 새 제스처 없음

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                g = self._infer_gesture_from_hand(hand_lms)
                print("[DEBUG] inferred gesture:", g)
                if g is not None:
                    new_gesture = g

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

def load_local_qwen_vlm(model_dir: str):
    """
    로컬 Qwen3-VL (LoRA merge 완료된 full model) 로드.
    - 프로그램 시작 시 1번만 호출해야 함 (매 프레임 로드 금지)
    """
    print(f"[INFO] Loading local VLM from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # GPU 사용 시 device_map을 쓰면 큰 모델도 비교적 안전하게 올라가요.
    if LOCAL_DEVICE.startswith("cuda"):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=LOCAL_DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=LOCAL_DTYPE,
            device_map="cpu",
            trust_remote_code=True,
        )

    model.eval()

    # 속도 옵션(선택)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    print("[INFO] Local VLM loaded.")
    return processor, model

def query_local_qwen_action_single(processor, model, pil_img, gesture, spoken_text):
    """
    pil_img: PIL.Image (현재 프레임 1장)
    return: next_action (str)
    """
    if pil_img is None:
        return "stop"

    gesture_str = _normalize_gesture_for_prompt(gesture)
    spoken_text = (spoken_text or "").strip()
    template = load_prompt_template("prompt.txt")

    image_token = processor.image_token  # 보통 "<|image_pad|>"
    vision_start = "<|vision_start|>"
    vision_end   = "<|vision_end|>"

    # ✅ 이미지 1장만 넣음
    image_block = (vision_start + image_token + vision_end) + "\n"

    prompt = (
        image_block +
        template.replace("{spoken_text}", spoken_text)
                .replace("{gesture_str}", gesture_str)
    )

    system_instruction = (
        "You are a short-horizon navigation policy for a mobile robot.\n"
        "Output ONLY one token among: forward, left, right, stop, goal\n"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ✅ images는 리스트로 1장 전달
    t0 = time.perf_counter()
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt")
    t1 = time.perf_counter()

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # sanity check: 이미지 토큰이 실제로 들어갔는지
    try:
        decoded_head = processor.decode(inputs["input_ids"][0][:300], skip_special_tokens=False)
        if image_token not in decoded_head:
            print("[FATAL] image token not found in tokenized prompt head!")
            print("[DEBUG] decoded_head:", decoded_head)
            return "stop"
    except Exception:
        pass

    try:
        t2 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=4,   # ✅ 액션 1토큰이면 충분 (속도 ↑)
                do_sample=False,
            )
        t3 = time.perf_counter()
        print(f"[TIME] processor: {(t1 - t0):.3f}s | generate: {(t3 - t2):.3f}s")
    except Exception as e:
        print("[ERROR] Local VLM generate error:", e)
        return "stop"

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    full_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    cand = re.split(r"\s+", full_text)[0].strip().strip('"\'').strip("[](),.")
    if cand not in ACTION_SPACE:
        print(f"[WARN] invalid action output: raw='{full_text}' -> cand='{cand}'")
        return "stop"
    return cand



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
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    processor, local_model = load_local_qwen_vlm(LOCAL_QWEN_VLM_DIR)


    # 1) 제스처 카메라 쓰레드 시작 (이미 열린 cap을 넘김)
    cam_thread = GestureCamera(cap=cap)
    cam_thread.start()

    # 2) STT 준비 (ReSpeaker 입력, 초기 한 번만 사용)
    stt = STTProcessor()
    tts = TTSProcessorPiper(
        model_path="/ros2_ws/ws/voices/ko/piper-kss-korean.onnx",
        config_path="/ros2_ws/ws/voices/ko/piper-kss-korean.onnx.json",
        piper_bin="piper",
    )


    dummy_volume = DummyVolumeSignal()

    # ==========================
    # 3-1) 초기 음성 instruction + 제스처 한 번만 획득
    # ==========================
    spoken_text_initial = None
    initial_gesture = None
    tts.speak("무엇을 도와드릴까요?")
    try:
        while True:
            print("\n[MAIN] 초기 instruction 발화를 기다리는 중... (ReSpeaker로 말해줘)")
            
            audio_path = stt.listen_once(volume_level_changed=dummy_volume)
            

            if audio_path is None:
                print("[MAIN] 녹음된 오디오가 없음, 다시 대기")
                continue
            t_stt0 = time.perf_counter() 
            spoken_text = stt.transcribe(audio_file=audio_path).strip()
            t_stt1 = time.perf_counter()
            print(f"[TIME] transcribe (STT)       : {(t_stt1 - t_stt0):.1f} s")
            if not spoken_text:
                print("[MAIN] STT 결과가 비어 있음, 다시 대기")
                continue

            tts.speak("알겠어요")
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

    # ==========================
    # 3-2) 현재 프레임 1장으로 next_action 업데이트
    # ==========================
    try:
        while True:
            frame_bgr = cam_thread.get_latest_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # BGR -> PIL(RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            next_action = query_local_qwen_action_single(
                processor=processor,
                model=local_model,
                pil_img=pil_img,
                gesture=initial_gesture,
                spoken_text=spoken_text_initial,
            )

            print(f"[RESULT] instruction : {spoken_text_initial}")
            print(f"[RESULT] gesture     : {initial_gesture}")
            print(f"[RESULT] next_action  : {next_action}")
            print("--------------------------------------------------------")

            time.sleep(0.3)  # 너무 빠른 루프 방지

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, 종료 중...")

    finally:
        cam_thread.stop()
        cam_thread.join()
        print("[INFO] 종료 완료")


if __name__ == "__main__":
    main()
