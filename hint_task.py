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
from collections import deque
from queue import Queue, Empty
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
            else:
                print("[DEBUG] no hand")
          
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


class AsyncFrameSamplerSaver(threading.Thread):
    """
    - 매 interval_sec마다 cam_thread의 latest frame을 복사해서 queue에 넣고
    - queue에서 받아 JPEG 저장 + 최근 3장(순서 보장) ring buffer(deque) 업데이트
    - 메인 루프는 get_last_k(3)로 최근 3장을 즉시 가져오기만 하면 됨 (가벼움)
    """
    def __init__(self, cam_thread: GestureCamera,
                 save_dir: str,
                 interval_sec: float = 2.0,
                 jpg_quality: int = 90,
                 keep_k: int = 3):
        super().__init__(daemon=True)
        self.cam_thread = cam_thread
        self.save_dir = save_dir
        self.interval_sec = interval_sec
        self.jpg_quality = jpg_quality
        self.keep_k = keep_k

        os.makedirs(self.save_dir, exist_ok=True)

        self._q = Queue(maxsize=8)   # 너무 커지면 backpressure (메모리 폭주 방지)
        self._running = True
        self._lock = threading.Lock()
        self._buf = deque(maxlen=keep_k)  # [(path, pil_img), ...]

        self._idx = 0

    def stop(self):
        self._running = False

    def get_last_k(self, k: int = 3):
        """최근 k장의 (path, pil_img) 리스트를 오래된->최신 순으로 반환"""
        with self._lock:
            items = list(self._buf)
        if len(items) >= k:
            return items[-k:]
        return items  # 아직 k장 안 모였으면 있는 만큼만

    def run(self):
        next_t = time.perf_counter()
        while self._running:
            # 1) 2초마다 최신 프레임을 "복사해서" 큐에 넣기 (가벼움)
            now = time.perf_counter()
            if now >= next_t:
                frame = self.cam_thread.get_latest_frame()
                if frame is not None:
                    try:
                        # copy는 이미 get_latest_frame에서 copy라서 추가 copy 불필요하지만 안전하게 두려면 .copy()
                        self._q.put(frame, timeout=0.1)
                    except Exception:
                        pass
                next_t += self.interval_sec

            # 2) 큐에서 프레임 꺼내서 저장 + 버퍼 업데이트 (여기가 비용 큰 부분, 메인과 분리됨)
            try:
                frame_bgr = self._q.get(timeout=0.02)
            except Empty:
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            self._idx += 1
            fname = f"frame_{ts}_{self._idx:06d}.jpg"
            fpath = os.path.join(self.save_dir, fname)

            # JPEG 저장 (OpenCV)
            try:
                cv2.imwrite(fpath, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
            except Exception:
                # 저장 실패해도 버퍼는 계속 굴러가게
                fpath = ""

            # PIL(=VLM 입력용) 변환은 여기서 1번만 해두면 메인 루프가 가벼워짐
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
            except Exception:
                pil_img = None

            with self._lock:
                if pil_img is not None:
                    self._buf.append((fpath, pil_img))

        # drain optional


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
            dtype=LOCAL_DTYPE,
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

# def query_chatgpt_action(
#     client: OpenAI,
#     model_name: str,
#     frame_bgr,
#     gesture,
#     spoken_text,
#     past_action,        # <==== ADD (list)
#     turn_satisfied,     # <==== ADD (string "true"/"false" or boolean)
# ):
#     """
#     input:
#       - frame_bgr, gesture, spoken_text, past_action(list), turn_satisfied("true"/"false")
#     output:
#       - next_action (str), reason (str)
#     """
#     t_total0 = time.perf_counter()

#     if frame_bgr is None:
#         print("[WARN] frame_bgr가 None → 안전하게 'stop' 반환")
#         return "stop", "No camera frame."

#     # --- encode image ---
#     try:
#         frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
#         ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#         if not ok:
#             raise RuntimeError("cv2.imencode failed")
#         data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
#     except Exception as e:
#         print(f"[ERROR] 이미지 인코딩 실패 ({e}) → 'stop' 반환")
#         return "stop", "JPEG encode failed."

#     gesture_str = _normalize_gesture_for_prompt(gesture)
#     spoken_text = (spoken_text or "").strip()

#     # past_action 길이가 너무 길어지면 LLM이 헷갈리니 최근 N개만 권장
#     past_action = past_action or []
#     past_action_tail = past_action[-10:]  # <==== RECOMMENDED: 최근 10개만

#     # turn_satisfied는 프롬프트 일관성을 위해 문자열 "true"/"false"로 통일 추천
#     if isinstance(turn_satisfied, bool):
#         turn_satisfied_str = "true" if turn_satisfied else "false"
#     else:
#         turn_satisfied_str = str(turn_satisfied).lower()
#         if turn_satisfied_str not in ("true", "false"):
#             turn_satisfied_str = "false"

#     # <==== CHANGED: JSON array로만 출력 강제
#     system_instruction = (
#         "You are a navigation decision module for a mobile robot.\n"
#         "You MUST follow the user's prompt instructions.\n"
#         "Return ONLY a valid JSON array with exactly 2 strings:\n"
#         '["<next_action>", "<reason>"]\n'
#         "Rules:\n"
#         "- <next_action> MUST be exactly one of: forward, left, right, stop, goal\n"
#         "- <reason> is a short explanation (1-3 sentences)\n"
#         "- Output ONLY the JSON array. No extra text, no markdown, no code block.\n"
#     )

#     template = load_prompt_template("prompt.txt")

#     # <==== CHANGED: prompt 템플릿에 새 placeholder 추가 (prompt.txt에도 반드시 넣어야 함)
#     prompt = (
#         template
#         .replace("{spoken_text}", spoken_text)
#         .replace("{gesture_str}", gesture_str)
#         .replace("{past_action}", json.dumps(past_action_tail, ensure_ascii=False))
#         .replace("{turn_satisfied}", turn_satisfied_str)
#     )

#     try:
#         t_llm0 = time.perf_counter()
#         resp = client.responses.create(
#             model=model_name,
#             # temperature=0,  # <==== OPTIONAL: 결정 문제면 0 권장 (지원되면 켜세요)
#             input=[
#                 {
#                     "role": "system",
#                     "content": [{"type": "input_text", "text": system_instruction}],
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "input_text", "text": prompt},
#                         {"type": "input_image", "image_url": data_url},
#                     ],
#                 },
#             ],
#         )
#         t_llm1 = time.perf_counter()
#     except Exception as e:
#         print(f"[ERROR] ChatGPT API 호출 실패: {e} → 'stop' 반환")
#         return "stop", f"ChatGPT API error: {e}"

#     full_text = (resp.output_text or "").strip()

#     # <==== NEW: JSON 파싱 (파싱 없애는 게 아니라, '항상 JSON만' 나오게 해서 파싱을 단순화)
#     next_action = "stop"
#     reason = "Failed to parse model output."

#     try:
#         arr = json.loads(full_text)
#         if isinstance(arr, list) and len(arr) == 2:
#             cand_action = str(arr[0]).strip().lower()
#             cand_reason = str(arr[1]).strip()
#             if cand_action in ACTION_SPACE:
#                 next_action = cand_action
#                 reason = cand_reason
#             else:
#                 next_action = "stop"
#                 reason = f"Invalid action token from model: {cand_action}"
#         else:
#             reason = f"Output is not a 2-element JSON array: {full_text}"
#     except Exception as e:
#         reason = f"JSON parse error: {e} | raw={full_text}"

#     t_total1 = time.perf_counter()
#     print(f"[TIME] total per-frame (encode+LLM+parse): {(t_total1 - t_total0):.3f} s")

#     return next_action, reason

def query_local_qwen_action(processor, model, pil_imgs, gesture, spoken_text):
    """
    pil_imgs: [PIL.Image, PIL.Image, PIL.Image] (오래된 -> 최신 순)
    return: next_action (str)
    """
    if not pil_imgs or len(pil_imgs) == 0:
        return "stop"

    gesture_str = _normalize_gesture_for_prompt(gesture)
    spoken_text = (spoken_text or "").strip()

    template = load_prompt_template("prompt.txt")

    # ✅ 학습 포맷 최대한 맞추기: <image> 토큰을 K번 붙이고 줄바꿈
    k = len(pil_imgs)
    prompt = (("<image>" * k) + "\n" +
              template.replace("{spoken_text}", spoken_text)
                      .replace("{gesture_str}", gesture_str))

    system_instruction = (
        "You are a short-horizon navigation policy for a mobile robot.\n"
        "Output ONLY one token among: forward, left, right, stop, goal\n"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=pil_imgs,              # ✅ 3장 그대로 넣기
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )
    except Exception as e:
        print("[ERROR] Local VLM generate error:", e)
        return "stop"

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    full_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

    cand = re.split(r"\s+", full_text)[0]
    cand = cand.strip().strip('"\'').strip("[](),.")

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
  
    # ✅ 2초마다 프레임 저장/버퍼링 쓰레드 시작
    sampler = AsyncFrameSamplerSaver(
        cam_thread=cam_thread,
        save_dir="./log",     # 네가 원하던 저장 폴더
        interval_sec=2.0,
        jpg_quality=90,
        keep_k=3,
    )
    sampler.start()

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
    # 3-2) 6초 주기로 카메라 뷰 기반 next_action 업데이트
    # ==========================
    i=0
    # past_action = []          # <==== ADD: 처음엔 빈 리스트
    # turn_satisfied = "false"  # <==== ADD: 문자열로 통일 ("true"/"false")
    try:
        while True:
            # frame_bgr = cam_thread.get_latest_frame()
            # gesture_now = cam_thread.get_gesture()  # freeze 이후엔 항상 initial_gesture와 동일
            # cv2.imwrite(f"./Hallway1/frame_{i}.jpg", frame_bgr)

            items = sampler.get_last_k(3)
            pil_imgs = [pil for (_, pil) in items]   # 3장
          
            # next_action과 전체 LLM 응답을 모두 받음
            next_action = query_local_qwen_action(
                processor=processor,
                model=local_model,
                pil_imgs=pil_imgs,
                gesture=initial_gesture,
                spoken_text=spoken_text_initial,
            )

            print(f"[RESULT] instruction    : {spoken_text_initial}")
            print(f"[RESULT] gesture: {initial_gesture}")
            print(f"[RESULT] next_action: {next_action}")
            print("--------------------------------------------------------")
            # 여기 if문으로 turn_satisfied 여부 검사하는 if문 넣어야함.
            i+=1
            #time.sleep(6.0)


    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, 종료 중...")

    finally:
        sampler.stop()
        cam_thread.stop()
        cam_thread.join()
        print("[INFO] 종료 완료")


if __name__ == "__main__":
    main()
