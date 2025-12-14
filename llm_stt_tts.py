import subprocess
import threading
import time
import re
import queue
from collections import deque
import math
import whisper                         # ✅ STT용 Whisper
from transformers import VitsModel, AutoTokenizer
import sounddevice as sd               # (TTS에서만 사용)
import numpy as np
import soundfile as sf
import torch

from google import genai

from config import (
    SAMPLE_RATE,
    BLOCK_DURATION,
    SILENCE_DURATION,
    OUTPUT_FILE,
    MODEL_NAME,          # ✅ 이거 꼭 필요
    LANG,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    PROMPT_TEMPLATE,
    PIPER_OUTPUT_FILE,
    MMS_TTS_MODEL_ID,
    MMS_TTS_OUTPUT_FILE,
)
# ================== STT (Speech-to-Text) ==================

class STTProcessor:
    """
    VAD 기반 STT:
    - (1) PortAudio 입력 장치 목록에서 ReSpeaker를 자동 탐색
    - (2) 배경 소음 RMS 측정 → silence_threshold 자동 설정
    - (3) 말 시작(avg_rms > START_THRESHOLD) 때부터 프레임 저장 시작
    - (4) 무음이 SILENCE_DURATION 이상 지속되면 녹음 종료
    - (5) OUTPUT_FILE로 저장 후 Whisper STT 수행
    """

    def __init__(self):
        print("[INFO] Whisper 모델 로딩 중...")
        self.whisper_model = whisper.load_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")

        # ---- VAD 관련 ----
        self.audio_queue = queue.Queue()
        self.interrupt_event = threading.Event()

        # ---- 입력 디바이스 자동 선택 (ReSpeaker 우선) ----
        self.card_number, self.device_number = self._detect_respeaker_card_device()   # <==== 추가
        self.alsa_device = f"plughw:{self.card_number},{self.device_number}"          # <==== 추가
        print(f"[INFO] Detected ReSpeaker ALSA device: {self.alsa_device}")           # <==== 추가
        
        self.input_device_id = self._map_alsa_card_to_sounddevice_id(
            card=self.card_number, dev=self.device_number
        )  # <==== 추가

        # ---- 채널 수 결정 (가능하면 1ch로 다운믹스) ----
        dev_info = sd.query_devices(self.input_device_id, "input")
        max_ch = int(dev_info.get("max_input_channels", 1))
        self.channels = 1 if max_ch >= 1 else 1  # <==== 안전하게 1ch
        print(f"[INFO] STT input channels: {self.channels}")

        # ---- 배경 소음 측정 → threshold 자동 설정 ----
        self.base_noise = self._measure_noise_level(duration=1.0)
        # 너무 낮게 잡히면(완전 무음) threshold가 0에 붙어서 오탐이 커짐 → floor
        self.base_noise = max(self.base_noise, 1e-4)

        self.silence_threshold = self.base_noise * 3.5
        print(f"[INFO] base_noise(RMS)={self.base_noise:.6f}, silence_threshold={self.silence_threshold:.6f}")

    def _detect_respeaker_card_device(self):
        """
        arecord -l 출력 예:

        card 1: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: USB Audio [USB Audio]
          Subdevices: 1/1
          Subdevice #0: subdevice #0
        card 3: Generic_1 [HD-Audio Generic], device 0: ALC1220 Analog [ALC1220 Analog]

        여기서 'ReSpeaker', 'ArrayUAC10', 'Mic Array', 'USB Audio' 같은 키워드가
        포함된 라인에서 card / device 번호를 뽑는다.
        """
        print("[INFO] ReSpeaker ALSA 디바이스 자동 탐색: `arecord -l` 실행")

        try:
            proc = subprocess.run(
                ["arecord", "-l"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        except Exception as e:
            raise RuntimeError(f"[ERROR] `arecord -l` 실행 실패: {e}")

        output_lines = proc.stdout.splitlines()

        preferred_keywords = [
            "ReSpeaker",
            "ArrayUAC10",
            "Mic Array",
            "USB Audio",
        ]

        candidate = None

        for line in output_lines:
            line = line.strip()
            if not line.startswith("card "):
                continue

            # ReSpeaker 관련 키워드가 포함된 card 라인만 필터링
            if not any(kw in line for kw in preferred_keywords):
                continue

            # 예: "card 1: ArrayUAC10 [...], device 0: USB Audio [...]"
            m = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
            if m:
                card = int(m.group(1))
                dev = int(m.group(2))
                candidate = (card, dev)
                print(f"[INFO] ReSpeaker 후보 디바이스 발견: card={card}, device={dev}, line='{line}'")
                break

        if candidate is None:
            # 진짜 ReSpeaker가 안 붙어 있으면 여기서 에러
            raise RuntimeError(
                "[ERROR] `arecord -l` 출력에서 ReSpeaker 디바이스를 찾지 못했습니다.\n"
                "ReSpeaker가 제대로 연결되어 있는지, 컨테이너에 /dev/snd 가 마운트되어 있는지,\n"
                "`arecord -l` 출력에 'ReSpeaker 4 Mic Array (UAC1.0)' 가 보이는지 확인하세요."
            )

        return candidate

    def _map_alsa_card_to_sounddevice_id(self, card: int, dev: int) -> int:
        """
        arecord -l로 얻은 (card, dev)을 sounddevice(PortAudio) 입력 장치 id로 매핑.
        - 우선순위:
          1) device name에 "hw:card,dev" / "plughw:card,dev" / "card,dev" 같은 패턴이 포함된 입력장치
          2) 그게 없으면 'respeaker/mic array/arrayuac' 키워드 매칭 입력장치
        - hostapi가 ALSA인 장치를 약간 우대
        """
        targets = [
            f"hw:{card},{dev}",
            f"plughw:{card},{dev}",
            f"{card},{dev}",
            f"card {card}",
        ]
        kw_fallback = ["respeaker", "mic array", "arrayuac", "uac1", "usb audio"]
    
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    
        scored = []
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = (d.get("name", "") or "").lower()
    
            score = 0
            for t in targets:
                if t in name:
                    score += 10
            for kw in kw_fallback:
                if kw in name:
                    score += 2
    
            host = (hostapis[d["hostapi"]]["name"] or "").lower()
            if "alsa" in host:
                score += 1  # ALSA hostapi면 약간 우대
    
            if score > 0:
                scored.append((score, idx, d.get("name", ""), host))
    
        if scored:
            scored.sort(reverse=True, key=lambda x: x[0])
            best = scored[0]
            print(f"[INFO] Matched input device: idx={best[1]}, score={best[0]}, hostapi={best[3]}, name='{best[2]}'")
            return best[1]
    
        # ---- 여기까지 왔으면 매핑 실패: 디버그용으로 입력 장치 목록을 출력하고 fallback ----
        print("[WARN] ALSA(card,dev) -> sounddevice 매핑 실패. 현재 입력 장치 목록:")
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                host = (hostapis[d["hostapi"]]["name"] or "")
                print(f"  [{idx}] hostapi={host} name='{d.get('name','')}' in_ch={d.get('max_input_channels')}")
    
        # 마지막 fallback: 기본 입력 장치
        default_in = sd.default.device[0]
        if default_in is None:
            print("[WARN] sd.default.device[0] is None -> 0 fallback")
            return 0
        print(f"[WARN] fallback to default input device_id={default_in}")
        return int(default_in)

    def request_interrupt(self):
        """외부에서 STT 대기를 끊고 싶을 때 사용"""
        self.interrupt_event.set()

    # -------------------------
    # Device detect
    # -------------------------
    # -------------------------
    # Noise measure
    # -------------------------
    def _measure_noise_level(self, duration=1.0):
        print("[INFO] 배경 소음 측정 중...")
        frames = int(SAMPLE_RATE * duration)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=self.channels,
            dtype="float32",
            device=self.input_device_id,
            blocksize=frames,   # <==== 한 번에 읽기
        ) as stream:
            data, _ = stream.read(frames)

        # (N, C)일 경우 mono로 다운믹스
        if data.ndim == 2 and data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)

        rms = float(np.sqrt(np.mean(np.square(data))))
        print(f"[INFO] 배경 RMS: {rms:.6f}")
        return rms

    # -------------------------
    # Audio callback
    # -------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # underrun/overrun 등 경고는 로그만
            # print(f"[WARN] audio status: {status}")
            pass
        self.audio_queue.put(indata.copy())

    # -------------------------
    # VAD listen
    # -------------------------
    def listen_once_vad(
        self,
        volume_level_changed,
        on_speech_start=None,
        on_speech_end=None,
        max_wait_sec=60.0,      # <==== 말 시작을 최대 얼마나 기다릴지
        max_record_sec=20.0,    # <==== 말이 계속 이어질 때 무한 녹음 방지
    ):
        """
        말 시작/끝 자동 감지로 한 번 발화만 녹음해서 OUTPUT_FILE로 저장하고 경로 반환.
        - 말 시작: avg_rms > START_THRESHOLD
        - 말 끝  : (avg_rms < silence_threshold) 상태가 SILENCE_DURATION 이상 지속
        """
        self.interrupt_event.clear()

        recorded_frames = []
        rms_buffer = deque(maxlen=10)

        started = False
        t0 = time.time()
        start_time = None
        last_sound_time = None

        START_THRESHOLD = self.silence_threshold * 1.6
        MIN_DURATION = 0.6  # 너무 짧은 오탐 방지

        # 볼륨 표시(디버깅/UI용) 변환 파라미터
        MIN_DB = -60.0
        MAX_DB = 0.0

        blocksize = int(SAMPLE_RATE * BLOCK_DURATION)  # <==== config의 BLOCK_DURATION 사용

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=self.channels,
            dtype="float32",
            device=self.input_device_id,
            blocksize=blocksize,
            callback=self._audio_callback,
        ):
            while True:
                # 외부 인터럽트
                if self.interrupt_event.is_set():
                    print("[INFO] STT listen_once_vad 인터럽트 → 중단")
                    return None

                # 말 시작 최대 대기 시간
                if (time.time() - t0) > max_wait_sec and (not started):
                    print("[WARN] 말 시작 대기 timeout")
                    return None

                # 녹음 최대 시간 (말이 계속 이어지거나 threshold가 잘못돼도 안전 탈출)
                if started and (time.time() - start_time) > max_record_sec:
                    print("[WARN] max_record_sec 도달 → 강제 종료")
                    break

                try:
                    block = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # stereo/multi면 mono로 다운믹스
                if block.ndim == 2 and block.shape[1] > 1:
                    block_mono = np.mean(block, axis=1, keepdims=True)
                else:
                    block_mono = block

                rms = float(np.sqrt(np.mean(block_mono ** 2)))
                rms_buffer.append(rms)
                avg_rms = float(np.mean(rms_buffer))

                # volume_level_changed (있으면)
                try:
                    db = 20.0 * math.log10(rms + 1e-8)
                    db_clamped = max(MIN_DB, min(MAX_DB, db))
                    level = int((db_clamped - MIN_DB) / (MAX_DB - MIN_DB) * 100)
                    volume_level_changed.emit(level, db_clamped + 50)
                except Exception:
                    pass

                # ---- 말 시작 감지 ----
                if not started:
                    if avg_rms > START_THRESHOLD:
                        started = True
                        start_time = time.time()
                        last_sound_time = start_time
                        print("[INFO] VAD: speech start")

                        if callable(on_speech_start):
                            try:
                                on_speech_start()
                            except Exception as e:
                                print(f"[WARN] on_speech_start callback error: {e}")

                        recorded_frames.append(block_mono)
                    continue

                # ---- 녹음 중 ----
                recorded_frames.append(block_mono)

                if avg_rms > self.silence_threshold:
                    last_sound_time = time.time()
                else:
                    now = time.time()
                    if (now - last_sound_time > SILENCE_DURATION) and (now - start_time > MIN_DURATION):
                        print("[INFO] VAD: speech end")
                        if callable(on_speech_end):
                            try:
                                on_speech_end()
                            except Exception as e:
                                print(f"[WARN] on_speech_end callback error: {e}")
                        break

        if not recorded_frames:
            return None

        audio_data = np.concatenate(recorded_frames, axis=0).astype(np.float32)

        # OUTPUT_FILE 저장 (float32 → int16)
        audio_int16 = np.int16(np.clip(audio_data, -1.0, 1.0) * 32767)
        sf.write(OUTPUT_FILE, audio_int16, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        print(f"[INFO] 녹음 완료(VAD): {OUTPUT_FILE}")
        return OUTPUT_FILE

    def transcribe(self, audio_file):
        print(f"[INFO] '{audio_file}' STT 처리 중...")
        result = self.whisper_model.transcribe(audio_file, language=LANG, fp16=False)
        text = (result.get("text", "") or "").strip()
        print(f"[INFO] STT 결과: {text}")
        return text

    # -------------------------
    # Hint task 편의를 위해: "한 번 호출로 말 기다림+녹음+STT" 제공
    # -------------------------
    def listen_and_transcribe_once(
        self,
        volume_level_changed,
        on_speech_start=None,
        on_speech_end=None,
    ):
        """
        hint_task.py에서 while 루프 없애기 위한 원샷 API.
        - 내부에서 VAD로 말 시작/끝까지 녹음하고
        - 바로 STT 결과(text)를 반환
        """
        audio_path = self.listen_once_vad(
            volume_level_changed=volume_level_changed,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )
        if audio_path is None:
            return ""
        return self.transcribe(audio_file=audio_path).strip()
