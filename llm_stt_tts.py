import subprocess
import threading
import time
import re

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
    - hint_task.py 호환 유지:
      listen_once(...) -> OUTPUT_FILE 경로 반환 (또는 None)
      transcribe(audio_file) -> text 반환
    - 개선점:
      1) callback 로직 중복 제거 (중복 append/중복 카운트 버그 제거)
      2) 연속 N블록(start_blocks)로 시작 오탐 방지
      3) pre-roll(preroll_sec)로 말 시작 초반 잘림 방지
      4) hysteresis(stop_ratio)로 종료 안정화
      5) optional 자동 노이즈 캘리브레이션 (silence_threshold=None이면 자동)
      6) Whisper 디코딩 옵션으로 속도 개선
    """

    def __init__(self, record_duration_sec: int = 5):
        print("[INFO] Whisper 모델 로딩 중...")
        self.whisper_model = whisper.load_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")

        self.record_duration_sec = record_duration_sec
        self.interrupt_event = threading.Event()

        # Whisper fp16: GPU 있으면 True가 보통 더 빠름
        self._use_fp16 = bool(torch.cuda.is_available())

        # sounddevice에서 ReSpeaker 입력 디바이스를 최대한 자동 선택
        self.sd_input_device = self._detect_sounddevice_input_device()
        if self.sd_input_device is not None:
            devinfo = sd.query_devices(self.sd_input_device)
            print(f"[INFO] Selected sounddevice input: #{self.sd_input_device} '{devinfo['name']}'")
        else:
            print("[WARN] ReSpeaker sounddevice input을 특정 못함 -> system default input 사용")

    def request_interrupt(self):
        self.interrupt_event.set()

    def _detect_sounddevice_input_device(self):
        """
        PortAudio(sounddevice) 입력 디바이스 목록에서 ReSpeaker 후보 찾기.
        못 찾으면 None 반환 -> default input 사용.
        """
        preferred_keywords = ["respeaker", "arrayuac10", "mic array", "uac", "usb audio"]
        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"[WARN] sd.query_devices 실패: {e}")
            return None

        candidates = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = str(d.get("name", "")).lower()
            if any(kw in name for kw in preferred_keywords):
                candidates.append((i, d))

        if not candidates:
            return None

        # 간단한 우선순위: input 채널 많은 걸 선호
        candidates.sort(key=lambda x: x[1].get("max_input_channels", 0), reverse=True)
        return candidates[0][0]

    def listen_once(
        self,
        volume_level_changed,
        duration_sec: int | None = None,      # 기존 인터페이스 유지용 (실제 로직에선 직접 사용 X)
        silence_threshold: float = 0.01,      # RMS start threshold (None이면 자동 캘리브레이션)
        max_duration: float = 30.0,           # (대기+발화 포함) 전체 타임아웃
        min_speech_sec: float = 0.25,         # 이 이상 말해야 유효 발화로 인정
        # --- 아래는 “추가 파라미터”지만 기본값 제공 -> hint_task.py 수정 없이 작동 ---
        start_blocks: int = 3,                # 시작 오탐 방지: 연속 N블록 이상 threshold 초과해야 speaking 시작
        preroll_sec: float = 0.30,            # 말 시작 직전 N초를 버퍼로 포함 (초반 잘림 방지)
        stop_ratio: float = 0.60,             # 종료 임계값 = start_threshold * stop_ratio (hysteresis)
        calib_sec: float = 0.0,               # silence_threshold=None일 때만 사용(권장 0.5~1.0). 기본은 0 (추가 지연 없음)
    ):
        """
        VAD 스타일 녹음:
        - 발화 시작 전: threshold 넘길 때까지 기다림 (단, max_duration 넘으면 종료)
        - 시작: threshold 이상이 start_blocks 연속 발생하면 speaking=True
        - 종료: speaking 이후 rms < (threshold*stop_ratio) 인 무음이 SILENCE_DURATION 지속되면 종료
        - pre-roll 포함하여 OUTPUT_FILE 저장 후 경로 반환
        """

        self.interrupt_event.clear()

        # --- 블록 설정 ---
        block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        preroll_blocks = max(0, int(preroll_sec / BLOCK_DURATION))

        # pre-roll ring buffer
        from collections import deque
        preroll = deque(maxlen=preroll_blocks)

        # 상태 변수
        audio_chunks = []
        speaking = False
        speech_time = 0.0
        silence_time = 0.0
        above_cnt = 0

        # threshold 자동 캘리브레이션(옵션)
        # - silence_threshold=None일 때만 수행
        threshold = silence_threshold

        print(
            f"[INFO] VAD listen_once 시작: threshold={threshold}, "
            f"silence={SILENCE_DURATION}s, max={max_duration}s, "
            f"start_blocks={start_blocks}, preroll={preroll_sec}s, file={OUTPUT_FILE}"
        )

        try:
            volume_level_changed.emit(0, 0.0)
        except Exception:
            pass

        t0 = time.perf_counter()
        started_at = None

        # --- callback: 최대한 가볍게 (print 금지) ---
        def callback(indata, frames, time_info, status):
            nonlocal speaking, speech_time, silence_time, above_cnt, threshold, started_at

            # indata: (frames, channels) float32
            # rms 계산 (빠르게)
            x = indata
            rms = float(np.sqrt(np.mean(x * x)))

            try:
                volume_level_changed.emit(0, rms)
            except Exception:
                pass

            # 아직 threshold가 None이면 (캘리브레이션 중) 일단 프리롤만 쌓음
            if threshold is None:
                preroll.append(indata.copy())
                return

            start_th = float(threshold)
            stop_th = float(threshold) * float(stop_ratio)

            if not speaking:
                # 말 시작 전에는 프리롤만 저장
                if preroll_blocks > 0:
                    preroll.append(indata.copy())

                if rms >= start_th:
                    above_cnt += 1
                else:
                    above_cnt = 0

                if above_cnt >= start_blocks:
                    speaking = True
                    started_at = time.perf_counter()

                    # pre-roll 먼저 넣고, 현재 블록도 넣음
                    if preroll_blocks > 0 and len(preroll) > 0:
                        audio_chunks.extend(list(preroll))
                    audio_chunks.append(indata.copy())

                    # 상태 초기화
                    preroll.clear()
                    silence_time = 0.0
                    speech_time = BLOCK_DURATION
            else:
                # speaking 중이면 항상 저장
                audio_chunks.append(indata.copy())
                speech_time += BLOCK_DURATION

                # 종료 판정은 더 낮은 stop_th로 (hysteresis)
                if rms < stop_th:
                    silence_time += BLOCK_DURATION
                else:
                    silence_time = 0.0

        # --- (옵션) 자동 캘리브레이션 ---
        def run_calibration(stream_device):
            nonlocal threshold
            if threshold is not None or calib_sec <= 0.0:
                return

            # calib_sec 동안 rms 통계 수집
            rms_vals = []
            tcal0 = time.perf_counter()
            while (time.perf_counter() - tcal0) < calib_sec:
                time.sleep(BLOCK_DURATION)

                # preroll에 쌓인 마지막 블록으로 rms 추정(간단)
                if len(preroll) > 0:
                    x = preroll[-1]
                    r = float(np.sqrt(np.mean(x * x)))
                    rms_vals.append(r)

            if len(rms_vals) == 0:
                threshold = 0.01
                return

            mu = float(np.mean(rms_vals))
            sigma = float(np.std(rms_vals))
            # 보수적으로: 노이즈 평균 + 3시그마, 최소 0.005
            threshold = max(mu + 3.0 * sigma, 0.005)
            print(f"[INFO] Auto-calibrated threshold={threshold:.4f} (mu={mu:.4f}, sigma={sigma:.4f})")

        # --- 스트림 실행 ---
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=callback,
                device=self.sd_input_device,  # None이면 default input
            ):
                # threshold=None이면 캘리브레이션 수행(옵션)
                run_calibration(self.sd_input_device)

                while True:
                    if self.interrupt_event.is_set():
                        print("[INFO] interrupt_event set -> 녹음 중단")
                        break

                    time.sleep(BLOCK_DURATION)

                    # 전체 타임아웃
                    if (time.perf_counter() - t0) > max_duration:
                        if not speaking:
                            print("[WARN] 발화 시작 없이 max_duration 초과 -> 종료")
                        else:
                            print("[WARN] speaking 중 max_duration 초과 -> 종료")
                        break

                    # speaking 시작 후 종료 조건
                    if speaking and silence_time >= SILENCE_DURATION:
                        if speech_time < min_speech_sec:
                            # 너무 짧은 잡음 -> 리셋하고 다시 대기 (단, 전체 max_duration 안에서)
                            print("[WARN] 발화가 너무 짧음(오탐) -> 다시 대기")
                            speaking = False
                            speech_time = 0.0
                            silence_time = 0.0
                            above_cnt = 0
                            audio_chunks.clear()
                            preroll.clear()
                            started_at = None
                            continue

                        print("[INFO] 무음 감지 -> 녹음 종료")
                        break

        except Exception as e:
            print(f"[ERROR] sounddevice 입력 스트림 열기 실패: {e}")
            print("[HINT] 도커/권한 문제면 /dev/snd 마운트 또는 Pulse/ALSA 설정을 확인해야 해요.")
            return None

        if not audio_chunks:
            print("[WARN] 녹음된 오디오 없음")
            return None

        audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)  # (T,1)

        # WAV 저장 (hint_task.py 호환 유지)
        sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
        print(f"[INFO] VAD 녹음 완료: {OUTPUT_FILE} (samples={len(audio)})")
        return OUTPUT_FILE

    def transcribe(self, audio_file):
        print(f"[INFO] '{audio_file}' STT 처리 중...")

        # Whisper 속도/안정성 옵션:
        # - beam_size=1, best_of=1, temperature=0.0  -> 빠르고 결정적
        # - condition_on_previous_text=False         -> 짧은 오디오에서 이상한 누적 줄임
        # - fp16: GPU 있으면 보통 더 빠름
        result = self.whisper_model.transcribe(
            audio_file,
            language=LANG,
            fp16=self._use_fp16,
            temperature=0.0,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            verbose=False,
        )

        text = (result.get("text") or "").strip()
        print(f"[INFO] STT 결과: {text}")
        return text

