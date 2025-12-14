#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import threading
import time
import re
import math
from collections import deque
from typing import Optional, Tuple

import whisper  # STT
import numpy as np
import soundfile as sf

# (프로젝트에서 쓰고 있던 것들 - STT만 쓸 때 미사용이어도 import 유지)
from transformers import VitsModel, AutoTokenizer  # noqa: F401
import torch  # noqa: F401
import sounddevice as sd  # noqa: F401  # TTS에서만 쓴다고 했으니 유지(원하면 삭제 가능)
from google import genai  # noqa: F401

from config import (
    SAMPLE_RATE,
    BLOCK_DURATION,
    SILENCE_DURATION,
    OUTPUT_FILE,
    MODEL_NAME,
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
    arecord(=ALSA direct) 기반 VAD + Whisper STT

    핵심 의도:
    - 버전1에서 정상 동작하던 ALSA 디바이스(plughw:card,dev)를 그대로 사용
    - sounddevice/PortAudio 매핑을 STT 경로에서 제거해서
      "열리긴 열리는데 무음(0)" 이슈를 원천 차단
    """

    def __init__(self):
        print("[INFO] Whisper 모델 로딩 중...")
        self.whisper_model = whisper.load_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")

        self.interrupt_event = threading.Event()

        # 1) ReSpeaker ALSA card/device 자동 탐지
        self.card_number, self.device_number = self._detect_respeaker_card_device()
        self.alsa_device = f"plughw:{self.card_number},{self.device_number}"
        print(f"[INFO] Detected ReSpeaker ALSA device: {self.alsa_device}")

        # 2) STT는 mono로 처리 (필요하면 나중에 4ch도 가능하지만 VAD 안정성이 우선)
        self.channels = 1
        print(f"[INFO] STT input channels (forced): {self.channels}")

        # 3) 배경 소음 측정 -> threshold 설정
        self.base_noise = self._measure_noise_level(duration=1.0)
        self.base_noise = max(self.base_noise, 1e-4)  # 너무 작으면 오탐/불안정

        self.silence_threshold = self.base_noise * 3.5
        print(f"[INFO] base_noise(RMS)={self.base_noise:.6f}, silence_threshold={self.silence_threshold:.6f}")

    def request_interrupt(self):
        """외부에서 STT 대기를 끊고 싶을 때 사용"""
        self.interrupt_event.set()

    # -------------------------
    # ALSA device detect (arecord -l)
    # -------------------------
    def _detect_respeaker_card_device(self) -> Tuple[int, int]:
        """
        arecord -l 출력에서 ReSpeaker로 보이는 card/device를 추출
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
            raise RuntimeError(
                "[ERROR] `arecord -l` 출력에서 ReSpeaker 디바이스를 찾지 못했습니다.\n"
                "1) ReSpeaker 연결 확인\n"
                "2) 컨테이너면 /dev/snd 마운트 확인\n"
                "3) `arecord -l` 출력에 ReSpeaker 관련 라인이 있는지 확인"
            )

        return candidate

    # -------------------------
    # arecord raw stream
    # -------------------------
    def _arecord_raw_stream(self, rate: int, channels: int) -> subprocess.Popen:
        """
        arecord로 raw PCM(S16_LE) 스트림을 stdout으로 받는다.
        """
        cmd = [
            "arecord",
            "-D", self.alsa_device,
            "-q",                 # quiet
            "-f", "S16_LE",        # 16-bit little endian
            "-r", str(rate),
            "-c", str(channels),
            "-t", "raw",           # raw PCM to stdout
        ]
        # stderr는 디버깅 위해 PIPE로 받되, read로 막히지 않게 주의
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _terminate_proc(self, p: subprocess.Popen):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    p.kill()
        except Exception:
            pass

    # -------------------------
    # Noise measure (arecord 기반)
    # -------------------------
    def _measure_noise_level(self, duration: float = 1.0) -> float:
        print("[INFO] 배경 소음 측정 중...(arecord raw)")
        p = self._arecord_raw_stream(rate=SAMPLE_RATE, channels=1)

        try:
            if p.stdout is None:
                print("[WARN] arecord stdout is None -> base_noise=0")
                return 0.0

            n_samples = int(SAMPLE_RATE * duration)
            n_bytes = n_samples * 2  # int16

            raw = p.stdout.read(n_bytes)
            if not raw:
                print("[WARN] raw audio empty -> base_noise=0")
                return 0.0

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[INFO] 배경 RMS: {rms:.6f}")
            return rms

        finally:
            self._terminate_proc(p)

    # -------------------------
    # VAD listen (arecord 기반)
    # -------------------------
    def listen_once_vad(
        self,
        volume_level_changed,
        on_speech_start=None,
        on_speech_end=None,
        max_wait_sec: float = 60.0,
        max_record_sec: float = 20.0,
        debug_rms: bool = False,   # <==== 필요하면 True로 켜서 RMS 확인
    ) -> Optional[str]:
        """
        말 시작/끝 자동 감지로 한 번 발화만 녹음해서 OUTPUT_FILE로 저장하고 경로 반환.
        - 입력: arecord raw (S16_LE)
        - 말 시작: avg_rms > START_THRESHOLD
        - 말 끝  : avg_rms < silence_threshold 상태가 SILENCE_DURATION 이상 지속
        """
        self.interrupt_event.clear()

        START_THRESHOLD = self.silence_threshold * 1.6
        MIN_DURATION = 0.6

        block_samples = int(SAMPLE_RATE * BLOCK_DURATION)
        block_bytes = block_samples * 2  # int16

        recorded = []
        rms_buffer = deque(maxlen=10)

        started = False
        t0 = time.time()
        start_time = None
        last_sound_time = None

        p = self._arecord_raw_stream(rate=SAMPLE_RATE, channels=1)

        try:
            if p.stdout is None:
                print("[ERROR] arecord stdout is None (stream not available)")
                return None

            while True:
                if self.interrupt_event.is_set():
                    print("[INFO] STT listen_once_vad 인터럽트 → 중단")
                    return None

                if (time.time() - t0) > max_wait_sec and (not started):
                    print("[WARN] 말 시작 대기 timeout")
                    return None

                if started and start_time is not None and (time.time() - start_time) > max_record_sec:
                    print("[WARN] max_record_sec 도달 → 강제 종료")
                    break

                raw = p.stdout.read(block_bytes)
                if not raw:
                    # 프로세스가 죽었는지 체크
                    if p.poll() is not None:
                        err = ""
                        try:
                            if p.stderr is not None:
                                err = p.stderr.read().decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                        print("[ERROR] arecord 종료됨 / raw read 실패")
                        if err.strip():
                            print("[ERROR] arecord stderr:\n" + err)
                        return None
                    continue

                block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(block ** 2)))
                rms_buffer.append(rms)
                avg_rms = float(np.mean(rms_buffer))

                if debug_rms:
                    print(f"[DEBUG] rms={rms:.6f} avg={avg_rms:.6f} start_th={START_THRESHOLD:.6f} sil_th={self.silence_threshold:.6f}")

                # 볼륨 표시 (UI용)
                try:
                    db = 20.0 * math.log10(rms + 1e-8)
                    level = int(np.clip((db + 60.0) / 60.0 * 100.0, 0, 100))
                    volume_level_changed.emit(level, db + 50.0)
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

                        recorded.append(block.copy())
                    continue

                # ---- 녹음 중 ----
                recorded.append(block.copy())

                # 마지막 소리 시점 갱신/종료 조건 체크
                if avg_rms > self.silence_threshold:
                    last_sound_time = time.time()
                else:
                    now = time.time()
                    if (
                        last_sound_time is not None
                        and (now - last_sound_time > SILENCE_DURATION)
                        and (start_time is not None)
                        and (now - start_time > MIN_DURATION)
                    ):
                        print("[INFO] VAD: speech end")

                        if callable(on_speech_end):
                            try:
                                on_speech_end()
                            except Exception as e:
                                print(f"[WARN] on_speech_end callback error: {e}")
                        break

        finally:
            self._terminate_proc(p)

        if not recorded:
            return None

        audio = np.concatenate(recorded, axis=0).astype(np.float32)
        audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

        sf.write(OUTPUT_FILE, audio_i16, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        print(f"[INFO] 녹음 완료(VAD/arecord): {OUTPUT_FILE}")
        return OUTPUT_FILE

    def transcribe(self, audio_file: str) -> str:
        print(f"[INFO] '{audio_file}' STT 처리 중...")
        result = self.whisper_model.transcribe(audio_file, language=LANG, fp16=False)
        text = (result.get("text", "") or "").strip()
        print(f"[INFO] STT 결과: {text}")
        return text

    def listen_and_transcribe_once(
        self,
        volume_level_changed,
        on_speech_start=None,
        on_speech_end=None,
        debug_rms: bool = False,
    ) -> str:
        """
        한 번 호출로:
        - VAD로 말 시작/끝까지 녹음
        - Whisper로 STT
        """
        audio_path = self.listen_once_vad(
            volume_level_changed=volume_level_changed,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            debug_rms=debug_rms,
        )
        if audio_path is None:
            return ""
        return self.transcribe(audio_file=audio_path).strip()


# ================== (선택) TTS / 기타 클래스들 ==================
# 너 원본 llm_stt_tts.py에 TTSProcessor, Piper/MMS TTS 로직이 있었다면
# 아래에 그대로 유지하면 돼. (STT는 위 STTProcessor만 쓰면 됨)
