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
    - 처음 생성될 때 `arecord -l` 결과에서
      'ReSpeaker 4 Mic Array (UAC1.0)'가 있는 card/device를 자동으로 찾는다.
    - 이후 listen_once()를 부르면
      arecord -D plughw:{card},{device} -f cd -d {duration_sec} OUTPUT_FILE
      로 WAV를 녹음한 뒤 Whisper STT 수행.
    """

    def __init__(self, record_duration_sec: int = 5):
        print("[INFO] Whisper 모델 로딩 중...")
        self.whisper_model = whisper.load_model(MODEL_NAME)
        print("[INFO] 모델 로드 완료.")

        self.record_duration_sec = record_duration_sec
        self.interrupt_event = threading.Event()

        # ReSpeaker ALSA card/device 자동 탐지
        self.card_number, self.device_number = self._detect_respeaker_card_device()
        self.alsa_device = f"plughw:{self.card_number},{self.device_number}"
        print(f"[INFO] Detected ReSpeaker ALSA device: {self.alsa_device}")

    def request_interrupt(self):
        """
        옛날 sounddevice 기반 VAD 인터페이스와 호환용.
        지금 구조에서는 arecord를 동기 호출하므로 별도 중단은 구현 안 했고,
        그냥 플래그만 세팅해둠.
        """
        self.interrupt_event.set()

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

    def listen_once(
        self,
        volume_level_changed,
        duration_sec: int | None = None,   # <- 인터페이스 유지 (hint_task.py 수정 최소화)
        silence_threshold: float = 0.01,   # RMS 임계값 (환경 따라 튜닝)
        max_duration: float = 8.0,         # 최장 녹음 시간(초) - 무한 대기 방지
        min_speech_sec: float = 0.25,      # 이만큼은 말해야 "발화로 인정"
    ):
        """
        VAD 스타일 녹음:
        - (발화 시작 전) 무한정 대기 가능
        - 발화(energy > threshold) 감지되면 speaking=True
        - speaking 이후 무음이 SILENCE_DURATION 초 지속되면 자동 종료
        - 결과를 OUTPUT_FILE로 저장하고 경로 반환
        """
        # duration_sec는 기존 코드 호환용으로만 남김 (여기선 기본 사용 안 함)
        self.interrupt_event.clear()
    
        print(
            f"[INFO] VAD listen_once 시작: threshold={silence_threshold}, "
            f"silence={SILENCE_DURATION}s, max={max_duration}s, file={OUTPUT_FILE}"
        )
    
        # 볼륨 미터 신호 (있으면 RMS를 계속 emit)
        try:
            volume_level_changed.emit(0, 0.0)
        except Exception:
            pass
    
        audio_chunks = []
        speaking = False
        speech_time = 0.0
        silence_time = 0.0
    
        block_size = int(SAMPLE_RATE * BLOCK_DURATION)
    
        def callback(indata, frames, time_info, status):
            nonlocal speaking, speech_time, silence_time
    
            if status:
                # 과도한 출력은 지연을 키울 수 있으니 필요 시만
                # print(f"[WARN] sounddevice status: {status}")
                pass
    
            # indata shape: (frames, channels)
            # channels=1 가정
            rms = float(np.sqrt(np.mean(indata ** 2)))
    
            # UI용 (없어도 됨)
            try:
                volume_level_changed.emit(0, rms)
            except Exception:
                pass
    
            audio_chunks.append(indata.copy())
    
            if rms >= silence_threshold:
                speaking = True
                speech_time += BLOCK_DURATION
                silence_time = 0.0
            else:
                if speaking:
                    silence_time += BLOCK_DURATION
    
        t0 = time.perf_counter()
    
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=callback,
            ):
                # 루프는 callback이 버퍼를 채우는 동안 조건만 체크
                while True:
                    if self.interrupt_event.is_set():
                        print("[INFO] interrupt_event set -> 녹음 중단")
                        break
    
                    time.sleep(BLOCK_DURATION)
    
                    # 아직 말 시작 안 했으면 계속 대기
                    # (원하면 여기서 "초기 무음 N초면 종료" 같은 정책도 가능)
                    if not speaking:
                        # max_duration은 "전체 대기 시간"으로도 작동
                        if (time.perf_counter() - t0) > max_duration:
                            print("[WARN] 발화 시작 없이 max_duration 초과 -> 종료")
                            break
                        continue
    
                    # speaking 이후 무음이 SILENCE_DURATION 이상이면 종료
                    if silence_time >= SILENCE_DURATION:
                        # 너무 짧게 툭 소리만 난 경우 필터
                        if speech_time < min_speech_sec:
                            # 발화로 인정 못할 정도로 짧으면 리셋하고 다시 대기
                            print("[WARN] 발화가 너무 짧음 -> 다시 대기")
                            speaking = False
                            speech_time = 0.0
                            silence_time = 0.0
                            audio_chunks.clear()
                            t0 = time.perf_counter()
                            continue
    
                        print("[INFO] 무음 감지 -> 녹음 종료")
                        break
    
                    # speaking 상태에서 max_duration 초과하면 강제 종료
                    if (time.perf_counter() - t0) > max_duration:
                        print("[WARN] speaking 상태에서 max_duration 초과 -> 종료")
                        break
    
        except Exception as e:
            print(f"[ERROR] sounddevice 입력 스트림 열기 실패: {e}")
            print("[HINT] 도커/권한 문제면 /dev/snd 마운트 또는 Pulse/ALSA 설정을 확인해야 해요.")
            return None
    
        if not audio_chunks:
            print("[WARN] 녹음된 오디오 없음")
            return None
    
        audio = np.concatenate(audio_chunks, axis=0)  # shape (T, 1)
    
        # float32 -> wav 저장
        sf.write(OUTPUT_FILE, audio, SAMPLE_RATE)
        print(f"[INFO] VAD 녹음 완료: {OUTPUT_FILE} (samples={len(audio)})")
        return OUTPUT_FILE


    def transcribe(self, audio_file):
        print(f"[INFO] '{audio_file}' STT 처리 중...")
        result = self.whisper_model.transcribe(audio_file, language=LANG, fp16=False)
        text = result["text"].strip()
        print(f"[INFO] STT 결과: {text}")
        return text
