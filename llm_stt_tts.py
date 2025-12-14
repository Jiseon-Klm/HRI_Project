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
        duration_sec: int | None = None,   # <- hint_task.py 호환용(실제론 VAD가 종료 결정)
        # ---- VAD 파라미터(기본값만으로도 대부분 동작) ----
        start_threshold: float = 0.006,    # RMS 시작 임계값 (환경 따라 0.003~0.02 튜닝)
        stop_threshold: float | None = None,  # None이면 start_threshold*0.6 사용 (hysteresis)
        start_blocks: int = 3,             # 연속 N블록 이상 넘어야 "발화 시작" 인정
        min_speech_sec: float = 0.25,      # 너무 짧은 소리는 무시(오탐 제거)
        max_wait_sec: float = 30.0,        # 말 시작을 최대 몇 초 기다릴지
        max_record_sec: float = 12.0,      # 말 시작 후 최대 녹음 길이(무한 녹음 방지)
        preroll_sec: float = 0.20,         # 발화 시작 직전 N초를 포함(초반 잘림 방지)
    ):
        """
        arecord 스트리밍 + RMS 기반 VAD:
        - 발화 시작 전: RMS가 start_threshold 이상인 블록이 start_blocks 연속이면 시작
        - 시작 후: RMS가 stop_threshold 미만인 무음이 SILENCE_DURATION 지속되면 종료
        - preroll_sec 만큼의 직전 버퍼를 포함해 WAV 저장
        - 반환: OUTPUT_FILE 경로 or None
        """
    
        import io
        import select
        from collections import deque
    
        self.interrupt_event.clear()
    
        if stop_threshold is None:
            stop_threshold = start_threshold * 0.6  # hysteresis: 종료는 더 낮은 임계로
    
        # ---- 오디오 포맷 (속도/안정성) ----
        # Whisper 표준: 16kHz mono PCM 16bit
        # 이렇게 받으면 파일도 작아지고 Whisper도 빨라짐.
        SR = 16000
        CH = 1
        SAMPLE_WIDTH = 2  # int16
    
        block_frames = max(160, int(SR * BLOCK_DURATION))  # 최소 10ms 정도 확보
        block_bytes = block_frames * CH * SAMPLE_WIDTH
    
        preroll_blocks = max(0, int(preroll_sec / BLOCK_DURATION))
        preroll = deque(maxlen=preroll_blocks)
    
        # 상태
        speaking = False
        above_cnt = 0
        speech_time = 0.0
        silence_time = 0.0
    
        # 최종 저장 버퍼 (bytes)
        captured = bytearray()
    
        print(
            f"[INFO] VAD arecord listen_once 시작: start_th={start_threshold:.4f}, "
            f"stop_th={stop_threshold:.4f}, start_blocks={start_blocks}, "
            f"silence={SILENCE_DURATION}s, wait<={max_wait_sec}s, rec<={max_record_sec}s, "
            f"file={OUTPUT_FILE}"
        )
    
        # 볼륨 미터 초기 emit
        try:
            volume_level_changed.emit(0, 0.0)
        except Exception:
            pass
    
        # ---- arecord: ReSpeaker 디바이스에서 raw PCM을 stdout으로 ----
        # -t raw: 헤더 없이 raw stream
        # -f S16_LE: 16-bit little-endian
        # -c 1: mono
        # -r 16000: 16k
        cmd = [
            "arecord",
            "-D", self.alsa_device,
            "-q",                  # quieter
            "-t", "raw",
            "-f", "S16_LE",
            "-c", str(CH),
            "-r", str(SR),
        ]
    
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as e:
            print(f"[ERROR] arecord Popen 실패: {e}")
            return None
    
        t_wait0 = time.perf_counter()
        t_speech0 = None
    
        def _rms_int16(block: bytes) -> float:
            # int16 -> float RMS (빠르게 numpy 사용)
            x = np.frombuffer(block, dtype=np.int16).astype(np.float32)
            if x.size == 0:
                return 0.0
            return float(np.sqrt(np.mean((x / 32768.0) ** 2)))
    
        try:
            assert proc.stdout is not None
            stdout = proc.stdout
    
            while True:
                if self.interrupt_event.is_set():
                    print("[INFO] interrupt_event set -> 녹음 중단")
                    break
    
                # 시작 전 최대 대기 시간
                if not speaking and (time.perf_counter() - t_wait0) > max_wait_sec:
                    print("[WARN] 발화 시작 없이 max_wait_sec 초과 -> 종료")
                    break
    
                # speaking 이후 최대 녹음 시간
                if speaking and t_speech0 is not None and (time.perf_counter() - t_speech0) > max_record_sec:
                    print("[WARN] speaking 상태에서 max_record_sec 초과 -> 종료")
                    break
    
                # stdout에서 block_bytes 만큼 읽기 (ready 될 때만)
                rlist, _, _ = select.select([stdout], [], [], 0.5)
                if not rlist:
                    continue
    
                block = stdout.read(block_bytes)
                if not block or len(block) == 0:
                    # arecord 종료/에러
                    break
    
                rms = _rms_int16(block)
    
                # UI용 emit (있으면)
                try:
                    volume_level_changed.emit(0, rms)
                except Exception:
                    pass
    
                if not speaking:
                    # preroll 채우기
                    if preroll_blocks > 0:
                        preroll.append(block)
    
                    if rms >= start_threshold:
                        above_cnt += 1
                    else:
                        above_cnt = 0
    
                    if above_cnt >= start_blocks:
                        speaking = True
                        t_speech0 = time.perf_counter()
                        silence_time = 0.0
                        speech_time = 0.0
    
                        # preroll 포함
                        if preroll_blocks > 0 and len(preroll) > 0:
                            for b in preroll:
                                captured.extend(b)
                        preroll.clear()
    
                        captured.extend(block)
                        speech_time += BLOCK_DURATION
                else:
                    # speaking 중: 항상 저장
                    captured.extend(block)
                    speech_time += BLOCK_DURATION
    
                    if rms < stop_threshold:
                        silence_time += BLOCK_DURATION
                    else:
                        silence_time = 0.0
    
                    # 종료 조건: 무음이 SILENCE_DURATION 지속
                    if silence_time >= SILENCE_DURATION:
                        if speech_time < min_speech_sec:
                            # 너무 짧은 오탐 -> 리셋하고 다시 대기
                            print("[WARN] 발화가 너무 짧음(오탐) -> 다시 대기")
                            speaking = False
                            above_cnt = 0
                            speech_time = 0.0
                            silence_time = 0.0
                            captured.clear()
                            preroll.clear()
                            t_wait0 = time.perf_counter()
                            t_speech0 = None
                            continue
    
                        print("[INFO] 무음 감지 -> 녹음 종료")
                        break
    
        finally:
            # arecord 종료
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
    
        if len(captured) == 0:
            print("[WARN] 녹음된 오디오 없음")
            return None
    
        # ---- raw PCM -> WAV 저장 ----
        try:
            audio = np.frombuffer(bytes(captured), dtype=np.int16).astype(np.float32) / 32768.0
            audio = audio.reshape(-1, 1)  # (T,1)
            sf.write(OUTPUT_FILE, audio, SR)
            print(f"[INFO] VAD 녹음 완료: {OUTPUT_FILE} (sec≈{len(audio)/SR:.2f})")
            return OUTPUT_FILE
        except Exception as e:
            print(f"[ERROR] WAV 저장 실패: {e}")
            return None

    def transcribe(self, audio_file):
        print(f"[INFO] '{audio_file}' STT 처리 중...")
        result = self.whisper_model.transcribe(audio_file, language=LANG, fp16=False)
        text = result["text"].strip()
        print(f"[INFO] STT 결과: {text}")
        return text
