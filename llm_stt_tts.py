import subprocess
import threading
import time
import re
import os
import tempfile
# import whisper                         # ✅ STT용 Whisper
from faster_whisper import WhisperModel
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
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    PROMPT_TEMPLATE,
    PIPER_OUTPUT_FILE,
    MMS_TTS_MODEL_ID,
    MMS_TTS_OUTPUT_FILE,
)
# ================== STT (Speech-to-Text) ==================
os.environ["COQUI_TOS_AGREED"] = "1"

# === transformers export monkey patch ===
import transformers
try:
    # 최신 transformers에서는 여기 위치에 존재함(네가 확인한 경로)
    from transformers.generation.beam_search import BeamSearchScorer
    if not hasattr(transformers, "BeamSearchScorer"):
        transformers.BeamSearchScorer = BeamSearchScorer
except Exception as e:
    print(f"[WARN] Failed to patch BeamSearchScorer: {e}")


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
        # ✅ Faster-Whisper 모델 로드
        self.whisper_model = WhisperModel(
            model_size_or_path=MODEL_NAME,
            device=WHISPER_DEVICE,             # config.py에서 가져옴 (cpu)
            compute_type=WHISPER_COMPUTE_TYPE  # config.py에서 가져옴 (int8)
        )
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

    def listen_once(self, volume_level_changed, duration_sec: int | None = None):
        """
        - arecord를 사용해서 지정된 ALSA 디바이스에서 고정 길이 녹음.
        - duration_sec가 None이면 self.record_duration_sec 사용.
        - 녹음이 끝나면 OUTPUT_FILE 경로를 반환.
        """
        if duration_sec is None:
            duration_sec = self.record_duration_sec

        # 인터럽트 플래그 초기화
        self.interrupt_event.clear()

        print(
            f"[INFO] arecord 시작: device={self.alsa_device}, "
            f"duration={duration_sec}s, file={OUTPUT_FILE}"
        )

        # 볼륨 미터용 신호는 여기선 제대로 계산 못 하니 0으로 한 번 쏴줌
        try:
            volume_level_changed.emit(0, 0.0)
        except Exception:
            pass

        # cmd = [
        #     "arecord",
        #     "-D",
        #     self.alsa_device,
        #     "-f",
        #     "cd",
        #     "-d",
        #     str(duration_sec),
        #     OUTPUT_FILE,
        # ]

        cmd = [
            "arecord",
            "-D", self.alsa_device,          # plughw:card,dev
            "-t", "wav",
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(SAMPLE_RATE),          # config.py의 16000
            "-d", str(duration_sec),
            OUTPUT_FILE,
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            # 필요하면 proc.stdout 로그 찍어서 디버깅 가능
            # print(proc.stdout)
        except subprocess.CalledProcessError as e:
            print("[ERROR] arecord 실행 실패:")
            print(e.stdout)
            return None

        print(f"[INFO] 녹음 완료: {OUTPUT_FILE}")
        data, sr = sf.read(OUTPUT_FILE)
        print("[DEBUG] wav sr/ch:", sr, data.shape)
        print("[DEBUG] rms:", float(np.sqrt(np.mean(np.square(data)))))
        return OUTPUT_FILE

    def transcribe(self, audio_file):
            print(f"[INFO] '{audio_file}' STT 처리 중...")
            
            # ✅ Faster-Whisper 추론
            # segments는 generator라서 루프를 돌아야 텍스트가 나옵니다.
            segments, info = self.whisper_model.transcribe(
                audio_file, 
                language=LANG,
                beam_size=5  # 정확도를 위해 빔 사이즈 5 정도 추천
            )
            
            # 쪼개진 문장(segment)들을 하나로 합치기
            text = " ".join([segment.text for segment in segments]).strip()
            
            print(f"[INFO] STT 결과: {text}")
            return text



class TTSProcessorPiper:
    """
    Piper 기반 로컬 TTS.
    - text -> wav 생성 -> aplay로 재생
    - 가장 통합 쉽고 의존성 적음
    """
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        piper_bin: str = "piper",
        aplay_bin: str = "aplay",
        sample_rate: int = 22050,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        if config_path is not None and (not os.path.exists(config_path)):
            raise FileNotFoundError(f"Piper config not found: {config_path}")

        self.model_path = model_path
        self.config_path = config_path
        self.piper_bin = piper_bin
        self.aplay_bin = aplay_bin
        self.sample_rate = sample_rate

    def speak(self, text: str, out_wav: str | None = None) -> str:
        if out_wav is None:
            fd, out_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # piper 실행: stdin으로 text 전달, wav 파일 생성
        cmd = [self.piper_bin, "--model", self.model_path, "--output_file", out_wav]
        if self.config_path is not None:
            cmd += ["--config", self.config_path]

        subprocess.run(cmd, input=text, text=True, check=True)

        # 재생
        subprocess.run([self.aplay_bin, "-q", out_wav], check=False)
        return out_wav

