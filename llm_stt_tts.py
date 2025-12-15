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
import shutil
import torch
import soundfile as sf

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
    ✅ Piper로 wav 생성 -> sounddevice로 즉시 재생 (ALSA 직결)
    - paplay/pactl/pipewire/bt 전부 제거
    - 지연 최소 + 컨테이너에서 가장 잘 됨
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        piper_bin: str = "piper",
        force_out_sr: int = 48000,      # ✅ 대부분 ALSA/HDMI가 48k를 좋아함
        output_device: str | int | None = None,  # ✅ 이름(부분문자열) 또는 index
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"piper model not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"piper config not found: {config_path}")

        self.model_path = model_path
        self.config_path = config_path
        self.piper_bin = piper_bin
        self.force_out_sr = force_out_sr

        # 환경변수로도 바로 바꿀 수 있게
        # 예: export SD_OUTPUT_DEVICE="hdmi"  또는 "2"
        env_dev = os.environ.get("SD_OUTPUT_DEVICE")
        if output_device is None and env_dev:
            # 숫자면 index로
            if env_dev.isdigit():
                output_device = int(env_dev)
            else:
                output_device = env_dev

        self.output_device = output_device

    def _pick_device_index(self, device_query: str | int | None):
        if device_query is None:
            return None  # default device

        if isinstance(device_query, int):
            return device_query

        # 문자열이면 "이름 부분매칭"으로 output device 찾기
        q = device_query.lower()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0 and q in d["name"].lower():
                return i

        print(f"[WARN] sounddevice output device '{device_query}' not found. Using default.")
        return None

    def _resample_linear(self, x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return x
        # 간단 선형 보간 (지연 최소, 의존성 0)
        n_in = x.shape[0]
        n_out = int(round(n_in * (sr_out / sr_in)))
        xp = np.linspace(0.0, 1.0, n_in, endpoint=False)
        fp = x.astype(np.float32)
        xq = np.linspace(0.0, 1.0, n_out, endpoint=False)
        y = np.interp(xq, xp, fp).astype(np.float32)
        return y

    def speak(self, text: str, out_wav: str | None = None):
        if out_wav is None:
            fd, out_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # 1) Piper로 wav 생성
        cmd = [
            self.piper_bin,
            "--model", self.model_path,
            "--config", self.config_path,
            "--output_file", out_wav,
        ]
        subprocess.run(cmd, input=(text.strip() + "\n"), text=True, check=True)

        # 2) wav 읽기
        audio, sr = sf.read(out_wav, dtype="float32")
        if audio.ndim == 2:
            # piper는 보통 mono지만 혹시 몰라서 mono로 다운믹스
            audio = audio.mean(axis=1)

        # 3) (권장) output SR로 리샘플
        audio_play = self._resample_linear(audio, sr_in=sr, sr_out=self.force_out_sr)

        # 4) sounddevice로 재생 (ALSA 직결)
        dev_index = self._pick_device_index(self.output_device)
        try:
            sd.play(audio_play, samplerate=self.force_out_sr, device=dev_index, blocking=True)
        except Exception as e:
            print(f"[ERROR] sounddevice play failed: {e}")
            print("[HINT] sd.query_devices()로 출력 디바이스 이름/인덱스 확인 후 SD_OUTPUT_DEVICE를 지정해봐요.")
            raise

        return out_wav
