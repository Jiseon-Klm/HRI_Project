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
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 piper_bin: str = "piper",
                 sample_rate: int = 22050):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"piper model not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"piper config not found: {config_path}")

        self.model_path = model_path
        self.config_path = config_path
        self.piper_bin = piper_bin
        self.sample_rate = sample_rate

    def _find_pulse_sink_by_description(self, target_desc: str) -> str | None:
        """
        PulseAudio/PipeWire sink 중에서 Description(혹은 device.description)에
        target_desc가 포함된 sink의 'Name'을 찾아 반환.
        예: bluez_output.XX_XX_...a2dp-sink
        """
        if shutil.which("pactl") is None:
            print("[WARN] pactl not found. Install pulseaudio-utils in Dockerfile.")
            return None

        try:
            proc = subprocess.run(
                ["pactl", "list", "sinks"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )

        except Exception as e:
            print(f"[WARN] pactl list sinks failed: {e}")
            return None

        text = proc.stdout
        blocks = text.split("Sink #")
        target = target_desc.lower()

        for b in blocks:
            # sink name 라인 찾기: "Name: xxx"
            m_name = re.search(r"^\s*Name:\s*(.+)$", b, flags=re.MULTILINE)
            if not m_name:
                continue
            sink_name = m_name.group(1).strip()

            # description 후보들
            desc_candidates = []
            m_desc = re.search(r"^\s*Description:\s*(.+)$", b, flags=re.MULTILINE)
            if m_desc:
                desc_candidates.append(m_desc.group(1).strip())
            # PipeWire에서는 properties 안에 device.description이 있는 경우도 많아요
            m_devdesc = re.search(r"device\.description\s*=\s*\"(.+?)\"", b)
            if m_devdesc:
                desc_candidates.append(m_devdesc.group(1).strip())

            joined = " | ".join(desc_candidates).lower()
            if target in joined:
                return sink_name

        return None

    def _play_wav(self, wav_path: str, prefer_sink_desc: str | None = None):
        """
        1) prefer_sink_desc가 있으면 해당 Description을 가진 sink를 찾아 paplay로 출력
        2) 못 찾으면 paplay(기본 sink)
        3) paplay가 없으면 aplay로 fallback
        """
        # 1) paplay 우선
        if shutil.which("paplay") is not None:
            sink_name = None
            if prefer_sink_desc:
                sink_name = self._find_pulse_sink_by_description(prefer_sink_desc)
                if sink_name:
                    print(f"[AUDIO] Using Pulse sink: {sink_name} (desc contains '{prefer_sink_desc}')")
                    subprocess.run(["paplay", "--device", sink_name, wav_path], check=False)
                    return
                else:
                    print(f"[WARN] Target BT sink not found by desc='{prefer_sink_desc}'. Fallback to default sink.")

            subprocess.run(["paplay", wav_path], check=False)
            return

        # 2) paplay 없으면 aplay fallback (BT에서는 보통 실패)
        print("[WARN] paplay not found. Falling back to aplay (may fail for Bluetooth).")
        subprocess.run(["aplay", "-q", wav_path], check=False)
    
    def _convert_wav_for_bt(self, in_wav: str, out_wav: str):
        # ffmpeg 필요 (없으면 Dockerfile에 ffmpeg 설치)
        if shutil.which("ffmpeg") is None:
            print("[WARN] ffmpeg not found; skip conversion")
            return in_wav

        cmd = [
            "ffmpeg", "-y",
            "-i", in_wav,
            "-ar", "48000",      # 48kHz
            "-ac", "2",          # stereo
            "-sample_fmt", "s16",
            out_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return out_wav

    def speak(self, text: str, out_wav: str | None = None):
        """
        - piper로 wav 생성
        - 생성 wav 메타(sr/ch) 로깅
        - (BT 괴음 방지) 가능하면 ffmpeg로 48kHz/2ch/s16로 변환한 wav를 재생
        - paplay로 특정 BT sink(Description 매칭)로 출력
        """
        if out_wav is None:
            fd, out_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # 1) Piper: stdin으로 텍스트 넣고 wav 출력
        cmd = [
            self.piper_bin,
            "--model", self.model_path,
            "--config", self.config_path,
            "--output_file", out_wav,
        ]

        subprocess.run(
            cmd,
            input=(text.strip() + "\n"),
            text=True,
            check=True,
        )

        # 2) 생성된 wav 메타 확인
        print(f"[TTS] using model={self.model_path}")
        print(f"[TTS] using config={self.config_path}")

        try:
            data, sr = sf.read(out_wav)
            shape = getattr(data, "shape", None)
            rms = float(np.sqrt(np.mean(np.square(data)))) if data is not None else None
            print(f"[TTS] raw wav sr={sr}, shape={shape}, rms={rms:.6f}")
        except Exception as e:
            print(f"[WARN] failed to read generated wav: {e}")

        # 3) (중요) BT에서 괴음 방지: 48kHz/2ch/s16로 강제 변환해서 재생
        play_wav = out_wav
        try:
            if shutil.which("ffmpeg") is not None:
                bt_wav = out_wav[:-4] + ".bt.wav" if out_wav.lower().endswith(".wav") else out_wav + ".bt.wav"
                conv_cmd = [
                    "ffmpeg", "-y",
                    "-i", out_wav,
                    "-ar", "48000",      # BT(A2DP)에서 안정적인 편
                    "-ac", "2",          # stereo로 맞춤
                    "-sample_fmt", "s16",
                    bt_wav
                ]
                subprocess.run(
                    conv_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

                # 변환 성공/유효성 간단 체크
                if os.path.exists(bt_wav) and os.path.getsize(bt_wav) > 44:
                    try:
                        _, sr2 = sf.read(bt_wav)
                        print(f"[TTS] converted wav -> {bt_wav} (sr={sr2})")
                    except Exception:
                        print(f"[TTS] converted wav -> {bt_wav}")
                    play_wav = bt_wav
                else:
                    print("[WARN] ffmpeg conversion failed or produced invalid file. Using raw wav.")
            else:
                print("[WARN] ffmpeg not found. Using raw wav (BT may sound weird).")
        except Exception as e:
            print(f"[WARN] conversion step failed: {e}. Using raw wav.")

        # 4) 재생 (BT sink Description 매칭)
        self._play_wav(play_wav, prefer_sink_desc="Mi Portable BT Speaker 16W")
        return play_wav

