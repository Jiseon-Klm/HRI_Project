# config.py

import os
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.3
SILENCE_DURATION = 1.0
OUTPUT_FILE = "speech.wav"
MODEL_NAME = "medium" # Whisper 모델
LANG = "ko" # 한국어
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
INPUT_DEVICE_ID = 1
sd.default.device = (INPUT_DEVICE_ID, None)

# --- LLM (Large Language Model) 설정 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA-AIzaSyB7pAB6GMAsSsqphUZQWHuhrFG_R7zwWlI")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# LLM 프롬프트 템플릿
PROMPT_TEMPLATE = '''
From now on, whenever I ask you something,
please give me two versions of your answer:

1. The normal answer in Korean (depending on my question)
2. The English phonetic transcription of that same answer

Format:
---
Answer: <your normal answer>
Phonetic: <the English phonetic transcription>
---
'''

# --- TTS (Text-to-Speech) 설정 ---
# PIPER_MODEL_PATH = "/home/jhs/Desktop/jhs/HRI/version_2/en_US-lessac-medium.onnx"
PIPER_OUTPUT_FILE = "test.wav"

MMS_TTS_MODEL_ID = "facebook/mms-tts-kor"
MMS_TTS_OUTPUT_FILE = "output_kor.wav"

CAMERA_DEVICE_ID = 2 # OpenCV VideoCapture 장치 ID
