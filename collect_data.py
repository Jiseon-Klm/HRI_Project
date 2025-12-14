#!/usr/bin/env python3
import os
import time
import json
from collections import deque
from dataclasses import dataclass
import cv2

@dataclass
class SampleMeta:
    t: float
    img_path: str

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def build_query(instruction: str, gesture: str, k: int) -> str:
    # Qwen2-VL(ms-swift) 포맷: 이미지 수만큼 <image>를 query에 포함 :contentReference[oaicite:2]{index=2}
    img_tokens = "<image>" * k
    # K=3에서 “거리감/시간감” 학습을 돕기 위해 시간 간격을 명시적으로 넣어주는 게 좋아요.
    return (
        f"{img_tokens}\n"
        "You are a short-horizon navigation policy for a mobile robot.\n"
        "You will be given K sequential front-facing RGB frames sampled at 0.5 Hz "
        "(i.e., 2 seconds between frames). The last frame is the most recent.\n"
        f"Instruction: {instruction}\n"
        f"Gesture: {gesture}  # one of {{left,right,forward,none}}\n"
        "Task: Predict ONE action to execute for the NEXT ~2 seconds.\n"
        "Action space: {forward, left, right, stop, goal}\n"
        "Output ONLY the action token.\n"
    )

def main():
    # ====== 사용자 설정 ======
    out_root = "./nav_dataset/traj#"
    img_dir  = os.path.join(out_root, "images")
    jsonl_path = os.path.join(out_root, "train_unlabeled.jsonl")
    K = 3
    hz = 0.5
    period = 1.0 / hz  # 2.0 sec

    ensure_dir(out_root)
    ensure_dir(img_dir)

    # ====== 여기만 네 코드 객체로 교체 ======
    # frame_provider는 "현재 BGR 프레임 반환" 함수/객체면 충분
    # 예: frame_bgr = cam_thread.get_latest_frame()
    from hint_task import open_realsense_capture, GestureCamera, _normalize_gesture_for_prompt
    cap, _, _ = open_realsense_capture()
    cam_thread = GestureCamera(cap=cap)
    cam_thread.start()

    # 초기 instruction/gesture는 네 파이프라인에서 이미 얻는다고 했으니
    # 여기서는 예시로 환경변수/입력으로 받게 해둘게요.
    instruction = os.environ.get("NAV_INSTRUCTION", "저쪽으로 가")
    gesture_raw = os.environ.get("NAV_GESTURE", "left")  # left/right/none 등
    gesture = _normalize_gesture_for_prompt(gesture_raw)

    # ====== K-window 버퍼 ======
    buf = deque(maxlen=K)

    # ====== 루프 ======
    frame_idx = 1
    sample_idx = 1
    try:
        while True:
            t0 = time.time()
            frame_bgr = cam_thread.get_latest_frame()
            if frame_bgr is None:
                time.sleep(0.05)
                continue
    
            ts = time.time()
            img_name = f"frame{frame_idx}.jpg"
            img_path = os.path.join(img_dir, img_name)
            ok = cv2.imwrite(img_path, frame_bgr)
            if not ok:
                print("[WARN] cv2.imwrite failed:", img_path)
                time.sleep(period)
                continue
    
            buf.append(SampleMeta(t=ts, img_path=img_path))
    
            if len(buf) == K:
                images = [m.img_path for m in list(buf)]
                query = build_query(instruction, gesture, K)
    
                row = {
                    "id": f"nav_{sample_idx:08d}",
                    "query": query,
                    "response": "",
                    "images": images,
                    "meta": {
                        "instruction": instruction,
                        "gesture": gesture,
                        "hz": hz,
                        "k": K,
                        "timestamps": [m.t for m in list(buf)],
                    },
                }
    
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
                print(f"[SAVE] sample id={row['id']} images={len(images)} -> {jsonl_path}")
                sample_idx += 1
    
            frame_idx += 1
    
            dt = time.time() - t0
            sleep_s = max(0.0, period - dt)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[INFO] stop")
    finally:
        cam_thread.stop()
        cam_thread.join()

if __name__ == "__main__":
    main()
