#!/usr/bin/env python3
import os
import time
import json
from collections import deque
from dataclasses import dataclass
import cv2

# ======================
# ===== 사용자 설정 =====
# ======================
traj_idx = 1   # <==== ★ 매 실행마다 여기만 수동으로 바꾸면 됨
K = 3
hz = 0.5
period = 1.0 / hz  # 2.0 sec

out_root = "./nav_dataset"
img_dir  = os.path.join(out_root, "images")
jsonl_path = os.path.join(out_root, "train_unlabeled.jsonl")

# ======================

@dataclass
class SampleMeta:
    t: float
    img_path: str

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def build_query(instruction: str, gesture: str, k: int) -> str:
    img_tokens = "<image>" * k
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
    ensure_dir(out_root)
    ensure_dir(img_dir)

    # ===== 네 파이프라인 재사용 =====
    from hint_task import open_realsense_capture, GestureCamera, _normalize_gesture_for_prompt
    cap, _, _ = open_realsense_capture()
    cam_thread = GestureCamera(cap=cap)
    cam_thread.start()

    instruction = os.environ.get("NAV_INSTRUCTION", "저쪽으로 가")
    gesture_raw = os.environ.get("NAV_GESTURE", "left")
    gesture = _normalize_gesture_for_prompt(gesture_raw)

    buf = deque(maxlen=K)
    frame_idx = 1

    try:
        while True:
            t0 = time.time()
            frame_bgr = cam_thread.get_latest_frame()
            if frame_bgr is None:
                time.sleep(0.05)
                continue

            ts = time.time()
            img_name = f"traj{traj_idx}_frame{frame_idx:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)

            if not cv2.imwrite(img_path, frame_bgr):
                print("[WARN] cv2.imwrite failed:", img_path)
                time.sleep(period)
                continue

            buf.append(SampleMeta(t=ts, img_path=f"images/{img_name}"))

            if len(buf) == K:
                images = [m.img_path for m in buf]
                query = build_query(instruction, gesture, K)

                row = {
                    "id": f"traj{traj_idx}_{frame_idx:06d}",
                    "query": query,
                    "response": "",  # 나중에 수동 라벨링
                    "images": images,
                    "meta": {
                        "traj_idx": traj_idx,
                        "instruction": instruction,
                        "gesture": gesture,
                        "hz": hz,
                        "k": K,
                        "timestamps": [m.t for m in buf],
                    },
                }

                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(f"[SAVE] traj={traj_idx} frame={frame_idx} images={images}")

            frame_idx += 1
            time.sleep(max(0.0, period - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n[INFO] stop")
    finally:
        cam_thread.stop()
        cam_thread.join()

if __name__ == "__main__":
    main()
