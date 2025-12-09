class GestureCamera(threading.Thread):
    """
    - 카메라 프레임에서 MediaPipe Hands로 손 검출
    - '오른쪽을 가리키는 손짓' ⇒ "turn right"
    - '왼쪽을 가리키는 손짓' ⇒ "turn left"
    - 그 외(수직/애매/손 없음)는 '제스처 없음' ⇒ gesture 갱신 안 함 (이전 값 유지)
    - gesture 변수에는 절대 'stop' 같은 값 안 넣음
    - main에서 한 번 freeze 요청이 오면, 그 이후로는 제스처를 더 이상 업데이트하지 않음
    """

    def __init__(
        self,
        cap,
        horizontal_ratio_threshold: float = 1.3,
        min_vec_norm: float = 0.08,
        max_angle_deg: float = 25.0,
        history_len: int = 7,
        min_stable_ratio: float = 0.7,
    ):
        """
        cap: open_realsense_capture() 에서 이미 '프레임까지' 확인하고 넘겨준
             cv2.VideoCapture 객체 (release 미호출 상태)

        horizontal_ratio_threshold:
            |dx| >= horizontal_ratio_threshold * |dy| 일 때만
            '수평에 가깝게 뻗은 손가락'으로 간주.
            값이 클수록 '옆으로 더 확실히 뻗은' 제스처만 인식.

        min_vec_norm:
            손목→검지 끝 벡터 길이 (0~1 정규화 기준)가
            이 값보다 작으면 노이즈로 간주.

        max_angle_deg:
            수평(0°)에서 ±max_angle_deg 이내일 때만 좌/우 제스처로 인정.

        history_len:
            최근 몇 프레임을 다수결에 사용할지.

        min_stable_ratio:
            history 내에서 특정 제스처가 이 비율 이상 나와야
            최종 제스처로 인정.
        """
        super().__init__(daemon=True)
        self.cap = cap
        self.running = True

        self._lock = threading.Lock()
        self._latest_frame_bgr = None

        # gesture는 "turn right", "turn left" 또는 None
        self._gesture = None

        self.horizontal_ratio_threshold = horizontal_ratio_threshold
        self.min_vec_norm = min_vec_norm
        self.max_angle_deg = max_angle_deg
        self.min_stable_ratio = min_stable_ratio

        # 제스처 freeze 플래그
        self._gesture_frozen = False

        # 최근 인식된 제스처 히스토리 (None은 넣지 않음)
        self._recent_gestures = deque(maxlen=history_len)

        # MediaPipe Hands 설정을 조금 더 보수적으로 (conf 높임)
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,   # 기존 0.5 → 0.7
            min_tracking_confidence=0.7,    # 기존 0.5 → 0.7
        )

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        """가장 최근 프레임(BGR)을 복사해서 반환 (없으면 None)"""
        with self._lock:
            if self._latest_frame_bgr is None:
                return None
            return self._latest_frame_bgr.copy()

    def get_gesture(self):
        """
        현재까지 인식된(또는 freeze된) 제스처 반환.
        - "turn right"
        - "turn left"
        - None (아직 아무 제스처도 안정적으로 인식된 적 없음)
        """
        with self._lock:
            return self._gesture

    def freeze_gesture_once(self):
        """
        제스처를 한 번만 freeze.
        이후에는 _update_gesture에서 더 이상 값이 바뀌지 않음.
        """
        with self._lock:
            if not self._gesture_frozen:
                self._gesture_frozen = True
                print(f"[GESTURE] 초기 제스처 freeze: {self._gesture}")

    def _update_gesture(self, new_gesture: str | None):
        """
        gesture 값 갱신 (다수결 + 안정화 적용).
        - new_gesture가 None이면 무시
        - frozen 상태면 무시
        - 최근 history에서 일정 비율 이상 나온 제스처만 확정
        """
        if new_gesture is None:
            return

        with self._lock:
            if self._gesture_frozen:
                return

            # 최근 제스처 히스토리에 추가
            self._recent_gestures.append(new_gesture)

            # 최소 3프레임 이상 쌓일 때부터 다수결 적용
            if len(self._recent_gestures) < 3:
                return

            # 다수결 (numpy 이용)
            vals, counts = np.unique(list(self._recent_gestures), return_counts=True)
            best_idx = int(np.argmax(counts))
            best_label = vals[best_idx]
            best_ratio = counts[best_idx] / len(self._recent_gestures)

            # 충분히 안정적이지 않으면 반영 안 함
            if best_ratio < self.min_stable_ratio:
                return

            if best_label != self._gesture:
                self._gesture = best_label
                print(f"[GESTURE] 안정화된 제스처: {self._gesture} (ratio={best_ratio:.2f})")

    def _infer_gesture_from_hand(self, hand_landmarks):
        """
        단순 규칙 기반 제스처 해석 (좌/우만):

        - 손목(WRIST) → 검지 손끝(INDEX_TIP) 방향 벡터 (dx, dy) 사용
        - 벡터 길이가 너무 짧으면 노이즈로 간주
        - 벡터 각도가 수평(0도)에서 너무 멀면 (수직에 가깝거나 애매) 제스처 없음
        - |dx|가 |dy|보다 충분히 크지 않으면 제스처 없음
        - 최종적으로:
            dx > 0  : turn right
            dx < 0  : turn left

        ※ 이미지 좌표 기준: x 오른쪽 증가, y 아래쪽 증가
        """
        WRIST = 0
        INDEX_TIP = 8

        wrist = hand_landmarks.landmark[WRIST]
        tip = hand_landmarks.landmark[INDEX_TIP]

        dx = tip.x - wrist.x
        dy = tip.y - wrist.y  # 아래로 증가

        # 1) 벡터 길이가 너무 짧으면 (손가락 거의 안 뻗음) → 노이즈
        vec_norm = math.hypot(dx, dy)
        if vec_norm < self.min_vec_norm:
            return None

        # 2) 수평 각도 체크 (수평에서 ±max_angle_deg 안쪽만 인정)
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if angle_deg > self.max_angle_deg:
            return None

        # 3) 수평 성분 비율 체크 (dy가 너무 크면 수직에 가까운 제스처)
        if abs(dx) < self.horizontal_ratio_threshold * abs(dy):
            return None

        # 여기까지 왔으면 '옆으로 꽤 뻗은' 손가락이라고 간주
        if dx > 0:
            # 화면 오른쪽을 가리킴 ⇒ 로봇 에고 프레임 기준 "turn right"
            return "turn right"
        else:
            return "turn left"

    def run(self):
        if not self.cap.isOpened():
            print("[ERROR] GestureCamera: 전달받은 cap이 이미 닫혀 있습니다.")
            return

        print("[INFO] GestureCamera 쓰레드 시작")

        while self.running:
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                print("[WARN] 카메라 프레임 읽기 실패")
                time.sleep(0.05)
                continue

            # 최신 프레임 저장 (freeze 여부와 관계 없이 계속 업데이트)
            with self._lock:
                self._latest_frame_bgr = frame_bgr

            # 제스처가 이미 freeze 되었더라도, 카메라 프레임은 계속 갱신해야 함
            if self._gesture_frozen:
                time.sleep(0.03)
                continue

            # MediaPipe Hands는 RGB 입력
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(frame_rgb)

            new_gesture = None  # 기본값: 이번 프레임에서는 새 제스처 없음

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                g = self._infer_gesture_from_hand(hand_lms)
                if g is not None:
                    new_gesture = g

            # 제스처 갱신 (None이면 무시 / freeze면 무시 / 다수결 안정화)
            self._update_gesture(new_gesture)

            # 약 30fps
            time.sleep(0.03)

        self.cap.release()
        self._mp_hands.close()
        print("[INFO] GestureCamera 쓰레드 종료")
