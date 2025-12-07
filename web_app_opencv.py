"""
Robot Control Web Application - OpenCV Color Detection
- Sử dụng OpenCV HSV để nhận dạng 3 màu (red, green, blue) thay vì YOLO
- Cho phép điều chỉnh màu nhận diện qua giao diện web
- Video stream từ camera
- Điều khiển robot qua giao diện web
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import math
import requests
import time
import threading
import json
import os
import logging

# Tắt Flask request logging để giảm spam console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ================== ROBOT CONFIG ==================
ROBOT_IP = "192.168.1.23"
ROBOT_PW = "5613"
ROBOT_ENABLED = True
HEARTBEAT_INTERVAL = 10.0

# ================== CONTAINER/BIN POSITION ==================
CONTAINER_POS = (200, 360)
CONTAINER_MARGIN = 120

# ================== MARKER DISTANCE CONFIG ==================
# Khoảng cách tối thiểu giữa red và blue marker (pixel)
# Nếu < MIN -> nhận diện sai (2 marker quá gần)
# Nếu > MAX -> robot quá xa camera
MARKER_MIN_DIST = 150
MARKER_MAX_DIST = 600

# ================== PICKUP DISTANCE CONFIG ==================
# Khoảng cách từ robot đến green target để dừng lại pick (pixel)
# Được đo từ tâm robot (giữa red-blue) đến tâm green
PICKUP_DISTANCE = 80
PICKUP_RANGE = 10  # Dung sai ± cho pickup distance

# ================== ROBOT STATE ==================
ROBOT_STATE = "SEEKING_GREEN"

# ================== CAMERA CONFIG ==================
CAM_INDEX = 1       # Camera index
FRAME_WIDTH = 1280  # Độ phân giải gốc
FRAME_HEIGHT = 720  # Độ phân giải gốc
JPEG_QUALITY = 80   # Chất lượng JPEG cao
TARGET_FPS = 30     # FPS mục tiêu
PROCESS_EVERY_N_FRAMES = 1  # Xử lý mỗi N frame (1 = tất cả, 2 = bỏ qua 1)

# ================== COLOR CONFIG ==================
# File lưu cấu hình màu
COLOR_CONFIG_FILE = "color_config.json"

# Cấu hình màu mặc định (RGB)
DEFAULT_COLORS_RGB = {
    "red": (255, 80, 80),      # Đỏ
    "green": (80, 200, 80),    # Xanh lá
    "blue": (80, 150, 255),    # Xanh dương
}

# Dung sai HSV mặc định
DEFAULT_HSV_TOLERANCE = {
    "h_tol": 15,   # ±15 đơn vị Hue (0..179)
    "s_tol": 80,   # ±80 đơn vị Saturation (0..255)
    "v_tol": 80,   # ±80 đơn vị Value (0..255)
}

# Diện tích tối thiểu để nhận dạng
MIN_AREA = 500

# Màu vẽ bbox (BGR)
BOX_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 200, 0),
}

# Morphology kernel
MORPH_KERNEL_SIZE = 5
MORPH_KERNEL = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

# ================== GLOBAL STATE ==================
_last_sent_cmd = None
_last_sent_time = 0
_current_frame = None
_frame_lock = threading.Lock()
_command_queue = []  # Queue lệnh gửi robot (non-blocking)
_command_lock = threading.Lock()
_detection_info = {
    "state": "SEEKING_GREEN",
    "command": "STOP",
    "robot_enabled": True,
    "markers_ok": False,
    "green_detected": False,
    "robot_pos": None,
    "detections": [],
    "marker_dist": 0,
    "marker_dist_ok": False,
    "fps": 0,
}

# Color configuration (sẽ được load từ file hoặc dùng default)
colors_rgb = {}
colors_hsv = {}
hsv_tolerance = {}

# ================== COLOR UTILITIES ==================
def rgb_to_hsv_opencv(r, g, b):
    """Chuyển RGB (0..255) sang HSV (OpenCV: H 0..179, S/V 0..255)."""
    patch = np.uint8([[[b, g, r]]])  # OpenCV dùng BGR
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0, :]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])

def inrange_hsv_tolerant(hsv_img, center_hsv, tol_h, tol_s, tol_v):
    """Tạo mask quanh tâm HSV với dung sai, xử lý cả wrap-around Hue."""
    Hc, Sc, Vc = center_hsv
    H_lo = (Hc - tol_h) % 180
    H_hi = (Hc + tol_h) % 180

    S_lo = max(0, Sc - tol_s)
    S_hi = min(255, Sc + tol_s)
    V_lo = max(0, Vc - tol_v)
    V_hi = min(255, Vc + tol_v)

    if H_lo <= H_hi:
        lower = np.array([H_lo, S_lo, V_lo], dtype=np.uint8)
        upper = np.array([H_hi, S_hi, V_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
    else:
        # Wrap quanh 180 → ghép 2 khoảng
        lower1 = np.array([0, S_lo, V_lo], dtype=np.uint8)
        upper1 = np.array([H_hi, S_hi, V_hi], dtype=np.uint8)
        lower2 = np.array([H_lo, S_lo, V_lo], dtype=np.uint8)
        upper2 = np.array([179, S_hi, V_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    return mask

def centroid(contour):
    """Tính tâm của contour."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def load_color_config():
    """Load TẤT CẢ cấu hình từ file."""
    global colors_rgb, colors_hsv, hsv_tolerance
    global ROBOT_IP, ROBOT_PW, CONTAINER_POS, CONTAINER_MARGIN
    global MARKER_MIN_DIST, MARKER_MAX_DIST, PICKUP_DISTANCE, PICKUP_RANGE
    
    if os.path.exists(COLOR_CONFIG_FILE):
        try:
            with open(COLOR_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
                # Color config
                colors_rgb = {k: tuple(v) for k, v in config.get('colors_rgb', DEFAULT_COLORS_RGB).items()}
                hsv_tolerance = config.get('hsv_tolerance', DEFAULT_HSV_TOLERANCE)
                
                # Robot config
                ROBOT_IP = config.get('robot_ip', ROBOT_IP)
                ROBOT_PW = config.get('robot_pw', ROBOT_PW)
                
                # Container/BIN config
                container = config.get('container_pos', list(CONTAINER_POS))
                CONTAINER_POS = tuple(container) if isinstance(container, list) else CONTAINER_POS
                CONTAINER_MARGIN = config.get('container_margin', CONTAINER_MARGIN)
                
                # Marker distance config
                MARKER_MIN_DIST = config.get('marker_min_dist', MARKER_MIN_DIST)
                MARKER_MAX_DIST = config.get('marker_max_dist', MARKER_MAX_DIST)
                
                # Pickup distance config
                PICKUP_DISTANCE = config.get('pickup_distance', PICKUP_DISTANCE)
                PICKUP_RANGE = config.get('pickup_range', PICKUP_RANGE)
                
                print(f"[INFO] Loaded all config from {COLOR_CONFIG_FILE}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            colors_rgb = DEFAULT_COLORS_RGB.copy()
            hsv_tolerance = DEFAULT_HSV_TOLERANCE.copy()
    else:
        colors_rgb = DEFAULT_COLORS_RGB.copy()
        hsv_tolerance = DEFAULT_HSV_TOLERANCE.copy()
        print("[INFO] Using default config")
    
    # Chuyển sang HSV
    colors_hsv = {name: rgb_to_hsv_opencv(*rgb) for name, rgb in colors_rgb.items()}
    print(f"[INFO] Colors RGB: {colors_rgb}")
    print(f"[INFO] Robot IP: {ROBOT_IP}, Container: {CONTAINER_POS}, Margin: {CONTAINER_MARGIN}")
    print(f"[INFO] Marker dist: {MARKER_MIN_DIST}-{MARKER_MAX_DIST}, Pickup: {PICKUP_DISTANCE}")

def save_color_config():
    """Lưu cấu hình màu ra file."""
    save_all_config()

def save_all_config():
    """Lưu TẤT CẢ cấu hình ra file (màu + measure tools + robot config)."""
    config = {
        'colors_rgb': {k: list(v) for k, v in colors_rgb.items()},
        'hsv_tolerance': hsv_tolerance,
        'robot_ip': ROBOT_IP,
        'robot_pw': ROBOT_PW,
        'container_pos': list(CONTAINER_POS),
        'container_margin': CONTAINER_MARGIN,
        'marker_min_dist': MARKER_MIN_DIST,
        'marker_max_dist': MARKER_MAX_DIST,
        'pickup_distance': PICKUP_DISTANCE,
        'pickup_range': PICKUP_RANGE,
    }
    try:
        with open(COLOR_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[INFO] Saved all config to {COLOR_CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")

def update_color(color_name, r, g, b):
    """Cập nhật màu và tính lại HSV."""
    global colors_rgb, colors_hsv
    colors_rgb[color_name] = (r, g, b)
    colors_hsv[color_name] = rgb_to_hsv_opencv(r, g, b)
    save_color_config()

def update_tolerance(h_tol, s_tol, v_tol):
    """Cập nhật dung sai HSV."""
    global hsv_tolerance
    hsv_tolerance = {
        'h_tol': h_tol,
        's_tol': s_tol,
        'v_tol': v_tol,
    }
    save_color_config()

# ================== ROBOT HTTP CONTROL (NON-BLOCKING) ==================
# Timeout dài hơn cho combo commands (có nhiều delay trong Arduino)
COMBO_COMMANDS = {"COMBO_PICK", "COMBO_DROP", "BOW_DOWN", "BOW_UP", "GRIP_OPEN", "GRIP_CLOSE"}
TIMEOUT_NORMAL = 2.0   # 2 giây cho lệnh thông thường
TIMEOUT_COMBO = 10.0   # 10 giây cho combo commands (thực tế ~7s)
COOLDOWN_AFTER_ERROR = 5.0  # Tạm dừng 5 giây sau khi timeout
COMBO_WAIT_TIME = 7.0  # Thời gian đợi combo hoàn thành (giây)

# State cho command sending
_last_error_time = 0       # Thời điểm xảy ra lỗi gần nhất
_pending_command = None    # Lệnh đang chờ gửi (chỉ giữ lệnh mới nhất)
_is_sending = False        # Đang có request đang gửi?
_send_lock = threading.Lock()  # Lock để tránh race condition
_combo_start_time = 0      # Thời điểm bắt đầu combo (để đợi)

# HTTP Session với Keep-Alive (giảm overhead tạo connection mới)
_http_session = None

def _get_http_session():
    """Lấy hoặc tạo HTTP session với keep-alive."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Giữ connection alive, giảm timeout cho connect
        _http_session.headers.update({'Connection': 'keep-alive'})
    return _http_session

def _reset_http_session():
    """Reset session khi có lỗi connection."""
    global _http_session
    if _http_session:
        try:
            _http_session.close()
        except:
            pass
    _http_session = None

def _is_in_cooldown():
    """Kiểm tra có đang trong cooldown không."""
    return time.time() - _last_error_time < COOLDOWN_AFTER_ERROR

def _send_command_worker(cmd, url):
    """Worker thread gửi lệnh HTTP (không block main thread)."""
    global _last_sent_cmd, _last_sent_time, _last_error_time, _is_sending
    
    with _send_lock:
        _is_sending = True
    
    try:
        # Dùng session với keep-alive
        session = _get_http_session()
        timeout = TIMEOUT_COMBO if cmd in COMBO_COMMANDS else TIMEOUT_NORMAL
        resp = session.get(url, timeout=timeout)
        
        with _send_lock:
            _last_sent_cmd = cmd
            _last_sent_time = time.time()
        print(f"[ROBOT] {cmd} -> {resp.text}")
        
    except requests.exceptions.RequestException as e:
        with _send_lock:
            _last_error_time = time.time()  # Đánh dấu thời điểm lỗi
        print(f"[ROBOT] Error: {e}")
        print(f"[ROBOT] Cooldown {COOLDOWN_AFTER_ERROR}s - không gửi lệnh mới")
        # Reset session khi có lỗi để tạo connection mới
        _reset_http_session()
    finally:
        with _send_lock:
            _is_sending = False

# Endpoint mapping cho các lệnh robot
ENDPOINT_MAP = {
    "MOVE_FORWARD": "/forward",
    "MOVE_BACKWARD": "/backward",
    "TURN_LEFT": "/left",
    "TURN_RIGHT": "/right",
    "STOP": "/stop",
    "SEARCH_TARGET": "/stop",
    "SPIN_LEFT": "/spin-left",
    "SPIN_RIGHT": "/spin-right",
    "BOW_DOWN": "/bow-down",
    "BOW_UP": "/bow-up",
    "GRIP_OPEN": "/grip-open",
    "GRIP_CLOSE": "/grip-close",
    "COMBO_PICK": "/combo-pick",
    "COMBO_DROP": "/combo-drop",
}

def send_robot_command(cmd, force=False):
    """Gửi lệnh HTTP đến robot (NON-BLOCKING - chạy trong thread riêng).
    Chỉ hoạt động khi ROBOT_ENABLED = True (Auto mode).
    - Không gửi nếu đang trong cooldown sau lỗi
    - Không gửi nếu đang có request pending
    - Chỉ gửi khi lệnh thay đổi (trừ force=True)
    """
    global _last_sent_cmd, _last_sent_time, ROBOT_ENABLED, _pending_command
    
    if not ROBOT_ENABLED:
        return False
    
    with _send_lock:
        now = time.time()
        in_cooldown = now - _last_error_time < COOLDOWN_AFTER_ERROR
        is_busy = _is_sending
        same_cmd = (cmd == _last_sent_cmd)
        time_since_last = now - _last_sent_time
    
    # Kiểm tra cooldown sau lỗi
    if in_cooldown:
        # Trong cooldown, chỉ lưu lệnh mới nhất (không gửi)
        _pending_command = cmd
        return True
    
    # Nếu đang gửi request khác, lưu lệnh mới nhất
    if is_busy:
        _pending_command = cmd
        return True
    
    # Chỉ gửi khi lệnh THAY ĐỔI (không heartbeat khi đang timeout liên tục)
    if not force and same_cmd:
        # Không gửi heartbeat nữa - chỉ gửi khi lệnh thay đổi
        return True
    
    endpoint = ENDPOINT_MAP.get(cmd)
    if not endpoint:
        return False
    
    url = f"http://{ROBOT_IP}{endpoint}?pw={ROBOT_PW}"
    
    # Cập nhật trước khi gửi
    with _send_lock:
        _last_sent_cmd = cmd
        _last_sent_time = time.time()
    _pending_command = None  # Clear pending vì đang gửi lệnh này
    
    # Gửi trong thread riêng để không block video
    thread = threading.Thread(target=_send_command_worker, args=(cmd, url), daemon=True)
    thread.start()
    
    return True

def process_pending_command():
    """Xử lý lệnh pending sau khi hết cooldown. Gọi từ main loop."""
    global _pending_command
    
    if _pending_command is None:
        return
    
    with _send_lock:
        in_cooldown = _is_in_cooldown()
        is_busy = _is_sending
        last_cmd = _last_sent_cmd
    
    # Không gửi nếu giống lệnh đã gửi (tránh spam)
    if _pending_command == last_cmd:
        _pending_command = None
        return
    
    if not in_cooldown and not is_busy:
        cmd = _pending_command
        _pending_command = None
        print(f"[ROBOT] Hết cooldown, gửi lệnh pending: {cmd}")
        send_robot_command(cmd, force=True)

def send_manual_command(cmd):
    """Gửi lệnh thủ công đến robot - hoạt động khi Auto OFF.
    Cũng tuân theo cooldown để tránh spam khi robot không phản hồi."""
    global _pending_command
    
    endpoint = ENDPOINT_MAP.get(cmd)
    if not endpoint:
        print(f"[MANUAL] Unknown command: {cmd}")
        return False
    
    # Kiểm tra cooldown - manual cũng cần tôn trọng cooldown
    with _send_lock:
        in_cooldown = _is_in_cooldown()
        is_busy = _is_sending
    
    if in_cooldown:
        print(f"[MANUAL] Đang cooldown, lưu lệnh: {cmd}")
        _pending_command = cmd
        return True
    
    if is_busy:
        print(f"[MANUAL] Đang gửi lệnh khác, lưu: {cmd}")
        _pending_command = cmd
        return True
    
    url = f"http://{ROBOT_IP}{endpoint}?pw={ROBOT_PW}"
    print(f"[MANUAL] Sending {cmd} to {url}")
    
    # Gửi trong thread riêng để không block
    thread = threading.Thread(target=_send_command_worker, args=(cmd, url), daemon=True)
    thread.start()
    
    return True

# ================== HELPER FUNCTIONS ==================
def is_in_container_zone(cx, cy):
    """Kiểm tra tọa độ có nằm trong vùng container không."""
    dist = math.hypot(cx - CONTAINER_POS[0], cy - CONTAINER_POS[1])
    return dist <= CONTAINER_MARGIN

def determine_orientation(blue_obj, red_obj, margin=25):
    """Xác định hướng từ 2 markers: blue (trái) và red (phải)."""
    if not blue_obj or not red_obj:
        return "MISSING_MARKER", None, None
    
    blue_center = (blue_obj["cx"], blue_obj["cy"])
    red_center = (red_obj["cx"], red_obj["cy"])
    dx = red_center[0] - blue_center[0]
    
    if dx > margin:
        status = "LEFT_RIGHT_OK"
    elif dx < -margin:
        status = "LEFT_RIGHT_FLIPPED"
    else:
        status = "AMBIGUOUS"
    
    return status, blue_center, red_center

def check_balanced_triangle(red_obj, blue_obj, green_obj, max_rel_diff=0.25):
    """Kiểm tra 3 điểm tạo thành tam giác cân."""
    if not (red_obj and blue_obj and green_obj):
        return False

    def dist(a, b):
        return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])

    d_rb = dist(red_obj, blue_obj)
    d_rg = dist(red_obj, green_obj)
    d_bg = dist(blue_obj, green_obj)

    mean_d = (d_rb + d_rg + d_bg) / 3.0
    if mean_d < 1e-3:
        return False

    for d in (d_rb, d_rg, d_bg):
        if abs(d - mean_d) > max_rel_diff * mean_d:
            return False
    return True

def decide_move_to_target(target_pos, heading_info, frame_w, frame_h):
    """
    Di chuyển robot về phía target.
    - Tính góc giữa hướng robot và hướng đến target
    - Góc > 45° → SPIN (quay nhanh tại chỗ)
    - Góc ≤ 45° → TURN (rẽ nhẹ)
    - Target ở phía sau (góc > 90°) → cũng dùng SPIN
    """
    if target_pos is None:
        return "SEARCH_TARGET"
    
    tx, ty = target_pos
    
    if heading_info is not None:
        origin, direction = heading_info
        hx, hy = origin
        dir_x, dir_y = direction
        
        # Vector từ robot đến target
        vec_x = tx - hx
        vec_y = ty - hy
        vec_len = math.hypot(vec_x, vec_y)
        
        if vec_len < 50:
            return "STOP"
        
        # Chuẩn hóa vector
        vec_x /= vec_len
        vec_y /= vec_len
        
        # Dot product: cos(angle) giữa hướng robot và hướng đến target
        # dot = 1 → cùng hướng (0°)
        # dot = 0 → vuông góc (90°)
        # dot = -1 → ngược hướng (180°)
        dot = dir_x * vec_x + dir_y * vec_y
        
        # Cross product: xác định quay trái hay phải
        # cross > 0 → target bên phải → quay phải
        # cross < 0 → target bên trái → quay trái
        cross = dir_x * vec_y - dir_y * vec_x
        
        # Tính góc (độ)
        # Clamp dot để tránh lỗi acos do floating point
        dot_clamped = max(-1.0, min(1.0, dot))
        angle_deg = math.degrees(math.acos(dot_clamped))
        
        # Ngưỡng góc
        ANGLE_FORWARD = 10.0   # Đủ thẳng để tiến
        ANGLE_SPIN = 45.0      # Góc lớn → spin nhanh
        
        if angle_deg <= ANGLE_FORWARD:
            # Gần như thẳng hướng → tiến tới
            return "MOVE_FORWARD"
        elif angle_deg <= ANGLE_SPIN:
            # Góc nhỏ (10-45°) → rẽ nhẹ (vừa tiến vừa quay)
            return "TURN_RIGHT" if cross > 0 else "TURN_LEFT"
        else:
            # Góc lớn (>45°) hoặc target ở phía sau → spin tại chỗ
            return "SPIN_RIGHT" if cross > 0 else "SPIN_LEFT"
    else:
        # Không có heading info → dùng vị trí trong frame
        center_x = frame_w / 2
        offset_x = tx - center_x
        margin = frame_w * 0.1
        
        if abs(offset_x) > margin:
            return "TURN_LEFT" if offset_x < 0 else "TURN_RIGHT"
        
        return "MOVE_FORWARD"

# ================== COLOR DETECTION ==================
def detect_color(hsv_img, frame, color_name, kernel):
    """Phát hiện một màu trong ảnh HSV (chỉ trả về object lớn nhất)."""
    if color_name not in colors_hsv:
        return None
    
    hsv_c = colors_hsv[color_name]
    h_tol = hsv_tolerance.get('h_tol', 15)
    s_tol = hsv_tolerance.get('s_tol', 80)
    v_tol = hsv_tolerance.get('v_tol', 80)
    
    # Tạo mask
    raw_mask = inrange_hsv_tolerant(hsv_img, hsv_c, h_tol, s_tol, v_tol)
    
    # Làm sạch mask
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Tìm contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        ctr = centroid(c)
        if ctr is None:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if best is None or area > best[0]:
            best = (area, x, y, w, h, ctr)
    
    if best is not None:
        area, x, y, w, h, ctr = best
        return {
            "label": color_name,
            "cx": ctr[0],
            "cy": ctr[1],
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": area,
        }
    
    return None

def detect_all_of_color(hsv_img, frame, color_name, kernel):
    """Phát hiện TẤT CẢ objects của một màu trong ảnh HSV."""
    if color_name not in colors_hsv:
        return []
    
    hsv_c = colors_hsv[color_name]
    h_tol = hsv_tolerance.get('h_tol', 15)
    s_tol = hsv_tolerance.get('s_tol', 80)
    v_tol = hsv_tolerance.get('v_tol', 80)
    
    # Tạo mask
    raw_mask = inrange_hsv_tolerant(hsv_img, hsv_c, h_tol, s_tol, v_tol)
    
    # Làm sạch mask
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Tìm contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        ctr = centroid(c)
        if ctr is None:
            continue
        x, y, w, h = cv2.boundingRect(c)
        results.append({
            "label": color_name,
            "cx": ctr[0],
            "cy": ctr[1],
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": area,
        })
    
    # Sắp xếp theo diện tích giảm dần
    results.sort(key=lambda d: d["area"], reverse=True)
    return results

# ================== VIDEO PROCESSING ==================
def process_frame(frame):
    global ROBOT_STATE, _detection_info, _combo_start_time
    
    # Xử lý lệnh pending (sau cooldown)
    process_pending_command()
    
    H, W = frame.shape[:2]
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Phát hiện red và blue markers (chỉ cần 1)
    top_red = detect_color(hsv, frame, "red", MORPH_KERNEL)
    top_blue = detect_color(hsv, frame, "blue", MORPH_KERNEL)
    
    # Phát hiện TẤT CẢ green objects
    all_greens = detect_all_of_color(hsv, frame, "green", MORPH_KERNEL)
    
    # Tạo detections dict cho vẽ
    detections = {}
    if top_red:
        detections["red"] = top_red
    if top_blue:
        detections["blue"] = top_blue
    
    # Heading từ blue/red markers (cần tính trước để chọn green gần nhất)
    heading_info = None
    marker_base_len = None
    orientation, blue_center, red_center = determine_orientation(top_blue, top_red)
    robot_pos = None
    
    if blue_center and red_center:
        mid_x = int((blue_center[0] + red_center[0]) / 2)
        mid_y = int((blue_center[1] + red_center[1]) / 2)
        dx = red_center[0] - blue_center[0]
        dy = red_center[1] - blue_center[1]
        base_len = math.hypot(dx, dy)
        marker_base_len = base_len
        
        if base_len > 1e-3:
            perp_x = dy / base_len
            perp_y = -dx / base_len
            heading_info = ((mid_x, mid_y), (perp_x, perp_y))
            robot_pos = (mid_x, mid_y)
    
    # Lọc green ngoài container zone và chọn GREEN GẦN NHẤT với robot
    greens_outside = []
    for g in all_greens:
        if not is_in_container_zone(g["cx"], g["cy"]):
            # Tính khoảng cách đến robot
            if robot_pos:
                g["dist_to_robot"] = math.hypot(robot_pos[0] - g["cx"], robot_pos[1] - g["cy"])
            else:
                g["dist_to_robot"] = float('inf')
            greens_outside.append(g)
    
    # Sắp xếp theo khoảng cách tăng dần (gần nhất trước)
    greens_outside.sort(key=lambda g: g["dist_to_robot"])
    
    # Green gần nhất (target)
    green_target = greens_outside[0] if greens_outside else None
    
    # Vẽ TẤT CẢ green objects
    for i, g in enumerate(all_greens):
        x, y, w, h = g["bbox"]["x"], g["bbox"]["y"], g["bbox"]["w"], g["bbox"]["h"]
        cx, cy = g["cx"], g["cy"]
        
        in_bin = is_in_container_zone(cx, cy)
        is_target = (green_target and g["cx"] == green_target["cx"] and g["cy"] == green_target["cy"])
        
        if in_bin:
            label = "green(BIN)"
            box_color = (0, 100, 0)
        elif is_target:
            label = "GREEN*"  # Đánh dấu target
            box_color = (0, 255, 0)
        else:
            label = f"green#{i+1}"
            box_color = (0, 200, 0)
        
        # Vẽ khoảng cách nếu có robot_pos
        dist_text = ""
        if robot_pos and not in_bin:
            dist = math.hypot(robot_pos[0] - cx, robot_pos[1] - cy)
            dist_text = f" d={dist:.0f}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3 if is_target else 2)
        cv2.drawMarker(frame, (cx, cy), box_color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(frame, f"{label}{dist_text}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        detections[f"green_{i}"] = g
    
    # Vẽ red và blue markers
    for color_name in ["red", "blue"]:
        det = detections.get(color_name)
        if det:
            x, y, w, h = det["bbox"]["x"], det["bbox"]["y"], det["bbox"]["w"], det["bbox"]["h"]
            cx, cy = det["cx"], det["cy"]
            box_color = BOX_COLORS.get(color_name, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.drawMarker(frame, (cx, cy), box_color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(frame, f"{color_name} ({cx},{cy})", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Vẽ container
    cv2.circle(frame, CONTAINER_POS, CONTAINER_MARGIN, (255, 0, 255), 2)
    cv2.putText(frame, "BIN", (CONTAINER_POS[0] - 20, CONTAINER_POS[1] - CONTAINER_MARGIN - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Vẽ heading arrow
    if blue_center and red_center and heading_info:
        cv2.line(frame, blue_center, red_center, (0, 255, 255), 3)
        origin, (perp_x, perp_y) = heading_info
        heading_len = max(60, int(marker_base_len * 0.6))
        head_end = (int(origin[0] + perp_x * heading_len), int(origin[1] + perp_y * heading_len))
        cv2.arrowedLine(frame, origin, head_end, (255, 255, 0), 3, tipLength=0.2)
    
    # State machine
    dist_to_green = None
    dist_to_bin = None
    angle_to_bin = None  # Góc từ hướng robot đến tâm BIN
    green_count = len(greens_outside)  # Số green còn lại ngoài BIN
    
    if robot_pos:
        dist_to_bin = math.hypot(robot_pos[0] - CONTAINER_POS[0], robot_pos[1] - CONTAINER_POS[1])
    
    move_cmd = "STOP"
    
    if ROBOT_STATE == "SEEKING_GREEN":
        if green_target:
            green_pos = (green_target["cx"], green_target["cy"])
            
            # Tính khoảng cách từ robot đến green
            if robot_pos:
                dist_to_green = math.hypot(robot_pos[0] - green_pos[0], robot_pos[1] - green_pos[1])
            
            # Dùng PICKUP_DISTANCE: khi gần hơn hoặc bằng thì pick
            if dist_to_green is not None and dist_to_green <= PICKUP_DISTANCE:
                print(f"[STATE] SEEKING_GREEN -> PICKING (dist={dist_to_green:.0f}px <= {PICKUP_DISTANCE}px)")
                ROBOT_STATE = "PICKING"
                move_cmd = "STOP"
            else:
                move_cmd = decide_move_to_target(green_pos, heading_info, W, H)
        else:
            move_cmd = "SEARCH_TARGET"
    
    elif ROBOT_STATE == "PICKING":
        # Gửi COMBO_PICK và chờ hoàn thành
        print("[STATE] PICKING -> WAIT_PICK (sending COMBO_PICK)")
        send_robot_command("COMBO_PICK", force=True)
        _combo_start_time = time.time()
        ROBOT_STATE = "WAIT_PICK"
        move_cmd = "STOP"
    
    elif ROBOT_STATE == "WAIT_PICK":
        # Đợi combo pick hoàn thành theo thời gian thực
        elapsed = time.time() - _combo_start_time
        
        if elapsed >= COMBO_WAIT_TIME:
            print(f"[STATE] WAIT_PICK -> RETURNING (waited {elapsed:.1f}s)")
            ROBOT_STATE = "RETURNING"
        move_cmd = "STOP"
    
    elif ROBOT_STATE == "RETURNING":
        # Tính góc từ hướng robot đến tâm BIN
        ready_to_drop = False
        
        if robot_pos and heading_info and dist_to_bin is not None:
            origin, direction = heading_info
            dir_x, dir_y = direction
            
            # Vector từ robot đến tâm BIN
            vec_x = CONTAINER_POS[0] - robot_pos[0]
            vec_y = CONTAINER_POS[1] - robot_pos[1]
            vec_len = dist_to_bin
            
            if vec_len > 1e-3:
                vec_x /= vec_len
                vec_y /= vec_len
                
                # Dot product để tính góc
                dot = dir_x * vec_x + dir_y * vec_y
                dot_clamped = max(-1.0, min(1.0, dot))
                angle_to_bin = math.degrees(math.acos(dot_clamped))
                
                # Điều kiện nhả: khoảng cách đủ gần VÀ đang hướng vào tâm BIN (góc nhỏ)
                ANGLE_THRESHOLD = 30.0  # Phải hướng vào tâm (góc < 30°)
                if dist_to_bin <= PICKUP_DISTANCE and angle_to_bin <= ANGLE_THRESHOLD:
                    ready_to_drop = True
                    print(f"[STATE] RETURNING -> DROPPING (dist={dist_to_bin:.0f}px, angle={angle_to_bin:.0f}°)")
        
        if ready_to_drop:
            ROBOT_STATE = "DROPPING"
            move_cmd = "STOP"
        else:
            move_cmd = decide_move_to_target(CONTAINER_POS, heading_info, W, H)
    
    elif ROBOT_STATE == "DROPPING":
        # Gửi COMBO_DROP và chờ hoàn thành
        print("[STATE] DROPPING -> WAIT_DROP (sending COMBO_DROP)")
        send_robot_command("COMBO_DROP", force=True)
        _combo_start_time = time.time()
        ROBOT_STATE = "WAIT_DROP"
        move_cmd = "STOP"
    
    elif ROBOT_STATE == "WAIT_DROP":
        # Đợi combo drop hoàn thành theo thời gian thực
        elapsed = time.time() - _combo_start_time
        
        if elapsed >= COMBO_WAIT_TIME:
            print(f"[STATE] WAIT_DROP -> SEEKING_GREEN (waited {elapsed:.1f}s)")
            ROBOT_STATE = "SEEKING_GREEN"
        move_cmd = "STOP"
    
    # Safety: stop if missing markers or distance invalid
    marker_dist_ok = True
    if not (top_red and top_blue):
        move_cmd = "STOP"
        marker_dist_ok = False
    elif marker_base_len:
        if marker_base_len < MARKER_MIN_DIST:
            move_cmd = "STOP"  # Quá gần = nhận diện sai
            marker_dist_ok = False
        elif marker_base_len > MARKER_MAX_DIST:
            move_cmd = "STOP"  # Quá xa camera
            marker_dist_ok = False
    
    # Send command - KHÔNG gửi khi đang đợi combo hoàn thành
    if ROBOT_STATE not in ("WAIT_PICK", "WAIT_DROP"):
        send_robot_command(move_cmd)
    
    # Update detection info
    _detection_info.update({
        "state": ROBOT_STATE,
        "command": move_cmd,
        "robot_enabled": ROBOT_ENABLED,
        "markers_ok": orientation == "LEFT_RIGHT_OK",
        "green_detected": green_target is not None,
        "green_count": green_count,  # Số green còn lại ngoài BIN
        "robot_pos": robot_pos,
        "detections": [det for det in detections.values()],
        "marker_dist": round(marker_base_len, 1) if marker_base_len else 0,
        "marker_dist_ok": marker_dist_ok,
        "dist_to_green": round(dist_to_green, 1) if dist_to_green else 0,
        "dist_to_bin": round(dist_to_bin, 1) if dist_to_bin else 0,
        "angle_to_bin": round(angle_to_bin, 1) if angle_to_bin else 0,
        "pickup_distance": PICKUP_DISTANCE,
    })
    
    # Draw status
    status_color = (0, 255, 0) if ROBOT_ENABLED else (0, 0, 255)
    enabled_text = "ON" if ROBOT_ENABLED else "OFF"
    cv2.putText(frame, f"STATE: {ROBOT_STATE} | CMD: {move_cmd} | SEND: {enabled_text}",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Draw color info
    y_offset = 30
    for color_name, rgb in colors_rgb.items():
        hsv_val = colors_hsv.get(color_name, (0, 0, 0))
        text = f"{color_name}: RGB({rgb[0]},{rgb[1]},{rgb[2]}) HSV({hsv_val[0]},{hsv_val[1]},{hsv_val[2]})"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLORS.get(color_name, (255,255,255)), 1)
        y_offset += 20
    
    return frame

def generate_frames():
    global _current_frame, _detection_info
    
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm độ trễ
    
    # Thử bật MJPEG codec nếu camera hỗ trợ (nhanh hơn)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    frame_count = 0
    last_processed_frame = None
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Tính FPS mỗi giây
        now = time.time()
        if now - fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = now
            _detection_info["fps"] = current_fps
        
        # Xử lý mỗi N frame để tăng FPS
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            frame = process_frame(frame)
            # Vẽ FPS lên frame
            cv2.putText(frame, f"FPS: {current_fps}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_processed_frame = frame
            
            with _frame_lock:
                _current_frame = frame.copy()
        else:
            # Dùng lại frame đã xử lý trước đó
            if last_processed_frame is not None:
                frame = last_processed_frame
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    return render_template('index_opencv.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify(_detection_info)

@app.route('/api/toggle_enabled', methods=['POST'])
def toggle_enabled():
    global ROBOT_ENABLED
    ROBOT_ENABLED = not ROBOT_ENABLED
    if not ROBOT_ENABLED:
        send_robot_command("STOP", force=True)
    return jsonify({"enabled": ROBOT_ENABLED})

@app.route('/api/reset_state', methods=['POST'])
def reset_state():
    global ROBOT_STATE
    ROBOT_STATE = "SEEKING_GREEN"
    return jsonify({"state": ROBOT_STATE})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    send_robot_command("STOP", force=True)
    return jsonify({"success": True})

@app.route('/api/manual_command', methods=['POST'])
def manual_command():
    """Gửi lệnh thủ công - hoạt động khi Auto OFF."""
    data = request.json
    cmd = data.get('command', 'STOP')
    print(f"[API] Manual command received: {cmd}")
    success = send_manual_command(cmd)
    return jsonify({"success": success, "command": cmd})

@app.route('/api/set_container', methods=['POST'])
def set_container():
    global CONTAINER_POS
    data = request.json
    x = data.get('x', CONTAINER_POS[0])
    y = data.get('y', CONTAINER_POS[1])
    CONTAINER_POS = (int(x), int(y))
    return jsonify({"container_pos": CONTAINER_POS})

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    global ROBOT_IP, ROBOT_PW, CONTAINER_POS, CONTAINER_MARGIN, MARKER_MIN_DIST, MARKER_MAX_DIST, PICKUP_DISTANCE, PICKUP_RANGE
    
    if request.method == 'POST':
        data = request.json
        if 'robot_ip' in data:
            ROBOT_IP = data['robot_ip']
        if 'robot_pw' in data:
            ROBOT_PW = data['robot_pw']
        if 'container_x' in data and 'container_y' in data:
            CONTAINER_POS = (int(data['container_x']), int(data['container_y']))
        if 'container_margin' in data:
            CONTAINER_MARGIN = int(data['container_margin'])
        if 'marker_min_dist' in data:
            MARKER_MIN_DIST = int(data['marker_min_dist'])
        if 'marker_max_dist' in data:
            MARKER_MAX_DIST = int(data['marker_max_dist'])
        if 'pickup_distance' in data:
            PICKUP_DISTANCE = int(data['pickup_distance'])
        if 'pickup_range' in data:
            PICKUP_RANGE = int(data['pickup_range'])
        
        # Lưu config vào file
        save_all_config()
    
    return jsonify({
        "robot_ip": ROBOT_IP,
        "robot_pw": ROBOT_PW,
        "container_pos": CONTAINER_POS,
        "container_margin": CONTAINER_MARGIN,
        "marker_min_dist": MARKER_MIN_DIST,
        "marker_max_dist": MARKER_MAX_DIST,
        "pickup_distance": PICKUP_DISTANCE,
        "pickup_range": PICKUP_RANGE,
    })

# ================== COLOR CONFIG API ==================
@app.route('/api/colors', methods=['GET'])
def get_colors():
    """Lấy cấu hình màu hiện tại."""
    return jsonify({
        "colors_rgb": {k: list(v) for k, v in colors_rgb.items()},
        "colors_hsv": {k: list(v) for k, v in colors_hsv.items()},
        "hsv_tolerance": hsv_tolerance,
    })

@app.route('/api/colors/<color_name>', methods=['POST'])
def set_color(color_name):
    """Cập nhật một màu."""
    if color_name not in ["red", "green", "blue"]:
        return jsonify({"error": "Invalid color name"}), 400
    
    data = request.json
    r = int(data.get('r', colors_rgb[color_name][0]))
    g = int(data.get('g', colors_rgb[color_name][1]))
    b = int(data.get('b', colors_rgb[color_name][2]))
    
    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    update_color(color_name, r, g, b)
    
    return jsonify({
        "color": color_name,
        "rgb": [r, g, b],
        "hsv": list(colors_hsv[color_name]),
    })

@app.route('/api/tolerance', methods=['POST'])
def set_tolerance():
    """Cập nhật dung sai HSV."""
    data = request.json
    h_tol = int(data.get('h_tol', hsv_tolerance['h_tol']))
    s_tol = int(data.get('s_tol', hsv_tolerance['s_tol']))
    v_tol = int(data.get('v_tol', hsv_tolerance['v_tol']))
    
    # Clamp values
    h_tol = max(1, min(90, h_tol))
    s_tol = max(1, min(127, s_tol))
    v_tol = max(1, min(127, v_tol))
    
    update_tolerance(h_tol, s_tol, v_tol)
    
    return jsonify(hsv_tolerance)

@app.route('/api/pick_color', methods=['POST'])
def pick_color_from_frame():
    """Lấy màu từ một vùng hình chữ nhật trong frame hiện tại."""
    global _current_frame
    
    data = request.json
    # Hỗ trợ cả điểm đơn (x, y) và vùng chữ nhật (x1, y1, x2, y2)
    x1 = int(data.get('x1', data.get('x', 0)))
    y1 = int(data.get('y1', data.get('y', 0)))
    x2 = int(data.get('x2', x1 + 5))  # Mặc định 5x5 nếu chỉ có điểm
    y2 = int(data.get('y2', y1 + 5))
    color_name = data.get('color_name', None)
    
    with _frame_lock:
        if _current_frame is None:
            return jsonify({"error": "No frame available"}), 400
        
        frame = _current_frame.copy()
    
    H, W = frame.shape[:2]
    
    # Đảm bảo x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Clamp vào frame
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    
    # Đảm bảo vùng có kích thước tối thiểu
    if x2 - x1 < 1:
        x2 = x1 + 1
    if y2 - y1 < 1:
        y2 = y1 + 1
    
    # Lấy màu trung bình trong vùng
    region = frame[y1:y2, x1:x2]
    mean_bgr = np.mean(region, axis=(0, 1)).astype(int)
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    
    # Tính độ lệch chuẩn để biết màu có đồng nhất không
    std_bgr = np.std(region, axis=(0, 1))
    color_variance = float(np.mean(std_bgr))
    
    result = {
        "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "area_pixels": (x2 - x1) * (y2 - y1),
        "rgb": [r, g, b],
        "hsv": list(rgb_to_hsv_opencv(r, g, b)),
        "color_variance": round(color_variance, 2),  # Độ lệch màu (thấp = đồng nhất)
    }
    
    if color_name and color_name in colors_rgb:
        update_color(color_name, r, g, b)
        result["updated_color"] = color_name
    
    return jsonify(result)

@app.route('/api/reset_colors', methods=['POST'])
def reset_colors():
    """Reset về màu mặc định."""
    global colors_rgb, colors_hsv, hsv_tolerance
    
    colors_rgb = DEFAULT_COLORS_RGB.copy()
    colors_hsv = {name: rgb_to_hsv_opencv(*rgb) for name, rgb in colors_rgb.items()}
    hsv_tolerance = DEFAULT_HSV_TOLERANCE.copy()
    save_color_config()
    
    return jsonify({
        "colors_rgb": {k: list(v) for k, v in colors_rgb.items()},
        "hsv_tolerance": hsv_tolerance,
    })

if __name__ == '__main__':
    load_color_config()
    print("[INFO] Starting OpenCV Web App at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
