from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import sys
import math
import requests
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")

# ================== ROBOT CONFIG ==================
ROBOT_IP = "192.168.1.14"  # Thay bằng IP thực của ESP8266
ROBOT_PW = "5613"
ROBOT_ENABLED = True        # Set False để tắt gửi lệnh (chỉ xem)
COMMAND_INTERVAL = 0.3      # Thời gian tối thiểu giữa 2 lệnh (giây)

# ================== CONTAINER/BIN POSITION ==================
# Tọa độ pixel của thùng chứa trên màn hình camera (x, y)
# Ví dụ: góc trái trên = (100, 100), góc phải dưới = (1180, 620)
CONTAINER_POS = (100, 450)  # Tọa độ tâm thùng chứa (bên phải màn hình)
CONTAINER_MARGIN = 120      # Bán kính vùng coi như "đã tới thùng" và loại trừ green

# ================== PICKUP DISTANCE CONFIG ==================
# Điều kiện để robot dừng lại và gắp green
PICKUP_STOP_DIST = 60       # Khoảng cách pixel từ robot đến green để STOP (trong decide_move_to_target)
PICKUP_TRIANGLE_DIFF = 0.25 # Độ lệch cho phép của tam giác cân (0.25 = 25%), nhỏ hơn = yêu cầu chính xác hơn

# ================== ROBOT STATE MACHINE ==================
# States: SEEKING_GREEN -> PICKING -> RETURNING -> DROPPING -> SEEKING_GREEN
ROBOT_STATE = "SEEKING_GREEN"

# ================== YOLO CONFIG ==================
CONF_THRES = 0.6
RED_CONF_THRES = 0.85
GREEN_CONF_THRES = 0.75
BLUE_CONF_THRES = 0.85

IMGSZ = 960
CAM_INDEX = 1
DEFAULT_BACKEND = "auto"
CLASS_NAMES = ["red_square", "green_square", "blue_square"]
ORIENT_MARGIN = 25

TILES = (1, 1)
OVERLAP = 0.0

# Maximum allowed distance (pixels) between blue and red markers before stopping.
MARKER_MAX_DIST = 600.0

# ================== ROBOT HTTP CONTROL ==================
_last_sent_cmd = None  # Lệnh cuối cùng đã gửi thành công
_last_sent_time = 0    # Thời điểm gửi lệnh cuối
_last_error_time = 0   # Thời điểm xảy ra lỗi gần nhất

# Timeout settings
COMBO_COMMANDS = {"COMBO_PICK", "COMBO_DROP"}
TIMEOUT_NORMAL = 2.0       # 2 giây cho lệnh thông thường
TIMEOUT_COMBO = 10.0       # 10 giây cho combo commands
COOLDOWN_AFTER_ERROR = 5.0 # Tạm dừng 5 giây sau khi timeout

# HTTP Session với Keep-Alive
_http_session = None

def _get_http_session():
    """Lấy hoặc tạo HTTP session với keep-alive."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
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

def send_robot_command(cmd, force=False):
    """
    Gửi lệnh HTTP đến robot ESP8266.
    cmd: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP, SEARCH_TARGET
    force: True để bỏ qua kiểm tra trùng lệnh (dùng cho emergency stop/heartbeat)
    """
    global _last_sent_cmd, _last_sent_time, _last_error_time
    
    # Cho phép gửi lệnh khi force=True ngay cả khi DISABLED
    if not ROBOT_ENABLED and not force:
        return
    
    now = time.time()
    
    # Kiểm tra cooldown sau lỗi (trừ khi force)
    if not force and (now - _last_error_time < COOLDOWN_AFTER_ERROR):
        return
    
    # Không gửi nếu lệnh giống lệnh trước (trừ khi force)
    if not force and cmd == _last_sent_cmd:
        return
    
    endpoint_map = {
        "MOVE_FORWARD": "/forward",
        "TURN_LEFT": "/left",
        "TURN_RIGHT": "/right",
        "STOP": "/stop",
        "SEARCH_TARGET": "/stop",  # Khi tìm kiếm thì dừng lại
        "SPIN_LEFT": "/spin-left",
        "SPIN_RIGHT": "/spin-right",
        "COMBO_PICK": "/combo-pick",
        "COMBO_DROP": "/combo-drop",
    }
    
    endpoint = endpoint_map.get(cmd)
    if not endpoint:
        return
    
    url = f"http://{ROBOT_IP}{endpoint}?pw={ROBOT_PW}"
    try:
        # Dùng session với keep-alive và timeout phù hợp
        session = _get_http_session()
        timeout = TIMEOUT_COMBO if cmd in COMBO_COMMANDS else TIMEOUT_NORMAL
        resp = session.get(url, timeout=timeout)
        _last_sent_cmd = cmd
        _last_sent_time = now
        print(f"[ROBOT] {cmd} -> {resp.text}")
    except requests.exceptions.RequestException as e:
        _last_error_time = time.time()  # Đánh dấu thời điểm lỗi
        print(f"[ROBOT] Error: {e}")
        print(f"[ROBOT] Cooldown {COOLDOWN_AFTER_ERROR}s - không gửi lệnh mới")
        _reset_http_session()  # Reset session để tạo connection mới

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect coloured squares with YOLO and control robot heading to green square."
    )
    parser.add_argument("--cam-index", type=int, default=CAM_INDEX,
                        help="Index of the camera to open (e.g. 0, 1, 2).")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf"],
                        default=DEFAULT_BACKEND,
                        help="Force a specific OpenCV backend.")
    parser.add_argument("--list-cams", action="store_true",
                        help="Probe the first few indexes and exit.")
    parser.add_argument("--probe-count", type=int, default=5,
                        help="How many indexes to probe when using --list-cams.")
    return parser.parse_args()


def backend_flag(name):
    mapping = {
        "auto": 0,
        "dshow": getattr(cv2, "CAP_DSHOW", 0),
        "msmf": getattr(cv2, "CAP_MSMF", 0),
    }
    return mapping.get(name, 0)


def open_capture(index, backend_name):
    flag = backend_flag(backend_name)
    cap = cv2.VideoCapture(index, flag) if flag else cv2.VideoCapture(index)
    return cap


def probe_cameras(max_index, backend_name):
    usable = []
    for idx in range(max_index):
        cap = open_capture(idx, backend_name)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        status = "OK" if ok else "Opened-no-frame"
        usable.append((idx, status))
        cap.release()
    return usable


def pick_top_by_conf(items):
    return max(items, key=lambda d: d["conf"]) if items else None


def is_in_container_zone(obj):
    """Kiểm tra object có nằm trong vùng container không."""
    if not obj:
        return False
    cx = obj["center"]["cx"]
    cy = obj["center"]["cy"]
    dist = math.hypot(cx - CONTAINER_POS[0], cy - CONTAINER_POS[1])
    return dist <= CONTAINER_MARGIN


def filter_green_outside_container(green_list):
    """Lọc bỏ các green nằm trong vùng container."""
    return [g for g in green_list if not is_in_container_zone(g)]


def decide_move_to_target(target_pos, heading_info, frame_w, frame_h):
    """
    Di chuyển robot về phía target_pos (x, y) trên màn hình.
    Dùng cho cả green object và container position.
    
    Logic góc quay:
    - Góc ≤ 10° → MOVE_FORWARD (tiến thẳng)
    - Góc 10-45° → TURN_LEFT/RIGHT (rẽ nhẹ, vừa tiến vừa quay)
    - Góc > 45° → SPIN_LEFT/RIGHT (xoay tại chỗ nhanh)
    """
    if target_pos is None:
        return "SEARCH_TARGET"
    
    tx, ty = target_pos
    
    if heading_info is not None:
        # Có heading từ blue/red markers
        origin, direction = heading_info
        hx, hy = origin
        dir_x, dir_y = direction
        
        # Vector từ robot đến target
        vec_x = tx - hx
        vec_y = ty - hy
        vec_len = math.hypot(vec_x, vec_y)
        
        if vec_len < PICKUP_STOP_DIST:  # Đủ gần target
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
        
        # Tính góc (độ) - clamp dot để tránh lỗi acos
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
        # Fallback: dùng center màn hình
        center_x = frame_w / 2
        offset_x = tx - center_x
        margin = frame_w * 0.1
        
        if abs(offset_x) > margin:
            return "TURN_LEFT" if offset_x < 0 else "TURN_RIGHT"
        
        return "MOVE_FORWARD"


def determine_orientation(left_obj, right_obj, margin=ORIENT_MARGIN):
    """
    Determine orientation from two markers: left_obj (blue) and right_obj (red).
    Returns: status, left_center, right_center
    """
    if not left_obj or not right_obj:
        return "MISSING_MARKER", None, None
    left_center = (left_obj["center"]["cx"], left_obj["center"]["cy"])
    right_center = (right_obj["center"]["cx"], right_obj["center"]["cy"])
    dx = right_center[0] - left_center[0]
    if dx > margin:
        status = "LEFT_RIGHT_OK"
    elif dx < -margin:
        status = "LEFT_RIGHT_FLIPPED"
    else:
        status = "AMBIGUOUS"
    return status, left_center, right_center


def tile_coords(W, H, nx, ny, overlap):
    tiles = []
    step_x = int(W / nx)
    step_y = int(H / ny)
    ox = int(step_x * overlap)
    oy = int(step_y * overlap)
    for i in range(nx):
        for j in range(ny):
            x1 = max(i * step_x - ox, 0)
            y1 = max(j * step_y - oy, 0)
            x2 = min((i + 1) * step_x + ox, W)
            y2 = min((j + 1) * step_y + oy, H)
            tiles.append((x1, y1, x2, y2))
    return tiles


# ================== GEOMETRY HELPER: BALANCED TRIANGLE ==================

def check_balanced_triangle(red_obj, blue_obj, green_obj, max_rel_diff=0.25):
    """
    Check if distances (R-B), (R-G), (B-G) are approximately equal.
    max_rel_diff: allowed relative difference (e.g. 0.25 = 25%).
    Return True when the triangle is close to equilateral/isosceles.
    """
    if not (red_obj and blue_obj and green_obj):
        return False

    def dist(a, b):
        ax, ay = a["center"]["cx"], a["center"]["cy"]
        bx, by = b["center"]["cx"], b["center"]["cy"]
        return math.hypot(ax - bx, ay - by)

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


def class_conf_thresholds():
    """Build per-class confidence thresholds from fixed constants."""
    return {
        "red_square": RED_CONF_THRES,
        "green_square": GREEN_CONF_THRES,
        "blue_square": BLUE_CONF_THRES,
    }


# ================== CONTROL LOGIC: ALWAYS MOVE TOWARD GREEN ==================

def decide_move_fallback_center(green_obj, frame_w, frame_h, balanced_triangle=False):
    """
    Simple fallback when we don't have heading markers:
      - If balanced triangle already good -> STOP.
      - Else align green to image center using TURN_LEFT / TURN_RIGHT.
      - MOVE_FORWARD when centered but still far.
      - STOP when close enough.
    """
    if not green_obj:
        return "SEARCH_TARGET"

    if balanced_triangle:
        return "STOP"

    cx = green_obj["center"]["cx"]
    w = green_obj["bbox"]["w"]
    h = green_obj["bbox"]["h"]
    area = w * h

    center_x = frame_w / 2
    center_margin = frame_w * 0.1     # 10% width
    close_area = (frame_w * frame_h) * 0.05  # 5% of frame area

    offset_x = cx - center_x

    if abs(offset_x) > center_margin:
        if offset_x < 0:
            return "TURN_LEFT"
        else:
            return "TURN_RIGHT"

    if area < close_area:
        return "MOVE_FORWARD"
    else:
        return "STOP"


def decide_move_with_heading(green_obj, heading_origin, heading_dir, frame_w, frame_h, balanced_triangle=False):
    """
    Use robot heading (from blue/red markers) to turn toward green.

    heading_origin: (hx, hy) - point between blue/red markers.
    heading_dir:    (dx, dy) - normalized direction vector (robot forward).
    """
    if not green_obj:
        return "SEARCH_TARGET"

    # If triangle already balanced, we are in good pose -> STOP
    if balanced_triangle:
        return "STOP"

    gx = green_obj["center"]["cx"]
    gy = green_obj["center"]["cy"]

    hx, hy = heading_origin
    dir_x, dir_y = heading_dir

    # Vector from robot to green
    vec_x = gx - hx
    vec_y = gy - hy
    vec_len = math.hypot(vec_x, vec_y)
    if vec_len < 1e-3:
        # We are basically at the green center
        return "STOP"
    vec_x /= vec_len
    vec_y /= vec_len

    # Dot and cross product between heading and target direction
    dot = dir_x * vec_x + dir_y * vec_y         # cos(theta)
    cross = dir_x * vec_y - dir_y * vec_x       # sign -> left/right

    # Angle threshold in degrees
    angle_margin_deg = 8.0                      # when angle < 8°, consider aligned
    cos_margin = math.cos(math.radians(angle_margin_deg))

    # Area-based distance check (still keep for "near/far" tuning)
    w = green_obj["bbox"]["w"]
    h = green_obj["bbox"]["h"]
    area = w * h
    close_area = (frame_w * frame_h) * 0.05     # adjust this to tune stop distance

    # 1) If target is mostly behind robot (dot < 0), rotate to find it
    if dot < 0.0:
        # Choose a consistent direction; you can flip if needed
        return "TURN_LEFT"

    # 2) If not aligned enough → turn toward target
    if dot < cos_margin:
        # NOTE: we FLIP here because your robot is turning reversed now
        # cross > 0 => TURN_RIGHT, cross < 0 => TURN_LEFT
        if cross > 0:
            return "TURN_RIGHT"
        else:
            return "TURN_LEFT"

    # 3) Heading is aligned with green: move or stop
    if area < close_area:
        return "MOVE_FORWARD"
    else:
        return "STOP"


def decide_move_to_green(green_obj, heading_info, frame_w, frame_h, balanced_triangle=False):
    """
    Wrapper:
      - If we have heading_info (blue+red markers) → use heading logic.
      - Otherwise, fallback to simple center-based logic.
      - If balanced_triangle is True -> STOP in both cases.
    """
    if heading_info is not None:
        origin, direction = heading_info
        return decide_move_with_heading(green_obj, origin, direction, frame_w, frame_h, balanced_triangle)
    else:
        return decide_move_fallback_center(green_obj, frame_w, frame_h, balanced_triangle)

# ======================================================================


def main():
    args = parse_args()
    conf_thresholds = class_conf_thresholds()
    infer_conf = min(conf_thresholds.values())  # use min threshold so higher per-class filters still work
    if args.list_cams:
        print(f"Probing first {args.probe_count} camera indexes via backend '{args.backend}'...")
        hits = probe_cameras(args.probe_count, args.backend)
        if not hits:
            print("No cameras responded; start your virtual/USB camera and try again.")
        else:
            for idx, status in hits:
                print(f"Index {idx}: {status}")
        return

    model = YOLO(MODEL_PATH)
    cap = open_capture(args.cam_index, args.backend)
    if not cap.isOpened():
        print(f"Cannot open camera index {args.cam_index} via backend '{args.backend}'.", file=sys.stderr)
        print("Use --list-cams to find the correct index.", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit.")
    last_cmd = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # ==== DETECT WITH TILING ====
        dets_all = {name: [] for name in CLASS_NAMES}
        for (x1, y1, x2, y2) in tile_coords(W, H, TILES[0], TILES[1], OVERLAP):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            r = model.predict(crop, imgsz=IMGSZ, conf=infer_conf,
                              device="cpu", verbose=False)[0]
            if len(r.boxes) == 0:
                continue

            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                if cls >= len(CLASS_NAMES):
                    continue

                cx1, cy1, cx2, cy2 = map(int, b.xyxy[0])
                X1 = x1 + cx1
                Y1 = y1 + cy1
                X2 = x1 + cx2
                Y2 = y1 + cy2

                name = CLASS_NAMES[cls]
                if conf < conf_thresholds.get(name, CONF_THRES):
                    continue
                dets_all[name].append({
                    "bbox": {"x": int(X1), "y": int(Y1), "w": int(X2 - X1), "h": int(Y2 - Y1)},
                    "center": {"cx": int((X1 + X2) // 2), "cy": int((Y1 + Y2) // 2)},
                    "conf": round(conf, 4)
                })

        # ==== NMS PER CLASS ====
        def iou(a, b):
            ax1, ay1 = a["bbox"]["x"], a["bbox"]["y"]
            ax2, ay2 = ax1 + a["bbox"]["w"], ay1 + a["bbox"]["h"]
            bx1, by1 = b["bbox"]["x"], b["bbox"]["y"]
            bx2, by2 = bx1 + b["bbox"]["w"], by1 + b["bbox"]["h"]
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = iw * ih
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter + 1e-6
            return inter / union

        def nms_per_class(items, thr=0.45):
            items = sorted(items, key=lambda d: d["conf"], reverse=True)
            kept = []
            for obj in items:
                if all(iou(obj, k) < thr for k in kept):
                    kept.append(obj)
            return kept

        for k in dets_all:
            dets_all[k] = nms_per_class(dets_all[k], thr=0.45)

        # ==== PICK TOP-1 FOR EACH COLOR ====
        top_red = pick_top_by_conf(dets_all["red_square"])
        top_blue = pick_top_by_conf(dets_all["blue_square"])
        
        # Lọc green: loại bỏ green trong vùng container
        greens_outside = filter_green_outside_container(dets_all["green_square"])
        top_green = pick_top_by_conf(greens_outside)
        
        # Vẫn vẽ tất cả green (kể cả trong container) để debug
        all_greens = dets_all["green_square"]

        # ==== DRAW BOXES ====
        def draw_one(name, obj, color=(0, 255, 0)):
            x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]
            cx, cy = obj["center"]["cx"], obj["center"]["cy"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(frame, f"{name} {obj['conf']:.2f}",
                        (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if top_red:
            draw_one("red_square", top_red, (0, 0, 255))
        if top_blue:
            draw_one("blue_square", top_blue, (255, 200, 0))
        
        # Vẽ tất cả green, đánh dấu khác nhau cho green trong/ngoài container
        for g in all_greens:
            if is_in_container_zone(g):
                draw_one("green(BIN)", g, (0, 100, 0))  # Màu tối = trong bin
            else:
                draw_one("green_square", g, (0, 255, 0))  # Màu sáng = target
        
        # Vẽ vùng container
        cv2.circle(frame, CONTAINER_POS, CONTAINER_MARGIN, (255, 0, 255), 2)
        cv2.putText(frame, "BIN", (CONTAINER_POS[0] - 15, CONTAINER_POS[1] - CONTAINER_MARGIN - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # ==== HEADING FROM BLUE/RED MARKERS ====
        heading_info = None
        marker_base_len = None
        orientation, sky_center, red_center = determine_orientation(top_blue, top_red)
        if sky_center and red_center:
            cv2.line(frame, sky_center, red_center, (0, 255, 255), 3)
            label_y = max(0, min(sky_center[1], red_center[1]) - 12)
            cv2.putText(frame, "BLUE — RED",
                        (min(sky_center[0], red_center[0]), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            mid_x = int((sky_center[0] + red_center[0]) / 2)
            mid_y = int((sky_center[1] + red_center[1]) / 2)
            dx = red_center[0] - sky_center[0]
            dy = red_center[1] - sky_center[1]
            base_len = math.hypot(dx, dy)
            marker_base_len = base_len
            if base_len > 1e-3:
                # Perpendicular vector = robot forward direction
                perp_x = dy / base_len
                perp_y = -dx / base_len
                heading_len = max(60, int(base_len * 0.6))
                head_end = (
                    int(mid_x + perp_x * heading_len),
                    int(mid_y + perp_y * heading_len)
                )
                cv2.arrowedLine(frame, (mid_x, mid_y), head_end, (255, 255, 0), 3, tipLength=0.2)

                # Save heading info for control logic
                heading_info = ((mid_x, mid_y), (perp_x, perp_y))

        if orientation == "LEFT_RIGHT_OK":
            orient_text = "MARKERS OK: blue-left, red-right"
        elif orientation == "LEFT_RIGHT_FLIPPED":
            orient_text = "MARKERS FLIPPED: rotate robot"
        elif orientation == "AMBIGUOUS":
            orient_text = "MARKERS AMBIGUOUS"
        else:
            orient_text = "SEARCHING MARKERS"

        cv2.putText(frame, orient_text, (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ==== STATE MACHINE ====
        global ROBOT_STATE, ROBOT_ENABLED
        move_cmd = "STOP"
        
        # Tính robot position (giữa blue-red markers)
        robot_pos = None
        if heading_info:
            robot_pos = heading_info[0]  # (mid_x, mid_y)
        
        # Kiểm tra robot đã đến container chưa
        robot_at_container = False
        if robot_pos:
            dist_to_bin = math.hypot(robot_pos[0] - CONTAINER_POS[0], robot_pos[1] - CONTAINER_POS[1])
            robot_at_container = dist_to_bin < CONTAINER_MARGIN
        
        if ROBOT_STATE == "SEEKING_GREEN":
            # Tìm và đi tới green
            if top_green:
                green_pos = (top_green["center"]["cx"], top_green["center"]["cy"])
                balanced = check_balanced_triangle(top_red, top_blue, top_green, max_rel_diff=PICKUP_TRIANGLE_DIFF)
                
                if balanced:
                    # Đã tới green -> PICK
                    print("[STATE] SEEKING_GREEN -> PICKING")
                    ROBOT_STATE = "PICKING"
                    move_cmd = "STOP"
                else:
                    move_cmd = decide_move_to_target(green_pos, heading_info, W, H)
            else:
                move_cmd = "SEARCH_TARGET"
        
        elif ROBOT_STATE == "PICKING":
            # Gửi lệnh combo pick
            send_robot_command("COMBO_PICK", force=True)
            print("[STATE] PICKING -> RETURNING")
            ROBOT_STATE = "RETURNING"
            time.sleep(3)  # Chờ combo pick hoàn thành
            move_cmd = "STOP"
        
        elif ROBOT_STATE == "RETURNING":
            # Quay về container
            if robot_at_container:
                print("[STATE] RETURNING -> DROPPING")
                ROBOT_STATE = "DROPPING"
                move_cmd = "STOP"
            else:
                move_cmd = decide_move_to_target(CONTAINER_POS, heading_info, W, H)
        
        elif ROBOT_STATE == "DROPPING":
            # Gửi lệnh combo drop
            send_robot_command("COMBO_DROP", force=True)
            print("[STATE] DROPPING -> SEEKING_GREEN")
            ROBOT_STATE = "SEEKING_GREEN"
            time.sleep(3)  # Chờ combo drop hoàn thành
            move_cmd = "STOP"

        # Safety: stop if missing a marker or markers too far apart
        if not (top_red and top_blue):
            move_cmd = "STOP"
        elif MARKER_MAX_DIST > 0 and marker_base_len is not None and marker_base_len > MARKER_MAX_DIST:
            move_cmd = "STOP"

        # Print command only when it changes to avoid spamming
        if move_cmd != last_cmd:
            print(f"[CMD] {move_cmd}")
            last_cmd = move_cmd
        
        # Gửi lệnh đến robot (có heartbeat tự động mỗi 3s)
        send_robot_command(move_cmd)

        # Debug text: show state and command
        status_color = (0, 255, 0) if ROBOT_ENABLED else (0, 0, 255)
        cv2.putText(frame, f"STATE: {ROBOT_STATE} | CMD: {move_cmd}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Hiển thị hướng dẫn phím tắt và trạng thái
        enabled_text = "ENABLED" if ROBOT_ENABLED else "DISABLED"
        cv2.putText(frame, f"[R]eset | [T]oggle Send: {enabled_text} | [S]top | [Q]uit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("tiny-objects via tiling", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            send_robot_command("STOP", force=True)  # Dừng robot khi thoát
            break
        elif key == ord('s'):
            # Phím 's' để dừng khẩn cấp
            send_robot_command("STOP", force=True)
            print("[MANUAL] Emergency STOP")
        elif key == ord('r'):
            # Phím 'r' để reset về SEEKING_GREEN
            ROBOT_STATE = "SEEKING_GREEN"
            print("[MANUAL] Reset to SEEKING_GREEN")
        elif key == ord('t'):
            # Phím 't' để toggle gửi lệnh
            ROBOT_ENABLED = not ROBOT_ENABLED
            status = "ENABLED" if ROBOT_ENABLED else "DISABLED"
            print(f"[MANUAL] Robot command sending: {status}")
            if not ROBOT_ENABLED:
                # Gửi STOP trước khi tắt
                send_robot_command("STOP", force=True)

    # Dừng robot khi kết thúc
    send_robot_command("STOP", force=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

