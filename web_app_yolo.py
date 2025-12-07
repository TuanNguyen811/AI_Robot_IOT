"""
Robot Control Web Application
- Video stream từ camera với YOLO detection
- Điều khiển robot qua giao diện web
- State machine visualization
"""

from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
import math
import requests
import time
import threading

app = Flask(__name__)

# ================== ROBOT CONFIG ==================
ROBOT_IP = "192.168.1.23"
ROBOT_PW = "5613"
ROBOT_ENABLED = True
HEARTBEAT_INTERVAL = 10.0

# ================== CONTAINER/BIN POSITION ==================
CONTAINER_POS = (200, 360)
CONTAINER_MARGIN = 120

# ================== ROBOT STATE ==================
ROBOT_STATE = "SEEKING_GREEN"

# ================== YOLO CONFIG ==================
MODEL_PATH = "best.pt"
CONF_THRES = 0.6
RED_CONF_THRES = 0.75
GREEN_CONF_THRES = 0.75
BLUE_CONF_THRES = 0.75
IMGSZ = 960
CAM_INDEX = 1
CLASS_NAMES = ["red_square", "green_square", "blue_square"]
ORIENT_MARGIN = 25
MARKER_MAX_DIST = 600.0

# ================== GLOBAL STATE ==================
_last_sent_cmd = None
_last_sent_time = 0
_current_frame = None
_frame_lock = threading.Lock()
_detection_info = {
    "state": "SEEKING_GREEN",
    "command": "STOP",
    "robot_enabled": True,
    "markers_ok": False,
    "green_detected": False,
    "robot_pos": None,
}

# Load model
model = None

def load_model():
    global model
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded!")

# ================== ROBOT HTTP CONTROL ==================
def send_robot_command(cmd, force=False):
    global _last_sent_cmd, _last_sent_time, ROBOT_ENABLED
    
    if not ROBOT_ENABLED:
        return False
    
    now = time.time()
    
    if not force and cmd == _last_sent_cmd:
        if now - _last_sent_time < HEARTBEAT_INTERVAL:
            return True
        print(f"[HEARTBEAT] Resending {cmd}")
    
    endpoint_map = {
        "MOVE_FORWARD": "/forward",
        "TURN_LEFT": "/left",
        "TURN_RIGHT": "/right",
        "STOP": "/stop",
        "SEARCH_TARGET": "/stop",
        "SPIN_LEFT": "/spin-left",
        "SPIN_RIGHT": "/spin-right",
        "COMBO_PICK": "/combo-pick",
        "COMBO_DROP": "/combo-drop",
    }
    
    endpoint = endpoint_map.get(cmd)
    if not endpoint:
        return False
    
    url = f"http://{ROBOT_IP}{endpoint}?pw={ROBOT_PW}"
    try:
        resp = requests.get(url, timeout=1)
        _last_sent_cmd = cmd
        _last_sent_time = now
        print(f"[ROBOT] {cmd} -> {resp.text}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[ROBOT] Error: {e}")
        return False

# ================== HELPER FUNCTIONS ==================
def pick_top_by_conf(items):
    return max(items, key=lambda d: d["conf"]) if items else None

def is_in_container_zone(obj):
    if not obj:
        return False
    cx = obj["center"]["cx"]
    cy = obj["center"]["cy"]
    dist = math.hypot(cx - CONTAINER_POS[0], cy - CONTAINER_POS[1])
    return dist <= CONTAINER_MARGIN

def filter_green_outside_container(green_list):
    return [g for g in green_list if not is_in_container_zone(g)]

def determine_orientation(left_obj, right_obj, margin=ORIENT_MARGIN):
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

def check_balanced_triangle(red_obj, blue_obj, green_obj, max_rel_diff=0.25):
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

def decide_move_to_target(target_pos, heading_info, frame_w, frame_h):
    if target_pos is None:
        return "SEARCH_TARGET"
    tx, ty = target_pos
    if heading_info is not None:
        origin, direction = heading_info
        hx, hy = origin
        dir_x, dir_y = direction
        vec_x = tx - hx
        vec_y = ty - hy
        vec_len = math.hypot(vec_x, vec_y)
        if vec_len < 50:
            return "STOP"
        vec_x /= vec_len
        vec_y /= vec_len
        dot = dir_x * vec_x + dir_y * vec_y
        cross = dir_x * vec_y - dir_y * vec_x
        angle_margin_deg = 10.0
        cos_margin = math.cos(math.radians(angle_margin_deg))
        if dot < 0.0:
            return "TURN_LEFT"
        if dot < cos_margin:
            return "TURN_RIGHT" if cross > 0 else "TURN_LEFT"
        return "MOVE_FORWARD"
    else:
        center_x = frame_w / 2
        offset_x = tx - center_x
        margin = frame_w * 0.1
        if abs(offset_x) > margin:
            return "TURN_LEFT" if offset_x < 0 else "TURN_RIGHT"
        return "MOVE_FORWARD"

# ================== VIDEO PROCESSING ==================
def process_frame(frame):
    global ROBOT_STATE, _detection_info
    
    H, W = frame.shape[:2]
    conf_thresholds = {
        "red_square": RED_CONF_THRES,
        "green_square": GREEN_CONF_THRES,
        "blue_square": BLUE_CONF_THRES,
    }
    infer_conf = min(conf_thresholds.values())
    
    # Detect
    dets_all = {name: [] for name in CLASS_NAMES}
    r = model.predict(frame, imgsz=IMGSZ, conf=infer_conf, device="cpu", verbose=False)[0]
    
    if len(r.boxes) > 0:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            if cls >= len(CLASS_NAMES):
                continue
            cx1, cy1, cx2, cy2 = map(int, b.xyxy[0])
            name = CLASS_NAMES[cls]
            if conf < conf_thresholds.get(name, CONF_THRES):
                continue
            dets_all[name].append({
                "bbox": {"x": int(cx1), "y": int(cy1), "w": int(cx2 - cx1), "h": int(cy2 - cy1)},
                "center": {"cx": int((cx1 + cx2) // 2), "cy": int((cy1 + cy2) // 2)},
                "conf": round(conf, 4)
            })
    
    # Pick top detections
    top_red = pick_top_by_conf(dets_all["red_square"])
    top_blue = pick_top_by_conf(dets_all["blue_square"])
    greens_outside = filter_green_outside_container(dets_all["green_square"])
    top_green = pick_top_by_conf(greens_outside)
    all_greens = dets_all["green_square"]
    
    # Draw detections
    def draw_one(name, obj, color):
        x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]
        cx, cy = obj["center"]["cx"], obj["center"]["cy"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(frame, f"{name} {obj['conf']:.2f}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if top_red:
        draw_one("red", top_red, (0, 0, 255))
    if top_blue:
        draw_one("blue", top_blue, (255, 200, 0))
    for g in all_greens:
        if is_in_container_zone(g):
            draw_one("green(BIN)", g, (0, 100, 0))
        else:
            draw_one("green", g, (0, 255, 0))
    
    # Draw container
    cv2.circle(frame, CONTAINER_POS, CONTAINER_MARGIN, (255, 0, 255), 2)
    cv2.putText(frame, "BIN", (CONTAINER_POS[0] - 20, CONTAINER_POS[1] - CONTAINER_MARGIN - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Heading
    heading_info = None
    marker_base_len = None
    orientation, sky_center, red_center = determine_orientation(top_blue, top_red)
    
    if sky_center and red_center:
        cv2.line(frame, sky_center, red_center, (0, 255, 255), 3)
        mid_x = int((sky_center[0] + red_center[0]) / 2)
        mid_y = int((sky_center[1] + red_center[1]) / 2)
        dx = red_center[0] - sky_center[0]
        dy = red_center[1] - sky_center[1]
        base_len = math.hypot(dx, dy)
        marker_base_len = base_len
        if base_len > 1e-3:
            perp_x = dy / base_len
            perp_y = -dx / base_len
            heading_len = max(60, int(base_len * 0.6))
            head_end = (int(mid_x + perp_x * heading_len), int(mid_y + perp_y * heading_len))
            cv2.arrowedLine(frame, (mid_x, mid_y), head_end, (255, 255, 0), 3, tipLength=0.2)
            heading_info = ((mid_x, mid_y), (perp_x, perp_y))
    
    # State machine
    robot_pos = heading_info[0] if heading_info else None
    robot_at_container = False
    if robot_pos:
        dist_to_bin = math.hypot(robot_pos[0] - CONTAINER_POS[0], robot_pos[1] - CONTAINER_POS[1])
        robot_at_container = dist_to_bin < CONTAINER_MARGIN
    
    move_cmd = "STOP"
    
    if ROBOT_STATE == "SEEKING_GREEN":
        if top_green:
            green_pos = (top_green["center"]["cx"], top_green["center"]["cy"])
            balanced = check_balanced_triangle(top_red, top_blue, top_green, max_rel_diff=0.25)
            if balanced:
                print("[STATE] SEEKING_GREEN -> PICKING")
                ROBOT_STATE = "PICKING"
                move_cmd = "STOP"
            else:
                move_cmd = decide_move_to_target(green_pos, heading_info, W, H)
        else:
            move_cmd = "SEARCH_TARGET"
    
    elif ROBOT_STATE == "PICKING":
        send_robot_command("COMBO_PICK", force=True)
        print("[STATE] PICKING -> RETURNING")
        ROBOT_STATE = "RETURNING"
        time.sleep(3)
        move_cmd = "STOP"
    
    elif ROBOT_STATE == "RETURNING":
        if robot_at_container:
            print("[STATE] RETURNING -> DROPPING")
            ROBOT_STATE = "DROPPING"
            move_cmd = "STOP"
        else:
            move_cmd = decide_move_to_target(CONTAINER_POS, heading_info, W, H)
    
    elif ROBOT_STATE == "DROPPING":
        send_robot_command("COMBO_DROP", force=True)
        print("[STATE] DROPPING -> SEEKING_GREEN")
        ROBOT_STATE = "SEEKING_GREEN"
        time.sleep(3)
        move_cmd = "STOP"
    
    # Safety
    if not (top_red and top_blue):
        move_cmd = "STOP"
    elif MARKER_MAX_DIST > 0 and marker_base_len and marker_base_len > MARKER_MAX_DIST:
        move_cmd = "STOP"
    
    # Send command
    send_robot_command(move_cmd)
    
    # Update detection info
    _detection_info.update({
        "state": ROBOT_STATE,
        "command": move_cmd,
        "robot_enabled": ROBOT_ENABLED,
        "markers_ok": orientation == "LEFT_RIGHT_OK",
        "green_detected": top_green is not None,
        "robot_pos": robot_pos,
    })
    
    # Draw status
    status_color = (0, 255, 0) if ROBOT_ENABLED else (0, 0, 255)
    enabled_text = "ON" if ROBOT_ENABLED else "OFF"
    cv2.putText(frame, f"STATE: {ROBOT_STATE} | CMD: {move_cmd} | SEND: {enabled_text}",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    return frame

def generate_frames():
    global _current_frame
    
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if model is not None:
            frame = process_frame(frame)
        
        with _frame_lock:
            _current_frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# ================== FLASK ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

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
    data = request.json
    cmd = data.get('command', 'STOP')
    success = send_robot_command(cmd, force=True)
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
    global ROBOT_IP, ROBOT_PW, CONTAINER_POS, CONTAINER_MARGIN
    
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
    
    return jsonify({
        "robot_ip": ROBOT_IP,
        "robot_pw": ROBOT_PW,
        "container_pos": CONTAINER_POS,
        "container_margin": CONTAINER_MARGIN,
    })

if __name__ == '__main__':
    load_model()
    print("[INFO] Starting web server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
