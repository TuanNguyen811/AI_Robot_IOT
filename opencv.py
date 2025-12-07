import numpy as np
import cv2
import time
import json
import math

# =============== Tiện ích =================
def centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def draw_center(img, c, color=(255,255,255)):
    cv2.line(img, (c[0]-6, c[1]), (c[0]+6, c[1]), color, 2)
    cv2.line(img, (c[0], c[1]-6), (c[0], c[1]+6), color, 2)

def rgb_to_hsv_opencv(r, g, b):
    """Chuyển RGB (0..255) sang HSV (OpenCV: H 0..179, S/V 0..255)."""
    patch = np.uint8([[[b, g, r]]])  # OpenCV dùng BGR
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0,0,:]
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
        # không wrap
        lower = np.array([H_lo, S_lo, V_lo], dtype=np.uint8)
        upper = np.array([H_hi, S_hi, V_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
    else:
        # wrap quanh 180 → ghép 2 khoảng
        lower1 = np.array([0,    S_lo, V_lo], dtype=np.uint8)
        upper1 = np.array([H_hi, S_hi, V_hi], dtype=np.uint8)
        lower2 = np.array([H_lo, S_lo, V_lo], dtype=np.uint8)
        upper2 = np.array([179,  S_hi, V_hi], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    return mask

# =============== Cấu hình màu thực tế =================
# Bạn đưa 4 màu (RGB):
colors_rgb = {
    "sky_blue":  (113, 176, 244),  # rgb(113,176,244)
    "salmon":    (249, 150, 137),  # rgb(249,150,137)
    "mint":      (143, 185, 164),  # rgb(143,185,164)
    "lavender":  (143, 141, 208),  # rgb(143,141,208)
}

# chuyển sang HSV (OpenCV)
colors_hsv = {name: rgb_to_hsv_opencv(*rgb) for name, rgb in colors_rgb.items()}

# Dung sai HSV (hãy tinh chỉnh ở hiện trường nếu cần)
H_TOL = 10     # ±10 đơn vị Hue (0..179)
S_TOL = 60     # ±60 đơn vị Saturation (0..255)
V_TOL = 60     # ±60 đơn vị Value (0..255)

# màu vẽ bbox cho từng label (BGR)
BOX_COLOR = {
    "sky_blue": (255, 180, 60),   # cam nhạt
    "salmon":   (0, 140, 255),    # cam
    "mint":     (60, 220, 60),    # xanh lá
    "lavender": (200, 100, 255),  # tím
}

kernel = np.ones((5,5), np.uint8)

def process_mask(frame, mask, label, box_color, min_area=250):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        ctr = centroid(c)
        if ctr is None:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if (best is None) or (area > best[0]):
            best = (area, x,y,w,h, ctr)

    result = None
    if best is not None:
        area, x,y,w,h, ctr = best
        cv2.rectangle(frame, (x,y), (x+w, y+h), box_color, 2)
        draw_center(frame, ctr, box_color)
        cv2.putText(frame, f"{label} ({ctr[0]},{ctr[1]})", (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        result = {"label": label,
                  "rgb": {"r": colors_rgb[label][0], "g": colors_rgb[label][1], "b": colors_rgb[label][2]},
                  "center": {"x": int(ctr[0]), "y": int(ctr[1])},
                  "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}}
    return result

# =============== Webcam & loop =================
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_t = time.time()
fps = 0.0
t_last_json = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # (tuỳ chọn) giảm kích thước cho mượt
    # frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = []

    # tạo mask cho từng màu theo tâm HSV + dung sai
    for label, hsv_c in colors_hsv.items():
        raw_mask = inrange_hsv_tolerant(hsv, hsv_c, H_TOL, S_TOL, V_TOL)
        # làm sạch
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        det = process_mask(frame, mask, label, BOX_COLOR[label], min_area=250)
        if det:
            detections.append(det)

    # (tuỳ chọn) ví dụ tính góc nếu bạn quyết định một cặp màu đại diện (vd: lavender=đuôi, sky_blue=đầu)
    # if any(d["label"]=="lavender" for d in detections) and any(d["label"]=="sky_blue" for d in detections):
    #     rear = next(d for d in detections if d["label"]=="lavender")["center"]
    #     front = next(d for d in detections if d["label"]=="sky_blue")["center"]
    #     vx, vy = (front["x"]-rear["x"]), (front["y"]-rear["y"])
    #     angle = math.degrees(math.atan2(vy, vx));  angle = angle if angle>=0 else angle+360
    #     tip = (int(rear["x"] + 60*vx/max(1, math.hypot(vx,vy))),
    #            int(rear["y"] + 60*vy/max(1, math.hypot(vx,vy))))
    #     cv2.arrowedLine(frame, (rear["x"], rear["y"]), tip, (255,255,255), 2)
    #     cv2.putText(frame, f"angle: {angle:.1f} deg", (10, 28),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # FPS
    now = time.time()
    dt = now - prev_t
    if dt > 0:
        fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else (1.0/dt)
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Custom 4-color tracking (Press Q to quit)", frame)

    # (tuỳ chọn) in JSON mỗi 0.5s
    if now - t_last_json > 0.5:
        # print(json.dumps(detections, ensure_ascii=False))
        t_last_json = now

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
