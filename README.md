# 🤖 Robot IoT - Color Block Sorting

Dự án Robot IoT tự động nhận diện và phân loại các khối màu sử dụng Computer Vision (YOLO / OpenCV) và điều khiển robot qua WiFi.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![ESP8266](https://img.shields.io/badge/ESP8266-NodeMCU-orange.svg)

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Tính năng](#-tính-năng)
- [Linh kiện phần cứng](#-linh-kiện-phần-cứng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [So sánh YOLO vs OpenCV](#-so-sánh-yolo-vs-opencv)
- [API Reference](#-api-reference)

---

## 🎯 Tổng quan

Robot có khả năng:
1. **Tự nhận diện vị trí** thông qua 2 marker màu (đỏ + xanh dương) gắn trên thân
2. **Phát hiện khối màu xanh lá (green)** cần thu gom
3. **Tự động di chuyển** đến vị trí khối màu, gắp và đưa về **BIN** (thùng chứa)
4. **Điều khiển từ xa** qua giao diện web hoặc phím tắt

### Workflow

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────────┐    ┌──────────────┐
│ SEEKING_TARGET  │───▶│   PICKING    │───▶│     RETURNING       │───▶│   DROPPING   │
│ (Tìm green/     │    │ (Gắp vật)    │    │ (Về BIN của màu     │    │ (Nhả vào     │
│  yellow gần    │    │              │    │  đang gắp)          │    │  đúng BIN)   │
│  nhất)         │    │              │    │                     │    │              │
└─────────────────┘    └──────────────┘    └─────────────────────┘    └──────────────┘
       ▲                                                                      │
       └──────────────────────────────────────────────────────────────────────┘
```

### Dual BIN System

Robot hỗ trợ 2 thùng chứa (BIN) riêng biệt cho 2 màu:
- **GREEN BIN**: Chứa các khối màu xanh lá
- **YELLOW BIN**: Chứa các khối màu vàng

Robot tự động:
1. Phát hiện khối màu gần nhất (green hoặc yellow)
2. Ghi nhớ màu đang gắp
3. Đưa về đúng BIN của màu đó

---

## ✨ Tính năng

### 🔍 Nhận diện màu sắc
- **YOLO (YOLOv8)**: Nhận diện object bằng deep learning
- **OpenCV HSV**: Nhận diện màu theo ngưỡng HSV, điều chỉnh realtime
- **Dual Target**: Hỗ trợ gắp cả khối GREEN và YELLOW

### 📦 Dual BIN System
- **GREEN BIN**: Thùng chứa khối xanh lá
- **YELLOW BIN**: Thùng chứa khối vàng
- Vẽ vùng BIN trực tiếp trên video
- Robot tự động đưa khối về đúng BIN của màu đó

### 🎮 Điều khiển Robot
- Điều khiển tự động (Auto mode)
- Điều khiển thủ công qua Web UI
- Emergency Stop
- Combo actions: Pick & Drop
- Backup after drop (lùi lại sau khi nhả)

### 📊 Giao diện Web
- Video stream realtime từ camera
- Cấu hình màu sắc bằng color picker
- Vẽ 2 vùng BIN riêng biệt trên video
- Đo khoảng cách pickup/drop
- Hiển thị màu đang gắp (carrying)

### 🔧 Cấu hình linh hoạt
- Lưu/load config từ file JSON
- Điều chỉnh ngưỡng HSV tolerance
- Cấu hình IP robot, khoảng cách pickup/drop
- Lưu vị trí 2 BIN

---

## 🔩 Linh kiện phần cứng

| Linh kiện | Mô tả | Số lượng |
|-----------|-------|----------|
| **ESP8266 NodeMCU** | Vi điều khiển WiFi | 1 |
| **L298N** | Module điều khiển động cơ | 1 |
| **DC Motor** | Động cơ bánh xe | 2 |
| **Servo SG90** | Servo điều khiển cánh tay/kẹp | 2 |
| **Webcam/IP Camera** | Camera nhận diện (1280x720) | 1 |
| **Pin Li-ion 18650** | Nguồn điện | 2-3 |
| **Khung robot** | Chassis 2WD | 1 |
| **Marker màu** | Đỏ (phải) + Xanh dương (trái) | 2 |

### Sơ đồ kết nối

```
ESP8266 NodeMCU
├── D1, D2 ──────▶ L298N (Motor A)
├── D3, D4 ──────▶ L298N (Motor B)
├── D5 ──────────▶ Servo 1 (Bow - cúi/ngẩng)
├── D6 ──────────▶ Servo 2 (Grip - kẹp/nhả)
└── VIN, GND ────▶ Power
```

---

## 📁 Cấu trúc thư mục

```
robot/
├── 📄 README.md                 # File này
├── 📄 requirements.txt          # Python dependencies
├── 📄 yolo_robot.py             # Chạy với YOLO detection
├── 📄 web_app_opencv.py         # Web app với OpenCV detection
│
├── 📁 arduino/
│   └── robot.ino                # Code ESP8266 điều khiển robot
│
├── 📁 models/
│   ├── best.pt                  # YOLO model đã train
│   ├── best2.pt
│   └── best3.pt
│
├── 📁 config/
│   └── color_config.json        # Cấu hình màu sắc + robot
│
├── 📁 notebooks/
│   ├── trainYolo.ipynb          # Notebook train YOLO model
│   └── trainV2.ipynb            # Notebook train version 2
│
├── 📁 web/
│   ├── templates/
│   │   └── index_opencv.html    # Giao diện web
│   └── static/
│       ├── css/opencv_style.css
│       └── js/opencv_app.js
│
└── 📁 data/
    ├── dataset/                 # Dataset cho training
    └── demo.mp4                 # Video demo
```

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd robot
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# hoặc
source .venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Upload code Arduino

1. Mở `arduino/robot.ino` trong Arduino IDE
2. Cài đặt board ESP8266 (nếu chưa có)
3. Cấu hình WiFi SSID/Password trong code
4. Upload lên ESP8266

### 5. Cấu hình

Sửa file `config/color_config.json` hoặc cấu hình qua Web UI:

```json
{
  "robot_ip": "192.168.1.23",
  "robot_pw": "5613",
  "pickup_distance": 80,
  "drop_distance": 100
}
```

---

## 💻 Sử dụng

### Chạy với OpenCV (khuyến nghị)

```bash
python web_app_opencv.py
```

Mở trình duyệt: `http://localhost:5001`

### Chạy với YOLO

```bash
python yolo_robot.py
```

**Phím tắt:**
- `Q` - Thoát
- `S` - Emergency Stop
- `R` - Reset state về SEEKING_GREEN
- `T` - Toggle Auto/Manual

---

## ⚖️ So sánh YOLO vs OpenCV

| Tiêu chí | YOLO | OpenCV HSV |
|----------|------|------------|
| **Độ chính xác** | ⭐⭐⭐⭐⭐ Cao | ⭐⭐⭐ Trung bình |
| **Tốc độ** | ⭐⭐⭐ Chậm hơn | ⭐⭐⭐⭐⭐ Nhanh |
| **Yêu cầu phần cứng** | GPU/CPU mạnh | CPU thường |
| **Điều kiện ánh sáng** | ⭐⭐⭐⭐ Ổn định | ⭐⭐ Nhạy cảm |
| **Tùy chỉnh** | Cần train lại | ⭐⭐⭐⭐⭐ Realtime |
| **Setup** | Phức tạp | Đơn giản |

### Khi nào dùng YOLO?
- ✅ Cần độ chính xác cao
- ✅ Môi trường ánh sáng thay đổi
- ✅ Có GPU hoặc không quan tâm tốc độ
- ✅ Object phức tạp, không chỉ là màu sắc

### Khi nào dùng OpenCV?
- ✅ Cần tốc độ xử lý nhanh (realtime)
- ✅ Phần cứng hạn chế (Raspberry Pi, laptop cũ)
- ✅ Môi trường ánh sáng ổn định
- ✅ Cần điều chỉnh nhanh theo điều kiện thực tế
- ✅ Object đơn giản (khối màu đồng nhất)

---

## 📡 API Reference

### Robot Endpoints (ESP8266)

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/forward` | GET | Tiến thẳng |
| `/backward` | GET | Lùi |
| `/left` | GET | Rẽ trái |
| `/right` | GET | Rẽ phải |
| `/spin-left` | GET | Xoay tại chỗ trái |
| `/spin-right` | GET | Xoay tại chỗ phải |
| `/stop` | GET | Dừng |
| `/combo-pick` | GET | Combo: Cúi + Kẹp + Ngẩng |
| `/combo-drop` | GET | Combo: Cúi + Nhả + Ngẩng |

### Web App API

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/video_feed` | GET | MJPEG stream |
| `/api/status` | GET | Trạng thái hiện tại |
| `/api/toggle_enabled` | POST | Bật/tắt Auto |
| `/api/update_bin` | POST | Cập nhật vị trí BIN |
| `/api/colors` | GET | Lấy cấu hình màu |
| `/api/pick_color` | POST | Chọn màu từ frame |

---

## 🎥 Demo

[Video Demo](data/demo.mp4)

---

## 📝 License

MIT License - Tự do sử dụng và chỉnh sửa cho mục đích học tập.

---

## 👥 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo Issue hoặc Pull Request.

---

**Made with ❤️ for IoT & Computer Vision learning**
