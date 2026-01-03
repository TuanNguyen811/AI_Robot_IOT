#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Servo.h>

// ================== CẤU HÌNH WIFI ==================
const char* ssid     = "MTP C8";            // tên WiFi
const char* password = "Mtp123456789@33";   // mật khẩu WiFi

// MẬT KHẨU ĐIỀU KHIỂN (QUERY): ?pw=5613
const char* CONTROL_PASSWORD = "5613";

ESP8266WebServer server(80);

// ================== SERVO ==================
// Bánh + cúi + kẹp
Servo wheelL;   // D1 = GPIO5  → bánh trái (servo 360°)
Servo wheelR;   // D2 = GPIO4  → bánh phải (servo 360°)
Servo bow;      // D3 = GPIO0  → cúi người (servo góc)
Servo gripper;  // D4 = GPIO2  → tay kẹp (servo góc)

// Tham số bánh (360°)
const int STOP_L = 87;  // trung tính bánh trái
const int STOP_R = 90;  // trung tính bánh phải
//tăng để
int FWD_L = 18;         // lệch trái khi tiến ()
int FWD_R = 14;         // lệch phải khi tiến (phải)
int BACK_L = 14;        // lệch trái khi lùi  ()
int BACK_R = 17;        // lệch phải khi lùi  ()

// Tham số cúi & kẹp
const int BOW_UP = 97;     // đứng thẳng
const int BOW_DOWN = 155;  // cúi xuống
const int GRIP_OPEN = 0;
const int GRIP_CLOSED = 180;

// ================== HÀM TIỆN ÍCH ==================
void sweep(Servo& s, int fromDeg, int toDeg, int step = 1, int stepDelay = 8) {
  if (fromDeg == toDeg) {
    s.write(toDeg);
    delay(stepDelay);
    return;
  }
  int dir = (toDeg > fromDeg) ? 1 : -1;
  for (int p = fromDeg; p != toDeg; p += dir) {
    s.write(p);
    delay(stepDelay);
  }
  s.write(toDeg);
}

// ================== BÁNH XE ==================
void wheels_stop() {
  wheelL.write(STOP_L);
  wheelR.write(STOP_R);
}

void wheels_forward() {
  wheelL.write(STOP_L - FWD_L);
  wheelR.write(STOP_R + FWD_R);
}

void wheels_backward() {
  wheelL.write(STOP_L + BACK_L);
  wheelR.write(STOP_R - BACK_R);
}

void wheels_turnRight() {
  wheelL.write(STOP_L);
  wheelR.write(STOP_R + FWD_R);
}

void wheels_turnLeft() {
  wheelL.write(STOP_L - FWD_L);
  wheelR.write(STOP_R);
}
const float SPIN_SCALE = 0.8;   // 40% tốc độ quay

void wheels_spinRight() {
  wheelL.write(STOP_L + (BACK_L * SPIN_SCALE));
  wheelR.write(STOP_R + (FWD_R * SPIN_SCALE));
}

void wheels_spinLeft() {
  wheelL.write(STOP_L - (FWD_L * SPIN_SCALE));
  wheelR.write(STOP_R - (BACK_R * SPIN_SCALE));
}

// ================== COMBO BOW + GRIP ==================
void comboPick() {
  // Bow down
  sweep(bow, BOW_UP, BOW_DOWN, 1, 10);
  delay(200);

  // Grip close
  sweep(gripper, GRIP_OPEN, GRIP_CLOSED, 3, 8);
  delay(300);

  // Bow up
  sweep(bow, BOW_DOWN, BOW_UP, 1, 10);
}

void comboDrop() {
  // Bow down
  sweep(bow, BOW_UP, BOW_DOWN, 1, 10);
  delay(200);

  // Grip open (release object)
  sweep(gripper, GRIP_CLOSED, GRIP_OPEN, 3, 8);
  delay(300);

  // Bow up
  sweep(bow, BOW_DOWN, BOW_UP, 1, 10);
}

// ================== TRANG WEB ĐIỀU KHIỂN ==================
const char MAIN_page[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Robot Controller</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      height: 100vh;
      margin: 0;
      padding-top: 30px;
    }
    h1 { margin-bottom: 10px; }
    #status {
      margin-bottom: 20px;
      font-size: 14px;
      color: #9f9;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 120px);
      grid-gap: 10px;
      justify-content: center;
      margin-bottom: 20px;
    }
    button {
      padding: 10px;
      font-size: 14px;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      transition: background 0.15s, transform 0.1s;
    }
    .move { background: #333; color: #fff; }
    .move.active { background: #0a7000; color: #fff; }
    .action { background: #444; color: #fff; }
    button:active { transform: scale(0.96); }

    #lock-screen {
      background: #000a;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      backdrop-filter: blur(3px);
    }
    #lock-screen input {
      padding: 10px;
      font-size: 16px;
      width: 160px;
      text-align: center;
      border-radius: 6px;
      border: 1px solid #666;
      margin-bottom: 10px;
    }
    #lock-screen button {
      padding: 10px 20px;
      font-size: 16px;
      border-radius: 6px;
      background: #444;
      color: #fff;
      border: none;
      cursor: pointer;
    }
  </style>
</head>
<body>

  <div id="lock-screen">
    <h2>Nhập mật khẩu điều khiển</h2>
    <input id="pw-input" type="password" placeholder="Password">
    <button onclick="unlock()">Unlock</button>
    <div id="pw-msg" style="margin-top:10px;color:#f88"></div>
  </div>

  <h1>Điều khiển Robot</h1>
  <div id="status">Status: Waiting for unlock...</div>

  <!-- Điều khiển di chuyển -->
  <div class="grid">
    <div></div>
    <button id="btn-forward" class="move" disabled onclick="toggleMove('forward')">Tiến</button>
    <div></div>

    <button id="btn-spin-left" class="move" disabled onclick="toggleMove('spin-left')">Quay trái</button>
    <button id="btn-backward" class="move" disabled onclick="toggleMove('backward')">Lùi</button>
    <button id="btn-spin-right" class="move" disabled onclick="toggleMove('spin-right')">Quay phải</button>

    <button id="btn-left" class="move" disabled onclick="toggleMove('left')">Rẽ trái</button>
    <div></div>
    <button id="btn-right" class="move" disabled onclick="toggleMove('right')">Rẽ phải</button>
  </div>

  <!-- Điều khiển cúi + kẹp + combo -->
  <div class="grid" style="grid-template-columns: repeat(3, 140px);">
    <button class="action" id="btn-bow-down"   disabled onclick="sendCommand('/bow-down')">Cúi xuống</button>
    <button class="action" id="btn-bow-up"     disabled onclick="sendCommand('/bow-up')">Cúi lên</button>
    <button class="action" id="btn-grip-close" disabled onclick="sendCommand('/grip-close')">Đóng kẹp</button>

    <button class="action" id="btn-grip-open"  disabled onclick="sendCommand('/grip-open')">Mở kẹp</button>
    <button class="action" id="btn-combo-pick" disabled onclick="sendCommand('/combo-pick')">Cúi + Kẹp + Nhấc</button>
    <button class="action" id="btn-combo-drop" disabled onclick="sendCommand('/combo-drop')">Cúi + Nhả + Ngẩng</button>
  </div>

  <script>
    let controlPassword = null;   // Mật khẩu sẽ được lưu sau khi unlock
    let currentMove = null;

    function setEnabledAllButtons(enabled) {
      document.querySelectorAll("button").forEach(btn => {
        if (btn.id.startsWith("btn-")) {
          btn.disabled = !enabled;
        }
      });
    }

    function unlock() {
      const pw = document.getElementById('pw-input').value.trim();
      if (pw.length === 0) {
        document.getElementById("pw-msg").innerText = "Password cannot be empty.";
        return;
      }

      // test bằng request nhẹ
      fetch("/stop?pw=" + encodeURIComponent(pw))
        .then(res => {
          if (res.status === 200) {
            // thành công
            controlPassword = pw;
            document.getElementById("lock-screen").style.display = "none";
            setEnabledAllButtons(true);
            setStatus("Unlocked");
          } else {
            document.getElementById("pw-msg").innerText = "Incorrect password!";
          }
        })
        .catch(() => {
          document.getElementById("pw-msg").innerText = "Network error.";
        });
    }

    const moveButtons = {
      'forward':   { id: 'btn-forward',    idle: 'Tiến',       active: 'Đang tiến' },
      'backward':  { id: 'btn-backward',   idle: 'Lùi',        active: 'Đang lùi' },
      'left':      { id: 'btn-left',       idle: 'Rẽ trái',    active: 'Đang rẽ trái' },
      'right':     { id: 'btn-right',      idle: 'Rẽ phải',    active: 'Đang rẽ phải' },
      'spin-left': { id: 'btn-spin-left',  idle: 'Quay trái',  active: 'Đang quay trái' },
      'spin-right':{ id: 'btn-spin-right', idle: 'Quay phải',  active: 'Đang quay phải' }
    };

    function setStatus(text) {
      document.getElementById('status').innerText = 'Status: ' + text;
    }

    function updateMoveButtons() {
      for (const key in moveButtons) {
        const cfg = moveButtons[key];
        const btn = document.getElementById(cfg.id);
        if (!btn) continue;

        if (currentMove === key) {
          btn.classList.add('active');
          btn.innerText = cfg.active;
        } else {
          btn.classList.remove('active');
          btn.innerText = cfg.idle;
        }
      }
    }

    function toggleMove(move) {
      if (!controlPassword) return;

      let path;
      if (currentMove === move) {
        currentMove = null;
        path = '/stop';
      } else {
        currentMove = move;
        switch (move) {
          case 'forward':    path = '/forward';    break;
          case 'backward':   path = '/backward';   break;
          case 'left':       path = '/left';       break;
          case 'right':      path = '/right';      break;
          case 'spin-left':  path = '/spin-left';  break;
          case 'spin-right': path = '/spin-right'; break;
          default:           currentMove = null; path = '/stop';
        }
      }

      updateMoveButtons();
      const url = path + '?pw=' + encodeURIComponent(controlPassword);
      setStatus('Sending ' + path + '...');
      fetch(url)
        .then(res => res.text())
        .then(t => setStatus(t))
        .catch(err => setStatus('Error: ' + err));
    }

    function sendCommand(path) {
      if (!controlPassword) return;
      const url = path + '?pw=' + encodeURIComponent(controlPassword);
      setStatus('Sending ' + path + '...');
      fetch(url)
        .then(res => res.text())
        .then(t => setStatus(t))
        .catch(err => setStatus('Error: ' + err));
    }

    // disable nút khi chưa unlock
    setEnabledAllButtons(false);
  </script>

</body>
</html>
)rawliteral";

// ================== AUTH CHECK ==================
bool checkAuth() {
  // yêu cầu query ?pw=5613
  if (!server.hasArg("pw")) {
    server.send(403, "text/plain", "Missing password");
    return false;
  }
  if (server.arg("pw") != CONTROL_PASSWORD) {
    server.send(403, "text/plain", "Invalid password");
    return false;
  }
  return true;
}

// ================== HANDLER WEB ==================
void handleRoot() {
  server.send_P(200, "text/html", MAIN_page);
}

void handleSimple(const String& msg, void (*action)()) {
  if (!checkAuth()) return;  // CHẶN LỆNH NẾU SAI MẬT KHẨU
  if (action) action();
  server.send(200, "text/plain", msg);
}

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  delay(200);

  // Servo
  wheelL.attach(D1);
  wheelR.attach(D2);
  bow.attach(D3, 500, 2400);
  gripper.attach(D4, 500, 2400);

  wheels_stop();
  bow.write(BOW_UP);
  gripper.write(GRIP_OPEN);
  delay(500);

  // WiFi
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Web server routes
  server.on("/", handleRoot);

  // Di chuyển
  server.on("/forward", []() {
    handleSimple("Forward", wheels_forward);
  });
  server.on("/backward", []() {
    handleSimple("Backward", wheels_backward);
  });
  server.on("/left", []() {
    handleSimple("Turn left", wheels_turnLeft);
  });
  server.on("/right", []() {
    handleSimple("Turn right", wheels_turnRight);
  });
  server.on("/spin-left", []() {
    handleSimple("Spin left", wheels_spinLeft);
  });
  server.on("/spin-right", []() {
    handleSimple("Spin right", wheels_spinRight);
  });
  server.on("/stop", []() {
    handleSimple("Stop", wheels_stop);
  });

  // Cúi & kẹp (điều khiển rời) – cũng yêu cầu pw
  server.on("/bow-down", []() {
    if (!checkAuth()) return;
    sweep(bow, BOW_UP, BOW_DOWN, 1, 10);
    server.send(200, "text/plain", "Bow down");
  });
  server.on("/bow-up", []() {
    if (!checkAuth()) return;
    sweep(bow, BOW_DOWN, BOW_UP, 1, 10);
    server.send(200, "text/plain", "Bow up");
  });
  server.on("/grip-open", []() {
    if (!checkAuth()) return;
    sweep(gripper, GRIP_CLOSED, GRIP_OPEN, 3, 8);
    server.send(200, "text/plain", "Gripper opened");
  });
  server.on("/grip-close", []() {
    if (!checkAuth()) return;
    sweep(gripper, GRIP_OPEN, GRIP_CLOSED, 3, 8);
    server.send(200, "text/plain", "Gripper closed");
  });

  // Combo: bow + grip
  server.on("/combo-pick", []() {
    if (!checkAuth()) return;
    comboPick();
    server.send(200, "text/plain", "Combo pick (bow down + grip close + bow up)");
  });

  server.on("/combo-drop", []() {
    if (!checkAuth()) return;
    comboDrop();
    server.send(200, "text/plain", "Combo drop (bow down + grip open + bow up)");
  });

  server.begin();
  Serial.println("HTTP server started");
}

// ================== LOOP ==================
void loop() {
  server.handleClient();
}
