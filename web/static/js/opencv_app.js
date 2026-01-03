/**
 * Robot Control - OpenCV App
 */

// ================== State ==================
let pickingColor = null;
let measureMode = null;
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let selectionRect = null;
let selectionLine = null;

const videoStream = document.getElementById('video-stream');
const videoContainer = document.querySelector('.video-container');

// ================== Init Selection Elements ==================
function initSelectionElements() {
    selectionRect = document.createElement('div');
    selectionRect.className = 'selection-rect';
    selectionRect.style.display = 'none';
    videoContainer.appendChild(selectionRect);

    selectionLine = document.createElement('div');
    selectionLine.className = 'selection-line';
    selectionLine.style.display = 'none';
    videoContainer.appendChild(selectionLine);
}

function getVideoCoords(e) {
    const rect = videoStream.getBoundingClientRect();
    const scaleX = videoStream.naturalWidth / rect.width;
    const scaleY = videoStream.naturalHeight / rect.height;
    return {
        x: Math.round((e.clientX - rect.left) * scaleX),
        y: Math.round((e.clientY - rect.top) * scaleY),
        displayX: e.clientX - rect.left,
        displayY: e.clientY - rect.top
    };
}

// ================== Mouse Events ==================
videoStream.addEventListener('mousedown', function(e) {
    if (!pickingColor && !measureMode) return;
    e.preventDefault();
    isDragging = true;
    dragStart = getVideoCoords(e);

    if (measureMode === 'marker' || measureMode === 'pickup' || measureMode === 'drop') {
        selectionLine.style.left = dragStart.displayX + 'px';
        selectionLine.style.top = dragStart.displayY + 'px';
        selectionLine.style.width = '0';
        selectionLine.style.transform = 'rotate(0deg)';
        selectionLine.className = 'selection-line mode-' + measureMode;
        selectionLine.style.display = 'block';
        selectionRect.style.display = 'none';
    } else {
        selectionRect.style.left = dragStart.displayX + 'px';
        selectionRect.style.top = dragStart.displayY + 'px';
        selectionRect.style.width = '0';
        selectionRect.style.height = '0';
        selectionRect.className = measureMode === 'bin' ? 'selection-rect mode-bin' : 'selection-rect';
        selectionRect.style.display = 'block';
        selectionLine.style.display = 'none';
    }
});

videoStream.addEventListener('mousemove', function(e) {
    if (!isDragging || (!pickingColor && !measureMode)) return;
    const coords = getVideoCoords(e);
    const rect = videoStream.getBoundingClientRect();

    if (measureMode === 'marker' || measureMode === 'pickup' || measureMode === 'drop') {
        const dx = coords.displayX - dragStart.displayX;
        const dy = coords.displayY - dragStart.displayY;
        const length = Math.sqrt(dx*dx + dy*dy);
        const angle = Math.atan2(dy, dx) * 180 / Math.PI;
        const scaleX = videoStream.naturalWidth / rect.width;
        selectionLine.style.width = length + 'px';
        selectionLine.style.transform = `rotate(${angle}deg)`;
        selectionLine.setAttribute('data-length', Math.round(length * scaleX) + 'px');
    } else {
        const left = Math.min(dragStart.displayX, coords.displayX);
        const top = Math.min(dragStart.displayY, coords.displayY);
        selectionRect.style.left = left + 'px';
        selectionRect.style.top = top + 'px';
        selectionRect.style.width = Math.abs(coords.displayX - dragStart.displayX) + 'px';
        selectionRect.style.height = Math.abs(coords.displayY - dragStart.displayY) + 'px';
    }
});

videoStream.addEventListener('mouseup', function(e) {
    if (!isDragging || (!pickingColor && !measureMode)) return;
    isDragging = false;
    selectionRect.style.display = 'none';
    selectionLine.style.display = 'none';

    const coords = getVideoCoords(e);

    if (measureMode === 'marker' || measureMode === 'pickup' || measureMode === 'drop') {
        const dx = coords.x - dragStart.x;
        const dy = coords.y - dragStart.y;
        const length = Math.round(Math.sqrt(dx*dx + dy*dy));
        handleMeasureResult(measureMode, length);
    } else if (measureMode === 'bin') {
        // Chế độ vẽ BIN - lấy tọa độ rectangle
        const x1 = Math.min(dragStart.x, coords.x);
        const y1 = Math.min(dragStart.y, coords.y);
        const x2 = Math.max(dragStart.x, coords.x);
        const y2 = Math.max(dragStart.y, coords.y);
        saveBinFromRect(x1, y1, x2, y2);
    } else if (pickingColor) {
        const x1 = Math.min(dragStart.x, coords.x);
        const y1 = Math.min(dragStart.y, coords.y);
        const x2 = Math.max(dragStart.x, coords.x);
        const y2 = Math.max(dragStart.y, coords.y);
        if (x2 - x1 < 5 && y2 - y1 < 5) {
            pickColorFromRegion(dragStart.x - 2, dragStart.y - 2, dragStart.x + 3, dragStart.y + 3, pickingColor);
        } else {
            pickColorFromRegion(x1, y1, x2, y2, pickingColor);
        }
    }
});

videoStream.addEventListener('mouseleave', function() {
    if (isDragging) {
        isDragging = false;
        selectionRect.style.display = 'none';
        selectionLine.style.display = 'none';
    }
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        cancelPicking();
        cancelMeasure();
    }
});

// ================== Measure ==================
function startMeasure(mode) {
    cancelPicking();
    if (measureMode === mode) { cancelMeasure(); return; }
    measureMode = mode;

    const ind = document.getElementById('picker-ind');
    ind.classList.remove('mode-color', 'mode-bin', 'mode-measure');
    ind.classList.add('active', 'mode-measure');

    const hints = {
        marker: 'Kẻ đường đo <strong>Marker Distance</strong>',
        pickup: 'Kẻ đường đo <strong>Pickup Distance</strong>',
        drop: 'Kẻ đường đo <strong>Drop Distance</strong>'
    };
    ind.innerHTML = hints[mode] + ' <small>(ESC để hủy)</small>';

    document.querySelectorAll('.measure-box').forEach(b => b.classList.remove('active'));
    const el = document.getElementById('measure-' + mode);
    if (el) el.classList.add('active');
}

function cancelMeasure() {
    measureMode = null;
    drawBinMode = false;  // Cũng hủy draw bin mode
    document.getElementById('picker-ind').classList.remove('active');
    document.querySelectorAll('.measure-box').forEach(b => b.classList.remove('active'));
}

function handleMeasureResult(mode, length) {
    if (mode === 'marker') {
        document.getElementById('marker-dist-tol').value = length;
    } else if (mode === 'pickup') {
        document.getElementById('pickup-dist-tol').value = length;
    } else if (mode === 'drop') {
        document.getElementById('drop-dist').value = length;
    }
    saveConfigSilent();
    cancelMeasure();
}

// ================== BIN Drawing ==================
let drawBinMode = false;
let currentBinType = 'green';  // 'green' or 'yellow'

function startDrawBin(binType, event) {
    if (event) event.stopPropagation();
    cancelPicking();
    cancelMeasure();
    
    if (drawBinMode && currentBinType === binType) {
        cancelDrawBin();
        return;
    }
    
    drawBinMode = true;
    currentBinType = binType;
    measureMode = 'bin';  // Dùng measureMode để xử lý mouse events
    
    const ind = document.getElementById('picker-ind');
    ind.classList.remove('mode-color', 'mode-measure');
    ind.classList.add('active', 'mode-bin');
    
    const colorText = binType === 'green' ? 'GREEN' : 'YELLOW';
    ind.innerHTML = `Kéo chuột vẽ <strong>${colorText} BIN</strong> trên video <small>(ESC để hủy)</small>`;
    
    // Clear all active measure boxes first
    document.querySelectorAll('.measure-box').forEach(b => b.classList.remove('active'));
    
    // Highlight the active bin button
    const btnId = binType === 'green' ? 'measure-green-bin' : 'measure-yellow-bin';
    const btn = document.getElementById(btnId);
    if (btn) btn.classList.add('active');
}

function cancelDrawBin() {
    drawBinMode = false;
    measureMode = null;
    document.getElementById('picker-ind').classList.remove('active');
    document.querySelectorAll('.measure-box').forEach(b => b.classList.remove('active'));
}

async function saveBinFromRect(x1, y1, x2, y2) {
    try {
        const res = await fetch('/api/update_bin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                bin_type: currentBinType,
                x1, y1, x2, y2 
            })
        });
        const data = await res.json();
        
        if (data.success) {
            updateBinStatus(data.bin_type, data.bin_pos, data.bin_margin);
            console.log(`[${currentBinType.toUpperCase()} BIN] Saved from drawing:`, data);
        } else {
            alert('Lỗi: ' + (data.error || 'Không thể lưu BIN'));
        }
    } catch (err) {
        console.error(err);
        alert('Lỗi kết nối!');
    }
    cancelDrawBin();
}

async function clearBin(binType, event) {
    if (event) event.stopPropagation();
    try {
        await fetch('/api/clear_bin', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bin_type: binType })
        });
        updateBinStatus(binType, null, null);
        console.log(`[${binType.toUpperCase()} BIN] Cleared`);
    } catch (err) {
        console.error(err);
    }
}

function updateBinStatus(binType, pos, margin) {
    const elId = binType === 'green' ? 'green-bin-status' : 'yellow-bin-status';
    const el = document.getElementById(elId);
    if (!el) return;
    
    if (pos && margin) {
        el.textContent = `(${pos[0]}, ${pos[1]}) r=${margin}`;
        el.classList.add('active');
    } else {
        el.textContent = 'Not set';
        el.classList.remove('active');
    }
}

// For backward compatibility
function updateBothBinStatus(greenPos, greenMargin, yellowPos, yellowMargin) {
    updateBinStatus('green', greenPos, greenMargin);
    updateBinStatus('yellow', yellowPos, yellowMargin);
}

// ================== Color Picking ==================
function startPicking(colorName) {
    cancelMeasure();
    if (pickingColor === colorName) { cancelPicking(); return; }
    pickingColor = colorName;

    const ind = document.getElementById('picker-ind');
    ind.classList.remove('mode-bin', 'mode-measure');
    ind.classList.add('active', 'mode-color');
    ind.innerHTML = `Chọn màu <strong>${colorName.toUpperCase()}</strong> từ video <small>(ESC để hủy)</small>`;

    document.querySelectorAll('.pick-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.color === colorName);
    });
}

function cancelPicking() {
    pickingColor = null;
    document.getElementById('picker-ind').classList.remove('active');
    document.querySelectorAll('.pick-btn').forEach(b => b.classList.remove('active'));
}

async function pickColorFromRegion(x1, y1, x2, y2, colorName) {
    try {
        const res = await fetch('/api/pick_color', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x1, y1, x2, y2, color_name: colorName })
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        // Show picked preview
        const box = document.getElementById('picked-box');
        box.classList.add('show');
        document.getElementById('picked-swatch').style.background = `rgb(${data.rgb.join(',')})`;
        document.getElementById('picked-rgb').textContent = `(${data.rgb.join(', ')})`;
        document.getElementById('picked-hsv').textContent = `(${data.hsv.join(', ')})`;

        if (data.updated_color) loadColors();
        cancelPicking();
    } catch (err) {
        console.error(err);
    }
}

// ================== Load Colors ==================
async function loadColors() {
    try {
        const res = await fetch('/api/colors');
        const data = await res.json();

        for (const [name, rgb] of Object.entries(data.colors_rgb)) {
            const swatch = document.getElementById(`swatch-${name}`);
            const rgbEl = document.getElementById(`rgb-${name}`);
            if (swatch) swatch.style.background = `rgb(${rgb.join(',')})`;
            if (rgbEl) rgbEl.textContent = `(${rgb.join(', ')})`;
        }

        document.getElementById('tol-h').value = data.hsv_tolerance.h_tol;
        document.getElementById('tol-s').value = data.hsv_tolerance.s_tol;
        document.getElementById('tol-v').value = data.hsv_tolerance.v_tol;
        updateTolDisplay();
    } catch (err) { console.error(err); }
}

// ================== Tolerance ==================
function updateTolDisplay() {
    document.getElementById('tol-h-val').textContent = '±' + document.getElementById('tol-h').value;
    document.getElementById('tol-s-val').textContent = '±' + document.getElementById('tol-s').value;
    document.getElementById('tol-v-val').textContent = '±' + document.getElementById('tol-v').value;
}

async function applyTolerance() {
    try {
        await fetch('/api/tolerance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                h_tol: +document.getElementById('tol-h').value,
                s_tol: +document.getElementById('tol-s').value,
                v_tol: +document.getElementById('tol-v').value
            })
        });
    } catch (err) { console.error(err); }
}

// ================== Robot Control ==================
// Cơ chế toggle giống Arduino - bấm 1 lần để bật, bấm lần nữa để tắt
let currentMove = null;

const moveButtons = {
    'MOVE_FORWARD':   { id: 'btn-forward',    idle: 'Tiến',       active: 'Đang tiến' },
    'MOVE_BACKWARD':  { id: 'btn-backward',   idle: 'Lùi',        active: 'Đang lùi' },
    'TURN_LEFT':      { id: 'btn-left',       idle: 'Rẽ trái',    active: 'Đang rẽ trái' },
    'TURN_RIGHT':     { id: 'btn-right',      idle: 'Rẽ phải',    active: 'Đang rẽ phải' },
    'SPIN_LEFT':      { id: 'btn-spin-left',  idle: 'Quay trái',  active: 'Đang quay trái' },
    'SPIN_RIGHT':     { id: 'btn-spin-right', idle: 'Quay phải',  active: 'Đang quay phải' }
};

function updateMoveButtons() {
    for (const cmd in moveButtons) {
        const cfg = moveButtons[cmd];
        const btn = document.getElementById(cfg.id);
        if (!btn) continue;

        if (currentMove === cmd) {
            btn.classList.add('active');
            btn.innerText = cfg.active;
        } else {
            btn.classList.remove('active');
            btn.innerText = cfg.idle;
        }
    }
}

async function toggleMove(cmd) {
    // Chỉ cho phép điều khiển khi Auto OFF
    if (autoEnabled) {
        console.log('Manual control disabled - Auto is ON');
        return;
    }
    
    let sendCmd;
    if (currentMove === cmd) {
        // Đang active, click lần nữa -> stop
        currentMove = null;
        sendCmd = 'STOP';
    } else {
        // Bật lệnh mới
        currentMove = cmd;
        sendCmd = cmd;
    }
    
    updateMoveButtons();
    
    try {
        await fetch('/api/manual_command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: sendCmd })
        });
    } catch (err) { console.error(err); }
}

async function sendCmd(cmd) {
    // Chỉ cho phép điều khiển khi Auto OFF
    if (autoEnabled) {
        console.log('Manual control disabled - Auto is ON');
        return;
    }
    
    // Các lệnh không phải movement (bow, grip, combo) -> gửi trực tiếp
    // Reset currentMove vì robot sẽ dừng sau các action này
    if (cmd === 'STOP') {
        currentMove = null;
        updateMoveButtons();
    }
    
    try {
        await fetch('/api/manual_command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: cmd })
        });
    } catch (err) { console.error(err); }
}

async function toggleAuto() {
    try {
        const res = await fetch('/api/toggle_enabled', { method: 'POST' });
        const data = await res.json();
        updateAutoToggle(data.enabled);
    } catch (err) { console.error(err); }
}

let autoEnabled = true;

function updateAutoToggle(enabled) {
    autoEnabled = enabled;
    const badge = document.getElementById('badge-auto');
    const btn = document.getElementById('btn-auto');
    const manualSection = document.getElementById('manual-section');
    const manualDisabled = document.getElementById('manual-disabled');
    
    if (enabled) {
        badge.textContent = 'AUTO: ON';
        badge.classList.add('on');
        badge.classList.remove('off');
        btn.textContent = 'Auto OFF';
        btn.classList.remove('on');
        btn.classList.add('off');
        // Hide manual controls when Auto is ON
        if (manualSection) manualSection.style.display = 'none';
        if (manualDisabled) manualDisabled.style.display = 'flex';
    } else {
        badge.textContent = 'AUTO: OFF';
        badge.classList.remove('on');
        badge.classList.add('off');
        btn.textContent = 'Auto ON';
        btn.classList.add('on');
        btn.classList.remove('off');
        // Show manual controls when Auto is OFF
        if (manualSection) manualSection.style.display = 'block';
        if (manualDisabled) manualDisabled.style.display = 'none';
    }
}

async function resetState() {
    try { await fetch('/api/reset_state', { method: 'POST' }); } catch (err) { console.error(err); }
}

async function emergencyStop() {
    // Dừng khẩn cấp - LUÔN hoạt động bất kể Auto ON/OFF
    console.log('[EMERGENCY] Sending STOP command');
    currentMove = null;
    updateMoveButtons();
    try {
        await fetch('/api/manual_command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'STOP' })
        });
    } catch (err) { console.error(err); }
}

// ================== Config ==================
async function saveConfig() {
    await saveConfigSilent();
    alert('Saved!');
}

async function saveConfigSilent() {
    const markerDist = +document.getElementById('marker-dist-tol').value;
    const markerRange = +document.getElementById('marker-dist-range').value;
    const cfg = {
        robot_ip: document.getElementById('cfg-ip').value,
        robot_pw: document.getElementById('cfg-pw').value,
        marker_min_dist: markerDist - markerRange,
        marker_max_dist: markerDist + markerRange,
        pickup_distance: +document.getElementById('pickup-dist-tol').value,
        drop_distance: +document.getElementById('drop-dist').value,
        drop_angle: +document.getElementById('drop-angle').value
    };
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg)
        });
    } catch (err) { console.error(err); }
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        document.getElementById('cfg-ip').value = data.robot_ip;
        document.getElementById('cfg-pw').value = data.robot_pw;
        
        const minDist = data.marker_min_dist || 130;
        const maxDist = data.marker_max_dist || 170;
        const markerCenter = Math.round((minDist + maxDist) / 2);
        const markerRange = Math.round((maxDist - minDist) / 2);
        document.getElementById('marker-dist-tol').value = markerCenter;
        document.getElementById('marker-dist-range').value = markerRange;
        
        document.getElementById('pickup-dist-tol').value = data.pickup_distance || 80;
        document.getElementById('drop-dist').value = data.drop_distance || 100;
        document.getElementById('drop-angle').value = data.drop_angle || 30;
    } catch (err) { console.error(err); }
}

// ================== Status Updates ==================
async function updateStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('stat-state').textContent = data.state;
        document.getElementById('stat-cmd').textContent = data.command;
        
        const markersEl = document.getElementById('stat-markers');
        markersEl.textContent = data.markers_ok ? 'OK' : 'Missing';
        markersEl.className = 'val ' + (data.markers_ok ? 'ok' : 'err');

        // Target (green or yellow)
        const targetEl = document.getElementById('stat-target');
        if (targetEl) {
            if (data.target_detected) {
                targetEl.textContent = `${data.target_color?.toUpperCase() || 'Found'}`;
                targetEl.className = 'val ok';
            } else {
                targetEl.textContent = 'None';
                targetEl.className = 'val';
            }
        }
        
        // Carrying color
        const carryingEl = document.getElementById('stat-carrying');
        if (carryingEl) {
            if (data.carrying_color) {
                carryingEl.textContent = data.carrying_color.toUpperCase();
                carryingEl.className = 'val ok';
            } else {
                carryingEl.textContent = 'None';
                carryingEl.className = 'val';
            }
        }
        
        // Counts
        const greenCountEl = document.getElementById('stat-green-count');
        if (greenCountEl) {
            greenCountEl.textContent = data.green_count || 0;
        }
        const yellowCountEl = document.getElementById('stat-yellow-count');
        if (yellowCountEl) {
            yellowCountEl.textContent = data.yellow_count || 0;
        }

        const markerDistEl = document.getElementById('stat-marker-dist');
        markerDistEl.textContent = data.marker_dist > 0 ? data.marker_dist + 'px' : '--';
        markerDistEl.className = 'val ' + (data.marker_dist_ok ? 'ok' : (data.marker_dist > 0 ? 'err' : ''));

        const targetDistEl = document.getElementById('stat-target-dist');
        if (targetDistEl) {
            targetDistEl.textContent = data.dist_to_target > 0 ? data.dist_to_target + 'px' : '--';
            const pickupDist = data.pickup_distance || 80;
            targetDistEl.className = 'val ' + (data.dist_to_target > 0 && data.dist_to_target <= pickupDist ? 'ok' : (data.dist_to_target > 0 ? 'warn' : ''));
        }

        // BIN distance and angle
        const binDistEl = document.getElementById('stat-bin-dist');
        binDistEl.textContent = data.dist_to_bin > 0 ? data.dist_to_bin + 'px' : '--';
        const pickupDist = data.pickup_distance || 80;
        binDistEl.className = 'val ' + (data.dist_to_bin > 0 && data.dist_to_bin <= pickupDist ? 'ok' : (data.dist_to_bin > 0 ? 'warn' : ''));

        const binAngleEl = document.getElementById('stat-bin-angle');
        binAngleEl.textContent = data.angle_to_bin > 0 ? data.angle_to_bin + '°' : '--';
        const angleOk = data.angle_to_bin > 0 && data.angle_to_bin <= 30;
        binAngleEl.className = 'val ' + (angleOk ? 'ok' : (data.angle_to_bin > 0 ? 'warn' : ''));

        // Update dual BIN status display
        updateBinStatus('green', data.green_bin_pos, data.green_bin_margin);
        updateBinStatus('yellow', data.yellow_bin_pos, data.yellow_bin_margin);

        document.getElementById('badge-fps').textContent = 'FPS: ' + (data.fps || '--');
        updateAutoToggle(data.robot_enabled);
    } catch (err) { console.error(err); }
}

// ================== Init ==================
document.addEventListener('DOMContentLoaded', function() {
    initSelectionElements();

    // Tolerance slider events
    ['h', 's', 'v'].forEach(c => {
        const el = document.getElementById('tol-' + c);
        if (el) {
            el.addEventListener('input', updateTolDisplay);
            el.addEventListener('change', applyTolerance);
        }
    });

    loadColors();
    loadConfig();
    updateStatus();
    setInterval(updateStatus, 3000);
});
