package com.example.ramblebotgateway

object WebUi {
    fun html(): String = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Ramblebot Console</title>
  <style>
    body { font-family: sans-serif; margin: 12px; background: #f5f5f5; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; max-width: 1200px; }
    @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; background: white; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    button { padding: 10px 14px; font-size: 14px; cursor: pointer; border-radius: 6px; border: 1px solid #ccc; }
    button:hover { background: #e8e8e8; }
    input[type=range] { width: 100%; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    #camWrap { display: inline-block; width: 100%; overflow: hidden; border-radius: 8px; border: 1px solid #ccc; }
    #cam { width: 100%; display:block; transform-origin: center center; }
    .hint { color:#666; font-size: 12px; }
    .small { font-size: 13px; }
    #mapCanvas { border: 1px solid #333; border-radius: 8px; background: #1a1a2e; }
    .pose-info { font-size: 13px; color: #333; }
    .tracking { color: #2e7d32; font-weight: bold; }
    .not-tracking { color: #c62828; font-weight: bold; }
  </style>
</head>
<body>
  <h2 style="margin-top:0;">🤖 Ramblebot Console</h2>
  <div class="grid">

    <!-- Video Feed -->
    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>📹 Video</b></div>
        <button id="kb">Capture keyboard</button>
      </div>
      <div id="camWrap" style="margin-top:8px;">
        <img id="cam" src="/stream.mjpeg" alt="mjpeg stream"/>
      </div>
      <div style="margin-top:8px;">
        <div class="row">
          <button id="rot0">0°</button>
          <button id="rot90">90°</button>
          <button id="rot180">180°</button>
          <button id="rot270">270°</button>
          <button id="mirror">Mirror</button>
          <button id="flash" title="Toggle flashlight">🔦</button>
          <span class="mono" id="camState"></span>
        </div>
      </div>
    </div>

    <!-- ARCore Map -->
    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>🗺️ ARCore Map</b></div>
        <div class="row">
          <button id="clearMap">Clear</button>
          <label class="small"><input id="mapAuto" type="checkbox" checked/> Live</label>
        </div>
      </div>
      <canvas id="mapCanvas" width="400" height="400" style="margin-top:8px; width:100%; max-width:400px;"></canvas>
      <div class="pose-info" style="margin-top:8px;">
        <div>Status: <span id="trackStatus" class="not-tracking">-</span></div>
        <div>Position: <span id="posXYZ" class="mono">-</span></div>
        <div>Rotation: <span id="rotQ" class="mono">-</span></div>
        <div>Points: <span id="pointCount" class="mono">0</span></div>
      </div>
    </div>

    <!-- Drive Controls -->
    <div class="card">
      <div><b>🎮 Drive</b> <span class="hint">(WASD + Space)</span></div>
      <div class="row" style="margin-top:8px;">
        <button id="fwd">↑ W</button>
        <button id="left">← A</button>
        <button id="stop">STOP</button>
        <button id="right">→ D</button>
        <button id="back">↓ S</button>
      </div>
      <div style="margin-top:10px;">
        <div>Left: <span id="lv">0</span></div>
        <input id="l" type="range" min="-255" max="255" value="0"/>
        <div>Right: <span id="rv">0</span></div>
        <input id="r" type="range" min="-255" max="255" value="0"/>
        <div class="row" style="margin-top:8px;">
          <button id="apply">Apply</button>
          <button id="center">Center</button>
        </div>
        <div class="hint mono" id="last" style="margin-top:6px;"></div>
      </div>
    </div>

    <!-- Head Controls -->
    <div class="card">
      <div><b>📐 Head Tilt</b></div>
      <div style="margin-top:8px;">
        <div>Position: <span id="hp">90</span>°</div>
        <input id="headPos" type="range" min="0" max="180" value="90"/>
        <div style="margin-top:8px;">Speed: <span id="hs">0</span></div>
        <input id="headSpeed" type="range" min="0" max="9" value="0"/>
        <div class="row" style="margin-top:8px;">
          <button id="headUp">Up</button>
          <button id="headDown">Down</button>
          <button id="headSend">Set</button>
        </div>
      </div>
    </div>

    <!-- Sensors -->
    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>📊 Sensors</b></div>
        <label class="small"><input id="sensorAuto" type="checkbox"/> Auto</label>
      </div>
      <pre id="sensorOut" class="mono" style="white-space:pre-wrap; max-height:150px; overflow:auto; margin-top:8px;"></pre>
    </div>

    <!-- ARCore Raw -->
    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>📍 ARCore Raw</b></div>
        <label class="small"><input id="arAuto" type="checkbox" checked/> Auto</label>
      </div>
      <pre id="arOut" class="mono" style="white-space:pre-wrap; max-height:150px; overflow:auto; margin-top:8px;"></pre>
    </div>

  </div>

<script>
const last = document.getElementById('last');
function api(url){
  last.textContent = url;
  fetch(url).catch(()=>{});
}
function drive(L,R){ api("/cmd?do=drive&l=" + L + "&r=" + R); }
function stop(){ api("/cmd?do=stop"); }

// Sliders
const l = document.getElementById('l');
const r = document.getElementById('r');
const lv = document.getElementById('lv');
const rv = document.getElementById('rv');
function setLabels(){ lv.textContent = l.value; rv.textContent = r.value; }
l.addEventListener('input', setLabels);
r.addEventListener('input', setLabels);
setLabels();
document.getElementById('apply').onclick = ()=> drive(l.value, r.value);
document.getElementById('center').onclick = ()=> { l.value=0; r.value=0; setLabels(); };
document.getElementById('stop').onclick = stop;

// Hold-to-run buttons
function hold(btn, onDown){
  let down = false;
  const start = (e)=>{ e.preventDefault(); down=true; onDown(); };
  const end = (e)=>{ e.preventDefault(); if(down){ down=false; stop(); } };
  btn.addEventListener('mousedown', start);
  btn.addEventListener('touchstart', start, {passive:false});
  window.addEventListener('mouseup', end);
  window.addEventListener('touchend', end);
}
hold(document.getElementById('fwd'),  ()=> drive(180,180));
hold(document.getElementById('back'), ()=> drive(-180,-180));
hold(document.getElementById('left'), ()=> drive(-140,140));
hold(document.getElementById('right'), ()=> drive(140,-140));

// WASD keyboard
const keys = new Set();
function recompute(){
  if(keys.has(' ')) return stop();
  if(keys.has('w')) return drive(180,180);
  if(keys.has('s')) return drive(-180,-180);
  if(keys.has('a')) return drive(-140,140);
  if(keys.has('d')) return drive(140,-140);
  stop();
}
window.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  if(['w','a','s','d',' '].includes(k)){
    e.preventDefault();
    if(!keys.has(k)){ keys.add(k); recompute(); }
  }
});
window.addEventListener('keyup', (e)=>{
  const k = e.key.toLowerCase();
  if(keys.has(k)){ keys.delete(k); recompute(); }
});
document.getElementById('kb').onclick = ()=>{
  document.body.tabIndex = 0;
  document.body.focus();
  last.textContent = "Keyboard captured!";
};

// Camera transform
const cam = document.getElementById('cam');
const camState = document.getElementById('camState');
let rot = 0, mir = false;
function applyCamTransform(){
  cam.style.transform = "rotate(" + rot + "deg) scaleX(" + (mir?-1:1) + ")";
  camState.textContent = rot + "° " + (mir?"mirrored":"");
}
document.getElementById('rot0').onclick = ()=>{ rot=0; applyCamTransform(); };
document.getElementById('rot90').onclick = ()=>{ rot=90; applyCamTransform(); };
document.getElementById('rot180').onclick = ()=>{ rot=180; applyCamTransform(); };
document.getElementById('rot270').onclick = ()=>{ rot=270; applyCamTransform(); };
document.getElementById('mirror').onclick = ()=>{ mir=!mir; applyCamTransform(); };
applyCamTransform();

// Flashlight toggle
const flashBtn = document.getElementById('flash');
let flashOn = false;
flashBtn.onclick = ()=>{
  fetch('/flash?toggle=1').then(r => r.text()).then(t => {
    flashOn = t.includes('ON');
    flashBtn.style.background = flashOn ? '#ffc107' : '';
    flashBtn.title = flashOn ? 'Flashlight ON' : 'Flashlight OFF';
  }).catch(()=>{});
};

// Head controls
const headPos = document.getElementById('headPos');
const headSpeed = document.getElementById('headSpeed');
const hp = document.getElementById('hp');
const hs = document.getElementById('hs');
function updateHeadLabels(){ hp.textContent = headPos.value; hs.textContent = headSpeed.value; }
headPos.oninput = updateHeadLabels;
headSpeed.oninput = updateHeadLabels;
updateHeadLabels();
function sendHead(){ api("/cmd?do=head&pos=" + headPos.value + "&speed=" + headSpeed.value); }
document.getElementById('headSend').onclick = sendHead;
document.getElementById('headUp').onclick = ()=>{ headPos.value = Math.min(180, +headPos.value + 10); updateHeadLabels(); sendHead(); };
document.getElementById('headDown').onclick = ()=>{ headPos.value = Math.max(0, +headPos.value - 10); updateHeadLabels(); sendHead(); };

// Sensor feed
const sensorOut = document.getElementById('sensorOut');
const sensorAuto = document.getElementById('sensorAuto');
let sensorTimer = null;
async function loadSensors(){
  try {
    const resp = await fetch('/sensors.json', {cache: 'no-store'});
    sensorOut.textContent = await resp.text();
  } catch (e) { sensorOut.textContent = "Error: " + e; }
}
function setSensorTimer(){
  if(sensorTimer) clearInterval(sensorTimer);
  if(sensorAuto.checked) sensorTimer = setInterval(loadSensors, 2000);
}
sensorAuto.onchange = setSensorTimer;
loadSensors();
setSensorTimer();

// ========== ARCore Map ==========
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const arOut = document.getElementById('arOut');
const arAuto = document.getElementById('arAuto');
const trackStatus = document.getElementById('trackStatus');
const posXYZ = document.getElementById('posXYZ');
const rotQ = document.getElementById('rotQ');
const pointCount = document.getElementById('pointCount');

// Trail history
let trail = [];
const maxTrail = 500;
let mapScale = 50; // pixels per meter
let mapCenterX = canvas.width / 2;
let mapCenterY = canvas.height / 2;

function clearMap(){
  trail = [];
  drawMap(null);
}
document.getElementById('clearMap').onclick = clearMap;

function worldToCanvas(x, z){
  return {
    x: mapCenterX + x * mapScale,
    y: mapCenterY - z * mapScale  // Z is forward, so negate for screen Y
  };
}

function drawMap(pose){
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid
  ctx.strokeStyle = '#2a2a4e';
  ctx.lineWidth = 0.5;
  const gridStep = mapScale; // 1 meter
  for(let x = 0; x < canvas.width; x += gridStep){
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for(let y = 0; y < canvas.height; y += gridStep){
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }

  // Origin marker
  ctx.fillStyle = '#4a4a6e';
  ctx.beginPath();
  ctx.arc(mapCenterX, mapCenterY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#888';
  ctx.font = '10px sans-serif';
  ctx.fillText('0,0', mapCenterX + 8, mapCenterY + 4);

  // Draw trail
  if(trail.length > 1){
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const p0 = worldToCanvas(trail[0].x, trail[0].z);
    ctx.moveTo(p0.x, p0.y);
    for(let i = 1; i < trail.length; i++){
      const p = worldToCanvas(trail[i].x, trail[i].z);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }

  // Draw robot
  if(pose && pose.position){
    const px = pose.position[0];
    const py = pose.position[1];
    const pz = pose.position[2];

    const cp = worldToCanvas(px, pz);

    // Robot body
    ctx.fillStyle = '#00e676';
    ctx.beginPath();
    ctx.arc(cp.x, cp.y, 8, 0, Math.PI * 2);
    ctx.fill();

    // Direction indicator (from quaternion)
    if(pose.rotation){
      const qx = pose.rotation[0];
      const qy = pose.rotation[1];
      const qz = pose.rotation[2];
      const qw = pose.rotation[3];

      // Extract yaw from quaternion
      const siny = 2 * (qw * qy - qz * qx);
      const cosy = 1 - 2 * (qx * qx + qy * qy);
      const yaw = Math.atan2(siny, cosy);

      // Draw direction arrow
      const arrowLen = 20;
      const ax = cp.x + Math.sin(yaw) * arrowLen;
      const ay = cp.y - Math.cos(yaw) * arrowLen;

      ctx.strokeStyle = '#00e676';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cp.x, cp.y);
      ctx.lineTo(ax, ay);
      ctx.stroke();

      // Arrow head
      ctx.fillStyle = '#00e676';
      ctx.beginPath();
      ctx.arc(ax, ay, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Add to trail
    trail.push({x: px, z: pz});
    if(trail.length > maxTrail) trail.shift();
  }

  // Scale indicator
  ctx.fillStyle = '#888';
  ctx.font = '11px sans-serif';
  ctx.fillText('1m', 10, canvas.height - 10);
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(10, canvas.height - 20);
  ctx.lineTo(10 + mapScale, canvas.height - 20);
  ctx.stroke();
}

let arTimer = null;
async function loadAr(){
  try {
    const resp = await fetch('/arcore.json', {cache: 'no-store'});
    const data = await resp.json();
    arOut.textContent = JSON.stringify(data, null, 2);

    // Update status
    const tracking = data.trackingState === 'TRACKING';
    trackStatus.textContent = data.trackingState || 'N/A';
    trackStatus.className = tracking ? 'tracking' : 'not-tracking';

    if(data.position){
      posXYZ.textContent = 'X:' + data.position[0].toFixed(2) +
                          ' Y:' + data.position[1].toFixed(2) +
                          ' Z:' + data.position[2].toFixed(2);
    }
    if(data.rotation){
      rotQ.textContent = data.rotation.map(v => v.toFixed(2)).join(', ');
    }
    pointCount.textContent = trail.length;

    // Draw map
    if(tracking){
      drawMap({position: data.position, rotation: data.rotation});
    } else {
      drawMap(null);
    }
  } catch (e) {
    arOut.textContent = "Error: " + e;
    trackStatus.textContent = 'ERROR';
    trackStatus.className = 'not-tracking';
  }
}

function setArTimer(){
  if(arTimer) clearInterval(arTimer);
  if(arAuto.checked) arTimer = setInterval(loadAr, 200);
}
arAuto.onchange = setArTimer;
loadAr();
setArTimer();

// Initial map draw
drawMap(null);
</script>
</body>
</html>
""".trim()
}
