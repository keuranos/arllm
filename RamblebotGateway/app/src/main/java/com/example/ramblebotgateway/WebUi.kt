package com.example.ramblebotgateway

object WebUi {
    fun html(): String = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Ramblebot</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, sans-serif; margin: 0; padding: 8px; background: #f0f0f0; font-size: 14px; }
    .header { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #333; color: white; border-radius: 8px; margin-bottom: 8px; }
    .header h1 { margin: 0; font-size: 18px; }
    .status-bar { display: flex; gap: 12px; font-size: 13px; }
    .status-item { display: flex; align-items: center; gap: 4px; }
    .battery { padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .battery.high { background: #4caf50; }
    .battery.mid { background: #ff9800; }
    .battery.low { background: #f44336; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 8px; background: white; }
    .card-title { font-weight: bold; font-size: 13px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
    .row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    button { padding: 8px 12px; font-size: 13px; cursor: pointer; border-radius: 4px; border: 1px solid #aaa; background: #f8f8f8; }
    button:hover { background: #e0e0e0; }
    button:active { background: #d0d0d0; }
    .btn-sm { padding: 4px 8px; font-size: 12px; }
    input[type=range] { width: 100%; margin: 4px 0; }
    .mono { font-family: ui-monospace, monospace; font-size: 11px; }
    #camWrap { width: 100%; overflow: hidden; border-radius: 6px; border: 1px solid #999; background: #000; }
    #cam { width: 100%; display: block; transform-origin: center; }
    #mapCanvas { border: 1px solid #444; border-radius: 6px; background: #1a1a2e; width: 100%; max-width: 300px; }
    .collapsible { cursor: pointer; }
    .collapsible:after { content: ' ▼'; font-size: 10px; }
    .collapsed:after { content: ' ►'; }
    .collapse-content { max-height: 100px; overflow: auto; transition: max-height 0.2s; }
    .collapse-content.hidden { max-height: 0; overflow: hidden; }
    .tracking { color: #2e7d32; }
    .not-tracking { color: #c62828; }
    .hint { color: #666; font-size: 11px; }
    .flex-col { display: flex; flex-direction: column; gap: 8px; }
  </style>
</head>
<body>
  <!-- Header with battery and status -->
  <div class="header">
    <h1>🤖 Ramblebot</h1>
    <div class="status-bar">
      <div class="status-item">🔋 <span id="batteryPct" class="battery high">--</span></div>
      <div class="status-item">📍 <span id="arStatus" class="not-tracking">--</span></div>
      <div class="status-item"><button id="kb" class="btn-sm">⌨️</button></div>
    </div>
  </div>

  <div class="grid">
    <!-- Left Column: Video + Controls -->
    <div class="flex-col">
      <!-- Video -->
      <div class="card">
        <div id="camWrap"><img id="cam" src="/stream.mjpeg" alt="stream"/></div>
        <div class="row" style="margin-top:6px;">
          <button class="btn-sm" id="rot0">0°</button>
          <button class="btn-sm" id="rot90">90°</button>
          <button class="btn-sm" id="rot180">180°</button>
          <button class="btn-sm" id="rot270">270°</button>
          <button class="btn-sm" id="mirror">⟷</button>
          <button class="btn-sm" id="flash">🔦</button>
          <span class="mono" id="camState"></span>
        </div>
      </div>

      <!-- Drive -->
      <div class="card">
        <div class="card-title">🎮 Drive <span class="hint">WASD+Space</span></div>
        <div class="row">
          <button id="fwd">↑W</button>
          <button id="left">←A</button>
          <button id="stop" style="background:#fcc;">■</button>
          <button id="right">D→</button>
          <button id="back">↓S</button>
        </div>
        <div style="margin-top:6px;">
          <span class="mono">L:<span id="lv">0</span></span>
          <input id="l" type="range" min="-255" max="255" value="0"/>
          <span class="mono">R:<span id="rv">0</span></span>
          <input id="r" type="range" min="-255" max="255" value="0"/>
          <div class="row"><button class="btn-sm" id="apply">Apply</button><button class="btn-sm" id="center">Center</button></div>
        </div>
        <div class="mono hint" id="last"></div>
      </div>

      <!-- Head -->
      <div class="card">
        <div class="card-title">📐 Head</div>
        <div class="row">
          <button class="btn-sm" id="headUp">↑</button>
          <span class="mono">Pos:<span id="hp">90</span>°</span>
          <button class="btn-sm" id="headDown">↓</button>
          <input id="headPos" type="range" min="0" max="180" value="90" style="flex:1;"/>
        </div>
      </div>
    </div>

    <!-- Right Column: Map + Data -->
    <div class="flex-col">
      <!-- ARCore Map -->
      <div class="card">
        <div class="card-title">🗺️ Map <div class="row"><button class="btn-sm" id="clearMap">Clear</button><label class="hint"><input id="mapAuto" type="checkbox" checked/> Live</label></div></div>
        <canvas id="mapCanvas" width="300" height="300"></canvas>
        <div class="mono" style="margin-top:4px;">
          <span id="trackStatus" class="not-tracking">-</span> |
          Pos: <span id="posXYZ">-</span> |
          Pts: <span id="pointCount">0</span>
        </div>
      </div>

      <!-- ARCore Raw (collapsible) -->
      <div class="card">
        <div class="card-title collapsible" id="arToggle">📍 ARCore <label class="hint"><input id="arAuto" type="checkbox" checked/> Auto</label></div>
        <pre id="arOut" class="mono collapse-content" style="margin:0;"></pre>
      </div>

      <!-- Sensors (collapsible) -->
      <div class="card">
        <div class="card-title collapsible" id="sensorToggle">📊 Sensors <label class="hint"><input id="sensorAuto" type="checkbox"/> Auto</label></div>
        <pre id="sensorOut" class="mono collapse-content hidden" style="margin:0;"></pre>
      </div>
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
const hp = document.getElementById('hp');
function updateHeadLabel(){ hp.textContent = headPos.value; }
headPos.oninput = ()=>{ updateHeadLabel(); sendHead(); };
updateHeadLabel();
function sendHead(){ api("/cmd?do=head&pos=" + headPos.value + "&speed=0"); }
document.getElementById('headUp').onclick = ()=>{ headPos.value = Math.min(180, +headPos.value + 15); updateHeadLabel(); sendHead(); };
document.getElementById('headDown').onclick = ()=>{ headPos.value = Math.max(0, +headPos.value - 15); updateHeadLabel(); sendHead(); };

// Battery status
const batteryPct = document.getElementById('batteryPct');
const arStatus = document.getElementById('arStatus');
async function loadBattery(){
  try {
    const resp = await fetch('/battery.json', {cache: 'no-store'});
    const data = await resp.json();
    const pct = data.percent;
    batteryPct.textContent = pct + '%' + (data.charging ? '⚡' : '');
    batteryPct.className = 'battery ' + (pct > 50 ? 'high' : pct > 20 ? 'mid' : 'low');
  } catch(e) { batteryPct.textContent = '--'; }
}
loadBattery();
setInterval(loadBattery, 10000);

// Collapsible sections
document.getElementById('arToggle').onclick = (e)=>{
  if(e.target.tagName === 'INPUT') return;
  document.getElementById('arOut').classList.toggle('hidden');
  e.currentTarget.classList.toggle('collapsed');
};
document.getElementById('sensorToggle').onclick = (e)=>{
  if(e.target.tagName === 'INPUT') return;
  document.getElementById('sensorOut').classList.toggle('hidden');
  e.currentTarget.classList.toggle('collapsed');
};

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
    arStatus.textContent = tracking ? '✓ Tracking' : data.trackingState || '--';
    arStatus.className = tracking ? 'tracking' : 'not-tracking';

    if(data.position){
      posXYZ.textContent = data.position[0].toFixed(2) + ',' +
                          data.position[1].toFixed(2) + ',' +
                          data.position[2].toFixed(2);
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
