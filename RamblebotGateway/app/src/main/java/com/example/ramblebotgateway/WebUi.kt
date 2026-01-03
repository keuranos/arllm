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
    body { font-family: sans-serif; margin: 12px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; max-width: 980px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    button { padding: 12px 16px; font-size: 16px; }
    input[type=range] { width: 100%; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    #camWrap { display: inline-block; width: 100%; max-width: 980px; overflow: hidden; border-radius: 8px; border: 1px solid #ccc; }
    #cam { width: 100%; display:block; transform-origin: center center; }
    .hint { color:#555; font-size: 13px; }
    .small { font-size: 14px; }
  </style>
</head>
<body>
  <div class="grid">

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>Video</b> (<span class="mono">/stream.mjpeg</span>)</div>
        <div class="row">
          <button id="kb">Capture keyboard</button>
          <span class="hint">Klikkaa tästä jos WASD ei reagoi</span>
        </div>
      </div>

      <div id="camWrap">
        <img id="cam" src="/stream.mjpeg" alt="mjpeg stream"/>
      </div>

      <div style="margin-top:10px;">
        <div class="row">
          <b class="small">Camera view</b>
          <button id="rot0">0°</button>
          <button id="rot90">90°</button>
          <button id="rot180">180°</button>
          <button id="rot270">270°</button>
          <button id="mirror">Mirror</button>
          <span class="mono" id="camState"></span>
        </div>
      </div>
    </div>

    <div class="card">
      <div><b>Drive</b> <span class="hint">(WASD = hold, Space = stop)</span></div>

      <div class="row" style="margin-top:8px;">
        <button id="fwd">Forward (W)</button>
        <button id="left">Left (A)</button>
        <button id="stop">STOP (Space)</button>
        <button id="right">Right (D)</button>
        <button id="back">Back (S)</button>
      </div>

      <div style="margin-top:10px;">
        <div>Left motor: <span id="lv">0</span></div>
        <input id="l" type="range" min="-255" max="255" value="0"/>
        <div>Right motor: <span id="rv">0</span></div>
        <input id="r" type="range" min="-255" max="255" value="0"/>

        <div class="row" style="margin-top:8px;">
          <button id="apply">Apply L/R</button>
          <button id="center">Center</button>
        </div>

        <div class="hint mono" id="last" style="margin-top:6px;"></div>
      </div>
    </div>

    <div class="card">
      <div><b>Head / Phone tilt</b> <span class="hint">(from .ino: command '3')</span></div>

      <div style="margin-top:8px;">
        <div>Position (0..180): <span id="hp">90</span></div>
        <input id="headPos" type="range" min="0" max="180" value="90"/>

        <div style="margin-top:8px;">Speed (0..9): <span id="hs">0</span></div>
        <input id="headSpeed" type="range" min="0" max="9" value="0"/>

        <div class="row" style="margin-top:10px;">
          <button id="headUp">Up</button>
          <button id="headDown">Down</button>
          <button id="headSend">Set tilt</button>
        </div>

        <div class="hint">Huom: pos=90 on yleensä “suoraan”.</div>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>Phone sensors</b> <span class="hint">(live JSON)</span></div>
        <div class="row">
          <button id="sensorRefresh">Refresh</button>
          <label class="small"><input id="sensorAuto" type="checkbox" checked/> Auto</label>
        </div>
      </div>
      <div class="hint mono">/sensors.json</div>
      <pre id="sensorOut" class="mono" style="white-space:pre-wrap;"></pre>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between;">
        <div><b>ARCore pose</b> <span class="hint">(room tracking)</span></div>
        <div class="row">
          <button id="arRefresh">Refresh</button>
          <label class="small"><input id="arAuto" type="checkbox" checked/> Auto</label>
        </div>
      </div>
      <div class="hint mono">/arcore.json</div>
      <pre id="arOut" class="mono" style="white-space:pre-wrap;"></pre>
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

// sliders
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

// hold-to-run buttons
function hold(btn, onDown){
  let down = false;
  const start = (e)=>{ e.preventDefault(); down=true; onDown(); };
  const end = (e)=>{ e.preventDefault(); if(down){ down=false; stop(); } };
  btn.addEventListener('mousedown', start);
  btn.addEventListener('touchstart', start, {passive:false});
  window.addEventListener('mouseup', end);
  window.addEventListener('touchend', end);
  window.addEventListener('touchcancel', end);
}
hold(document.getElementById('fwd'),  ()=> drive(180,180));
hold(document.getElementById('back'), ()=> drive(-180,-180));
hold(document.getElementById('left'), ()=> drive(-140,140));
hold(document.getElementById('right'), ()=> drive(140,-140));

// ✅ WASD (hold) — toimii vaikka fokus olisi sivulla
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
    if(!keys.has(k)){
      keys.add(k);
      recompute();
    }
  }
});
window.addEventListener('keyup', (e)=>{
  const k = e.key.toLowerCase();
  if(keys.has(k)){
    e.preventDefault();
    keys.delete(k);
    recompute();
  }
});

// “Capture keyboard” nappi varmistaa että käyttäjä klikkaa sivua (joillain selaimilla pakollinen)
document.getElementById('kb').onclick = ()=>{
  document.body.tabIndex = 0;
  document.body.focus();
  last.textContent = "Keyboard capture OK";
};

// ✅ Camera rotate/mirror (CSS transform)
const cam = document.getElementById('cam');
const camState = document.getElementById('camState');
let rot = 0;
let mir = false;

function applyCamTransform(){
  const scaleX = mir ? -1 : 1;
  cam.style.transform = "rotate(" + rot + "deg) scaleX(" + scaleX + ")";
  camState.textContent = "rotate=" + rot + " mirror=" + mir;
}
document.getElementById('rot0').onclick = ()=>{ rot=0; applyCamTransform(); };
document.getElementById('rot90').onclick = ()=>{ rot=90; applyCamTransform(); };
document.getElementById('rot180').onclick = ()=>{ rot=180; applyCamTransform(); };
document.getElementById('rot270').onclick = ()=>{ rot=270; applyCamTransform(); };
document.getElementById('mirror').onclick = ()=>{ mir=!mir; applyCamTransform(); };
applyCamTransform();

// ✅ Head tilt controls
const headPos = document.getElementById('headPos');
const headSpeed = document.getElementById('headSpeed');
const hp = document.getElementById('hp');
const hs = document.getElementById('hs');

function updateHeadLabels(){
  hp.textContent = headPos.value;
  hs.textContent = headSpeed.value;
}
headPos.oninput = updateHeadLabels;
headSpeed.oninput = updateHeadLabels;
updateHeadLabels();

function sendHead(){
  api("/cmd?do=head&pos=" + headPos.value + "&speed=" + headSpeed.value);
}
document.getElementById('headSend').onclick = sendHead;

document.getElementById('headUp').onclick = ()=>{
  headPos.value = Math.min(180, parseInt(headPos.value,10) + 5);
  updateHeadLabels();
  sendHead();
};
document.getElementById('headDown').onclick = ()=>{
  headPos.value = Math.max(0, parseInt(headPos.value,10) - 5);
  updateHeadLabels();
  sendHead();
};

// ✅ Sensor feed
const sensorOut = document.getElementById('sensorOut');
const sensorRefresh = document.getElementById('sensorRefresh');
const sensorAuto = document.getElementById('sensorAuto');
let sensorTimer = null;

async function loadSensors(){
  try {
    const resp = await fetch('/sensors.json', {cache: 'no-store'});
    const text = await resp.text();
    sensorOut.textContent = text;
  } catch (e) {
    sensorOut.textContent = "Sensor fetch failed: " + e;
  }
}

function setSensorTimer(){
  if(sensorTimer) clearInterval(sensorTimer);
  if(sensorAuto.checked){
    sensorTimer = setInterval(loadSensors, 1000);
  }
}

sensorRefresh.onclick = loadSensors;
sensorAuto.onchange = setSensorTimer;
loadSensors();
setSensorTimer();

// ✅ ARCore pose feed
const arOut = document.getElementById('arOut');
const arRefresh = document.getElementById('arRefresh');
const arAuto = document.getElementById('arAuto');
let arTimer = null;

async function loadAr(){
  try {
    const resp = await fetch('/arcore.json', {cache: 'no-store'});
    const text = await resp.text();
    arOut.textContent = text;
  } catch (e) {
    arOut.textContent = "ARCore fetch failed: " + e;
  }
}

function setArTimer(){
  if(arTimer) clearInterval(arTimer);
  if(arAuto.checked){
    arTimer = setInterval(loadAr, 1000);
  }
}

arRefresh.onclick = loadAr;
arAuto.onchange = setArTimer;
loadAr();
setArTimer();
</script>
</body>
</html>
""".trim()
}
