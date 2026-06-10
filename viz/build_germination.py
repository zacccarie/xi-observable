"""Génère viz/germination.html — le film du germe (pythia s41, test3).

Données réelles : 28 canaux Γ (7 couches × 4 positions) + val_acc, embarquées
inline. Three.js r128, fond noir, émissif, HUD mono. Autonome (double-clic).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/tmp/s41.json"

d = json.load(open(SRC))
anchors = d["gammas"]["L3_p1"]["anchors"]
gam = {ch: [round(v, 4) for v in d["gammas"][ch]["gamma"]] for ch in d["gammas"]}
traj = [{"s": t["step"], "v": round(t["val_acc"], 4), "t": round(t["train_acc"], 4)}
        for t in d["traj"]]
payload = {"anchors": anchors, "gam": gam, "traj": traj,
           "t_grok": d["t_grok"], "t_mem": d["t_mem"], "steps": d["steps"]}

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GERMINATION — naissance d'un circuit dans un LLM</title>
<style>
  :root { --orange:#ff6d00; --cyan:#00e5ff; --dim:#9aa0a6; --bg:#000; }
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { background:var(--bg); height:100%; overflow:hidden;
    font-family:ui-monospace,'SF Mono',Menlo,monospace; color:#e8eaed; }
  #scene { position:fixed; inset:0; }
  .hud { position:fixed; z-index:10; pointer-events:none; }
  #title { top:28px; left:32px; }
  #title h1 { font-size:15px; letter-spacing:.35em; font-weight:600; color:#fff; }
  #title p { font-size:10.5px; color:var(--dim); margin-top:7px; line-height:1.65;
    letter-spacing:.04em; }
  #counter { top:28px; right:32px; text-align:right;
    font-variant-numeric:tabular-nums; }
  #counter .step { font-size:26px; font-weight:600; color:#fff; }
  #counter .lbl { font-size:9.5px; color:var(--dim); letter-spacing:.25em; }
  #counter .val { font-size:12px; margin-top:6px; color:var(--cyan); }
  #phase { left:32px; bottom:118px; font-size:12px; letter-spacing:.14em;
    color:var(--orange); min-height:16px; transition:color .3s; }
  #controls { position:fixed; z-index:11; left:32px; right:32px; bottom:36px; }
  #spark { width:100%; height:34px; display:block; margin-bottom:14px;
    opacity:.95; }
  .bar { display:flex; align-items:center; gap:18px; }
  #play { width:44px; height:44px; background:none; border:1px solid #3c4043;
    color:#fff; font-size:13px; cursor:pointer; border-radius:2px;
    transition:border-color .15s ease-out; flex:0 0 auto; }
  #play:hover { border-color:var(--orange); }
  #play:focus-visible { outline:2px solid var(--cyan); outline-offset:2px; }
  #scrub { -webkit-appearance:none; appearance:none; flex:1; height:44px;
    background:transparent; cursor:pointer; }
  #scrub::-webkit-slider-runnable-track { height:2px; background:#2a2d31; }
  #scrub::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px;
    border-radius:50%; background:var(--orange); margin-top:-6px;
    box-shadow:0 0 12px var(--orange); }
  #legend { position:fixed; right:32px; bottom:118px; z-index:10; text-align:right;
    font-size:9.5px; color:var(--dim); letter-spacing:.06em; line-height:1.9;
    pointer-events:none; }
  .sw { display:inline-block; width:7px; height:7px; border-radius:50%;
    margin-left:7px; vertical-align:baseline; }
  @media (max-width:640px){ #legend{display:none} #title p{display:none} }
</style>
</head>
<body>
<div id="scene"></div>

<div class="hud" id="title">
  <h1>GERMINATION</h1>
  <p>naissance d'un circuit arithmétique dans pythia-70m<br>
     fine-tune «&nbsp;a+b=c mod 97&nbsp;» · run s41 · données réelles Γ<br>
     7 couches × 4 positions de prompt — l'émissif = croissance fidèle</p>
</div>

<div class="hud" id="counter">
  <div class="lbl">STEP</div>
  <div class="step" id="stepN">0</div>
  <div class="val" id="valN">val 0.0%</div>
</div>

<div class="hud" id="phase"></div>

<div id="legend">
  germe (Γ, croissance cohérente)<span class="sw" style="background:#ff6d00"></span><br>
  généralisation (val_acc)<span class="sw" style="background:#00e5ff"></span><br>
  glisser pour orbiter · espace = lecture
</div>

<div id="controls">
  <canvas id="spark" width="1200" height="34" aria-label="courbe de généralisation"></canvas>
  <div class="bar">
    <button id="play" aria-label="lecture / pause">▶</button>
    <input id="scrub" type="range" min="0" max="7500" value="0" step="10"
           aria-label="position temporelle (steps)">
  </div>
</div>

<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = __DATA__;
const POS = ['a','+','b','='];
const T_END = 7500;

// ---------- interpolation Γ(canal, t) entre ancres
const A = D.anchors;
function gammaAt(ch, t){
  const g = D.gam[ch];
  if (t <= A[0]) return g[0];
  if (t >= A[A.length-1]) return g[g.length-1];
  let i = A.findIndex(a => a > t); const a0 = A[i-1], a1 = A[i];
  const u = (t - a0) / (a1 - a0);
  return g[i-1] * (1-u) + g[i] * u;
}
function valAt(t){
  const tr = D.traj;
  if (t <= tr[0].s) return tr[0].v;
  for (let i=1;i<tr.length;i++) if (tr[i].s >= t){
    const u=(t-tr[i-1].s)/(tr[i].s-tr[i-1].s);
    return tr[i-1].v*(1-u)+tr[i].v*u; }
  return tr[tr.length-1].v;
}
// intensité visuelle : log pour dompter le pic 7.39
const GMAX = Math.max(...Object.values(D.gam).flat());
const vis = g => Math.log1p(g*60) / Math.log1p(GMAX*60);

// ---------- scène
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.028);
const camera = new THREE.PerspectiveCamera(46, innerWidth/innerHeight, .1, 200);
camera.position.set(10.5, 3.2, 13.5);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.getElementById('scene').appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = .06;
controls.autoRotate = !reduced; controls.autoRotateSpeed = .45;
controls.target.set(0, 0, 0);

// la tour : y = couche (0 emb → 6 sortie), x = position du token
const grp = new THREE.Group(); scene.add(grp);
const SX = 2.1, SY = 1.55;
const nodes = {};
const haloTex = (() => {                       // sprite radial émissif
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(64,64,0,64,64,64);
  g.addColorStop(0,'rgba(255,255,255,1)'); g.addColorStop(.25,'rgba(255,255,255,.55)');
  g.addColorStop(1,'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0,0,128,128);
  return new THREE.CanvasTexture(c);
})();
const geoN = new THREE.SphereGeometry(.10, 20, 20);
for (let L=0; L<7; L++) for (let p=0; p<4; p++){
  const x = (p-1.5)*SX, y = (L-3)*SY;
  const core = new THREE.Mesh(geoN,
    new THREE.MeshBasicMaterial({color:0x17181b}));
  core.position.set(x,y,0); grp.add(core);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map:haloTex, color:0xff6d00, transparent:true, opacity:0,
    blending:THREE.AdditiveBlending, depthWrite:false}));
  halo.position.set(x,y,0); halo.scale.set(.1,.1,1); grp.add(halo);
  nodes[`L${L}_p${p}`] = {core, halo};
}
// arêtes verticales discrètes (flux inter-couches)
const eMat = new THREE.LineBasicMaterial({color:0x1d2024, transparent:true, opacity:.8});
for (let p=0;p<4;p++){
  const pts=[]; for(let L=0;L<7;L++) pts.push(new THREE.Vector3((p-1.5)*SX,(L-3)*SY,0));
  grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), eMat));
}
// étiquettes tokens (sprites canvas)
function label(txt, x, y, size=120){
  const c=document.createElement('canvas'); c.width=256; c.height=128;
  const g=c.getContext('2d'); g.font='600 64px ui-monospace,Menlo,monospace';
  g.fillStyle='#5f6368'; g.textAlign='center'; g.fillText(txt,128,86);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),
    transparent:true, depthWrite:false}));
  s.scale.set(1.6,.8,1); s.position.set(x,y,0); grp.add(s);
}
for (let p=0;p<4;p++) label(POS[p], (p-1.5)*SX, -3*SY-1.15);
label('sortie →', -1.5*SX-1.9, 3*SY, 90);
label('embedding →', -1.5*SX-2.25, -3*SY, 90);

const colGerm = new THREE.Color(0xff6d00), colHot = new THREE.Color(0xffffff),
      colCyan = new THREE.Color(0x00e5ff), tmp = new THREE.Color();

// ---------- phases narratives
const PHASES = [
  [0,    'mémorisation — le réseau apprend par cœur'],
  [D.t_mem, 'plateau — silence extérieur, val_acc ≈ 0'],
  [6200, 'GERMINATION — couche 3, token « + »'],
  [6750, 'propagation vers les opérandes'],
  [7050, 'descente vers la sortie'],
  [D.t_grok, 'BASCULE — généralisation'],
  [D.t_grok+150, 'consolidation'],
];
function phaseAt(t){ let s=PHASES[0][1];
  for (const [a,b] of PHASES) if (t>=a) s=b; return s; }

// ---------- HUD + sparkline
const elStep=document.getElementById('stepN'), elVal=document.getElementById('valN'),
      elPhase=document.getElementById('phase'), spark=document.getElementById('spark'),
      sx=spark.getContext('2d'), scrub=document.getElementById('scrub'),
      btn=document.getElementById('play');
function drawSpark(t){
  const w=spark.width, h=spark.height;
  sx.clearRect(0,0,w,h);
  sx.strokeStyle='#2a2d31'; sx.beginPath();
  const xg=w*D.t_grok/T_END; sx.moveTo(xg,0); sx.lineTo(xg,h); sx.stroke();
  sx.strokeStyle='#00e5ff'; sx.lineWidth=1.6; sx.beginPath();
  for(let x=0;x<=w*t/T_END;x+=2){
    const v=valAt(x/w*T_END);
    const y=h-2-v*(h-5);
    x===0 ? sx.moveTo(x,y) : sx.lineTo(x,y);
  }
  sx.stroke();
  // Γ max (orange, sous la courbe)
  sx.strokeStyle='rgba(255,109,0,.85)'; sx.lineWidth=1.2; sx.beginPath();
  for(let x=0;x<=w*t/T_END;x+=2){
    let m=0; const tt=x/w*T_END;
    for(const ch in D.gam) m=Math.max(m,gammaAt(ch,tt));
    const y=h-2-vis(m)*(h-5);
    x===0 ? sx.moveTo(x,y) : sx.lineTo(x,y);
  }
  sx.stroke();
}

// ---------- boucle temporelle (interruptible)
let t = 0, playing = !reduced, grokFlash = 0;
const SPEED = T_END / 26;          // film ≈ 26 s
btn.textContent = playing ? '❚❚' : '▶';
btn.onclick = () => { playing = !playing; btn.textContent = playing?'❚❚':'▶'; };
addEventListener('keydown', e => { if(e.code==='Space'){e.preventDefault();btn.onclick();} });
scrub.oninput = () => { t = +scrub.value; playing = false; btn.textContent='▶'; };

let last = performance.now();
function frame(now){
  requestAnimationFrame(frame);
  const dt = Math.min((now-last)/1000, .1); last = now;
  if (playing){ t += dt*SPEED; if (t >= T_END){ t = T_END; playing=false; btn.textContent='▶'; } }
  scrub.value = t;

  const v = valAt(t);
  const post = t >= D.t_grok;
  if (post && grokFlash < 1) grokFlash = Math.min(1, grokFlash + dt*2.2);

  for (const ch in nodes){
    const I = vis(gammaAt(ch, t));
    const {core, halo} = nodes[ch];
    tmp.copy(colGerm).lerp(colHot, Math.max(0, I-.55)*2.2);
    if (post) tmp.lerp(colCyan, grokFlash*.55*v);
    halo.material.color.copy(tmp);
    halo.material.opacity = Math.min(1, I*1.25 + (post? .05*v : 0));
    const s = .25 + I*3.4 + (post? .18*v : 0);
    halo.scale.set(s, s, 1);
    core.material.color.setScalar(.09 + I*.9);
    if (post) core.material.color.lerp(colCyan, .25*v*grokFlash);
  }

  elStep.textContent = Math.round(t);
  elVal.textContent = 'val ' + (v*100).toFixed(1) + '%';
  const ph = phaseAt(t);
  if (elPhase.textContent !== ph){
    elPhase.textContent = ph;
    elPhase.style.color = ph.startsWith('BASCULE') ? '#00e5ff'
      : ph.startsWith('GERMINATION') ? '#ff6d00'
      : ph==='consolidation' ? '#e040fb' : '#9aa0a6';
  }
  drawSpark(t);
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(frame);

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>
"""

out = HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
dst = os.path.join(HERE, "germination.html")
open(dst, "w").write(out)
print(f"→ {dst}  ({len(out)//1024} KiB)")
