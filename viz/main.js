// EDGE OF CHAOS — viz émergence. Three.js r128, Coupled Map Lattice live + bloom.
// Dégradation gracieuse : HUD + courbes fonctionnent même si WebGL indisponible.

const L = 96;
let r = 3.70, eps = 0.30;
let state = new Float32Array(L * L);
let next = new Float32Array(L * L);
let ref = null;

for (let i = 0; i < L * L; i++) state[i] = Math.random();

function stepCML() {
  const f = (x) => r * x * (1 - x);
  for (let y = 0; y < L; y++) {
    const yu = ((y - 1 + L) % L) * L, yd = ((y + 1) % L) * L, yc = y * L;
    for (let x = 0; x < L; x++) {
      const xl = (x - 1 + L) % L, xr = (x + 1) % L;
      const fc = f(state[yc + x]);
      const nb = f(state[yu + x]) + f(state[yd + x]) + f(state[yc + xl]) + f(state[yc + xr]);
      next[yc + x] = (1 - eps) * fc + eps * 0.25 * nb;
    }
  }
  const t = state; state = next; next = t;
}

const C0 = [0.0, 0.55, 0.85], C1 = [1.0, 0.17, 0.84], C2 = [1.0, 0.83, 0.0];
function colormap(v, out) {
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  let a, b, t;
  if (v < 0.5) { a = C0; b = C1; t = v / 0.5; } else { a = C1; b = C2; t = (v - 0.5) / 0.5; }
  const g = 0.35 + 0.65 * v;
  out.setRGB((a[0] + (b[0] - a[0]) * t) * g, (a[1] + (b[1] - a[1]) * t) * g, (a[2] + (b[2] - a[2]) * t) * g);
}

// ---------- HUD (toujours actif) ----------
function getCSS(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
const rVal = document.getElementById('rVal'), eVal = document.getElementById('eVal');
const lyapVal = document.getElementById('lyapVal'), synVal = document.getElementById('synVal'), dimVal = document.getElementById('dimVal');
const regimeTag = document.getElementById('regime-tag');
const cc = document.getElementById('curveCanvas'), cx = cc.getContext('2d');

function interp(arr, rr) {
  if (!ref || !arr) return null;
  const rs = ref.r, n = rs.length;
  if (rr <= rs[0]) return arr[0];
  if (rr >= rs[n - 1]) return arr[n - 1];
  let i = 0; while (i < n - 1 && rs[i + 1] < rr) i++;
  const t = (rr - rs[i]) / (rs[i + 1] - rs[i]);
  return arr[i] + (arr[i + 1] - arr[i]) * t;
}

function setRegime(syn) {
  let label, color, glow = false;
  if (syn === null) { label = '— —'; color = '#6b7a99'; }
  else if (syn < -0.05) { label = 'ORDRE · redondance'; color = getCSS('--cyan'); }
  else if (syn > 0.05) { label = '◆ CHAOS · ÉMERGENCE'; color = getCSS('--magenta'); glow = true; }
  else { label = '◆ BASCULE'; color = getCSS('--yellow'); glow = true; }
  regimeTag.textContent = label; regimeTag.style.color = color;
  regimeTag.style.boxShadow = glow ? `0 0 18px ${color}` : 'none';
}

function updateHUD() {
  rVal.textContent = r.toFixed(3); eVal.textContent = eps.toFixed(2);
  const ly = interp(ref && ref.lyapunov, r), om = interp(ref && ref.omega, r), dm = interp(ref && ref.dim_eff, r);
  lyapVal.textContent = ly === null ? '—' : (ly >= 0 ? '+' : '') + ly.toFixed(3);
  synVal.textContent = om === null ? '—' : ((-om) >= 0 ? '+' : '') + (-om).toFixed(3);
  dimVal.textContent = dm === null ? '—' : dm.toFixed(0);
  setRegime(om === null ? null : -om);
}

function drawCurves() {
  const w = cc.width = cc.clientWidth * 2, h = cc.height = cc.clientHeight * 2;
  cx.clearRect(0, 0, w, h);
  if (!ref) { cx.fillStyle = '#6b7a99'; cx.font = '20px JetBrains Mono'; cx.fillText('chargement…', 12, h / 2); return; }
  const rs = ref.r, n = rs.length, rmin = ref.r_min, rmax = ref.r_max;
  const X = (rr) => (rr - rmin) / (rmax - rmin) * w;
  const norm = (arr) => { const mn = Math.min(...arr), mx = Math.max(...arr); return arr.map(v => (v - mn) / (mx - mn + 1e-9)); };
  const drawLine = (arr, color) => {
    const nn = norm(arr); cx.strokeStyle = color; cx.lineWidth = 2; cx.beginPath();
    for (let i = 0; i < n; i++) { const px = X(rs[i]), py = h - 8 - nn[i] * (h - 16); i ? cx.lineTo(px, py) : cx.moveTo(px, py); }
    cx.stroke();
  };
  drawLine(ref.lyapunov, getCSS('--cyan'));
  drawLine(ref.omega.map(v => -v), getCSS('--magenta'));
  const mx = X(r); cx.strokeStyle = getCSS('--yellow'); cx.lineWidth = 2; cx.beginPath();
  cx.moveTo(mx, 0); cx.lineTo(mx, h); cx.stroke();
}

// contrôles
const rSlider = document.getElementById('rSlider'), eSlider = document.getElementById('eSlider');
rSlider.addEventListener('input', e => { r = +e.target.value; updateHUD(); drawCurves(); });
eSlider.addEventListener('input', e => { eps = +e.target.value; updateHUD(); });
document.querySelectorAll('.preset').forEach(b => b.addEventListener('click', () => {
  r = +b.dataset.r; rSlider.value = r; updateHUD(); drawCurves();
}));

fetch('data.json').then(x => x.json()).then(d => { ref = d; updateHUD(); drawCurves(); })
  .catch(() => { updateHUD(); drawCurves(); });
updateHUD(); drawCurves();

// ---------- Three.js (optionnel, dégradation gracieuse) ----------
let webglOK = true, composer = null, mesh = null;
const col = new THREE.Color();
try {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  document.getElementById('app').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05060a);
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  cam.position.z = 2;

  const geo = new THREE.PlaneGeometry(0.92, 0.92);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  mesh = new THREE.InstancedMesh(geo, mat, L * L);
  const dummy = new THREE.Object3D();
  for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
    const i = y * L + x;
    dummy.position.set(x - L / 2 + 0.5, y - L / 2 + 0.5, 0); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    colormap(state[i], col); mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);

  composer = new THREE.EffectComposer(renderer);
  composer.addPass(new THREE.RenderPass(scene, cam));
  composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(1, 1), 1.25, 0.55, 0.0));

  const resize = () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h); composer.setSize(w, h);
    const aspect = w / h, span = L / 2 + 1;
    if (aspect >= 1) { cam.left = -span * aspect; cam.right = span * aspect; cam.top = span; cam.bottom = -span; }
    else { cam.left = -span; cam.right = span; cam.top = span / aspect; cam.bottom = -span / aspect; }
    cam.updateProjectionMatrix();
  };
  window.addEventListener('resize', resize); resize();
} catch (e) {
  webglOK = false;
  console.warn('WebGL indisponible — grille désactivée, HUD actif', e && e.message);
}

let frame = 0;
function loop() {
  requestAnimationFrame(loop);
  if (!webglOK) return;
  stepCML();
  if ((frame++ & 1) === 0) {
    for (let i = 0; i < L * L; i++) { colormap(state[i], col); mesh.setColorAt(i, col); }
    mesh.instanceColor.needsUpdate = true;
  }
  composer.render();
}
loop();
