import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const ACTION_COLORS = {
  SCREEN: 0x1f6fd1, COUNSEL: 0x8e44ad, SUPPLEMENT: 0xf2c94c,
  REFER: 0xd9452b, RESTOCK: 0x16a085, RECORD_KEEPING: 0x7f8c8d,
};
const HOUSE_RADIUS = 12;
const HOUSE_ANGLES = [-30, 15, 60, 120, 165, 210];
const PILLAR_MAX = 4.2;
const HEALTH_CENTRE = new THREE.Vector3(0, 0, -16);
const BASE = new THREE.Vector3(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById("scene").appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fc9e8);
scene.fog = new THREE.Fog(0x9fc9e8, 50, 110);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 300);
camera.position.set(-2, 25, 31);
const controls = new OrbitControls(camera, renderer.domElement);
// the log panel covers the right of the screen, so aim a little to the right of the village
controls.target.set(3, 0, -2);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.46;
controls.minDistance = 10;
controls.maxDistance = 80;

scene.add(new THREE.HemisphereLight(0xe4f1ff, 0x5d7a3a, 0.9));
const sun = new THREE.DirectionalLight(0xfff1d6, 1.7);
sun.position.set(20, 32, 14);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -32, right: 32, top: 32, bottom: -32, far: 90 });
scene.add(sun);

function mat(color, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.85, ...extra });
}

function mesh(geometry, material, { cast = true, receive = false } = {}) {
  const m = new THREE.Mesh(geometry, material);
  m.castShadow = cast;
  m.receiveShadow = receive;
  return m;
}

let seed = 7;
function rand() {
  seed = (seed * 16807) % 2147483647;
  return seed / 2147483647;
}

function buildGround() {
  const geo = new THREE.PlaneGeometry(220, 220, 110, 110);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = [];
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const ramp = THREE.MathUtils.smoothstep(Math.hypot(x, z), 22, 45);
    const h = ramp * (7 + 5 * Math.sin(x * 0.08) + 4 * Math.cos(z * 0.07 + 1) + 2 * Math.sin((x + z) * 0.2));
    pos.setY(i, h);
    // alternating shades give the hills a terraced look
    const terrace = Math.floor(h * 1.5) % 2 === 0 ? 0 : 0.04;
    c.setHSL(0.27 + terrace * 0.3, 0.45, 0.36 + terrace + rand() * 0.03);
    colors.push(c.r, c.g, c.b);
  }
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  scene.add(mesh(geo, mat(0xffffff, { vertexColors: true, flatShading: true, roughness: 1 }), { cast: false, receive: true }));
}

function buildPath(from, to) {
  const len = from.distanceTo(to);
  const path = mesh(new THREE.BoxGeometry(1.1, 0.04, len), mat(0xb08a5a, { roughness: 1 }), { cast: false, receive: true });
  path.position.copy(from).add(to).multiplyScalar(0.5);
  path.position.y = 0.02;
  path.lookAt(to.x, 0.02, to.z);
  scene.add(path);
}

function textSprite(lines, { width = 512, height = 200, scale = 5, bg = "rgba(255,255,255,0.92)" } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
  sprite.scale.set(scale, scale * height / width, 1);
  sprite.renderOrder = 10;
  sprite.userData.draw = (newLines, accent = "#1d2430") => {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.roundRect(4, 4, width - 8, height - 8, 22);
    ctx.fill();
    ctx.fillStyle = accent;
    ctx.fillRect(4, 4, 14, height - 8);
    ctx.textAlign = "center";
    newLines.forEach((line, k) => {
      ctx.fillStyle = k === 0 ? "#1d2430" : "#4a5566";
      ctx.font = k === 0 ? "bold 66px Segoe UI, Arial" : "bold 50px Segoe UI, Arial";
      ctx.fillText(line, width / 2 + 6, 80 + k * 76);
    });
    texture.needsUpdate = true;
  };
  sprite.userData.draw(lines);
  return sprite;
}

function hazColor(haz) {
  if (haz < -3) return 0x7a1010;
  if (haz < -2) return 0xd9452b;
  if (haz < -1) return 0xe3a72f;
  return 0x3fa34d;
}

function hazHeight(haz) {
  return THREE.MathUtils.clamp((haz + 4) / 5, 0.04, 1) * PILLAR_MAX;
}

function buildHouse(i) {
  const angle = THREE.MathUtils.degToRad(HOUSE_ANGLES[i]);
  const group = new THREE.Group();
  group.position.set(HOUSE_RADIUS * Math.cos(angle), 0, HOUSE_RADIUS * Math.sin(angle));
  group.lookAt(0, 0, 0);
  scene.add(group);

  const walls = mesh(new THREE.BoxGeometry(2.6, 1.7, 2.2), mat(0xb5652e), { receive: true });
  walls.position.y = 0.85;
  const roof = mesh(new THREE.ConeGeometry(2.25, 1.1, 4), mat(0x9aa3ab, { metalness: 0.55, roughness: 0.45 }));
  roof.rotation.y = Math.PI / 4;
  roof.position.y = 2.25;
  const door = mesh(new THREE.BoxGeometry(0.6, 1.1, 0.06), mat(0x4a2c17));
  door.position.set(0, 0.55, 1.11);
  const win = mesh(new THREE.BoxGeometry(0.5, 0.45, 0.06), mat(0x2b3a4a));
  win.position.set(-0.8, 1.05, 1.11);
  group.add(walls, roof, door, win);

  const pillarGroup = new THREE.Group();
  pillarGroup.position.set(1.9, 0, 1.0);
  group.add(pillarGroup);
  const pillarGeo = new THREE.CylinderGeometry(0.33, 0.33, 1, 20);
  pillarGeo.translate(0, 0.5, 0);
  const pillar = mesh(pillarGeo, mat(0x3fa34d, { roughness: 0.5 }));
  const ring = mesh(new THREE.TorusGeometry(0.55, 0.07, 10, 32), mat(0xffffff, { emissive: 0xffffff, emissiveIntensity: 0.5 }), { cast: false });
  ring.rotation.x = Math.PI / 2;
  const threshold = mesh(new THREE.TorusGeometry(0.46, 0.03, 6, 32), mat(0x222222), { cast: false });
  threshold.rotation.x = Math.PI / 2;
  threshold.position.y = hazHeight(-2);
  const plinth = mesh(new THREE.CylinderGeometry(0.5, 0.55, 0.12, 20), mat(0x6b6b6b), { receive: true });
  pillarGroup.add(pillar, ring, threshold, plinth);

  const crate = mesh(new THREE.BoxGeometry(0.6, 0.5, 0.5), mat(0xf2c94c));
  crate.position.set(-1.8, 0.25, 1.0);
  const sick = mesh(new THREE.SphereGeometry(0.28, 16, 12), mat(0xff3b3b, { emissive: 0xff0000, emissiveIntensity: 0.6 }));
  sick.position.set(-0.5, 3.4, 0);
  const cross = new THREE.Group();
  const crossMat = mat(0x2f80ed, { emissive: 0x2f80ed, emissiveIntensity: 0.4 });
  cross.add(mesh(new THREE.BoxGeometry(0.7, 0.22, 0.12), crossMat), mesh(new THREE.BoxGeometry(0.22, 0.7, 0.12), crossMat));
  cross.position.set(0.5, 3.4, 0);
  group.add(crate, sick, cross);

  const label = textSprite([`Child ${i + 1}`, ""]);
  label.position.set(0, 5.2, 0);
  group.add(label);

  const front = group.position.clone().multiplyScalar((HOUSE_RADIUS - 2.6) / HOUSE_RADIUS);
  return { group, pillar, ring, crate, sick, cross, label, front };
}

function buildHealthCentre() {
  const g = new THREE.Group();
  g.position.copy(HEALTH_CENTRE);
  const body = mesh(new THREE.BoxGeometry(8, 3, 4), mat(0xf4f4f0), { receive: true });
  body.position.y = 1.5;
  const roof = mesh(new THREE.BoxGeometry(8.6, 0.3, 4.6), mat(0x5b8db8));
  roof.position.y = 3.15;
  const crossMat = mat(0xd9452b, { emissive: 0xd9452b, emissiveIntensity: 0.3 });
  const c1 = mesh(new THREE.BoxGeometry(1.2, 0.36, 0.08), crossMat);
  const c2 = mesh(new THREE.BoxGeometry(0.36, 1.2, 0.08), crossMat);
  c1.position.set(0, 2.1, 2.04);
  c2.position.set(0, 2.1, 2.04);
  const door = mesh(new THREE.BoxGeometry(1.2, 1.6, 0.08), mat(0x2b3a4a));
  door.position.set(-2.4, 0.8, 2.04);
  g.add(body, roof, c1, c2, door);
  const label = textSprite(["Health Centre", "supplies & referrals"], { scale: 5.5 });
  label.position.set(0, 5.6, 0);
  g.add(label);
  scene.add(g);
}

function buildBase() {
  const g = new THREE.Group();
  const body = mesh(new THREE.BoxGeometry(2.2, 1.4, 1.8), mat(0xe8dcc0), { receive: true });
  body.position.y = 0.7;
  const roof = mesh(new THREE.ConeGeometry(1.8, 0.9, 4), mat(0x2f6f9f));
  roof.rotation.y = Math.PI / 4;
  roof.position.y = 1.85;
  g.add(body, roof);
  g.position.set(BASE.x, 0, BASE.z - 1.6);
  const label = textSprite(["CHW base", "records"], { scale: 3.6 });
  label.position.set(0, 3.6, 0);
  g.add(label);
  scene.add(g);
}

function buildTrees() {
  const trunkMat = mat(0x6b4f2a);
  const leafMat = mat(0x3f7f2f, { side: THREE.DoubleSide });
  for (let k = 0; k < 55; k++) {
    const r = 19 + rand() * 18;
    const a = rand() * Math.PI * 2;
    const x = r * Math.cos(a), z = r * Math.sin(a);
    if (Math.abs(x) < 6 && z < -10) continue;
    const tree = new THREE.Group();
    const h = 1.6 + rand() * 1.2;
    const trunk = mesh(new THREE.CylinderGeometry(0.12, 0.18, h, 6), trunkMat);
    trunk.position.y = h / 2;
    tree.add(trunk);
    // banana plants: a fan of long drooping leaves
    for (let l = 0; l < 6; l++) {
      const leaf = mesh(new THREE.PlaneGeometry(0.5, 1.8), leafMat);
      leaf.position.y = h;
      leaf.rotation.set(-0.9 - rand() * 0.4, (l / 6) * Math.PI * 2, 0, "YXZ");
      leaf.translateY(0.8);
      tree.add(leaf);
    }
    tree.position.set(x, 0, z);
    scene.add(tree);
  }
}

function buildChw() {
  const g = new THREE.Group();
  const body = mesh(new THREE.CylinderGeometry(0.3, 0.38, 1.1, 14), mat(0x0f9d8a));
  body.position.y = 0.75;
  const head = mesh(new THREE.SphereGeometry(0.27, 16, 12), mat(0x5a3a22));
  head.position.y = 1.55;
  const bag = mesh(new THREE.BoxGeometry(0.45, 0.5, 0.25), mat(0x8a5a2b));
  bag.position.set(0, 0.85, -0.35);
  const marker = mesh(new THREE.ConeGeometry(0.3, 0.55, 12), mat(0xffd400, { emissive: 0xffaa00, emissiveIntensity: 0.6 }), { cast: false });
  marker.rotation.x = Math.PI;
  marker.position.y = 2.4;
  g.add(body, head, bag, marker);
  g.userData.marker = marker;
  g.position.copy(BASE);
  scene.add(g);
  return g;
}

buildGround();
HOUSE_ANGLES.forEach((_, i) => {
  const a = THREE.MathUtils.degToRad(HOUSE_ANGLES[i]);
  buildPath(BASE, new THREE.Vector3(HOUSE_RADIUS * Math.cos(a), 0, HOUSE_RADIUS * Math.sin(a)));
});
buildPath(BASE, HEALTH_CENTRE);
const houses = HOUSE_ANGLES.map((_, i) => buildHouse(i));
buildHealthCentre();
buildBase();
buildTrees();
const chw = buildChw();

const pulses = [];
function pulse(position, color) {
  const ring = mesh(new THREE.RingGeometry(0.8, 1.1, 40), new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide }), { cast: false });
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(position.x, 0.08, position.z);
  scene.add(ring);
  pulses.push({ ring, t: 0 });
}

const arcs = [];
function referralArc(from) {
  const mid = from.clone().add(HEALTH_CENTRE).multiplyScalar(0.5);
  mid.y = 7;
  const curve = new THREE.QuadraticBezierCurve3(from.clone().setY(0.5), mid, HEALTH_CENTRE.clone().setY(2));
  const geo = new THREE.TubeGeometry(curve, 40, 0.09, 8);
  const tube = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xd9452b, transparent: true }));
  scene.add(tube);
  arcs.push({ tube, t: 0 });
}

let walk = null;
function walkTo(target, duration) {
  walk = { from: chw.position.clone(), to: target.clone(), t: 0, duration };
  chw.lookAt(target.x, 0, target.z);
}

const $ = (id) => document.getElementById(id);
const entries = $("entries");
const chart = $("chart");
let rewards = [];
let cumulative = 0;

function drawChart(maxSteps) {
  const ctx = chart.getContext("2d");
  const w = chart.width, h = chart.height, pad = 6;
  ctx.clearRect(0, 0, w, h);
  const lo = Math.min(-20, ...rewards), hi = Math.max(20, ...rewards);
  const y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad);
  ctx.strokeStyle = "#c5ccd6";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, y(0));
  ctx.lineTo(w, y(0));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#5b6575";
  ctx.font = "11px Segoe UI, Arial";
  ctx.fillText(hi.toFixed(0), 2, 12);
  ctx.fillText(lo.toFixed(0), 2, h - 4);
  if (rewards.length < 2) return;
  ctx.strokeStyle = cumulative >= 0 ? "#2e8540" : "#c0392b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rewards.forEach((v, k) => {
    const x = (k / maxSteps) * w;
    k === 0 ? ctx.moveTo(x, y(v)) : ctx.lineTo(x, y(v));
  });
  ctx.stroke();
}

function addEntry(f, kind) {
  const li = document.createElement("li");
  if (f.wasted) li.classList.add("wasted");
  const week = Math.floor((f.step - 1) / 3) + 1, day = ((f.step - 1) % 3) + 1;
  const r = f.reward ?? 0;
  li.innerHTML = `<span class="when">W${week} D${day}</span><span class="what a-${kind}">${f.action_name}${f.wasted ? " (wasted)" : ""}</span>` +
    `<span class="r ${r >= 0 ? "pos" : "neg"}">${r >= 0 ? "+" : ""}${r.toFixed(1)}</span>`;
  entries.prepend(li);
  for (const e of f.events) {
    const ev = document.createElement("li");
    ev.className = "event";
    ev.textContent = e;
    li.after(ev);
  }
  while (entries.children.length > 120) entries.lastChild.remove();
}

function applyFrame(f, duration) {
  const maxSteps = f.n_weeks * 3;
  if (f.action === null) {
    rewards = [0];
    cumulative = 0;
    entries.innerHTML = "";
    chw.position.copy(BASE);
    walk = null;
    $("current-action").textContent = "New caseload registered";
    $("current-action").className = "";
  } else {
    cumulative += f.reward;
    rewards.push(cumulative);
    const kind = f.action_name.split(" ")[0];
    const child = f.action_name.includes("child") ? parseInt(f.action_name.split(" ").pop(), 10) - 1 : null;
    let target = BASE;
    if (kind === "RESTOCK") target = HEALTH_CENTRE.clone().add(new THREE.Vector3(-2.4, 0, 3));
    else if (child !== null) target = houses[child].front;
    walkTo(target, duration);
    pulse(target, f.wasted ? 0x9aa0a6 : ACTION_COLORS[kind]);
    if (kind === "REFER" && !f.wasted) referralArc(houses[child].group.position);
    $("current-action").textContent = f.action_name + (f.wasted ? "  (wasted)" : "");
    $("current-action").className = `a-${kind}`;
    addEntry(f, kind);
  }

  houses.forEach((h, i) => {
    const haz = f.true_haz[i];
    const color = hazColor(haz);
    h.pillar.scale.y = hazHeight(haz);
    h.pillar.material.color.setHex(color);
    h.ring.position.y = hazHeight(f.measured_haz[i]);
    h.crate.visible = f.supplement_weeks[i] > 0;
    h.sick.visible = f.ill[i];
    h.cross.visible = f.treatment_weeks[i] > 0;
    const accent = "#" + color.toString(16).padStart(6, "0");
    h.label.userData.draw([`Child ${i + 1}`, `${f.age_months[i].toFixed(0)} mo · risk ${f.risk_score[i].toFixed(2)}`], accent);
  });

  $("scenario").textContent = `scenario: ${f.scenario.replace("_", " ")}`;
  const started = f.step > 0;
  $("week").textContent = `${started ? Math.floor((f.step - 1) / 3) + 1 : 1} / ${f.n_weeks}`;
  $("day").textContent = started ? `${((f.step - 1) % 3) + 1} / 3` : "-";
  $("stock").textContent = f.stock;
  $("referrals").textContent = f.referrals_left;
  $("stunted").textContent = `${f.n_stunted} / ${f.n_children}`;
  $("return").textContent = cumulative.toFixed(1);
  drawChart(maxSteps);
}

const queue = [];
let nextFrame = 0;
let lastApplied = 0;

async function poll() {
  try {
    const res = await fetch(`/frames?since=${nextFrame}`, { cache: "no-store" });
    const frames = await res.json();
    nextFrame += frames.length;
    queue.push(...frames);
    setTimeout(poll, 150);
  } catch {
    $("current-action").textContent = "Simulation finished";
  }
}
poll();

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  const now = performance.now();

  // speed up if the agent is running ahead of the animation
  const duration = queue.length > 3 ? 120 : 450;
  if (queue.length && now - lastApplied > duration) {
    applyFrame(queue.shift(), duration);
    lastApplied = now;
  }

  if (walk) {
    walk.t = Math.min(1, walk.t + (dt * 1000) / walk.duration);
    const e = walk.t * walk.t * (3 - 2 * walk.t);
    chw.position.lerpVectors(walk.from, walk.to, e);
    chw.position.y = Math.abs(Math.sin(walk.t * Math.PI * 4)) * 0.15;
    if (walk.t >= 1) walk = null;
  }
  chw.userData.marker.position.y = 2.4 + Math.sin(now / 300) * 0.12;
  houses.forEach((h) => { h.sick.position.y = 3.4 + Math.sin(now / 250) * 0.15; });

  for (let k = pulses.length - 1; k >= 0; k--) {
    const p = pulses[k];
    p.t += dt;
    p.ring.scale.setScalar(1 + p.t * 2.2);
    p.ring.material.opacity = Math.max(0, 1 - p.t / 1.2);
    if (p.t > 1.2) { scene.remove(p.ring); pulses.splice(k, 1); }
  }
  for (let k = arcs.length - 1; k >= 0; k--) {
    const a = arcs[k];
    a.t += dt;
    a.tube.material.opacity = Math.max(0, 1 - a.t / 2.5);
    if (a.t > 2.5) { scene.remove(a.tube); arcs.splice(k, 1); }
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
