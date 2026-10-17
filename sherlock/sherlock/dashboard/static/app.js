"use strict";

/* Sherlock board. Everything on it comes from the runtime: /events (stage events
   plus `event: stats`), the MJPEG streams, /reports, /cameras and /mode. The
   labels default to the mock runtime and switch to the live models only when
   /mode says mock=false. Every frame the Scout counts becomes one particle. */
const $ = id => document.getElementById(id);

const MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0.35 : 1;

const C = { accent: "#f2c14e", ok: "#8fe0a4", warn: "#ff9c62", err: "#ff6f6a", face: "#b4b9ff", dim: "#a3aab6" };

const CAMS = ["hero", "scout1"];

// The Scout card names the detector that runs (/mode scout_backend) in either mode:
// a synthetic scene is read as scene truth even when the models are live.
const SCOUT = {
  scene: ["scene truth", "synthetic scene · no detector runs"],
  yolo: ["YOLO", "measured per frame · beside the cameras"],
  motion: ["Motion detector", "background subtraction · no model"],
};

const LABELS = {
  mock: {
    badge: ["MOCK", "synthetic scene · rule director · canned VLM"],
    scout: SCOUT.scene,
    director: ["Rule director", "deterministic policy · no LLM"],
    vlm: ["Canned analyst + verifier", "recorded answers · no model"],
    fail: "Simulate: break the next tool call",
    cover: "Simulate: cover hero lens",
  },
  live: {
    badge: ["LIVE", "model servers not identified"],
    scout: ["Scout", "detector not reported"],
    director: ["Nemotron 3.5 Lightning", "30B total · 3B active · tool loop"],
    vlm: ["SmolVLM 500M", "analyst + verifier · one shared instance"],
    fail: "Break the next tool call",
    cover: "Cover hero lens",
  },
};

let MODE = "mock", L = LABELS.mock;

const COUNT = ["no", "one", "two", "three"];

// The badge uses the server's own labels (/mode servers): a runtime is named only
// when its server said so or MODEL_SERVER declares it; otherwise the host is shown.
function liveBadge(servers) {
  if (!servers) return LABELS.live.badge[1];

  const d = servers.director, a = servers.analyst, v = servers.verifier, n = servers.endpoints;

  if (d.name && [a, v].every(s => s.name === d.name)) return `${d.name} · ${COUNT[n] || n} model server${n === 1 ? "" : "s"}`;

  const vlm = a.label === v.label ? a.label : `${a.label} + ${v.label}`;

  return `Director on ${d.label} · VLM on ${vlm}`;
}

function setBadge([tag, text]) {
  const b = document.createElement("b");

  b.textContent = tag;
  $("mode").replaceChildren(b, ` · ${text}`);
}

function applyMode(m) {
  MODE = m && m.mock === false ? "live" : "mock";
  L = { ...LABELS[MODE] };

  if (m && SCOUT[m.scout_backend]) L.scout = SCOUT[m.scout_backend];

  if (MODE === "live") {
    // The table's names belong to the default models; a swapped model shows its own id.
    const base = id => String(id || "").split("/").pop();

    if (m.director_model && !/Nemotron-3\.5-Lightning/i.test(m.director_model)) L.director = [base(m.director_model), "tool loop"];

    if (m.vlm_model && base(m.vlm_model) !== "SmolVLM-500M-Instruct") L.vlm = [base(m.vlm_model), L.vlm[1]];

    if (m.servers && !m.servers.vlm_shared) L.vlm = [L.vlm[0], "analyst + verifier · separate instances"];

    L.badge = [L.badge[0], liveBadge(m.servers)];
  }

  document.body.dataset.mode = MODE;
  setBadge(L.badge);

  for (const [k, n, meta] of [["scout", "scout-name", "scout-meta"], ["director", "director-name", "director-meta"], ["vlm", "vlm-name", "vlm-meta"]]) {
    $(n).textContent = L[k][0];
    $(meta).textContent = L[k][1];
  }

  $("btn-fail").textContent = L.fail;
  setCovered("hero", covered.hero);
}

let stats = {
  frames: 0, scout_hits: 0, vlm_calls: 0, tool_errors: 0, recoveries: 0, confirmed: 0,
  alerts: 0, ignored: 0, logged: 0, tool_calls: 0,
};

let lastAlert = null, feedPaused = false, connected = false;

const covered = { hero: false, scout1: false };

// Cameras that did not open at startup (name -> reason), from /cameras.
const unavailable = {};

// The hero's own warning from /cameras (PTZ offline, tracking not switched off, gimbal
// moving on its own): its tile says so instead of a plain "PTZ" it cannot deliver.
let heroWarning = null;

const heroNote = () => covered.hero ? "· LENS COVERED" : heroWarning ? `· ${heroWarning.label}` : "· PTZ";

const live = c => !covered[c] && !(c in unavailable);

const feedBuffer = [];

// Analyst rows wait for the report, which shows their words or withholds them when they name
// another object than the verified claim. Shown on arrival, "Cake." sat under a knife alert.
const READING = "reading the crop…", WITHHELD = "named another object in the crop · not shown";

const held = []; // { ev, said, li } per Analyst row the next report settles

/* ----------------------------------------------------------------- feed */
const feed = $("feed");

function addRow(ev) {
  const d = ev.data || {};

  // Presenter feed: no planner echoes, no narration (it lives in the Director card),
  // no duplicate of the REPORT row, no second sighting (the system row says who handled it).
  if (ev.stage === "narration" || (ev.stage === "director" && ev.msg.startsWith("->"))) return;

  if (ev.stage === "tool" && d.tool === "report") return;

  if (ev.stage === "scout" && d.handled_via) return;

  if (feedPaused) {
    feedBuffer.push(ev);

    return;
  }

  const li = document.createElement("li");

  li.dataset.stage = ev.stage;

  if (ev.msg.startsWith("REPORT")) {
    li.dataset.kind = "report";
    li.dataset.action = d.action || "";
  } else if (ev.stage === "scout" && d.threat === "weapon") li.dataset.kind = "weapon";
  else if (ev.stage === "tool" && d.error) li.dataset.kind = "err";
  else if (ev.stage === "tool" && d.subject === "face") li.dataset.kind = "face";

  const stage = ev.stage === "tool" && d.step ? `tool ${d.step}` : ev.stage;

  for (const [cls, text] of [["st", stage], ["msg", ev.msg], ["ms", ms(ev.latency_ms)]]) {
    const s = document.createElement("span");

    s.className = cls;
    s.textContent = text;
    li.append(s);
  }

  const h = held.find(x => x.ev === ev);

  if (h) h.li = li;

  li.classList.add("fresh");
  feed.querySelector(".empty")?.remove();
  feed.prepend(li);

  // Bounded, never scrolled: drop rows that no longer fit.
  while (feed.children.length > 1 && feed.scrollHeight > feed.clientHeight) feed.lastChild.remove();
}

feed.addEventListener("mouseenter", () => {
  feedPaused = true;
  feed.classList.add("paused");
});

feed.addEventListener("mouseleave", () => {
  feedPaused = false;
  feed.classList.remove("paused");
  feedBuffer.splice(0).forEach(addRow);
});

/* --------------------------------------------------------------- status */
function watching() {
  const n = CAMS.filter(live).length;

  return `${n} CAMERA${n === 1 ? "" : "S"}`;
}

let pillTimer = null;

function setPill(state, text, holdMs) {
  $("pill").dataset.state = state;
  $("pill").textContent = text;
  document.body.dataset.state = state;
  clearTimeout(pillTimer);

  if (holdMs) pillTimer = setTimeout(() => setPill("clear", `MONITORING · ${watching()}`), holdMs);
}

function setState(id, text) {
  $(id).textContent = text;
}

function fmt(n) {
  return n.toLocaleString("en-US");
}

let sleepTimer = null;

function wake(level) {
  $("cascade").dataset.awake = level;
  clearTimeout(sleepTimer);
}

function sleepIn(delay) {
  clearTimeout(sleepTimer);
  sleepTimer = setTimeout(() => {
    $("cascade").dataset.awake = "none";
    $("cascade").dataset.gate = "clear";
    setState("vlm-state", "asleep · no crop in hand");
    setState("scout-state", `watching ${watching().toLowerCase()}`);
  }, delay);
}

/* -------------------------------------------------------------- numbers */
function zero(el, n) {
  el.dataset.zero = n === 0;
}

function render() {
  $("n-frames").textContent = fmt(stats.frames);
  $("n-flagged").textContent = stats.scout_hits;
  $("n-calls").textContent = stats.tool_calls || 0;
  $("n-errors").textContent = stats.tool_errors;
  zero($("n-errors").parentElement, stats.tool_errors);
  $("n-recovered").textContent = stats.recoveries;
  zero($("n-recovered").parentElement, stats.recoveries);
  $("n-vlm").textContent = stats.vlm_calls;
  $("n-confirmed").textContent = stats.confirmed;

  for (const [tier, key] of [["alert", "alerts"], ["log", "logged"], ["ignore", "ignored"]]) {
    $("t-" + tier).textContent = stats[key];
    zero($("tier-" + tier), stats[key]);
  }

  $("avoided").textContent = fmt(Math.max(0, stats.frames - stats.vlm_calls));
  $("m-frames").textContent = fmt(stats.frames);
  $("m-frames-bar").style.width = stats.frames ? "100%" : "0";
  $("m-vlm").textContent = stats.vlm_calls;
  $("m-vlm-bar").style.width = stats.frames ? (100 * stats.vlm_calls / stats.frames) + "%" : "0";

  // Live only: the Scout's measured per-frame time (the backend omits it in mock mode).
  if (MODE === "live" && stats.scout_ms != null) setState("scout-meta", `${stats.scout_ms} ms per frame (p50) · beside the cameras`);
}

/* ----------------------------------------------------- last flagged, chain */
const last = $("last");

function addLast(label, outcome) {
  last.querySelector(".empty")?.remove();

  const li = document.createElement("li");
  const b = document.createElement("b");
  const out = document.createElement("span");

  b.textContent = label;
  out.textContent = outcome;
  li.append(b, out);
  last.prepend(li);

  while (last.children.length > 2) last.lastChild.remove();

  return out;
}

function setOutcome(out, tier, vlmCalls) {
  const t = document.createElement("span");

  out.textContent = "";
  t.dataset.tier = tier;
  t.textContent = `${tier} · ${vlmCalls || 0} VLM calls`;
  out.append(t);
}

function resetChain(title) {
  $("chain-title").textContent = title;

  for (const s of $("steps").children) {
    s.dataset.state = "pending";
    s.querySelector("small").textContent = "";
  }
}

function stepEl(n) {
  return $("steps").children[n - 1];
}

function stepState(n, state, sub) {
  const s = stepEl(n);

  s.dataset.state = state;

  if (sub !== undefined) s.querySelector("small").textContent = sub;
}

function ms(v) {
  if (v == null) return "";

  return v < 1 ? "<1 ms" : `${Math.round(v)} ms`;
}

/* -------------------------------------------------------------- incident */
function tierOf(action) {
  if (action === "alert") return "alert";

  return action === "ignore" ? "ignore" : "log";
}

function evidenceUrl(incident, name) {
  return `/evidence/${encodeURIComponent(incident)}/${name}.jpg`;
}

function showIncident(d) {
  const inc = $("incident");
  const via = d.evidence_camera && d.evidence_camera !== d.camera ? ` · via ${d.evidence_camera}` : "";
  const meta = $("inc-meta");
  const v = document.createElement("b");

  inc.dataset.state = d.action === "alert" ? "alert" : "log";
  $("inc-title").textContent = (d.action === "alert" ? "WEAPON ALERT · " : "LOGGED · ") + d.label.toUpperCase();
  // The claim leads: it is what the Verifier answered. The Analyst's words follow, and the
  // server leaves them out when they name another object (a confirmed knife read as "Cake.").
  $("inc-claim").textContent = d.claim ? d.claim[0].toUpperCase() + d.claim.slice(1) : "";
  $("inc-said").hidden = !d.description;
  $("inc-said").querySelector("span").textContent = d.description || "";
  meta.textContent = "";
  v.dataset.ok = !!d.verified;
  v.textContent = d.verified ? "verified" : "not verified";
  meta.append(`${d.camera}${via} · `, v, ` · ${d.vlm_calls} VLM calls · ${(d.total_ms / 1000).toFixed(1)} s`);

  for (const name of ["weapon", "face"]) {
    const img = $(name + "-img"), has = (d.evidence || []).includes(name + "_crop");

    img.hidden = !has;

    if (has) img.src = evidenceUrl(d.incident, name + "_crop");

    // A narrow head crop must not frame the placeholder text behind it.
    img.parentElement.querySelector(".empty").hidden = has;
  }

  lastAlert = d.t ? d.t * 1000 : Date.now();
  tick();
}

function tick() {
  if (!lastAlert) return;

  const s = Math.max(0, Math.round((Date.now() - lastAlert) / 1000));

  if (s < 2) $("inc-time").textContent = "just now";
  else $("inc-time").textContent = s < 120 ? `${s} s ago` : `${Math.round(s / 60)} min ago`;
}

setInterval(tick, 1000);

/* --------------------------------------------------------------- cameras */
// The zoom panel plays /stream/crop while a chain runs and rests on the stored
// weapon crop after a report, so it always shows the evidence the decision used.
function zoomTo(src, subject, label) {
  const z = $("cam-zoom");

  if (subject) z.dataset.subject = subject;
  else delete z.dataset.subject;

  if (src && $("img-zoom").getAttribute("src") !== src) $("img-zoom").src = src;

  if (src) {
    $("img-zoom").hidden = false;
    $("zoom-empty").hidden = true;
  }

  if (label) $("zoom-label").textContent = label;
}

function zoomLabel(subject) {
  if (subject === "face") return "where the camera looked · review crop · not analyzed";

  return `where the camera looked · ${subject}`;
}

function looking(cam) {
  for (const c of CAMS) $("cam-" + c).dataset.looking = cam === c;
}

function setCovered(cam, on) {
  covered[cam] = on;

  if (cam !== "hero") return;

  $("btn-cover").setAttribute("aria-pressed", on);
  $("btn-cover").textContent = on ? L.cover.replace("cover", "uncover").replace("Cover", "Uncover") : L.cover;
  $("cam-hero").dataset.covered = on;
  $("hero-note").textContent = heroNote();

  if (document.body.dataset.state === "clear" && connected) setPill("clear", `MONITORING · ${watching()}`);
}

// A camera that would not open at startup: the tile says so (reason on hover),
// it gets no stream and no particles, and the hero's cover button goes inert.
function markUnavailable(cam, why) {
  unavailable[cam] = why;
  $("cam-" + cam).dataset.unavailable = true;
  $("cam-" + cam).title = why;
  $(cam + "-empty").textContent = "UNAVAILABLE · DID NOT OPEN";

  if (cam === "hero") $("btn-cover").disabled = true;
}

// Called on every (re)connect: an MJPEG stream does not reopen by itself after a restart.
function showStreams() {
  const fresh = `?t=${Date.now()}`;

  for (const c of CAMS) {
    if (c in unavailable) continue;  // its stream would 404; the tile keeps saying why

    $("img-" + c).src = `/stream/wide/${c}${fresh}`;
    $("img-" + c).hidden = false;
    $(c + "-empty").hidden = true;
  }

  if (($("img-zoom").getAttribute("src") || "").startsWith("/stream/crop")) $("img-zoom").src = `/stream/crop${fresh}`;
}

/* --------------------------------------------------------------- flow */
const stage = $("stage"), flow = $("flow"), ctx = flow.getContext("2d");

const particles = [];

let W = 0, H = 0, lanes = {}, lastT = performance.now();

function resize() {
  const r = stage.getBoundingClientRect(), dpr = devicePixelRatio || 1;

  W = r.width;
  H = r.height;
  flow.width = W * dpr;
  flow.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const box = id => {
    const b = $(id).getBoundingClientRect();

    return {
      l: b.left - r.left, t: b.top - r.top, r: b.right - r.left, b: b.bottom - r.top,
      cy: (b.top + b.bottom) / 2 - r.top, cx: (b.left + b.right) / 2 - r.left,
    };
  };

  const hero = box("cam-hero"), s1 = box("cam-scout1"), sc = box("node-scout"), di = box("node-director"), vl = box("node-vlm");
  const ta = box("tier-alert"), tl = box("tier-log"), ti = box("tier-ignore");
  const curve = (x1, y1, x2, y2, cx, cy) => ({ x1, y1, x2, y2, cx, cy });

  // A bent lane: a curve into the card's corner, then a straight run along its edge.
  const bent = (x1, y1, cx, cy, x2, y2, x3, y3) => {
    const a = Math.hypot(x2 - x1, y2 - y1), b = Math.hypot(x3 - x2, y3 - y2);

    return { x1, y1, cx, cy, x2, y2, x3, y3, seg: a / (a + b) };
  };

  const gate = sc.r + 3;

  // Frames are born inside the camera frame, cross the gutter, ride the Scout
  // card's top (hero) and bottom (scout1) edges and die on the gate bar.
  // Survivors leave the gate's midpoint for the Director.
  lanes = {
    hero: bent(hero.r - 150, hero.cy, sc.l - 40, sc.t + 4, sc.l + 6, sc.t + 4, gate, sc.t + 4),
    scout1: bent(s1.r - 150, s1.cy, sc.l - 40, sc.b - 4, sc.l + 6, sc.b - 4, gate, sc.b - 4),
    flag: curve(gate, sc.t + 0.5 * (sc.b - sc.t), di.l, di.cy, gate + 20, di.cy),
    down: curve(di.cx - 24, di.b, vl.cx - 24, vl.t, di.cx - 24, (di.b + vl.t) / 2),
    up: curve(vl.cx + 24, vl.t, di.cx + 24, di.b, vl.cx + 24, (di.b + vl.t) / 2),
    alert: curve(di.r, di.cy - 10, ta.l, ta.cy, di.r + 36, ta.cy),
    log: curve(di.r, di.cy, tl.l, tl.cy, di.r + 36, tl.cy),
    ignore: curve(di.r, di.cy + 10, ti.l, ti.cy, di.r + 36, ti.cy),
    gateY: [sc.t + 10, sc.b - 10],
  };
}

// Card heights change as rows arrive, so the lanes follow the boxes, not just the window.
const lanesObserver = new ResizeObserver(resize);

for (const id of ["stage", "node-scout", "node-director", "node-vlm", "incident"]) lanesObserver.observe($(id));

function at(l, t) {
  if (l.seg !== undefined && t >= l.seg) {
    const k = (t - l.seg) / (1 - l.seg);

    return { x: l.x2 + (l.x3 - l.x2) * k, y: l.y2 + (l.y3 - l.y2) * k };
  }

  if (l.seg !== undefined) t /= l.seg;

  const u = 1 - t;

  return { x: u * u * l.x1 + 2 * u * t * l.cx + t * t * l.x2, y: u * u * l.y1 + 2 * u * t * l.cy + t * t * l.y2 };
}

function end(l) {
  return l.seg !== undefined ? { x: l.x3, y: l.y3 } : { x: l.x2, y: l.y2 };
}

function length(l) {
  const a = Math.hypot(l.x2 - l.x1, l.y2 - l.y1);

  return l.seg !== undefined ? a + Math.hypot(l.x3 - l.x2, l.y3 - l.y2) : a;
}

function spawn(lane, color, opts = {}) {
  if (particles.length > 900) return;

  particles.push({ lane, t: 0, color, size: opts.size || 2.5, speed: (opts.speed || 190) * MOTION, die: opts.die || false, alpha: opts.alpha || 1 });
}

/* Frame particles. Each frame the backend counts (stats.frames) is one particle,
   released over the next half second and dealt across the uncovered cameras.
   Until stats arrive (or after they stop) the cameras emit a faint standby
   trickle, so the gate reads as a gate before anything is connected. */
let pendingFrames = 0, frameAcc = 0, idleAcc = 0, dealt = 0, statsAt = 0;

function spawnFrames(dt) {
  const open = CAMS.filter(live);

  if (!open.length) {
    pendingFrames = 0;

    return;
  }

  if (performance.now() - statsAt > 2500) {
    // Reduced motion slows particles (spawn applies MOTION to speed); it never thins the count.
    idleAcc += dt * 6 * open.length;

    for (; idleAcc >= 1; idleAcc--) spawn(open[dealt++ % open.length], C.accent, { die: true, alpha: .5 });

    return;
  }

  frameAcc += pendingFrames * Math.min(1, dt / 0.5);

  for (; frameAcc >= 1 && pendingFrames >= 1; frameAcc--, pendingFrames--) spawn(open[dealt++ % open.length], C.accent, { die: true, alpha: .85 });

  if (pendingFrames < 1) frameAcc = 0;
}

const ticks = [];

function laneColor(k) {
  if (k === "up" || k === "down") return "rgba(143,224,164,.35)";

  if (k === "alert") return "rgba(255,111,106,.35)";

  if (k === "log") return "rgba(255,156,98,.3)";

  return k === "ignore" ? "rgba(163,170,182,.3)" : "rgba(242,193,78,.3)";
}

function draw(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);

  lastT = now;
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 6]);

  for (const k in lanes) {
    if (k === "gateY") continue;

    const l = lanes[k];
    const start = (k === "hero" || k === "scout1") ? at(l, .25) : { x: l.x1, y: l.y1 };

    ctx.strokeStyle = laneColor(k);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.quadraticCurveTo(l.cx, l.cy, l.x2, l.y2);

    if (l.seg !== undefined) ctx.lineTo(l.x3, l.y3);

    ctx.stroke();
  }

  ctx.setLineDash([]);
  spawnFrames(dt);

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i], l = lanes[p.lane];

    if (!l) {
      particles.splice(i, 1);
      continue;
    }

    p.t += dt * p.speed / (length(l) || 1);

    if (p.t >= 1) {
      if (p.die) ticks.push({ y: end(l).y, a: 1 });

      particles.splice(i, 1);
      continue;
    }

    const q = at(l, p.t);

    // Faint while still inside the camera frame, bright once in the gutter and on the card edge.
    const a = p.die ? p.alpha * (p.t < .3 ? .3 : 1) : p.alpha;

    ctx.globalAlpha = a;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(q.x, q.y, p.size, 0, Math.PI * 2);
    ctx.fill();
  }

  // A frame dying on the gate leaves a short tick on the bar.
  const gx = lanes.flag ? lanes.flag.x1 : 0;

  for (let i = ticks.length - 1; i >= 0; i--) {
    const k = ticks[i];

    k.a -= dt * 3;

    if (k.a <= 0) {
      ticks.splice(i, 1);
      continue;
    }

    ctx.globalAlpha = k.a;
    ctx.fillStyle = C.accent;
    ctx.fillRect(gx - 4, k.y - 1, 10, 2);
  }

  ctx.globalAlpha = 1;
  requestAnimationFrame(draw);
}

requestAnimationFrame(draw);

/* -------------------------------------------------------- event handling */
let currentOut = null, currentCam = "hero", currentLabel = "", claim = "", fault = null;

function clearStatusSoon() {
  setTimeout(() => {
    $("ctl-status").textContent = "";
    $("ctl-status").dataset.state = "";
  }, 6000);
}

function onScout(d) {
  if (d.handled_via) return;  // the same object on the second camera: not a new incident

  spawn("flag", d.threat === "weapon" ? C.err : C.accent, { size: 4, speed: 240 });
  $("node-scout").dataset.active = true;
  setTimeout(() => $("node-scout").dataset.active = false, 900);
  setState("scout-state", `flagged ${d.label} on ${d.camera} · ${d.threat}`);
  currentOut = addLast(d.label, `looking · ${d.camera}`);
  currentCam = d.camera;
  currentLabel = d.label;
  $("cascade").dataset.gate = d.threat === "weapon" ? "weapon" : "clear";
  wake(d.threat === "benign" ? "director" : "all");
  $("node-director").dataset.active = true;
  $("node-director").dataset.fault = false;

  if (d.threat !== "benign") {
    setPill("busy", `INVESTIGATING ${d.label.toUpperCase()} · STEP 0/6`);
    resetChain(`${d.label} on ${d.camera} · ${d.threat}`);
  } else {
    resetChain(`${d.label} on ${d.camera} · benign · skipped to report`);

    for (let n = 1; n <= 5; n++) stepState(n, "skipped", "skipped");
  }
}

// Narration arrives as each chain step starts; the tool event arrives when it ends.
function onNarration(ev, d) {
  const cam = d.camera || currentCam;

  $("node-director").dataset.fault = false;
  setState("director-state", `step ${d.step}/6 · ${ev.msg}`);

  if (stepEl(d.step).dataset.state !== "done") stepState(d.step, "active", d.retry ? `↻ ${cam}` : undefined);

  if (document.body.dataset.state === "busy") {
    setPill("busy", `INVESTIGATING ${(d.label || currentLabel).toUpperCase()} · STEP ${d.step}/6${d.retry ? " · RETRY" : ""}`);
  }

  if (d.tool === "snapshot") {
    zoomTo("/stream/crop", d.subject, zoomLabel(d.subject));

    if (d.subject !== "face") {
      spawn("down", C.ok, { size: 4, speed: 200 });
      $("node-vlm").dataset.active = true;
      setState("vlm-state", `reading the ${d.subject} crop`);
    }
  }

  if (d.tool === "verify") {
    claim = d.claim || claim;
    spawn("down", C.ok, { size: 4, speed: 200 });
    $("node-vlm").dataset.active = true;
    setState("vlm-state", `checking “${claim}”`);
  }
}

function onTool(ev, d) {
  const cam = d.camera || currentCam;

  if (d.error) {
    spawn("flag", C.err, { size: 4, speed: 120 });
    $("node-director").dataset.fault = true;

    if (d.step) stepState(d.step, "failed", `${cam} failed`);

    if (/\(injected\)/.test(ev.msg)) {
      fault = cam;
      $("ctl-status").dataset.state = "fired";
      $("ctl-status").textContent = `fault injected on ${cam} · recovering`;
    }

    return;
  }

  if (!d.step || d.tool === "report") return;

  const sub = { 1: d.retry ? `↻ ${cam}` : ms(ev.latency_ms), 3: d.retry ? `↻ ${cam}` : ms(ev.latency_ms), 4: "no model" }[d.step];

  // Steps 2 and 5 take their sub-label from the analyst and verifier events.
  if (d.step !== 2 && d.step !== 5) stepState(d.step, "done", sub);
  else if (stepEl(d.step).dataset.state !== "done") stepState(d.step, "done");

  if (d.retry && fault) {
    $("ctl-status").textContent = `fault injected · recovered on ${cam}`;
    fault = null;
    clearStatusSoon();
  }

  if (d.tool === "look_at") {
    looking(cam);
    zoomTo(null, d.subject, zoomLabel(d.subject));
  }

  if (d.tool === "snapshot" && d.subject === "face") setState("vlm-state", "not consulted for the review crop");
}

function onReport(d) {
  const tier = tierOf(d.action);
  const el = $("tier-" + tier);

  spawn(tier, tier === "alert" ? C.err : tier === "log" ? C.warn : C.dim, { size: 4, speed: 240 });
  el.classList.remove("gain");
  void el.offsetWidth;
  el.classList.add("gain");

  if (currentOut) {
    setOutcome(currentOut, tier, d.vlm_calls);
    currentOut = null;
  }

  stepState(6, "done", tier);

  // A row still buffered behind a paused feed renders the settled ev.msg when it lands.
  for (const h of held.splice(0)) {
    h.ev.msg = d.analyst_mismatch ? WITHHELD : h.said;

    if (h.li) h.li.querySelector(".msg").textContent = h.ev.msg;
  }

  $("node-director").dataset.active = false;
  $("node-director").dataset.fault = false;
  looking(null);
  setState("director-state", `reported “${d.claim}” · ${d.action}`);

  if (fault) {
    // The broken call was not retried (a best-effort step): say so, then clear.
    $("ctl-status").textContent = `fault injected on ${fault} · reported without that step`;
    fault = null;
    clearStatusSoon();
  }

  if (d.action === "ignore") {
    setPill("ignored", `IGNORED ${d.label.toUpperCase()} · 0 VLM CALLS`, 4000);
    sleepIn(1500);
  } else if (d.threat === "weapon") {
    showIncident(d);

    // Rest the zoom panel on the evidence the decision used.
    if ((d.evidence || []).includes("weapon_crop")) {
      zoomTo(evidenceUrl(d.incident, "weapon_crop"), "weapon", `where the camera looked · weapon · ${d.verified ? "verified" : "not verified"}`);
    }

    setPill(d.action === "alert" ? "alert" : "busy", `${d.action === "alert" ? "ALERT" : "LOGGED"} · ${d.label.toUpperCase()} · ${d.verified ? "VERIFIED" : "NOT VERIFIED"}`, 9000);
    // Keep the lit path on screen long enough to read before the models go cold again.
    sleepIn(7000);
  } else {
    setPill("clear", `LOGGED ${d.label.toUpperCase()}`, 4000);
    sleepIn(1500);
  }
}

function onEvent(ev) {
  const d = ev.data || {};

  if (ev.stage === "analyst") {
    held.push({ ev, said: ev.msg });
    ev.msg = READING;
  }

  addRow(ev);

  if (ev.stage === "scout") onScout(d);

  if (ev.stage === "narration" && d.step) onNarration(ev, d);

  if (ev.stage === "director" && !ev.msg.startsWith("->")) {
    $("node-director").dataset.fault = !!d.fault;
    setState("director-state", ev.msg);
  }

  if (ev.stage === "tool") onTool(ev, d);

  if (ev.stage === "analyst") {
    spawn("up", C.ok, { size: 4, speed: 200 });
    stepState(2, "done", ms(ev.latency_ms));
    setState("vlm-state", "read the crop");
  }

  if (ev.stage === "verifier") {
    spawn("up", d.supported ? C.ok : C.err, { size: 4, speed: 200 });
    setState("vlm-state", `${d.supported ? "confirmed" : "rejected"} “${claim}”`);
    stepState(5, "done", d.supported ? "confirmed" : "rejected");
    setTimeout(() => {
      $("node-vlm").dataset.active = false;
    }, 1200);
  }

  if (ev.stage === "system") {
    const m = /^(\S+) (covered|uncovered)$/.exec(ev.msg);

    if (ev.msg.startsWith("REPORT")) onReport(d);

    if (m && m[1] in covered) setCovered(m[1], m[2] === "covered");
  }
}

function onStats(s) {
  if (statsAt && s.frames > stats.frames) pendingFrames += s.frames - stats.frames;

  statsAt = performance.now();
  stats = s;
  render();

  if (!connected) {
    connected = true;
    showStreams();
    setPill("clear", `MONITORING · ${watching()}`);
    setState("scout-state", `watching ${watching().toLowerCase()}`);
    refreshMode();
  }
}

/* ------------------------------------------------------------- runtime */
async function getJson(url) {
  const r = await fetch(url, { cache: "no-store" });

  if (!r.ok) throw new Error(`${url}: ${r.status}`);

  return r.json();
}

// Reload mid-demo: the last two flagged objects and the latest weapon incident.
function restore(reports) {
  for (const r of reports.slice(-2)) {
    const t = r.trigger || {};

    if (t.label) setOutcome(addLast(t.label, ""), tierOf(r.action), r.vlm_calls);
  }

  const w = [...reports].reverse().find(r => (r.trigger || {}).threat === "weapon");

  if (!w) return;

  const d = { ...w, label: w.trigger.label, camera: w.trigger.camera, threat: "weapon" };

  showIncident(d);

  if ((d.evidence || []).includes("weapon_crop")) {
    zoomTo(evidenceUrl(d.incident, "weapon_crop"), "weapon", `where the camera looked · weapon · ${d.verified ? "verified" : "not verified"}`);
  }
}

function connect() {
  const es = new EventSource("/events");

  es.onmessage = e => onEvent(JSON.parse(e.data));
  es.addEventListener("stats", e => onStats(JSON.parse(e.data)));
  es.onerror = () => {
    if (connected) setPill("clear", "STANDBY · RECONNECTING");

    connected = false;
  };
}

// Model servers can come up after the page loads, so this runs again on every (re)connect.
async function refreshMode() {
  try {
    applyMode(await getJson("/mode"));
  } catch {
    // Standalone (no runtime): the labels already shown stay.
  }
}

function showCameraWarnings(cameras) {
  const hero = cameras.find(c => c.name === "hero");

  heroWarning = hero && hero.warning ? { label: hero.warning, why: hero.tracking || hero.ptz || "" } : null;
  $("cam-hero").dataset.ptzWarning = !!heroWarning;
  $("hero-note").title = heroWarning ? heroWarning.why : "";
  $("hero-note").textContent = heroNote();
}

// A gimbal can fail or start moving on its own mid-demo, so the label is re-read.
async function refreshCameraWarnings() {
  try {
    showCameraWarnings((await getJson("/cameras")).cameras || []);
  } catch {
    // No runtime: keep the last label.
  }
}

setInterval(refreshCameraWarnings, 5000);

async function boot() {
  await refreshMode();

  try {
    const cams = await getJson("/cameras");

    for (const c of cams.cameras || []) if (c.name in covered) setCovered(c.name, !!c.covered);

    showCameraWarnings(cams.cameras || []);

    for (const c of cams.unavailable || []) if (CAMS.includes(c.name)) markUnavailable(c.name, c.error);
  } catch {
    // No runtime yet: nothing is covered.
  }

  try {
    restore(await getJson("/reports"));
  } catch {
    // Nothing to restore.
  }

  connect();
}

/* ------------------------------------------------------------ controls */
async function post(url) {
  const r = await fetch(url, { method: "POST" });

  if (!r.ok) throw new Error(`${url}: ${r.status}`);
}

function noAnswer() {
  $("ctl-status").dataset.state = "";
  $("ctl-status").textContent = "the runtime did not answer";
}

// Armed and covered show only once the runtime has acknowledged the request.
$("btn-fail").onclick = async () => {
  try {
    await post("/demo/fail/1");
    $("ctl-status").dataset.state = "armed";
    $("ctl-status").textContent = "armed · the next look_at fails";
  } catch {
    noAnswer();
  }
};

$("btn-cover").onclick = async () => {
  const on = !covered.hero;

  try {
    await post(`/demo/cover/hero/${on ? "on" : "off"}`);
    setCovered("hero", on);
  } catch {
    noAnswer();
  }
};

applyMode(null);

render();

boot();
