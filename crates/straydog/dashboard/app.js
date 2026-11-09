// The Straydog dashboard: polls /dashboard/status and /dashboard/logs, draws the hardware and
// the speed as scrolling charts, and chats through /v1/chat/completions. Text from the server
// and the model goes into DOM text nodes, never into HTML.

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

// ---- language ----

const STRINGS = {
  en: {
    connecting: "Connecting…", ready: "Ready", generating: "Generating", loading: "Loading", failed: "Failed",
    offline: "Server unreachable", unauthorized: "API key required", uptime: "Uptime", apiKey: "API key", language: "Language",
    apiKeyHelp: "This server requires its API key. It is kept in this tab only.", save: "Save", cancel: "Cancel",
    overview: "Overview", environment: "Environment", logsTab: "Log",
    decode: "Decode", prefill: "Prefill", firstToken: "First token", speculative: "Speculative acceptance", live: "live",
    lastRequest: "last request", cached: "from cache", off: "off", throughput: "Speed", idle: "Idle",
    prefilling: "Prefilling", decoding: "Generating", preparing: "Starting", queued: "{0} waiting",
    lastAt: "last request at {0}", requests: "Requests", decodePerRequest: "Decode tok/s",
    prefillPerRequest: "Prefill tok/s (256+ new tokens)", newTokens: "{0} new tokens", peak: "peak {0}", noRequests: "No requests yet. Open the chat and ask something.",
    chat: "Chat", effort: "Reasoning", effortNone: "none", effortLow: "low", effortMedium: "medium", effortHigh: "high",
    temperature: "Temp", maxTokens: "Max", auto: "auto", clear: "Clear", send: "Send", stop: "Stop",
    expand: "Expand", close: "Close",
    chatEmpty: "Messages go to this server's model. Reasoning appears folded above each answer.",
    promptPlaceholder: "Message Straydog (Enter to send, Shift+Enter for a new line)", reasoning: "Reasoning",
    thinking: "Thinking…", stopped: "stopped", length: "cut at the token limit",
    gpus: "GPUs", host: "Host", settings: "Settings", logs: "Service log", filter: "Filter",
    warningsOnly: "Warnings only", follow: "Follow", busy: "busy", power: "power", vram: "VRAM", usage: "usage",
    load: "load", memory: "Memory", used: "used", straydog: "Straydog", swap: "Swap", available: "available",
    disk: "Disk", diskRead: "disk read", serverReadIo: "Straydog read I/O", allVolumes: "all volumes",
    diskScope: "physical reads · all apps", ioScope: "I/O may include cached reads", ioScopeLinux: "storage reads by process",
    ioScopeUnknown: "process I/O", serverResident: "Straydog RSS total",
    rssScope: "Sum of process RSS, not a limit; shared pages may count more than once.",
    cpu: "CPU", gpu: "GPU", state: "State", placement: "Placement", fullyResident: "fully resident",
    partiallyResident: "GPU + CPU experts", kv: "K/V cache", vramKv: "VRAM", hostMemory: "host memory",
    layers: "Layers", layersValue: "{0}–{1} on GPU {2}", expertSlots: "Experts in VRAM", expertSlotsValue: "{0} slots", expertHost: "Host expert weights (model size)",
    context: "Context", tokens: "{0} tokens", prefillBatch: "Prefill batch", drafts: "{0} drafts",
    sequenceState: "Sequence state", loadedIn: "Loaded in", queue: "Queue", requestsDone: "Requests",
    failedCount: "{0} failed", promptTokens: "Prompt tokens", reused: "Reused", generated: "Generated",
    hostName: "Host name", os: "OS", kernel: "Kernel", architecture: "Architecture", cores: "Cores",
    coresValue: "{0} cores / {1} threads", loadAverage: "Load average", hostUptime: "Host uptime",
    cpuTemperature: "CPU temperature", serverProcesses: "Straydog processes", serverMemory: "Straydog process RSS total",
    serverCpu: "Straydog CPU", model: "Model", quantization: "Quantization", drafter: "MTP drafter", none: "none",
    draftMax: "MTP drafts", devices: "GPUs", vramBudget: "VRAM per GPU", splitMode: "Split",
    stateRam: "Host memory for state", gemm: "GEMM precision", gemmMin: "GEMM from batch", address: "Address",
    required: "required", notRequired: "not required", defaultTemp: "Default temperature", defaultSeed: "Default seed",
    defaultEffort: "Default reasoning", startupTimeout: "Startup timeout", topK: "Top-k", topP: "Top-p",
    minP: "Min-p", presencePenalty: "Presence penalty", modelCard: "model card", freshSeed: "fresh per request", pci: "PCI", fan: "Fan", clock: "Clock",
    powerCap: "Power limit", deviceMemory: "Device memory", allocated: "Straydog allocation",
    budgetOf: "{0} / {1} GiB budget", noSensors: "No sensors for this GPU.", waiting: "Waiting for the engine…",
    time: "Time", prompt: "Prompt", completion: "Completion", ttft: "First token", finish: "Finish",
    acceptance: "Speculative accepted", speculation: "Speculation", dspark: "DSpark", lookup: "lookup", mtpName: "MTP", stageAuthenticate: "authenticating the model", stageDrafter: "authenticating the drafter",
    stagePrepare: "placing the model on the GPUs", stageWarm: "warming up the GPU code", error: "Error",
  },
  ja: {
    connecting: "接続中…", ready: "待機中", generating: "生成中", loading: "読み込み中", failed: "停止",
    offline: "サーバーに接続できません", unauthorized: "API キーが必要です", uptime: "稼働時間", apiKey: "API キー", language: "言語",
    apiKeyHelp: "このサーバーには API キーが必要です。キーはこのタブの中だけに保存します。", save: "保存", cancel: "キャンセル",
    overview: "概要", environment: "環境", logsTab: "ログ",
    decode: "デコード", prefill: "プレフィル", firstToken: "最初のトークン", speculative: "投機採択率", live: "ライブ",
    lastRequest: "直近のリクエスト", cached: "キャッシュ", off: "オフ", throughput: "処理速度", idle: "待機中",
    prefilling: "プレフィル中", decoding: "生成中", preparing: "開始中", queued: "{0} 件待ち",
    lastAt: "直近 {0}", requests: "リクエスト", decodePerRequest: "デコード tok/s",
    prefillPerRequest: "プレフィル tok/s（新256トークン以上）", newTokens: "新 {0} トークン", peak: "最大 {0}", noRequests: "まだリクエストがありません。チャットを開いて話しかけてみてください。",
    chat: "チャット", effort: "思考", effortNone: "なし", effortLow: "低", effortMedium: "中", effortHigh: "高",
    temperature: "温度", maxTokens: "上限", auto: "自動", clear: "消去", send: "送信", stop: "停止",
    expand: "広げる", close: "閉じる",
    chatEmpty: "このサーバーのモデルと会話します。思考内容は各回答の上に折りたたんで表示します。",
    promptPlaceholder: "メッセージを入力（Enter で送信、Shift+Enter で改行）", reasoning: "思考",
    thinking: "考えています…", stopped: "停止しました", length: "トークン上限で終了",
    gpus: "GPU", host: "ホスト", settings: "設定", logs: "サービスログ", filter: "絞り込み",
    warningsOnly: "警告のみ", follow: "追従", busy: "使用率", power: "電力", vram: "VRAM", usage: "使用率",
    load: "負荷", memory: "メモリ", used: "使用", straydog: "Straydog", swap: "スワップ", available: "空き",
    disk: "ディスク", diskRead: "ディスク読み取り", serverReadIo: "Straydog 読取 I/O", allVolumes: "全ボリューム",
    diskScope: "実ディスク読取・全アプリ", ioScope: "I/O はキャッシュ読取も含む", ioScopeLinux: "プロセスによる実ストレージ読取",
    ioScopeUnknown: "プロセス I/O", serverResident: "Straydog RSS 合計",
    rssScope: "各プロセス RSS の合計です。確保上限ではなく、共有ページは重複計上される場合があります。",
    cpu: "CPU", gpu: "GPU", state: "状態", placement: "配置", fullyResident: "全常駐",
    partiallyResident: "GPU + CPU エキスパート", kv: "K/V キャッシュ", vramKv: "VRAM", hostMemory: "ホストメモリ",
    layers: "層", layersValue: "{0}–{1}（GPU {2}）", expertSlots: "VRAM のエキスパート", expertSlotsValue: "{0} スロット", expertHost: "ホスト上のエキスパート重み（モデルサイズ）",
    context: "コンテキスト", tokens: "{0} トークン", prefillBatch: "プレフィルバッチ", drafts: "{0} ドラフト",
    sequenceState: "系列状態", loadedIn: "読み込み", queue: "待ち行列", requestsDone: "リクエスト",
    failedCount: "失敗 {0}", promptTokens: "プロンプト", reused: "再利用", generated: "生成",
    hostName: "ホスト名", os: "OS", kernel: "カーネル", architecture: "アーキテクチャ", cores: "コア",
    coresValue: "{0} コア / {1} スレッド", loadAverage: "ロードアベレージ", hostUptime: "ホスト稼働時間",
    cpuTemperature: "CPU 温度", serverProcesses: "Straydog のプロセス", serverMemory: "Straydog のプロセス RSS 合計",
    serverCpu: "Straydog の CPU", model: "モデル", quantization: "量子化", drafter: "MTP drafter", none: "なし",
    draftMax: "MTP ドラフト", devices: "GPU", vramBudget: "GPU ごとの VRAM", splitMode: "分割",
    stateRam: "系列状態のホストメモリ", gemm: "GEMM 精度", gemmMin: "GEMM 開始バッチ", address: "アドレス",
    required: "必要", notRequired: "不要", defaultTemp: "既定の温度", defaultSeed: "既定のシード",
    defaultEffort: "既定の思考", startupTimeout: "起動タイムアウト", topK: "Top-k", topP: "Top-p",
    minP: "Min-p", presencePenalty: "Presence penalty", modelCard: "モデルカード準拠", freshSeed: "リクエストごとに新規", pci: "PCI", fan: "ファン", clock: "クロック",
    powerCap: "電力上限", deviceMemory: "デバイスメモリ", allocated: "Straydog の確保",
    budgetOf: "予算 {1} GiB 中 {0}", noSensors: "この GPU のセンサーは読めません。", waiting: "エンジンを待っています…",
    time: "時刻", prompt: "プロンプト", completion: "生成", ttft: "最初のトークン", finish: "終了理由",
    acceptance: "投機採択", speculation: "投機的デコード", dspark: "DSpark", lookup: "照合", mtpName: "MTP", stageAuthenticate: "モデルを検証中", stageDrafter: "drafter を検証中",
    stagePrepare: "モデルを GPU に配置中", stageWarm: "GPU のコードを準備中", error: "エラー",
  },
};
// English unless the viewer chose Japanese with the top bar's switch (kept in this browser). A link may
// name the language (`?lang=ja` or `?lang=en`, as an app that knows its own opens it): that wins, and is kept.
const LANG_STORAGE = "straydog.dashboard.lang";
let LANG = "en";
try { if (localStorage.getItem(LANG_STORAGE) === "ja") LANG = "ja"; } catch { /* storage unavailable */ }
{
  const asked = new URLSearchParams(location.search).get("lang");
  if (asked === "ja" || asked === "en") {
    LANG = asked;
    try { localStorage.setItem(LANG_STORAGE, LANG); } catch { /* storage unavailable */ }
  }
}

function t(key, ...args) {
  let text = STRINGS[LANG][key] ?? STRINGS.en[key] ?? key;
  args.forEach((arg, index) => { text = text.replace(`{${index}}`, arg); });
  return text;
}

function translatePage() {
  document.documentElement.lang = LANG;
  // (the switch names the language it switches to, in that language)
  const switcher = document.getElementById("lang-button");
  if (switcher) switcher.textContent = LANG === "en" ? "日本語" : "English";
  for (const element of document.querySelectorAll("[data-i18n]")) element.textContent = t(element.dataset.i18n);
  for (const element of document.querySelectorAll("[data-i18n-placeholder]")) element.placeholder = t(element.dataset.i18nPlaceholder);
  for (const element of document.querySelectorAll("[data-i18n-title]")) {
    element.title = t(element.dataset.i18nTitle);
    element.setAttribute("aria-label", element.title);
  }
}

// ---- formatting ----

const numberFormats = new Map();
function number(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (!numberFormats.has(digits)) {
    numberFormats.set(digits, new Intl.NumberFormat(LANG, { minimumFractionDigits: digits, maximumFractionDigits: digits }));
  }
  return numberFormats.get(digits).format(value);
}
const gib = (bytes) => number(bytes / 2 ** 30, 1);
const sizeGiB = (bytes) => Number.isFinite(bytes) ? `${gib(bytes)} GiB` : "—";
const mibRate = (bytes) => (Number.isFinite(bytes) ? bytes / 2 ** 20 : null);
const percent = (part, whole) => (whole > 0 && Number.isFinite(part) ? (100 * part) / whole : null);
const clock = (seconds) => new Date(seconds * 1000).toLocaleTimeString(LANG, { hour12: false });
const celsius = (value) => (Number.isFinite(value) ? `${number(value)}°C` : "");

function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const [d, h, m] = [Math.floor(s / 86400), Math.floor((s % 86400) / 3600), Math.floor((s % 3600) / 60)];
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attributes = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  return node;
}

function facts(list, rows) {
  list.replaceChildren(...rows.filter(Boolean).flatMap(([label, value, kind]) => {
    const detail = element("dd", kind || "", value ?? "—");
    return [element("dt", "", label), detail];
  }));
}

// ---- colours from the stylesheet ----

let colors = {};
function readColors() {
  const style = getComputedStyle(document.documentElement);
  for (const name of ["decode", "prefill", "heat", "power", "grid", "muted", "text"]) {
    colors[name] = style.getPropertyValue(`--${name}`).trim();
  }
}
matchMedia("(prefers-color-scheme: light)").addEventListener("change", readColors);

function withAlpha(color, alpha) {
  const hex = /^#([\da-f]{6})$/i.exec(color);
  if (!hex) return color;
  const value = parseInt(hex[1], 16);
  return `rgba(${value >> 16}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

// ---- scrolling time charts ----

const WINDOW = 120;
const charts = new Set();

/// A canvas chart of the last two minutes; `series` name a colour, an axis and a style.
class TimeChart {
  constructor(container, series, axes) {
    this.container = container;
    this.canvas = container.querySelector("canvas");
    this.series = series;
    this.axes = axes;
    this.data = new Map(series.map((item) => [item.key, []]));
    this.onAxes = null;
    new ResizeObserver(() => this.resize()).observe(container);
    charts.add(this);
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.container.clientWidth * ratio));
    this.canvas.height = Math.max(1, Math.round(this.container.clientHeight * ratio));
  }

  push(key, time, value) {
    const points = this.data.get(key);
    // Samples arrive in order; one seen before is skipped.
    if (points.length && points.at(-1)[0] >= time) return;
    points.push([time, Number.isFinite(value) ? value : null]);
    while (points.length && points[0][0] < time - WINDOW - 4) points.shift();
  }

  visible() {
    return this.container.offsetParent !== null && this.canvas.width > 1;
  }

  scale(axis, start) {
    const config = this.axes[axis];
    if (config.max !== "auto") return config.max;
    let top = 0;
    for (const item of this.series) {
      if (item.axis !== axis) continue;
      for (const [time, value] of this.data.get(item.key)) if (time >= start && value > top) top = value;
    }
    return Math.max(top * 1.2, config.floor || 1);
  }

  draw(now) {
    const context = this.canvas.getContext("2d");
    const [width, height] = [this.canvas.width, this.canvas.height];
    const ratio = window.devicePixelRatio || 1;
    const pad = 4 * ratio;
    context.clearRect(0, 0, width, height);
    context.lineWidth = ratio;
    context.strokeStyle = colors.grid;
    for (const level of [0.25, 0.5, 0.75]) {
      const y = Math.round(pad + (height - 2 * pad) * level) + 0.5;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }
    const start = now - WINDOW;
    const maxima = { left: this.scale("left", start), right: this.axes.right ? this.scale("right", start) : null };
    const x = (time) => ((time - start) / WINDOW) * width;
    for (const item of this.series) {
      const top = maxima[item.axis];
      const y = (value) => height - pad - (Math.min(value, top) / top) * (height - 2 * pad);
      const color = colors[item.color];
      const runs = [];
      let run = [];
      for (const [time, value] of this.data.get(item.key)) {
        if (value === null) { if (run.length) runs.push(run); run = []; continue; }
        if (time >= start - 2) run.push([x(time), y(value)]);
      }
      if (run.length) runs.push(run);
      for (const points of runs) {
        if (points.length < 2) continue;
        const trace = () => {
          context.beginPath();
          context.moveTo(points[0][0], points[0][1]);
          for (let index = 1; index < points.length; index += 1) {
            const [px, py] = points[index - 1];
            const [cx, cy] = points[index];
            context.quadraticCurveTo(px, py, (px + cx) / 2, (py + cy) / 2);
          }
          context.lineTo(points.at(-1)[0], points.at(-1)[1]);
        };
        if (item.fill) {
          trace();
          context.lineTo(points.at(-1)[0], height);
          context.lineTo(points[0][0], height);
          context.closePath();
          const gradient = context.createLinearGradient(0, 0, 0, height);
          gradient.addColorStop(0, withAlpha(color, 0.38));
          gradient.addColorStop(1, withAlpha(color, 0));
          context.fillStyle = gradient;
          context.fill();
        }
        trace();
        context.setLineDash(item.dash ? [4 * ratio, 4 * ratio] : []);
        context.lineWidth = (item.width || 1.8) * ratio;
        context.strokeStyle = color;
        context.shadowColor = item.glow ? color : "transparent";
        context.shadowBlur = item.glow ? 10 * ratio : 0;
        context.stroke();
        context.shadowBlur = 0;
        context.setLineDash([]);
      }
      const last = runs.at(-1)?.at(-1);
      if (last && item.glow) {
        context.fillStyle = color;
        context.shadowColor = color;
        context.shadowBlur = 12 * ratio;
        context.beginPath();
        context.arc(last[0], last[1], 3 * ratio, 0, Math.PI * 2);
        context.fill();
        context.shadowBlur = 0;
      }
    }
    this.onAxes?.(maxima);
  }
}

let lastFrame = 0;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
function animate(time) {
  requestAnimationFrame(animate);
  if (document.hidden || time - lastFrame < (reducedMotion.matches ? 1000 : 33)) return;
  lastFrame = time;
  const now = Date.now() / 1000 + clockOffset;
  for (const chart of charts) if (chart.visible()) chart.draw(now);
}

// ---- the API key ----

const KEY_STORAGE = "straydog.dashboard.key";
let apiKey = "";
try { apiKey = sessionStorage.getItem(KEY_STORAGE) || ""; } catch { /* storage unavailable */ }

function authorized(headers = {}) {
  return apiKey ? { ...headers, Authorization: `Bearer ${apiKey}` } : headers;
}

function openKeyDialog() {
  const dialog = $("key-dialog");
  if (dialog.open) return;
  $("key-input").value = apiKey;
  dialog.showModal();
  $("key-input").focus();
}

$("key-button").addEventListener("click", openKeyDialog);
$("key-cancel").addEventListener("click", () => $("key-dialog").close());
$("key-form").addEventListener("submit", (event) => {
  event.preventDefault();
  apiKey = $("key-input").value.trim();
  try { sessionStorage.setItem(KEY_STORAGE, apiKey); } catch { /* storage unavailable */ }
  $("key-dialog").close();
  poll();
  pollLogs();
});

// ---- tabs ----

const TABS = ["overview", "environment", "logs"];
function showTab(name) {
  const tab = TABS.includes(name) ? name : "overview";
  for (const button of document.querySelectorAll(".tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === tab));
  }
  for (const item of TABS) $(`tab-${item}`).hidden = item !== tab;
  if (tab === "logs" && $("log-follow").checked) $("log").scrollTop = $("log").scrollHeight;
}
for (const button of document.querySelectorAll(".tabs button")) {
  button.addEventListener("click", () => { history.replaceState(null, "", `#${button.dataset.tab}`); showTab(button.dataset.tab); });
}
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ---- status ----

let pollTimer = 0;
// The server's samples (one a second), and how far the server's clock is ahead of this one.
const samples = [];
let lastSample = 0;
let clockOffset = 0;

function setState(state, text, progress = null) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
  $("status-bar").style.setProperty("width", progress === null ? "0" : `${progress}%`);
}

async function poll() {
  clearTimeout(pollTimer);
  let delay = 1000;
  try {
    const response = await fetch(`/dashboard/status?since=${lastSample}`, { headers: authorized(), cache: "no-store" });
    if (response.status === 401) {
      setState("unauthorized", t("unauthorized"));
      openKeyDialog();
      delay = 5000;
    } else if (!response.ok) {
      throw new Error(String(response.status));
    } else {
      const status = await response.json();
      render(status);
      if (status.engine.activity.live) delay = 500;
    }
  } catch {
    setState("offline", t("offline"));
    delay = 3000;
  }
  pollTimer = setTimeout(poll, document.hidden ? Math.max(delay, 5000) : delay);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { poll(); pollLogs(); }
});

function stageText(progress) {
  return { authenticate: t("stageAuthenticate"), drafter: t("stageDrafter"), prepare: t("stagePrepare"), warm: t("stageWarm") }[progress.stage]
    || progress.stage || "";
}

function render({ now, server, engine, host, samples: fresh = [] }) {
  const ready = engine.ready || {};
  const activity = engine.activity;
  clockOffset = now - Date.now() / 1000;
  const added = fresh.filter((sample) => sample.time > lastSample);
  for (const sample of added) {
    samples.push(sample);
    lastSample = sample.time;
  }
  while (samples.length > WINDOW + 30) samples.shift();
  $("model-line").textContent = [server.alias, `${server.backend} · v${server.version}`].join(" · ");
  $("uptime").textContent = duration(now - server.started);
  if (engine.state === "loading") {
    const progress = engine.progress || {};
    const share = progress.total ? percent(progress.done, progress.total) : null;
    setState("loading", [t("loading"), stageText(progress), share === null ? "" : `${number(share)}%`].filter(Boolean).join(" · "), share);
  } else if (engine.state === "failed") {
    setState("failed", `${t("failed")} · ${engine.error}`);
  } else if (activity.live) {
    setState("busy", engine.queued ? `${t("generating")} · ${t("queued", engine.queued)}` : t("generating"));
  } else {
    setState("ready", t("ready"));
  }
  const rebuilt = renderHardware(engine, host);
  for (const sample of rebuilt ? samples : added) feed(sample);
  renderSpeed(activity, ready, engine.queued, samples.at(-1));
  renderRequests(activity.recent);
  renderEngine(engine, ready, activity);
  renderEnvironment(engine, host, server);
  if (!effortChosen) $("effort").value = server.reasoning_effort;
  $("temperature").placeholder = server.temperature == null ? t("auto") : number(server.temperature, 1);
  $("chat-open").classList.toggle("busy", Boolean(controller));
}

// ---- hardware ----

/// The GPUs Straydog uses, with their sensors; before the engine reports its devices, every
/// GPU with sensors.
function gpuList(engine, host) {
  const sensors = host?.gpus || [];
  const devices = engine.ready?.gpu?.devices || [];
  const matched = devices.map((device) => ({ device, sensor: sensors.find((sensor) => sensor.pci === device.pci) }));
  if (matched.length && matched.some((item) => item.sensor)) return matched;
  if (matched.length && !sensors.length) return matched;
  return sensors.map((sensor, index) => ({ device: devices[index] || null, sensor }));
}

let hardwareKey = "";
const hardware = { gpus: [], cpu: null, memory: null, disk: null };

function metric(className, label, unit) {
  const node = element("div", `metric ${className}`);
  const value = element("b", "", "—");
  node.append(value, element("span", "", unit), element("small", "", label));
  return { node, value, unit: node.querySelector("span") };
}

function deviceCard(kind, legend) {
  const card = element("article", "card glass device");
  const head = element("div", "device-head");
  const kindLabel = element("span", "kind", kind);
  const name = element("span", "name");
  const temp = element("span", "temp");
  head.append(kindLabel, name, temp);
  const metrics = element("div", "metrics");
  const chart = element("div", "timechart");
  chart.append(element("canvas"));
  const legendRow = element("div", "series-legend");
  for (const [color, label] of legend) {
    const item = element("span");
    const swatch = element("i");
    swatch.style.setProperty("background", `var(--${color})`);
    item.append(swatch, label);
    legendRow.append(item);
  }
  const foot = element("div", "device-foot");
  const footLeft = element("span");
  const footRight = element("span");
  foot.append(footLeft, footRight);
  card.append(head, metrics, chart, legendRow);
  return { card, kindLabel, name, temp, metrics, chart, foot, footLeft, footRight };
}

function heat(value) {
  return value >= 80 ? "hot" : value >= 60 ? "warm" : "cool";
}

function buildHardware(gpus) {
  charts.clear();
  const container = $("hardware");
  container.replaceChildren();
  hardware.gpus = gpus.map(({ device, sensor }) => {
    const view = deviceCard(t("gpu"), [["decode", t("busy")], ["heat", "°C"], ["power", t("power")]]);
    view.pci = sensor?.pci || device?.pci;
    view.busy = metric("primary", t("busy"), "%");
    view.power = metric("power", t("power"), "W");
    view.vram = metric("memory", t("vram"), "GiB");
    view.metrics.append(view.busy.node, view.power.node, view.vram.node);
    view.bar = element("div", "bar");
    view.used = element("span", "used");
    view.other = element("span", "other");
    view.mark = element("span", "mark");
    view.bar.append(view.used, view.other, view.mark);
    view.card.append(view.bar, view.foot);
    view.timechart = new TimeChart(view.chart, [
      { key: "busy", color: "decode", axis: "left", fill: true, glow: true },
      { key: "temp", color: "heat", axis: "left", width: 1.6 },
      { key: "power", color: "power", axis: "left", width: 1.4, dash: true },
    ], { left: { max: 100 } });
    container.append(view.card);
    return view;
  });

  const cpu = deviceCard(t("cpu"), [["decode", t("usage")], ["heat", "°C"]]);
  cpu.usage = metric("primary", t("usage"), "%");
  cpu.load = metric("", t("load"), "");
  cpu.server = metric("power", t("straydog"), "%");
  cpu.metrics.append(cpu.usage.node, cpu.load.node, cpu.server.node);
  cpu.cores = element("div", "cores");
  cpu.card.append(cpu.cores, cpu.foot);
  cpu.timechart = new TimeChart(cpu.chart, [
    { key: "usage", color: "decode", axis: "left", fill: true, glow: true },
    { key: "temp", color: "heat", axis: "left", width: 1.6 },
  ], { left: { max: 100 } });
  container.append(cpu.card);
  hardware.cpu = cpu;

  const memory = deviceCard(t("memory"), [["prefill", t("used")], ["decode", t("serverResident")]]);
  memory.card.classList.add("memory-card");
  memory.used = metric("memory", t("used"), "%");
  memory.amount = metric("", t("used"), "GiB");
  memory.server = metric("primary", t("serverResident"), "GiB");
  memory.metrics.append(memory.used.node, memory.amount.node, memory.server.node);
  memory.bar = element("div", "bar");
  memory.usedBar = element("span", "used");
  memory.bar.append(memory.usedBar);
  memory.scope = element("div", "memory-scope", t("rssScope"));
  memory.card.append(memory.bar, memory.scope, memory.foot);
  memory.timechart = new TimeChart(memory.chart, [
    { key: "used", color: "prefill", axis: "left", fill: true, glow: true },
    { key: "server", color: "decode", axis: "left", width: 1.6 },
  ], { left: { max: 100 } });
  container.append(memory.card);
  hardware.memory = memory;

  const disk = deviceCard(t("disk"), [["decode", t("diskRead")], ["prefill", t("serverReadIo")]]);
  disk.card.classList.add("disk");
  disk.name.textContent = t("allVolumes");
  disk.read = metric("primary", t("diskRead"), "MiB/s");
  disk.server = metric("memory", t("serverReadIo"), "MiB/s");
  disk.metrics.append(disk.read.node, disk.server.node);
  disk.timechart = new TimeChart(disk.chart, [
    { key: "read", color: "decode", axis: "left", fill: true, glow: true },
    { key: "server", color: "prefill", axis: "left", width: 1.6 },
  ], { left: { max: "auto", floor: 1 } });
  disk.footLeft.textContent = t("diskScope");
  disk.footRight.textContent = t("ioScope");
  disk.card.append(disk.foot);
  container.append(disk.card);
  hardware.disk = disk;

  container.style.setProperty("--cards", String(gpus.length + 3));
  charts.add(speedChart);
}

/// Charts get their points from the samples: every one after a rebuild, new ones otherwise.
function feed(sample) {
  for (const view of hardware.gpus) {
    const gpu = sample.gpus.find((item) => item.pci === view.pci);
    view.timechart.push("busy", sample.time, gpu?.busy);
    view.timechart.push("temp", sample.time, gpu?.temperature);
    view.timechart.push("power", sample.time, gpu?.power);
  }
  hardware.cpu?.timechart.push("usage", sample.time, sample.cpu);
  hardware.cpu?.timechart.push("temp", sample.time, sample.cpu_temperature);
  hardware.memory?.timechart.push("used", sample.time, sample.memory);
  hardware.memory?.timechart.push("server", sample.time, sample.server_memory);
  hardware.disk?.timechart.push("read", sample.time, sample.disk_read);
  hardware.disk?.timechart.push("server", sample.time, sample.server_read_io);
  speedChart.push("decode", sample.time, sample.decode);
  speedChart.push("prefill", sample.time, sample.prefill);
}

/// The hardware cards; true when they were built anew.
function renderHardware(engine, host) {
  if (!host || !host.server) return false;
  const gpus = gpuList(engine, host);
  const key = gpus.map((item) => item.sensor?.pci || item.device?.pci || item.device?.device).join("|");
  const rebuilt = key !== hardwareKey || !hardware.cpu;
  if (rebuilt) {
    hardwareKey = key;
    buildHardware(gpus);
  }
  gpus.forEach(({ device, sensor }, index) => {
    const view = hardware.gpus[index];
    view.kindLabel.textContent = device ? `${t("gpu")} ${device.device}` : t("gpu");
    view.name.textContent = device?.name || sensor?.pci || "";
    const temps = (sensor?.temperatures || []).map((item) => item.celsius).filter(Number.isFinite);
    const hottest = temps.length ? Math.max(...temps) : null;
    view.temp.textContent = celsius(hottest);
    view.temp.dataset.heat = heat(hottest);
    view.busy.value.textContent = number(sensor?.busy_percent);
    view.power.value.textContent = number(sensor?.power_watts);
    const total = sensor?.vram_total ?? device?.total_bytes;
    const used = sensor?.vram_used ?? (device ? device.total_bytes - device.free_bytes : null);
    view.vram.value.textContent = gib(used);
    view.vram.unit.textContent = `/ ${gib(total)} GiB`;
    const allocated = device?.allocated_bytes ?? 0;
    view.used.style.setProperty("width", `${percent(allocated, total) ?? 0}%`);
    view.other.style.setProperty("left", `${percent(allocated, total) ?? 0}%`);
    view.other.style.setProperty("width", `${percent(Math.max(0, used - allocated), total) ?? 0}%`);
    view.mark.hidden = !device?.budget_bytes;
    view.mark.style.setProperty("left", `${Math.min(99.5, percent(device?.budget_bytes, total) ?? 0)}%`);
    view.footLeft.textContent = (sensor?.temperatures || []).map((item) => `${item.label} ${number(item.celsius)}°`).join(" · ")
      || (sensor ? "" : t("noSensors"));
    view.footRight.textContent = [sensor?.fan_rpm != null ? `${number(sensor.fan_rpm)} rpm` : null,
      sensor?.fan_percent != null ? `${t("fan")} ${number(sensor.fan_percent)}%` : null,
      sensor?.clock_mhz != null ? `${number(sensor.clock_mhz)} MHz` : null].filter(Boolean).join(" · ");
  });

  const cpu = hardware.cpu;
  cpu.name.textContent = (host.cpu || "").replace(/\(R\)|\(TM\)|\s+\d+-Core Processor|\s+Processor/gi, "");
  cpu.temp.textContent = celsius(host.cpu_temperature);
  cpu.temp.dataset.heat = heat(host.cpu_temperature);
  cpu.usage.value.textContent = number(host.cpu_usage);
  cpu.load.value.textContent = host.load_average ? number(host.load_average[0], 2) : "—";
  const serverShare = host.server.cpu_usage / Math.max(host.logical_cores, 1);
  cpu.server.value.textContent = number(serverShare, 1);
  const cores = host.cpu_cores || [];
  if (cpu.cores.childElementCount !== cores.length) cpu.cores.replaceChildren(...cores.map(() => element("i")));
  cores.forEach((usage, index) => cpu.cores.children[index].style.setProperty("--level", `${Math.min(100, usage)}%`));
  cpu.footLeft.textContent = t("coresValue", host.physical_cores ?? "—", host.logical_cores);
  cpu.footRight.textContent = host.load_average ? host.load_average.map((value) => number(value, 2)).join(" / ") : "";

  const memory = hardware.memory;
  const used = host.memory_total - host.memory_available;
  memory.name.textContent = `${gib(host.memory_total)} GiB`;
  memory.used.value.textContent = number(percent(used, host.memory_total));
  memory.amount.value.textContent = gib(used);
  memory.server.value.textContent = Number.isFinite(host.server.memory) ? gib(host.server.memory) : "—";
  memory.server.unit.textContent = Number.isFinite(host.server.memory) ? "GiB" : "";
  memory.usedBar.style.setProperty("width", `${Math.min(100, Math.max(0, percent(used, host.memory_total) ?? 0))}%`);
  memory.footLeft.textContent = `${t("available")} ${gib(host.memory_available)} GiB`;
  memory.footRight.textContent = `${t("swap")} ${gib(host.swap_used)} / ${gib(host.swap_total)} GiB`;
  hardware.disk.read.value.textContent = number(mibRate(host.disk_read_bytes_per_second), 1);
  hardware.disk.server.value.textContent = number(mibRate(host.server_read_io_bytes_per_second), 1);
  hardware.disk.footRight.textContent = t(host.server_read_io_source === "proc_read_bytes" ? "ioScopeLinux"
    : host.server_read_io_source === "io_counters" ? "ioScope" : "ioScopeUnknown");
  return rebuilt;
}

// ---- speed ----

const speedChart = new TimeChart($("speed-chart"), [
  { key: "prefill", color: "prefill", axis: "right", width: 1.6 },
  { key: "decode", color: "decode", axis: "left", fill: true, glow: true, width: 2.2 },
], { left: { max: "auto", floor: 20 }, right: { max: "auto", floor: 200 } });
speedChart.onAxes = (maxima) => {
  $("speed-axis-left").textContent = `${number(maxima.left / 1.2)} tok/s`;
  $("speed-axis-right").textContent = `${number(maxima.right / 1.2)} tok/s`;
};

// A request that generated at most one token has no decode rate.
const decodeRate = (entry) => (entry.completion_tokens > 1 ? entry.decode_tokens_per_second : null);
// A prompt's new tokens and their time; their rate only from 256 on (a few tokens' time is the request's fixed cost).
const PREFILL_RATE_TOKENS = 256;
const prefillFresh = (entry) => entry.prefill_tokens ?? Math.max(0, entry.prompt_tokens - entry.cached_tokens);
const prefillMs = (entry) => (Number.isFinite(entry.prefill_seconds) ? entry.prefill_seconds * 1000 : null);
const prefillRate = (entry) => {
  const fresh = prefillFresh(entry);
  if (fresh < PREFILL_RATE_TOKENS) return null;
  return entry.prefill_seconds > 0 ? fresh / entry.prefill_seconds : entry.prefill_tokens_per_second ?? null;
};
// "新 29 トークン · 108 ms" (with the rate from 256 tokens on), or "cached".
const prefillText = (entry) => {
  const fresh = prefillFresh(entry);
  if (!fresh) return t("cached");
  const ms = prefillMs(entry), rate = prefillRate(entry);
  return [t("newTokens", number(fresh)), ms === null ? null : `${number(ms)} ms`, rate === null ? null : `${number(rate)} tok/s`]
    .filter(Boolean).join(" · ");
};
// Speculative acceptance: the drafts kept of the drafts verified, whichever drafters made them.
const acceptance = (entry) => (entry.drafted_tokens > 0 ? percent(entry.accepted_tokens, entry.drafted_tokens) : null);
const drafterName = (kind) => ({ dspark: t("dspark"), lookup: t("lookup"), mtp: t("mtpName") })[kind] || kind;
// "DSpark 78% · lookup 97%": each drafter's share of its own drafts kept in a request.
const drafterShares = (entry) => (entry.drafters || [])
  .map((drafter) => `${drafterName(drafter.kind)} ${number(percent(drafter.accepted, drafter.drafted))}%`).join(" · ");

// What the engine runs, as its ready frame says ("" before it is ready).
function speculationLabel(ready, detail = false) {
  if (!ready || (ready.speculation === undefined && ready.speculative_drafts === undefined)) return "";
  const kinds = Array.isArray(ready.speculation) ? ready.speculation
    : ready.speculative_drafts ? [{ kind: "mtp", drafts: ready.speculative_drafts }] : [];
  if (!kinds.length) return t("off");
  return kinds.map((kind) => {
    if (kind.kind === "mtp") return `${drafterName(kind.kind)} ${t("drafts", kind.drafts)}`;
    return detail && kind.resident ? `${drafterName(kind.kind)} ${kind.resident.join("/")}` : drafterName(kind.kind);
  }).join(" · ");
}

function renderSpeed(activity, ready, queued, sample) {
  const live = activity.live;
  const decodeNow = live && live.tokens > 0 && sample?.decode > 0 ? sample.decode : null;
  const box = $("live");
  const waiting = queued ? ` · ${t("queued", queued)}` : "";
  let barWidth = "0";
  if (live && live.tokens > 0) {
    box.dataset.phase = "decode";
    $("live-phase").textContent = t("decoding");
    $("live-detail").textContent = `${t("tokens", number(live.tokens))} · ${number(decodeNow, 1)} tok/s${waiting}`;
    barWidth = "100%";
  } else if (live && live.prefill_total > 0) {
    const share = percent(live.prefill_done, live.prefill_total);
    box.dataset.phase = "prefill";
    $("live-phase").textContent = t("prefilling");
    $("live-detail").textContent = `${number(live.prefill_done)} / ${number(live.prefill_total)} · ${number(share)}%${waiting}`;
    barWidth = `${share}%`;
  } else if (live) {
    box.dataset.phase = "prefill";
    $("live-phase").textContent = t("preparing");
    $("live-detail").textContent = `${number(live.seconds, 1)} s${waiting}`;
  } else {
    box.dataset.phase = "idle";
    $("live-phase").textContent = t("idle");
    const last = activity.recent.at(-1);
    $("live-detail").textContent = last ? t("lastAt", clock(last.time)) : "";
  }
  $("live-bar").style.setProperty("width", barWidth);

  const last = activity.recent.at(-1);
  $("decode-value").textContent = number(decodeNow ?? (last ? decodeRate(last) : null), 1);
  $("decode-note").textContent = decodeNow !== null ? t("live") : last ? t("lastRequest") : "";
  $("prefill-value").textContent = number(last && prefillFresh(last) ? prefillMs(last) : null);
  $("prefill-note").textContent = last ? prefillText(last) : "";
  $("ttft-value").textContent = number(last?.ttft_seconds, 2);
  $("ttft-note").textContent = last ? t("lastRequest") : "";
  $("speculative-value").textContent = number(last ? acceptance(last) : null);
  $("speculative-note").textContent = (last && drafterShares(last)) || speculationLabel(ready);
}

// ---- requests ----

let chartEntries = [];
let hovered = null;

function prepareChart(id) {
  const node = $(id).querySelector("svg");
  node.replaceChildren();
  if (id === "decode-chart") {
    const gradient = svg("linearGradient", { id: "decode-fill", x1: 0, y1: 0, x2: 0, y2: 1 });
    gradient.append(svg("stop", { offset: 0, class: "stop-a" }), svg("stop", { offset: 1, class: "stop-b" }));
    const defs = svg("defs");
    defs.append(gradient);
    node.append(defs);
  }
  for (const level of [0.25, 0.5, 0.75, 1]) {
    node.append(svg("line", { x1: 0, x2: 600, y1: 120 - level * 110, y2: 120 - level * 110, class: "grid" }));
  }
  return node;
}

function renderRequests(recent) {
  chartEntries = recent.slice(-40);
  $("chart-empty").hidden = chartEntries.length > 0;
  $("request-count").textContent = recent.length ? `${number(recent.length)}` : "";
  const slot = 600 / Math.max(chartEntries.length, 1);
  const x = (index) => slot * index + slot / 2;

  const decode = prepareChart("decode-chart");
  const decodes = chartEntries.map(decodeRate);
  const decodeTop = Math.max(...decodes.filter(Number.isFinite), 0) * 1.1 || 1;
  $("decode-axis").textContent = chartEntries.length ? t("peak", number(decodeTop / 1.1, 1)) : "";
  const points = decodes.map((value, index) => [index, value]).filter(([, value]) => Number.isFinite(value));
  const y = (value) => 120 - (value / decodeTop) * 110;
  if (points.length) {
    const line = points.map(([index, value], at) => `${at ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join("");
    if (points.length > 1) {
      decode.append(svg("path", { d: `${line}L${x(points.at(-1)[0]).toFixed(1)},120L${x(points[0][0]).toFixed(1)},120Z`, class: "area" }));
    }
    decode.append(svg("path", { d: line, class: "line" }));
    for (const [index, value] of points) {
      decode.append(svg("circle", { cx: x(index), cy: y(value), r: 2.6, class: "dot", "data-index": index }));
    }
  }

  const prefill = prepareChart("prefill-chart");
  const prefills = chartEntries.map(prefillRate);
  const prefillTop = Math.max(...prefills.filter(Number.isFinite), 0) * 1.1 || 1;
  $("prefill-axis").textContent = chartEntries.length ? t("peak", number(prefillTop / 1.1)) : "";
  prefills.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    const height = Math.max((value / prefillTop) * 110, 1);
    const width = Math.min(slot * 0.64, 24);
    prefill.append(svg("rect", { x: x(index) - width / 2, width, y: 120 - height, height, rx: 2, class: "bar-rect", "data-index": index }));
  });
  if (hovered !== null) highlight(hovered);
}

function highlight(index) {
  for (const node of document.querySelectorAll(".requests [data-index]")) {
    node.classList.toggle("active", Number(node.dataset.index) === index);
  }
}

function showTooltip(event) {
  if (!chartEntries.length) return;
  const card = document.querySelector(".requests");
  const area = event.currentTarget.querySelector("svg").getBoundingClientRect();
  const index = Math.min(chartEntries.length - 1, Math.max(0, Math.floor(((event.clientX - area.left) / area.width) * chartEntries.length)));
  const entry = chartEntries[index];
  hovered = index;
  highlight(index);
  const tooltip = $("tooltip");
  const list = element("dl");
  const accepted = acceptance(entry);
  facts(list, [
    [t("prompt"), `${number(entry.prompt_tokens)} (${t("cached")} ${number(entry.cached_tokens)})`],
    [t("completion"), number(entry.completion_tokens)],
    [t("prefill"), prefillText(entry)],
    [t("decode"), decodeRate(entry) === null ? "—" : `${number(decodeRate(entry), 1)} tok/s`],
    [t("ttft"), `${number(entry.ttft_seconds, 2)} s`],
    accepted === null ? null : [t("acceptance"), `${number(accepted)}% (${drafterShares(entry)})`],
    [t("finish"), entry.finish_reason],
  ]);
  tooltip.replaceChildren(element("b", "", clock(entry.time)), list);
  tooltip.hidden = false;
  const box = card.getBoundingClientRect();
  const left = Math.min(event.clientX - box.left + 14, box.width - tooltip.offsetWidth - 8);
  tooltip.style.setProperty("left", `${Math.max(8, left)}px`);
  // Inside the card: a later card would cover it.
  const below = event.clientY - box.top + 14;
  const top = below + tooltip.offsetHeight > box.height - 8 ? event.clientY - box.top - tooltip.offsetHeight - 14 : below;
  tooltip.style.setProperty("top", `${Math.max(8, top)}px`);
}

for (const chart of document.querySelectorAll(".requests .chart")) {
  chart.addEventListener("pointermove", showTooltip);
  chart.addEventListener("pointerleave", () => { hovered = null; highlight(-1); $("tooltip").hidden = true; });
}

// ---- engine, environment ----

function renderEngine(engine, ready, activity) {
  const placement = ready.placement || {};
  const stateLabel = { ready: activity.live ? t("generating") : t("ready"), loading: t("loading"), failed: t("failed") }[engine.state];
  const reuse = percent(activity.cached_tokens, activity.prompt_tokens);
  const isReady = engine.state === "ready";
  // A model read on demand: engines over the GPUs (each over one or several, splitting its layers),
  // routed experts in VRAM slots and a host tier.
  const tiered = Array.isArray(placement.expert_slots);
  const items = [
    [t("state"), engine.state === "failed" ? `${stateLabel} · ${engine.error}` : stateLabel, "state"],
    isReady && !tiered && [t("placement"), placement.resident ? t("fullyResident") : t("partiallyResident")],
    isReady && !tiered && [t("kv"), placement.host_kv ? t("hostMemory") : t("vramKv")],
    isReady && tiered && [t("layers"), (placement.layers || [])
      .map(([first, end], index) => t("layersValue", first, end - 1, ((placement.devices || [])[index] || []).join("+") || "-")).join(" · ")],
    isReady && tiered && [t("expertSlots"), t("expertSlotsValue", placement.expert_slots.map((slots) => number(slots)).join(" + "))],
    isReady && tiered && [t("expertHost"), sizeGiB(placement.host_expert_bytes)],
    isReady && [t("context"), number(ready.context)],
    isReady && Array.isArray(placement.vram_budgets) && [t("vramBudget"), `${placement.vram_budgets.map((bytes) => gib(bytes)).join(" + ")} GiB`],
    isReady && [t("prefillBatch"), number(placement.prefill_batch_size)],
    isReady && [t("speculation"), speculationLabel(ready, true)],
    isReady && [t("sequenceState"), `${gib(ready.state_bytes)} GiB`],
    isReady && [t("loadedIn"), `${number(ready.loaded_seconds)} s`],
    [t("queue"), number(engine.queued)],
    [t("requestsDone"), activity.failed ? `${number(activity.requests)} · ${t("failedCount", number(activity.failed))}` : number(activity.requests)],
    [t("promptTokens"), number(activity.prompt_tokens)],
    [t("reused"), reuse === null ? "—" : `${number(reuse)}%`],
    [t("generated"), number(activity.completion_tokens)],
  ].filter(Boolean);
  $("engine-strip").replaceChildren(...items.map(([label, value, kind]) => {
    const item = element("div", kind || "");
    if (kind === "state") item.dataset.state = engine.state;
    item.append(element("small", "", label), element("b", "", value));
    return item;
  }));
}

let settingsShown = "";

function renderEnvironment(engine, host, server) {
  if (host) {
    $("host-chip").textContent = host.host_name || "";
    facts($("host-facts"), [
      [t("os"), host.os],
      [t("kernel"), host.kernel],
      [t("architecture"), host.architecture],
      [t("cpu"), host.cpu],
      [t("cores"), t("coresValue", host.physical_cores ?? "—", host.logical_cores)],
      [t("cpuTemperature"), celsius(host.cpu_temperature) || "—"],
      host.load_average && [t("loadAverage"), host.load_average.map((value) => number(value, 2)).join(" / ")],
      [t("memory"), `${gib(host.memory_total - host.memory_available)} / ${gib(host.memory_total)} GiB`],
      [t("swap"), `${gib(host.swap_used)} / ${gib(host.swap_total)} GiB`],
      [t("hostUptime"), duration(host.uptime_seconds)],
      [t("serverProcesses"), `${number(host.server.processes)} (PID ${host.server.pid})`],
      [t("serverMemory"), sizeGiB(host.server.memory)],
      engine.state === "ready" && Array.isArray(engine.ready?.placement?.expert_slots)
        && [t("expertHost"), sizeGiB(engine.ready.placement.host_expert_bytes)],
      [t("diskRead"), `${number(mibRate(host.disk_read_bytes_per_second), 1)} MiB/s`],
      [t("serverReadIo"), `${number(mibRate(host.server_read_io_bytes_per_second), 1)} MiB/s`],
      [t("serverCpu"), `${number(host.server.cpu_usage)}%`],
    ]);
    const gpus = gpuList(engine, host);
    const names = gpus.map((item) => item.device?.name).filter(Boolean);
    $("gpu-summary").textContent = names.length > 1 && names.every((name) => name === names[0]) ? `${names.length}× ${names[0]}` : "";
    $("gpu-facts").replaceChildren(...gpus.map(({ device, sensor }) => {
      const section = element("section");
      section.append(element("h3", "", device ? `${t("gpu")} ${device.device} · ${device.name}` : sensor?.pci || t("gpu")));
      const list = element("dl", "facts");
      facts(list, [
        [t("pci"), device?.pci || sensor?.pci],
        ...(sensor?.temperatures || []).map((item) => [item.label, `${number(item.celsius)}°C${item.critical ? ` / ${number(item.critical)}°C` : ""}`]),
        sensor && [t("busy"), `${number(sensor.busy_percent)}%`],
        sensor && [t("power"), `${number(sensor.power_watts, 1)} W`],
        sensor?.power_cap_watts != null && [t("powerCap"), `${number(sensor.power_cap_watts)} W`],
        sensor?.fan_rpm != null && [t("fan"), `${number(sensor.fan_rpm)} rpm`],
        sensor?.fan_percent != null && [t("fan"), `${number(sensor.fan_percent)}%`],
        sensor?.clock_mhz != null && [t("clock"), `${number(sensor.clock_mhz)} MHz`],
        sensor?.vram_total != null && [t("deviceMemory"), `${gib(sensor.vram_used)} / ${gib(sensor.vram_total)} GiB`],
        device && [t("allocated"), t("budgetOf", gib(device.allocated_bytes), gib(device.budget_bytes))],
        !sensor && [t("noSensors"), ""],
      ]);
      section.append(list);
      return section;
    }));
  }
  const speculation = speculationLabel(engine.ready, true);
  const key = JSON.stringify([server, speculation]);
  if (key === settingsShown) return;
  settingsShown = key;
  $("version-chip").textContent = `v${server.version} · ${server.backend}`;
  facts($("settings-facts"), [
    [t("model"), server.model, "path"],
    [t("quantization"), server.quantization],
    // (the MTP drafter is the Qwen3.8 bundles'; DeepSeek-V4.1's DSpark is a file beside its model)
    server.mtp && [t("drafter"), server.model_draft || t("none"), server.model_draft ? "path" : ""],
    server.mtp && [t("draftMax"), server.model_draft ? number(server.draft_max) : t("off")],
    [t("speculation"), speculation || "—"],
    [t("context"), t("tokens", number(server.context))],
    [t("prefillBatch"), server.batch_size ? number(server.batch_size) : t("auto")],
    [t("devices"), server.devices.join(", ")],
    [t("vramBudget"), server.vram_gib ? `${server.vram_gib.map((value) => number(value, 1)).join(" + ")} GiB` : t("auto")],
    [t("splitMode"), server.split_mode],
    [t("stateRam"), server.state_ram_gib > 0 ? `${number(server.state_ram_gib, 1)} GiB` : t("auto")],
    [t("gemm"), server.gemm],
    [t("gemmMin"), number(server.gemm_min_batch)],
    [t("address"), server.address],
    [t("apiKey"), server.api_key ? t("required") : t("notRequired")],
    [t("defaultTemp"), server.temperature == null ? t("modelCard") : number(server.temperature, 2)],
    [t("topK"), server.top_k == null ? t("modelCard") : number(server.top_k)],
    [t("topP"), server.top_p == null ? t("modelCard") : number(server.top_p, 2)],
    [t("minP"), server.min_p == null ? t("modelCard") : number(server.min_p, 2)],
    [t("presencePenalty"), server.presence_penalty == null ? t("modelCard") : number(server.presence_penalty, 2)],
    [t("defaultSeed"), server.seed == null ? t("freshSeed") : String(server.seed)],
    [t("defaultEffort"), server.reasoning_effort],
    [t("startupTimeout"), `${number(server.startup_timeout)} s`],
  ]);
}

// ---- the service log ----

let logNext = 0;
let logTimer = 0;
const LOG_DOM_LINES = 2000;
const LOG_LINE = /^(\S+)\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+(.*)$/s;

function logVisible(item) {
  const filter = $("log-filter").value.trim().toLowerCase();
  const warnings = $("log-warnings").checked;
  return (!warnings || item.dataset.level === "WARN" || item.dataset.level === "ERROR")
    && (!filter || item.dataset.text.includes(filter));
}

function logItem(line) {
  const match = LOG_LINE.exec(line);
  const [stamp, level, text] = match ? [match[1], match[2], match[3]] : ["", "", line];
  const date = stamp ? new Date(stamp) : null;
  const item = element("li", level);
  const time = element("time", "", date && !Number.isNaN(date.getTime()) ? date.toLocaleTimeString(LANG, { hour12: false }) : "");
  if (stamp) time.dateTime = stamp;
  item.dataset.level = level;
  item.dataset.text = line.toLowerCase();
  item.append(time, element("span", "level", level), element("span", "text", text));
  item.hidden = !logVisible(item);
  return item;
}

async function pollLogs() {
  clearTimeout(logTimer);
  let delay = $("tab-logs").hidden ? 4000 : 1500;
  try {
    const response = await fetch(`/dashboard/logs?after=${logNext}`, { headers: authorized(), cache: "no-store" });
    if (response.ok) {
      const { next, lines } = await response.json();
      if (next < logNext) $("log").replaceChildren();  // the server restarted
      logNext = next;
      if (lines.length) {
        const log = $("log");
        log.append(...lines.map(logItem));
        while (log.childElementCount > LOG_DOM_LINES) log.firstElementChild.remove();
        if ($("log-follow").checked) log.scrollTop = log.scrollHeight;
        if (lines.length >= 500) delay = 50;
      }
    }
  } catch { /* the status poll reports an unreachable server */ }
  logTimer = setTimeout(pollLogs, document.hidden ? 5000 : delay);
}

function refilterLog() {
  for (const item of $("log").children) item.hidden = !logVisible(item);
  if ($("log-follow").checked) $("log").scrollTop = $("log").scrollHeight;
}

$("log-filter").addEventListener("input", refilterLog);
$("log-warnings").addEventListener("change", refilterLog);
$("log-follow").addEventListener("change", refilterLog);
$("log").addEventListener("scroll", () => {
  const log = $("log");
  $("log-follow").checked = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
});

// ---- chat ----

const conversation = [];
let controller = null;
let effortChosen = false;

$("effort").addEventListener("change", () => { effortChosen = true; });

function setChatOpen(open) {
  document.body.classList.toggle("chat-open", open);
  $("chat").setAttribute("aria-hidden", String(!open));
  if (open) $("prompt").focus();
}
$("chat-open").addEventListener("click", () => setChatOpen(true));
$("chat-close").addEventListener("click", () => setChatOpen(false));
$("chat-expand").addEventListener("click", () => $("chat").classList.toggle("wide"));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("chat-open") && !$("key-dialog").open) setChatOpen(false);
});

function renderText(node, text) {
  // Fenced code becomes a block; everything else stays text.
  node.replaceChildren(...text.split("```").map((part, index) => {
    if (index % 2 === 0) return document.createTextNode(part);
    const block = element("pre");
    block.append(element("code", "", part.replace(/^[\w+-]*\n/, "")));
    return block;
  }));
}

function messageElement(role) {
  $("chat-empty").hidden = true;
  const node = element("div", `message ${role}`);
  $("messages").append(node);
  return node;
}

function nearBottom(node) {
  return node.scrollHeight - node.scrollTop - node.clientHeight < 80;
}

function setBusy(busy) {
  const button = $("send");
  button.textContent = busy ? t("stop") : t("send");
  button.classList.toggle("stop", busy);
  button.type = busy ? "button" : "submit";
  $("chat-open").classList.toggle("busy", busy);
}

async function send() {
  const text = $("prompt").value.trim();
  if (!text || controller) return;
  $("prompt").value = "";
  resizePrompt();
  conversation.push({ role: "user", content: text });
  messageElement("user").textContent = text;

  const view = messageElement("assistant");
  const reasoningBox = element("details");
  const summary = element("summary", "", t("reasoning"));
  const reasoningText = element("div");
  reasoningBox.append(summary, reasoningText);
  reasoningBox.hidden = true;
  const content = element("div", "typing", t("thinking"));
  const meta = element("small", "meta");
  view.append(reasoningBox, content, meta);
  const messages = $("messages");
  messages.scrollTop = messages.scrollHeight;

  const answer = { role: "assistant", content: "", reasoning: "" };
  let timings = null;
  let usage = null;
  let finish = null;
  let failure = null;
  let frame = 0;
  const draw = () => {
    frame = 0;
    const follow = nearBottom(messages);
    if (answer.reasoning) {
      reasoningBox.hidden = false;
      reasoningText.textContent = answer.reasoning;
      if (!answer.content && !reasoningBox.dataset.touched) reasoningBox.open = true;
    }
    if (answer.content) {
      if (!reasoningBox.dataset.touched) reasoningBox.open = false;
      renderText(content, answer.content);
    }
    if (follow) messages.scrollTop = messages.scrollHeight;
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(draw); };
  summary.addEventListener("click", () => { reasoningBox.dataset.touched = "1"; });

  const body = {
    messages: conversation.map((message) => (message.role === "assistant"
      ? { role: "assistant", content: message.content, ...(message.reasoning ? { reasoning_content: message.reasoning } : {}) }
      : message)),
    stream: true,
    stream_options: { include_usage: true },
    reasoning_effort: $("effort").value,
  };
  const temperature = $("temperature").value.trim();
  if (temperature) body.temperature = Number(temperature);
  const limit = $("max-tokens").value.trim();
  if (limit) body.max_tokens = Number(limit);

  controller = new AbortController();
  setBusy(true);
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authorized({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      if (response.status === 401) openKeyDialog();
      const error = await response.json().catch(() => null);
      throw new Error(error?.error?.message || `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let errorEvent = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, "");
        buffer = buffer.slice(newline + 1);
        if (line.startsWith(":")) {
          const progress = /prefill (\d+)\/(\d+)/.exec(line);
          if (progress && !answer.content && !answer.reasoning) {
            content.textContent = `${t("prefilling")} ${number(percent(Number(progress[1]), Number(progress[2])))}%`;
          }
        } else if (line.startsWith("event: error")) {
          errorEvent = true;
        } else if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (errorEvent) throw new Error(chunk.error?.message || t("error"));
          const choice = chunk.choices?.[0];
          if (choice?.delta?.reasoning_content) answer.reasoning += choice.delta.reasoning_content;
          if (choice?.delta?.content) answer.content += choice.delta.content;
          if (choice?.finish_reason) finish = choice.finish_reason;
          if (chunk.timings) timings = chunk.timings;
          if (chunk.usage) usage = chunk.usage;
          schedule();
        }
      }
    }
  } catch (error) {
    failure = error.name === "AbortError" ? t("stopped") : error.message;
  } finally {
    controller = null;
    setBusy(false);
  }
  if (frame) cancelAnimationFrame(frame);
  draw();
  content.classList.remove("typing");
  if (!answer.content && !answer.reasoning) content.textContent = "";
  const notes = [];
  if (usage) {
    notes.push(t("tokens", number(usage.completion_tokens)));
    const cached = usage.prompt_tokens_details?.cached_tokens;
    if (cached) notes.push(`${t("cached")} ${number(cached)} / ${number(usage.prompt_tokens)}`);
  }
  if (timings?.decode_tokens_per_second && usage?.completion_tokens > 1) notes.push(`${number(timings.decode_tokens_per_second, 1)} tok/s`);
  if (timings?.ttft_seconds) notes.push(`${t("ttft")} ${number(timings.ttft_seconds, 2)} s`);
  if (finish === "length") notes.push(t("length"));
  if (failure) {
    notes.push(failure);
    view.classList.toggle("error", !answer.content && !answer.reasoning);
  }
  meta.textContent = notes.join(" · ");
  if (answer.content || answer.reasoning) conversation.push(answer);
  else conversation.pop();
  poll();
}

function resizePrompt() {
  const prompt = $("prompt");
  prompt.style.setProperty("height", "auto");
  prompt.style.setProperty("height", `${Math.min(prompt.scrollHeight + 2, 200)}px`);
}

$("composer").addEventListener("submit", (event) => { event.preventDefault(); send(); });
$("send").addEventListener("click", () => { if (controller) controller.abort(); });
$("prompt").addEventListener("input", resizePrompt);
$("prompt").addEventListener("keydown", (event) => {
  // Enter while an IME composes confirms the conversion, it does not send.
  if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  send();
});
$("clear").addEventListener("click", () => {
  controller?.abort();
  conversation.length = 0;
  $("messages").replaceChildren($("chat-empty"));
  $("chat-empty").hidden = false;
});

document.getElementById("lang-button").addEventListener("click", () => {
  LANG = LANG === "en" ? "ja" : "en";
  try { localStorage.setItem(LANG_STORAGE, LANG); } catch { /* storage unavailable */ }
  numberFormats.clear();
  translatePage();
  hardwareKey = "";
  poll();
});

translatePage();
readColors();
showTab(location.hash.slice(1));
setState("connecting", t("connecting"));
requestAnimationFrame(animate);
poll();
pollLogs();
