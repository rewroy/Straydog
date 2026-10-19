// The Straydog dashboard: polls /dashboard/status and /dashboard/logs, and chats through
// /v1/chat/completions. Everything is rendered with DOM text nodes, never as HTML.

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

// ---- language ----

const STRINGS = {
  en: {
    connecting: "Connecting…", ready: "Ready", generating: "Generating", loading: "Loading", failed: "Failed",
    offline: "Server unreachable", unauthorized: "API key required", uptime: "Uptime", apiKey: "API key",
    apiKeyHelp: "This server requires its API key. It is kept in this tab only.", save: "Save", cancel: "Cancel",
    decode: "Decode", prefill: "Prefill", firstToken: "First token", mtp: "MTP acceptance", live: "live",
    lastRequest: "last request", cached: "from cache", off: "off", throughput: "Throughput", idle: "Idle",
    prefilling: "Prefilling", decoding: "Generating", preparing: "Starting", queued: "{0} waiting",
    lastAt: "last request at {0}", decodePerRequest: "Decode tok/s per request", prefillPerRequest: "Prefill tok/s per request",
    peak: "peak {0}", noRequests: "No requests yet. Ask something in the chat.", chat: "Chat", effort: "Reasoning",
    effortNone: "none", effortLow: "low", effortMedium: "medium", effortHigh: "high", temperature: "Temp",
    maxTokens: "Max", auto: "auto", clear: "Clear", send: "Send", stop: "Stop",
    chatEmpty: "Messages go to this server's model. Reasoning appears folded above each answer.",
    promptPlaceholder: "Message Straydog (Enter to send, Shift+Enter for a new line)", reasoning: "Reasoning",
    thinking: "Thinking…", stopped: "stopped", length: "cut at the token limit", gpus: "GPUs", waiting: "Waiting for the engine…",
    engine: "Engine", host: "Host", cpu: "CPU", memory: "Memory", serverMemory: "Straydog memory", settings: "Settings",
    logs: "Service log", filter: "Filter", warningsOnly: "Warnings only", follow: "Follow",
    state: "State", placement: "Placement", fullyResident: "fully resident", partiallyResident: "GPU and CPU experts",
    kv: "K/V cache", vram: "VRAM", hostMemory: "host memory", context: "Context", tokens: "{0} tokens",
    prefillBatch: "Prefill batch", drafts: "{0} drafts", sequenceState: "Sequence state", loadedIn: "Loaded in",
    queue: "Queue", requests: "Requests", failedCount: "{0} failed", promptTokens: "Prompt tokens", reused: "Reused from cache",
    generated: "Generated tokens", hostName: "Host name", os: "OS", kernel: "Kernel", architecture: "Architecture",
    cores: "Cores", coresValue: "{0} cores / {1} threads", load: "Load average", swap: "Swap", hostUptime: "Host uptime",
    serverProcesses: "Straydog processes", serverCpu: "Straydog CPU", model: "Model", quantization: "Quantization",
    drafter: "MTP drafter", none: "none", draftMax: "MTP drafts", devices: "GPUs", vramBudget: "VRAM per GPU",
    splitMode: "Split", stateRam: "Host memory for state", gemm: "GEMM precision", gemmMin: "GEMM from batch",
    address: "Address", required: "required", notRequired: "not required", defaultTemp: "Default temperature",
    defaultSeed: "Default seed", defaultEffort: "Default reasoning", startupTimeout: "Startup timeout",
    budget: "{0} of {1} GiB budget", free: "{0} GiB free of {1}", gpuExperts: "GPU experts", cpuExperts: "CPU experts",
    swapped: "Swapped", time: "Time", prompt: "Prompt", completion: "Completion", ttft: "First token", finish: "Finish",
    acceptance: "MTP accepted", stageAuthenticate: "authenticating the model", stageDrafter: "authenticating the drafter",
    stagePrepare: "placing the model on the GPUs", error: "Error",
  },
  ja: {
    connecting: "接続中…", ready: "待機中", generating: "生成中", loading: "読み込み中", failed: "停止",
    offline: "サーバーに接続できません", unauthorized: "API キーが必要です", uptime: "稼働時間", apiKey: "API キー",
    apiKeyHelp: "このサーバーには API キーが必要です。キーはこのタブの中だけに保存します。", save: "保存", cancel: "キャンセル",
    decode: "デコード", prefill: "プレフィル", firstToken: "最初のトークン", mtp: "MTP 採択率", live: "ライブ",
    lastRequest: "直近のリクエスト", cached: "キャッシュ", off: "オフ", throughput: "処理速度", idle: "待機中",
    prefilling: "プレフィル中", decoding: "生成中", preparing: "開始中", queued: "{0} 件待ち",
    lastAt: "直近のリクエスト {0}", decodePerRequest: "リクエストごとのデコード tok/s", prefillPerRequest: "リクエストごとのプレフィル tok/s",
    peak: "最大 {0}", noRequests: "まだリクエストがありません。チャットで話しかけてみてください。", chat: "チャット",
    effort: "思考", effortNone: "なし", effortLow: "低", effortMedium: "中", effortHigh: "高", temperature: "温度",
    maxTokens: "上限", auto: "自動", clear: "消去", send: "送信", stop: "停止",
    chatEmpty: "このサーバーのモデルと会話します。思考内容は各回答の上に折りたたんで表示します。",
    promptPlaceholder: "メッセージを入力（Enter で送信、Shift+Enter で改行）", reasoning: "思考",
    thinking: "考えています…", stopped: "停止しました", length: "トークン上限で終了", gpus: "GPU", waiting: "エンジンを待っています…",
    engine: "エンジン", host: "ホスト", cpu: "CPU", memory: "メモリ", serverMemory: "Straydog のメモリ", settings: "設定",
    logs: "サービスログ", filter: "絞り込み", warningsOnly: "警告のみ", follow: "追従",
    state: "状態", placement: "配置", fullyResident: "全常駐", partiallyResident: "GPU と CPU のエキスパート",
    kv: "K/V キャッシュ", vram: "VRAM", hostMemory: "ホストメモリ", context: "コンテキスト", tokens: "{0} トークン",
    prefillBatch: "プレフィルバッチ", drafts: "{0} ドラフト", sequenceState: "系列状態", loadedIn: "読み込み時間",
    queue: "待ち行列", requests: "リクエスト", failedCount: "失敗 {0}", promptTokens: "プロンプト", reused: "キャッシュ再利用",
    generated: "生成トークン", hostName: "ホスト名", os: "OS", kernel: "カーネル", architecture: "アーキテクチャ",
    cores: "コア", coresValue: "{0} コア / {1} スレッド", load: "ロードアベレージ", swap: "スワップ", hostUptime: "ホスト稼働時間",
    serverProcesses: "Straydog のプロセス", serverCpu: "Straydog の CPU", model: "モデル", quantization: "量子化",
    drafter: "MTP drafter", none: "なし", draftMax: "MTP ドラフト", devices: "GPU", vramBudget: "GPU ごとの VRAM",
    splitMode: "分割", stateRam: "系列状態のホストメモリ", gemm: "GEMM 精度", gemmMin: "GEMM 開始バッチ",
    address: "アドレス", required: "必要", notRequired: "不要", defaultTemp: "既定の温度",
    defaultSeed: "既定のシード", defaultEffort: "既定の思考", startupTimeout: "起動タイムアウト",
    budget: "予算 {1} GiB 中 {0}", free: "{1} GiB 中 {0} GiB 空き", gpuExperts: "GPU エキスパート", cpuExperts: "CPU エキスパート",
    swapped: "入れ替え", time: "時刻", prompt: "プロンプト", completion: "生成", ttft: "最初のトークン", finish: "終了理由",
    acceptance: "MTP 採択", stageAuthenticate: "モデルを検証中", stageDrafter: "drafter を検証中",
    stagePrepare: "モデルを GPU に配置中", error: "エラー",
  },
};
const LANG = (navigator.language || "en").toLowerCase().startsWith("ja") ? "ja" : "en";

function t(key, ...args) {
  let text = STRINGS[LANG][key] ?? STRINGS.en[key] ?? key;
  args.forEach((arg, index) => { text = text.replace(`{${index}}`, arg); });
  return text;
}

function translatePage() {
  document.documentElement.lang = LANG;
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
const percent = (part, whole) => (whole > 0 ? (100 * part) / whole : null);
const clock = (seconds) => new Date(seconds * 1000).toLocaleTimeString(LANG, { hour12: false });

function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const [d, h, m] = [Math.floor(s / 86400), Math.floor((s % 86400) / 3600), Math.floor((s % 3600) / 60)];
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function svg(tag, attributes = {}) {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

function facts(list, rows) {
  list.replaceChildren(...rows.filter(Boolean).flatMap(([label, value, kind]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value ?? "—";
    if (kind) detail.className = kind;
    return [term, detail];
  }));
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

// ---- status ----

let status = null;
let pollTimer = 0;
const liveSamples = [];
let lastLive = null;

function setState(state, text, progress = null) {
  const pill = $("status");
  pill.dataset.state = state;
  $("status-text").textContent = text;
  $("status-bar").style.setProperty("width", progress === null ? "0" : `${progress}%`);
}

async function poll() {
  clearTimeout(pollTimer);
  let delay = 1000;
  try {
    const response = await fetch("/dashboard/status", { headers: authorized(), cache: "no-store" });
    if (response.status === 401) {
      setState("unauthorized", t("unauthorized"));
      openKeyDialog();
      delay = 5000;
    } else if (!response.ok) {
      throw new Error(String(response.status));
    } else {
      status = await response.json();
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
  const name = { authenticate: t("stageAuthenticate"), drafter: t("stageDrafter"), prepare: t("stagePrepare") }[progress.stage];
  return name || progress.stage || "";
}

function render({ now, server, engine, host }) {
  const ready = engine.ready || {};
  const activity = engine.activity;
  $("model-line").textContent = [server.alias, server.quantization, `${server.backend} · v${server.version}`].join(" · ");
  $("uptime").textContent = duration(now - server.started);
  if (engine.state === "loading") {
    const progress = engine.progress || {};
    const share = progress.total ? percent(progress.done, progress.total) : null;
    const label = [t("loading"), stageText(progress), share === null ? "" : `${number(share)}%`].filter(Boolean).join(" · ");
    setState("loading", label, share);
  } else if (engine.state === "failed") {
    setState("failed", `${t("failed")} · ${engine.error}`);
  } else if (activity.live) {
    setState("busy", engine.queued ? `${t("generating")} · ${t("queued", engine.queued)}` : t("generating"));
  } else {
    setState("ready", t("ready"));
  }
  renderLive(now, activity, engine.queued);
  renderTiles(activity, ready);
  renderCharts(activity.recent);
  renderGpus(engine);
  renderEngine(engine, ready, activity);
  renderHost(host);
  renderSettings(server);
  if (!effortChosen) $("effort").value = server.reasoning_effort;
  $("temperature").placeholder = number(server.temperature, 1);
}

// ---- speed ----

function sparkline(element, values) {
  const points = values.map((value, index) => [index, value]).filter(([, value]) => Number.isFinite(value));
  element.replaceChildren();
  if (points.length < 2) return;
  const [width, height] = element.getAttribute("viewBox").split(" ").slice(2).map(Number);
  const top = Math.max(...points.map(([, value]) => value)) * 1.15 || 1;
  const x = (index) => (index / Math.max(values.length - 1, 1)) * width;
  const y = (value) => height - (value / top) * height;
  const line = points.map(([index, value], at) => `${at ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join("");
  const first = x(points[0][0]).toFixed(1);
  const last = x(points.at(-1)[0]).toFixed(1);
  element.append(svg("path", { d: `${line}L${last},${height}L${first},${height}Z`, class: "fill" }),
                 svg("path", { d: line, class: "stroke" }));
}

function liveRate(now, live) {
  // A new generation restarts the samples.
  if (!lastLive || live.seconds < lastLive.seconds) liveSamples.length = 0;
  let rate = null;
  if (lastLive && live.seconds >= lastLive.seconds && live.tokens > lastLive.tokens && now > lastLive.now) {
    rate = (live.tokens - lastLive.tokens) / (now - lastLive.now);
    liveSamples.push(rate);
    if (liveSamples.length > 90) liveSamples.shift();
  }
  lastLive = { now, seconds: live.seconds, tokens: live.tokens };
  return rate ?? (live.decode_seconds > 0 ? live.tokens / live.decode_seconds : null);
}

let currentLiveRate = null;

function renderLive(now, activity, queued) {
  const live = activity.live;
  const box = $("live");
  const waiting = queued ? ` · ${t("queued", queued)}` : "";
  if (live) {
    currentLiveRate = live.tokens > 0 ? liveRate(now, live) : null;
    if (live.tokens > 0) {
      box.dataset.phase = "decode";
      $("live-phase").textContent = t("decoding");
      $("live-detail").textContent = `${t("tokens", number(live.tokens))} · ${number(currentLiveRate, 1)} tok/s · ${number(live.seconds, 1)} s${waiting}`;
      $("live-bar").style.setProperty("width", "100%");
    } else if (live.prefill_total > 0) {
      const share = percent(live.prefill_done, live.prefill_total);
      box.dataset.phase = "prefill";
      $("live-phase").textContent = t("prefilling");
      $("live-detail").textContent = `${number(live.prefill_done)} / ${number(live.prefill_total)} · ${number(share)}%${waiting}`;
      $("live-bar").style.setProperty("width", `${share}%`);
    } else {
      box.dataset.phase = "prefill";
      $("live-phase").textContent = t("preparing");
      $("live-detail").textContent = `${number(live.seconds, 1)} s${waiting}`;
      $("live-bar").style.setProperty("width", "0");
    }
  } else {
    lastLive = null;
    currentLiveRate = null;
    box.dataset.phase = "idle";
    $("live-phase").textContent = t("idle");
    const last = activity.recent.at(-1);
    $("live-detail").textContent = last ? t("lastAt", clock(last.time)) : "";
    $("live-bar").style.setProperty("width", "0");
  }
  sparkline($("live-spark"), liveSamples);
}

// A request that generated at most one token has no decode rate.
const decodeRate = (entry) => (entry.completion_tokens > 1 ? entry.decode_tokens_per_second : null);
const acceptance = (entry) => (entry.drafted_tokens > 0 ? percent(entry.accepted_tokens, entry.drafted_tokens) : null);
const prefillRate = (entry) => (entry.prompt_tokens > entry.cached_tokens ? entry.prefill_tokens_per_second : null);

function renderTiles(activity, ready) {
  const recent = activity.recent;
  const last = recent.at(-1);
  const series = recent.slice(-24);
  if (currentLiveRate !== null) {
    $("decode-value").textContent = number(currentLiveRate, 1);
    $("decode-note").textContent = t("live");
  } else {
    $("decode-value").textContent = number(last ? decodeRate(last) : null, 1);
    $("decode-note").textContent = last ? t("lastRequest") : "";
  }
  sparkline($("decode-spark"), series.map(decodeRate));

  const prefill = last ? prefillRate(last) : null;
  $("prefill-value").textContent = number(prefill);
  $("prefill-note").textContent = last && prefill === null ? t("cached") : last ? t("lastRequest") : "";
  sparkline($("prefill-spark"), series.map(prefillRate));

  $("ttft-value").textContent = number(last?.ttft_seconds, 2);
  $("ttft-note").textContent = last ? t("lastRequest") : "";
  sparkline($("ttft-spark"), series.map((entry) => entry.ttft_seconds));

  const accepted = last ? acceptance(last) : null;
  $("mtp-value").textContent = number(accepted);
  $("mtp-note").textContent = ready.speculative_drafts === 0 ? t("off") : ready.speculative_drafts ? t("drafts", ready.speculative_drafts) : "";
  sparkline($("mtp-spark"), series.map(acceptance));
}

let chartEntries = [];

function prepareChart(id) {
  const element = $(id).querySelector("svg");
  element.replaceChildren();
  if (id === "decode-chart") {
    const gradient = svg("linearGradient", { id: "decode-fill", x1: 0, y1: 0, x2: 0, y2: 1 });
    gradient.append(svg("stop", { offset: 0, class: "stop-a" }), svg("stop", { offset: 1, class: "stop-b" }));
    const defs = svg("defs");
    defs.append(gradient);
    element.append(defs);
  }
  for (const level of [0.25, 0.5, 0.75, 1]) {
    element.append(svg("line", { x1: 0, x2: 600, y1: 120 - level * 110, y2: 120 - level * 110, class: "grid" }));
  }
  return element;
}

function renderCharts(recent) {
  chartEntries = recent.slice(-40);
  $("chart-empty").hidden = chartEntries.length > 0;
  const slot = 600 / Math.max(chartEntries.length, 1);
  const x = (index) => slot * index + slot / 2;

  const decode = prepareChart("decode-chart");
  const decodes = chartEntries.map(decodeRate);
  const decodeTop = Math.max(...decodes.filter(Number.isFinite), 0) * 1.1 || 1;
  $("decode-axis").textContent = chartEntries.length ? t("peak", `${number(decodeTop / 1.1, 1)} tok/s`) : "";
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
  $("prefill-axis").textContent = chartEntries.length ? t("peak", `${number(prefillTop / 1.1)} tok/s`) : "";
  prefills.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    const height = Math.max((value / prefillTop) * 110, 1);
    const width = Math.min(slot * 0.64, 24);
    prefill.append(svg("rect", { x: x(index) - width / 2, width, y: 120 - height, height, rx: 2, class: "bar", "data-index": index }));
  });
  if (hovered !== null) highlight(hovered);
}

let hovered = null;

function highlight(index) {
  for (const element of document.querySelectorAll(".throughput [data-index]")) {
    element.classList.toggle("active", Number(element.dataset.index) === index);
  }
}

function showTooltip(event) {
  const card = document.querySelector(".throughput");
  const chart = event.currentTarget.querySelector("svg").getBoundingClientRect();
  if (!chartEntries.length) return;
  const index = Math.min(chartEntries.length - 1, Math.max(0, Math.floor(((event.clientX - chart.left) / chart.width) * chartEntries.length)));
  const entry = chartEntries[index];
  hovered = index;
  highlight(index);
  const tooltip = $("tooltip");
  const title = document.createElement("b");
  title.textContent = clock(entry.time);
  const list = document.createElement("dl");
  const accepted = acceptance(entry);
  facts(list, [
    [t("prompt"), `${number(entry.prompt_tokens)} (${t("cached")} ${number(entry.cached_tokens)})`],
    [t("completion"), number(entry.completion_tokens)],
    [t("prefill"), prefillRate(entry) === null ? t("cached") : `${number(prefillRate(entry))} tok/s`],
    [t("decode"), decodeRate(entry) === null ? "—" : `${number(decodeRate(entry), 1)} tok/s`],
    [t("ttft"), `${number(entry.ttft_seconds, 2)} s`],
    accepted === null ? null : [t("acceptance"), `${number(accepted)}%`],
    [t("finish"), entry.finish_reason],
  ]);
  tooltip.replaceChildren(title, list);
  tooltip.hidden = false;
  const box = card.getBoundingClientRect();
  const left = Math.min(event.clientX - box.left + 14, box.width - tooltip.offsetWidth - 8);
  tooltip.style.setProperty("left", `${Math.max(8, left)}px`);
  // Inside the card: a later card would cover it.
  const below = event.clientY - box.top + 14;
  const top = below + tooltip.offsetHeight > box.height - 8 ? event.clientY - box.top - tooltip.offsetHeight - 14 : below;
  tooltip.style.setProperty("top", `${Math.max(8, top)}px`);
}

for (const chart of document.querySelectorAll(".throughput .chart")) {
  chart.addEventListener("pointermove", showTooltip);
  chart.addEventListener("pointerleave", () => { hovered = null; highlight(-1); $("tooltip").hidden = true; });
}

// ---- GPUs, engine, host, settings ----

function renderGpus(engine) {
  const list = $("gpu-list");
  const gpu = engine.ready?.gpu;
  if (!gpu) {
    const note = document.createElement("p");
    note.className = "empty";
    note.textContent = engine.state === "failed" ? engine.error : t("waiting");
    list.replaceChildren(note);
    $("gpu-summary").textContent = "";
    return;
  }
  const devices = Array.isArray(gpu.devices) && gpu.devices.length ? gpu.devices : [gpu];
  const names = devices.map((device) => device.name);
  $("gpu-summary").textContent = names.every((name) => name === names[0]) && names.length > 1 ? `${names.length}× ${names[0]}` : "";
  list.replaceChildren(...devices.map((device) => {
    const item = document.createElement("article");
    item.className = "gpu";
    const head = document.createElement("header");
    const name = document.createElement("span");
    name.textContent = device.name;
    const ordinal = document.createElement("small");
    ordinal.textContent = `GPU ${device.device}`;
    head.append(name, ordinal);
    const meter = document.createElement("div");
    meter.className = "meter";
    const total = device.total_bytes || 1;
    const other = Math.max(0, total - device.free_bytes - device.allocated_bytes);
    const used = document.createElement("span");
    used.className = "used";
    used.style.setProperty("width", `${(100 * device.allocated_bytes) / total}%`);
    const others = document.createElement("span");
    others.className = "other";
    others.style.setProperty("left", `${(100 * device.allocated_bytes) / total}%`);
    others.style.setProperty("width", `${(100 * other) / total}%`);
    const mark = document.createElement("span");
    mark.className = "mark";
    mark.style.setProperty("left", `${Math.min(99.6, (100 * device.budget_bytes) / total)}%`);
    meter.append(used, others, mark);
    const line = document.createElement("p");
    const budget = document.createElement("span");
    budget.textContent = t("budget", gib(device.allocated_bytes), gib(device.budget_bytes));
    const free = document.createElement("span");
    free.textContent = t("free", gib(device.free_bytes), gib(device.total_bytes));
    line.append(budget, free);
    item.append(head, meter, line);
    return item;
  }));
  if (engine.ready.placement?.partial) {
    const hybrid = document.createElement("div");
    hybrid.className = "hybrid";
    for (const [key, label] of [["gpu_experts", t("gpuExperts")], ["cpu_experts", t("cpuExperts")], ["swapped_experts", t("swapped")]]) {
      const cell = document.createElement("span");
      const value = document.createElement("b");
      value.textContent = number(gpu[key]);
      cell.append(value, label);
      hybrid.append(cell);
    }
    list.append(hybrid);
  }
}

function renderEngine(engine, ready, activity) {
  const placement = ready.placement || {};
  const stateLabel = { ready: activity.live ? t("generating") : t("ready"), loading: t("loading"), failed: t("failed") }[engine.state];
  $("engine-chip").textContent = ready.quantization || "";
  const reuse = percent(activity.cached_tokens, activity.prompt_tokens);
  facts($("engine-facts"), [
    [t("state"), engine.state === "failed" ? `${stateLabel} · ${engine.error}` : stateLabel],
    engine.state === "ready" && [t("placement"), placement.resident ? t("fullyResident") : t("partiallyResident")],
    engine.state === "ready" && [t("kv"), placement.host_kv ? t("hostMemory") : t("vram")],
    engine.state === "ready" && [t("context"), t("tokens", number(ready.context))],
    engine.state === "ready" && [t("prefillBatch"), number(placement.prefill_batch_size)],
    engine.state === "ready" && ["MTP", ready.speculative_drafts ? t("drafts", ready.speculative_drafts) : t("off")],
    engine.state === "ready" && [t("sequenceState"), `${gib(ready.state_bytes)} GiB`],
    engine.state === "ready" && [t("loadedIn"), `${number(ready.loaded_seconds)} s`],
    [t("queue"), number(engine.queued)],
    [t("requests"), activity.failed ? `${number(activity.requests)} · ${t("failedCount", number(activity.failed))}` : number(activity.requests)],
    [t("promptTokens"), number(activity.prompt_tokens)],
    [t("reused"), reuse === null ? "—" : `${number(activity.cached_tokens)} (${number(reuse)}%)`],
    [t("generated"), number(activity.completion_tokens)],
  ]);
}

function gauge(id, share, value, label) {
  const element = $(id);
  const circumference = 2 * Math.PI * 32;
  const filled = Number.isFinite(share) ? Math.min(Math.max(share, 0), 100) / 100 : 0;
  element.querySelector(".value").style.setProperty("stroke-dashoffset", String(circumference * (1 - filled)));
  element.querySelector("b").textContent = value;
  element.querySelector("small").textContent = label;
}

function renderHost(host) {
  if (!host) return;
  const used = host.memory_total - host.memory_available;
  gauge("cpu-gauge", host.cpu_usage, `${number(host.cpu_usage)}%`, t("cpu"));
  gauge("memory-gauge", percent(used, host.memory_total), `${number(percent(used, host.memory_total))}%`,
        `${t("memory")} · ${gib(used)} / ${gib(host.memory_total)} GiB`);
  gauge("server-gauge", percent(host.server.memory, host.memory_total), `${gib(host.server.memory)}`,
        `${t("serverMemory")} (GiB)`);
  $("host-chip").textContent = host.host_name || "";
  facts($("host-facts"), [
    [t("os"), host.os],
    [t("kernel"), host.kernel],
    [t("cpu"), host.cpu],
    [t("cores"), t("coresValue", host.physical_cores ?? "—", host.logical_cores)],
    host.load_average && [t("load"), host.load_average.map((value) => number(value, 2)).join(" / ")],
    [t("swap"), `${gib(host.swap_used)} / ${gib(host.swap_total)} GiB`],
    [t("hostUptime"), duration(host.uptime_seconds)],
    [t("serverProcesses"), `${number(host.server.processes)} (PID ${host.server.pid})`],
    [t("serverCpu"), `${number(host.server.cpu_usage)}%`],
  ]);
}

let settingsShown = "";

function renderSettings(server) {
  const key = JSON.stringify(server);
  if (key === settingsShown) return;
  settingsShown = key;
  $("version-chip").textContent = `v${server.version} · ${server.backend}`;
  facts($("settings-facts"), [
    [t("model"), server.model, "path"],
    [t("quantization"), server.quantization],
    [t("drafter"), server.model_draft || t("none"), server.model_draft ? "path" : ""],
    [t("draftMax"), server.model_draft ? number(server.draft_max) : t("off")],
    [t("context"), t("tokens", number(server.context))],
    [t("prefillBatch"), number(server.batch_size)],
    [t("devices"), server.devices.join(", ")],
    [t("vramBudget"), `${number(server.vram_gib, 1)} GiB`],
    [t("splitMode"), server.split_mode],
    [t("stateRam"), server.state_ram_gib > 0 ? `${number(server.state_ram_gib, 1)} GiB` : t("auto")],
    [t("gemm"), server.gemm],
    [t("gemmMin"), number(server.gemm_min_batch)],
    [t("address"), server.address],
    [t("apiKey"), server.api_key ? t("required") : t("notRequired")],
    [t("defaultTemp"), number(server.temperature, 2)],
    [t("defaultSeed"), String(server.seed)],
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
  const item = document.createElement("li");
  const match = LOG_LINE.exec(line);
  const [stamp, level, text] = match ? [match[1], match[2], match[3]] : ["", "", line];
  const date = stamp ? new Date(stamp) : null;
  const time = document.createElement("time");
  time.textContent = date && !Number.isNaN(date.getTime()) ? date.toLocaleTimeString(LANG, { hour12: false }) : "";
  if (stamp) time.dateTime = stamp;
  const badge = document.createElement("span");
  badge.className = "level";
  badge.textContent = level;
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = text;
  item.className = level;
  item.dataset.level = level;
  item.dataset.text = line.toLowerCase();
  item.append(time, badge, body);
  item.hidden = !logVisible(item);
  return item;
}

async function pollLogs() {
  clearTimeout(logTimer);
  let delay = 1500;
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
  } catch { /* the status poll reports the server as unreachable */ }
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

function renderText(element, text) {
  // Fenced code becomes a block; everything else stays text.
  const parts = text.split("```");
  element.replaceChildren(...parts.map((part, index) => {
    if (index % 2 === 0) return document.createTextNode(part);
    const block = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = part.replace(/^[\w+-]*\n/, "");
    block.append(code);
    return block;
  }));
}

function messageElement(role) {
  $("chat-empty").hidden = true;
  const element = document.createElement("div");
  element.className = `message ${role}`;
  $("messages").append(element);
  return element;
}

function nearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 80;
}

function setBusy(busy) {
  const button = $("send");
  button.textContent = busy ? t("stop") : t("send");
  button.classList.toggle("stop", busy);
  button.type = busy ? "button" : "submit";
}

async function send() {
  const text = $("prompt").value.trim();
  if (!text || controller) return;
  $("prompt").value = "";
  resizePrompt();
  conversation.push({ role: "user", content: text });
  const user = messageElement("user");
  user.textContent = text;

  const view = messageElement("assistant");
  const reasoningBox = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = t("reasoning");
  const reasoningText = document.createElement("div");
  reasoningBox.append(summary, reasoningText);
  reasoningBox.hidden = true;
  const content = document.createElement("div");
  content.className = "typing";
  content.textContent = t("thinking");
  const meta = document.createElement("small");
  meta.className = "meta";
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
  if (timings?.decode_tokens_per_second) notes.push(`${number(timings.decode_tokens_per_second, 1)} tok/s`);
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

translatePage();
setState("connecting", t("connecting"));
poll();
pollLogs();
