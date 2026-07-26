const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("ask-button");
const messagesEl = document.getElementById("messages");
const suggestionsEl = document.getElementById("suggestions");
const threadEl = document.getElementById("thread");
const sourcesPanel = document.getElementById("sources-panel");
const sourcesList = document.getElementById("sources-list");
const sourcesItems = document.getElementById("sources-items");
const sourcesEmpty = document.getElementById("sources-empty");
const sourceCount = document.getElementById("source-count");
const composerCount = document.getElementById("composer-count");
const collapseSources = document.getElementById("collapse-sources");
const expandSources = document.getElementById("expand-sources");
const menuButton = document.getElementById("menu-button");
const menu = document.getElementById("menu");
const reindexButton = document.getElementById("reindex-button");
const reindexStatus = document.getElementById("reindex-status");
const reindexWarnings = document.getElementById("reindex-warnings");
const newConversationButton = document.getElementById("new-conversation-button");

const GREETING =
  "Bonjour Medhy, c'est moi Dona, que puis-je pour toi aujourd'hui ?\n\n"
  + "Je cherche dans tes documents et tes agendas, puis je te réponds en citant "
  + "les sources utilisées - elles s'affichent dans le panneau de gauche.";

const STARTERS = [
  "Quel est mon numéro de SIRET ?",
  "J'ai quoi de prévu aujourd'hui ?",
  "Combien d'impôts j'ai payé en 2024 ?",
];

// Radar axes: the agent's capabilities, including the ones not wired yet -
// an unused axis simply stays flat until its tools exist
const AXES = ["Réflexion", "Documents", "Agenda pro", "Agenda perso",
              "Recherche web", "Qonto", "Mail pro", "Mail perso"];

// Tool name -> axes it lights up, and the phrase shown while it runs.
// Anchored patterns, most specific first: the web route is the catch-all
const TOOL_ROUTES = [
  [/^rag_/, [1], "Je cherche dans tes documents"],
  [/^calendar_find_event/, [2, 3], "Je cherche dans tes agendas"],
  [/^calendar_pro/, [2], "Je consulte ton agenda pro"],
  [/^calendar_perso/, [3], "Je consulte ton agenda perso"],
  [/^qonto/, [5], "Je vérifie tes transactions"],
  [/^(gmail|mail)_pro/, [6], "Je lis ta boîte mail pro"],
  [/^(gmail|mail)_perso/, [7], "Je lis ta boîte mail perso"],
  [/^(web|search|tavily|brave)/, [4], "Je cherche sur le web"],
];

const PHASES = {
  thinking: (event) => [[0], "Je réfléchis à ta demande", `${event.tokens} tokens`],
  tool_prep: () => [[0], "Je prépare un appel d'outil", "…"],
  answer: (event) => [[0], "Je rédige ta réponse", `${event.tokens} tokens`],
};

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) el.setAttribute(key, value);
  return el;
}

// One thread per conversation: kept across reloads, renewed from the menu
let threadId = localStorage.getItem("thread_id") || crypto.randomUUID();
localStorage.setItem("thread_id", threadId);

const citedSources = new Map();

function scrollDown() {
  requestAnimationFrame(() => { threadEl.scrollTop = threadEl.scrollHeight; });
}

function addUserMessage(text) {
  const message = document.createElement("div");
  message.className = "message-user";
  message.textContent = text;
  messagesEl.appendChild(message);
  scrollDown();
}

function addBotMessage(html, { plainText = false, error = false } = {}) {
  const message = document.createElement("div");
  message.className = error ? "message-bot error" : "message-bot";
  if (plainText) {
    message.textContent = html;
  } else {
    message.innerHTML = html;
  }
  messagesEl.appendChild(message);
  scrollDown();
  return message;
}

function addWarning(text) {
  const warning = document.createElement("div");
  warning.className = "message-warning";
  warning.textContent = text;
  messagesEl.appendChild(warning);
}

function showSuggestions(questions) {
  suggestionsEl.textContent = "";
  for (const question of questions) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = question;
    chip.addEventListener("click", () => ask(question));
    suggestionsEl.appendChild(chip);
  }
}

function updateSourceCount() {
  const count = citedSources.size;
  const label = `${count} ${count === 1 ? "source" : "sources"}`;
  sourceCount.textContent = label;
  composerCount.textContent = label;
  sourcesList.hidden = count === 0;
  sourcesEmpty.hidden = count > 0;
}

function addSources(sources) {
  for (const source of sources || []) {
    const key = `${source.name}#${source.page}`;
    if (citedSources.has(key)) continue;
    citedSources.set(key, source);

    const card = document.createElement("a");
    card.className = "source-card";
    // #page=N opens the browser PDF viewer directly on the cited page
    card.href = source.url + (source.page !== null ? `#page=${source.page}` : "");
    card.target = "_blank";
    card.title = "Ouvrir le document";

    const icon = document.createElement("span");
    icon.className = "source-icon";
    const body = document.createElement("span");
    body.className = "source-body";
    const title = document.createElement("span");
    title.className = "source-title";
    title.textContent = source.name;
    const desc = document.createElement("span");
    desc.className = "source-desc";
    desc.textContent = source.page !== null ? `Page ${source.page}` : "Document";

    body.append(title, desc);
    card.append(icon, body);
    sourcesItems.appendChild(card);
  }
  updateSourceCount();
}

// Live activity radar: the shape morphs toward the axis of whatever the
// agent is doing right now (thinking, or the tool being called)
let radarSeq = 0;

function startRadar() {
  const size = AXES.length;
  const RADIUS = 92;
  const gradientId = `radarFill${radarSeq}`;
  const glowId = `radarGlow${radarSeq}`;
  radarSeq += 1;

  const values = AXES.map((_, i) => (i === 0 ? 70 : 12));
  let from = values.slice();
  let to = values.slice();
  let morphStart = performance.now();
  const visited = new Set([0]);
  let active = [0];
  let ghost = "";
  let running = true;

  const point = (index, value) => {
    const angle = (index * Math.PI * 2) / size;
    const r = (RADIUS * Math.max(0, Math.min(105, value))) / 100;
    return [r * Math.sin(angle), -r * Math.cos(angle)];
  };

  // Closed Catmull-Rom spline, as smooth as the mock's curve
  const spline = (points) => {
    const t = 1 / 6;
    let d = `M ${points[0][0].toFixed(2)} ${points[0][1].toFixed(2)}`;
    for (let i = 0; i < points.length; i += 1) {
      const p0 = points[(i - 1 + size) % size];
      const p1 = points[i];
      const p2 = points[(i + 1) % size];
      const p3 = points[(i + 2) % size];
      const c1 = [p1[0] + (p2[0] - p0[0]) * t, p1[1] + (p2[1] - p0[1]) * t];
      const c2 = [p2[0] - (p3[0] - p1[0]) * t, p2[1] - (p3[1] - p1[1]) * t];
      d += ` C ${c1[0].toFixed(2)} ${c1[1].toFixed(2)} ${c2[0].toFixed(2)} ${c2[1].toFixed(2)}`
         + ` ${p2[0].toFixed(2)} ${p2[1].toFixed(2)}`;
    }
    return `${d} Z`;
  };

  const radar = document.createElement("div");
  radar.className = "radar";
  const avatar = document.createElement("div");
  avatar.className = "radar-avatar";
  const card = document.createElement("div");
  card.className = "radar-card";

  const head = document.createElement("div");
  head.className = "radar-head";
  const dot = document.createElement("span");
  dot.className = "radar-dot";
  const title = document.createElement("span");
  title.className = "radar-title";
  title.textContent = "Je réfléchis à ta demande";
  const elapsed = document.createElement("span");
  elapsed.className = "radar-elapsed";
  elapsed.textContent = "0:00";
  head.append(dot, title, elapsed);

  const body = document.createElement("div");
  body.className = "radar-body";
  const plot = document.createElement("div");
  plot.className = "radar-plot";
  const tip = document.createElement("div");
  tip.className = "radar-tip";
  tip.textContent = AXES[0];

  const svg = svgEl("svg", { viewBox: "-160 -128 320 256" });
  const defs = svgEl("defs");
  const gradient = svgEl("radialGradient", { id: gradientId, cx: "50%", cy: "50%", r: "50%" });
  gradient.append(
    svgEl("stop", { offset: "0%", "stop-color": "#6fb3ab", "stop-opacity": "0.26" }),
    svgEl("stop", { offset: "100%", "stop-color": "#6fb3ab", "stop-opacity": "0.05" }));
  const filter = svgEl("filter", { id: glowId, x: "-60%", y: "-60%", width: "220%", height: "220%" });
  filter.appendChild(svgEl("feGaussianBlur", { stdDeviation: "4.5", result: "b" }));
  const merge = svgEl("feMerge");
  merge.append(svgEl("feMergeNode", { in: "b" }), svgEl("feMergeNode", { in: "SourceGraphic" }));
  filter.appendChild(merge);
  defs.append(gradient, filter);

  const rings = svgEl("g", { stroke: "#dde3e7", fill: "none" });
  [23, 46, 69, 92].forEach((r, i) => {
    rings.appendChild(svgEl("circle", { r, "stroke-opacity": i === 3 ? "0.11" : "0.06" }));
  });
  const spokes = svgEl("g", { stroke: "#dde3e7", "stroke-opacity": "0.08" });
  AXES.forEach((_, i) => {
    const [x, y] = point(i, 100);
    spokes.appendChild(svgEl("line", { x1: 0, y1: 0, x2: x.toFixed(2), y2: y.toFixed(2) }));
  });

  const ghostPath = svgEl("path", {
    fill: "none", stroke: "#6fb3ab", "stroke-width": "1.2",
    "stroke-opacity": "0", "stroke-linejoin": "round" });
  const areaPath = svgEl("path", { fill: `url(#${gradientId})`, stroke: "none" });
  const strokePath = svgEl("path", {
    fill: "none", stroke: "#6fb3ab", "stroke-width": "2",
    "stroke-linejoin": "round", filter: `url(#${glowId})` });
  const dots = AXES.map(() => svgEl("circle", {
    r: "2.4", fill: "#1a1f24", stroke: "#8fcac2",
    "stroke-width": "1.7", "stroke-opacity": "0.42" }));

  svg.append(defs, rings, spokes, ghostPath, areaPath, strokePath, ...dots);
  plot.append(tip, svg);
  body.appendChild(plot);

  const detail = document.createElement("div");
  detail.className = "radar-detail";
  const progress = document.createElement("div");
  progress.className = "radar-progress";
  progress.appendChild(document.createElement("span"));

  card.append(head, body, detail, progress);
  radar.append(avatar, card);
  messagesEl.appendChild(radar);
  scrollDown();

  const started = performance.now();
  const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

  const frame = (now) => {
    if (!running) return;
    const raw = Math.min(1, (now - morphStart) / 900);
    const stagger = 0.6;
    const span = 1 / (1 + stagger);
    const points = values.map((_, i) => {
      const start = (i / (size - 1)) * stagger * span;
      const p = Math.max(0, Math.min(1, (raw - start) / span));
      values[i] = from[i] + (to[i] - from[i]) * ease(p);
      return point(i, values[i]);
    });

    const d = spline(points);
    areaPath.setAttribute("d", d);
    strokePath.setAttribute("d", d);
    ghostPath.setAttribute("d", ghost);
    ghostPath.setAttribute("stroke-opacity", ((1 - raw) * 0.35).toFixed(3));
    points.forEach(([x, y], i) => {
      const isActive = active.includes(i);
      dots[i].setAttribute("cx", x.toFixed(2));
      dots[i].setAttribute("cy", y.toFixed(2));
      dots[i].setAttribute("r", isActive ? "3.6" : "2.4");
      dots[i].setAttribute("stroke-opacity", isActive ? "1" : "0.42");
    });

    const [tx, ty] = point(active[0], 118);
    tip.style.left = `${(((tx + 160) / 320) * 100).toFixed(2)}%`;
    tip.style.top = `${(((ty + 128) / 256) * 100).toFixed(2)}%`;
    tip.style.transform = "translate(-50%, -50%)";

    const shown = Math.floor((now - started) / 1000);
    elapsed.textContent = `${Math.floor(shown / 60)}:${String(shown % 60).padStart(2, "0")}`;
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  // Target shape: active axes spike, already-used ones keep a memory bump,
  // untouched ones stay flat - the curve mirrors what the agent really does
  const morphTo = (axes) => {
    axes.forEach((axis) => visited.add(axis));
    active = axes;
    ghost = spline(values.map((v, i) => point(i, v)));
    from = values.slice();
    to = AXES.map((_, i) => {
      if (axes.includes(i)) return 92;
      if (i === 0) return 52;
      return visited.has(i) ? 40 : 12;
    });
    morphStart = performance.now();
    tip.textContent = AXES[axes[0]];
  };

  return {
    update(axes, statusTitle, detailText) {
      if (axes && (axes[0] !== active[0] || axes.length !== active.length)) morphTo(axes);
      if (statusTitle) title.textContent = statusTitle;
      if (detailText !== undefined) detail.textContent = detailText;
    },
    remove() {
      running = false;
      radar.remove();
    },
  };
}

// Confirmation card for a side-effecting tool: shows the arguments that
// would really be sent, never the model's paraphrase
const ARG_LABELS = {
  summary: "Titre", start: "Début", end: "Fin", location: "Lieu",
  description: "Description", attendees: "Invités", timeZone: "Fuseau",
  recurrence: "Récurrence", calendarId: "Calendrier",
};

const ACCOUNT_LABELS = { calendar_pro: "Agenda pro", calendar_perso: "Agenda perso" };

function showConfirmation(actions) {
  const card = document.createElement("div");
  card.className = "confirm-card";

  const title = document.createElement("div");
  title.className = "confirm-title";
  const account = Object.entries(ACCOUNT_LABELS)
    .find(([prefix]) => actions[0].tool.startsWith(prefix));
  title.textContent = account
    ? `Créer cet événement — ${account[1]}`
    : `Confirmer : ${actions[0].tool}`;
  card.appendChild(title);

  const list = document.createElement("dl");
  list.className = "confirm-args";
  for (const action of actions) {
    for (const [key, value] of Object.entries(action.args || {})) {
      if (key === "account" || value === null || value === "") continue;
      const term = document.createElement("dt");
      term.textContent = ARG_LABELS[key] || key;
      const def = document.createElement("dd");
      def.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
      list.append(term, def);
    }
  }
  card.appendChild(list);

  const buttons = document.createElement("div");
  buttons.className = "confirm-buttons";
  const approve = document.createElement("button");
  approve.type = "button";
  approve.className = "confirm-approve";
  approve.textContent = "Confirmer la création";
  const reject = document.createElement("button");
  reject.type = "button";
  reject.textContent = "Annuler";
  buttons.append(reject, approve);
  card.appendChild(buttons);

  messagesEl.appendChild(card);
  scrollDown();

  const decide = (approved) => {
    buttons.remove();
    const outcome = document.createElement("div");
    outcome.className = "confirm-outcome";
    outcome.textContent = approved ? "Création confirmée" : "Création annulée";
    card.appendChild(outcome);
    run("/confirm", { approved, thread_id: threadId });
  };
  approve.addEventListener("click", () => decide(true));
  reject.addEventListener("click", () => decide(false));
}

// Shared runner for a new question and for resuming after a confirmation:
// both consume the same NDJSON protocol
async function run(url, payload) {
  if (sendButton.disabled) return;
  const radar = startRadar();
  sendButton.disabled = true;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const failure = await res.json().catch(() => ({}));
      throw new Error(failure.error || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let data = null;
    let confirmation = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).trim();
        buffer = buffer.slice(end + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.response !== undefined) {
          data = event;
        } else if (event.confirm) {
          confirmation = event.confirm;
        } else if (event.phase) {
          const render = PHASES[event.phase];
          if (render) radar.update(...render(event));
        } else if (event.tool) {
          const route = TOOL_ROUTES.find(([pattern]) => pattern.test(event.tool));
          const args = Object.values(event.args || {}).flat().join(" · ");
          radar.update(route ? route[1] : [0],
                       route ? route[2] : `J'utilise ${event.tool}`,
                       `${event.tool} · ${args}`);
        } else if (event.retrieved) {
          radar.update(null, null, `${event.retrieved} extraits récupérés`);
        }
      }
    }

    radar.remove();

    if (confirmation) {
      showConfirmation(confirmation);
      return;
    }
    if (!data) throw new Error("réponse incomplète");

    if (data.status === "error") {
      addBotMessage(
        "Un outil n'a pas pu être utilisé (erreur d'accès ou technique) : la "
        + "réponse serait incomplète ou trompeuse, elle n'est pas affichée. "
        + "Réessaie plus tard.",
        { plainText: true, error: true });
      return;
    }

    if (data.status === "partial") {
      addWarning(
        `⚠ Échec de : ${(data.failed_tools || []).join(", ")} - la partie de `
        + "la réponse qui en dépend est manquante ou non fiable.");
    }

    addBotMessage(data.response);
    addSources(data.sources);
  } catch (err) {
    radar.remove();
    addBotMessage(`Une erreur est survenue : ${err.message}`,
                  { plainText: true, error: true });
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
}

function ask(question) {
  suggestionsEl.textContent = "";
  questionInput.value = "";
  addUserMessage(question);
  return run("/ask", { question, thread_id: threadId });
}


function startConversation() {
  messagesEl.textContent = "";
  sourcesItems.textContent = "";
  citedSources.clear();
  updateSourceCount();
  addBotMessage(GREETING, { plainText: true });
  showSuggestions(STARTERS);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) ask(question);
});

collapseSources.addEventListener("click", () => {
  sourcesPanel.hidden = true;
  expandSources.hidden = false;
});

expandSources.addEventListener("click", () => {
  sourcesPanel.hidden = false;
  expandSources.hidden = true;
});

menuButton.addEventListener("click", (event) => {
  event.stopPropagation();
  menu.hidden = !menu.hidden;
});

document.addEventListener("click", () => { menu.hidden = true; });

newConversationButton.addEventListener("click", () => {
  threadId = crypto.randomUUID();
  localStorage.setItem("thread_id", threadId);
  startConversation();
  questionInput.focus();
});

function showValidationWarnings(warnings) {
  reindexWarnings.hidden = warnings.length === 0;
  if (warnings.length === 0) return;
  reindexWarnings.querySelector("summary").textContent =
    `${warnings.length} alerte(s) de validation`;
  const list = reindexWarnings.querySelector("ul");
  list.textContent = "";
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = `${warning.file} : ${warning.message}`;
    list.appendChild(item);
  }
}

function showReindexState(state) {
  reindexButton.disabled = state.running;
  if (state.running) {
    reindexWarnings.hidden = true;
    // Detail lives in the terminal running the server
    reindexStatus.textContent = "Indexation en cours...";
    setTimeout(refreshReindexState, 3000);
  } else if (state.error) {
    reindexStatus.textContent = `Erreur : ${state.error}`;
  } else if (state.result) {
    const r = state.result;
    reindexStatus.textContent =
      `${r.added} ajouté(s), ${r.updated} mis à jour, ${r.removed} retiré(s)`;
    showValidationWarnings(r.warnings || []);
  } else {
    reindexStatus.textContent = "";
  }
}

async function refreshReindexState() {
  try {
    const res = await fetch("/reindex/status");
    showReindexState(await res.json());
  } catch {
    reindexStatus.textContent = "";
  }
}

reindexButton.addEventListener("click", async () => {
  reindexButton.disabled = true;
  const res = await fetch("/reindex", { method: "POST" });
  showReindexState(await res.json());
});

startConversation();
refreshReindexState();
