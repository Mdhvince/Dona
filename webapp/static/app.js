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

const PHASES = {
  thinking: (event) => `Réflexion... (${event.tokens} tokens)`,
  tool_prep: () => "Préparation d'un appel d'outil...",
  answer: (event) => `Rédaction de la réponse... (${event.tokens} tokens)`,
};

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

// Live activity block: blinking dots, a mutating phase label, and the
// discrete steps (tool calls, retrieved chunks) stacked underneath
function startActivity() {
  const activity = document.createElement("div");
  activity.className = "activity";

  const header = document.createElement("div");
  header.className = "activity-head";
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  const label = document.createElement("span");
  label.className = "activity-label";
  label.textContent = "L'assistant consulte les sources...";
  header.append(spinner, label);

  const steps = document.createElement("ul");
  steps.className = "activity-steps";

  activity.append(header, steps);
  messagesEl.appendChild(activity);
  scrollDown();

  return {
    setPhase(text) { label.textContent = text; },
    addStep(text) {
      const step = document.createElement("li");
      step.textContent = text;
      steps.appendChild(step);
      scrollDown();
    },
    remove() { activity.remove(); },
  };
}

async function ask(question) {
  if (sendButton.disabled) return;

  suggestionsEl.textContent = "";
  questionInput.value = "";
  addUserMessage(question);
  const activity = startActivity();
  sendButton.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
    });
    if (!res.ok) {
      const failure = await res.json().catch(() => ({}));
      throw new Error(failure.error || res.statusText);
    }

    // NDJSON stream: activity events as the agent works, then the answer
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let data = null;
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
        } else if (event.phase) {
          const render = PHASES[event.phase];
          if (render) activity.setPhase(render(event));
        } else if (event.tool) {
          const args = Object.values(event.args || {}).flat().join(" · ");
          activity.addStep(`[Calling ${event.tool}]: ${args}`);
        } else if (event.retrieved) {
          activity.addStep(`${event.retrieved} extraits récupérés`);
        }
      }
    }
    if (!data) throw new Error("réponse incomplète");

    activity.remove();

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
    activity.remove();
    addBotMessage(`Une erreur est survenue : ${err.message}`,
                  { plainText: true, error: true });
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
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
    if (state.total > 0) {
      const percent = Math.round((state.done / state.total) * 100);
      reindexStatus.textContent = `Indexation... ${state.done}/${state.total} (${percent}%)`;
    } else {
      reindexStatus.textContent = "Indexation en cours...";
    }
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
