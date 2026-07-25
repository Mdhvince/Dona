const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const button = document.getElementById("ask-button");
const progress = document.getElementById("progress");
const progressSteps = document.getElementById("progress-steps");
const rewrittenEl = document.getElementById("rewritten");
const answerEl = document.getElementById("answer");
const answerWarning = document.getElementById("answer-warning");
const answerSources = document.getElementById("answer-sources");
const reindexButton = document.getElementById("reindex-button");
const reindexStatus = document.getElementById("reindex-status");
const reindexWarnings = document.getElementById("reindex-warnings");
const newConversationButton = document.getElementById("new-conversation-button");

// One thread per conversation: kept across reloads, renewed by the button
let threadId = localStorage.getItem("thread_id") || crypto.randomUUID();
localStorage.setItem("thread_id", threadId);

function resetInteraction() {
  answerEl.hidden = true;
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  answerWarning.hidden = true;
  answerWarning.textContent = "";
  answerSources.hidden = true;
  answerSources.textContent = "";
  rewrittenEl.hidden = true;
  rewrittenEl.textContent = "";
}

let liveStep = null;

function addProgressStep(text) {
  const step = document.createElement("li");
  step.textContent = text;
  progressSteps.appendChild(step);
  liveStep = null;
}

// Single mutating line for the heartbeat phases; discrete steps stay appended
function setLiveStep(text) {
  if (!liveStep) {
    liveStep = document.createElement("li");
    progressSteps.appendChild(liveStep);
  }
  liveStep.textContent = text;
}

const PHASES = {
  thinking: (event) => `Réflexion... (${event.tokens} tokens)`,
  tool_prep: () => "Préparation d'un appel d'outil...",
  answer: (event) => `Rédaction de la réponse... (${event.tokens} tokens)`,
};

function showQueries(queries) {
  if (!queries || queries.length === 0) return;
  rewrittenEl.textContent = `Recherche : ${queries.join(" · ")}`;
  rewrittenEl.hidden = false;
}

function showSources(sources, consulted) {
  answerSources.textContent = "";
  if (!sources || sources.length === 0) {
    answerSources.hidden = true;
    return;
  }
  answerSources.append("Sources : ");
  sources.forEach((source, i) => {
    if (i > 0) answerSources.append(" · ");
    const link = document.createElement("a");
    // #page=N opens the browser PDF viewer directly on the cited page
    link.href = source.url + (source.page !== null ? `#page=${source.page}` : "");
    link.target = "_blank";
    link.textContent = source.page !== null
      ? `${source.name}, p.${source.page}`
      : source.name;
    answerSources.appendChild(link);
  });
  if (consulted > sources.length) {
    answerSources.append(` (${consulted} documents consultés)`);
  }
  answerSources.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  resetInteraction();
  progressSteps.textContent = "";
  liveStep = null;
  progress.hidden = false;
  button.disabled = true;

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

    // NDJSON stream: query batches as the agent searches, then the answer
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
          if (render) setLiveStep(render(event));
        } else if (event.tool) {
          const args = Object.values(event.args || {}).flat().join(" · ");
          addProgressStep(`[Calling ${event.tool}]: ${args}`);
        } else if (event.retrieved) {
          addProgressStep(`${event.retrieved} extraits récupérés`);
        }
      }
    }
    if (!data) throw new Error("réponse incomplète");

    if (data.status === "error") {
      answerEl.textContent =
        "Un outil n'a pas pu être utilisé (erreur d'accès ou technique) : "
        + "la réponse serait incomplète ou trompeuse, elle n'est pas affichée. "
        + "Réessaie plus tard.";
      answerEl.classList.add("error");
      answerEl.hidden = false;
      showQueries(data.queries);
      return;
    }

    if (data.status === "partial") {
      answerWarning.textContent =
        `⚠ Échec de : ${(data.failed_tools || []).join(", ")} - la partie de `
        + "la réponse qui en dépend est manquante ou non fiable.";
      answerWarning.hidden = false;
    }

    answerEl.innerHTML = data.response;
    answerEl.hidden = false;
    showQueries(data.queries);
    showSources(data.sources, data.consulted);
  } catch (err) {
    answerEl.textContent = `Une erreur est survenue : ${err.message}`;
    answerEl.classList.add("error");
    answerEl.hidden = false;
  } finally {
    progress.hidden = true;
    button.disabled = false;
  }
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
      reindexStatus.textContent =
        `Indexation... ${state.done}/${state.total} (${percent}%)`;
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

newConversationButton.addEventListener("click", () => {
  threadId = crypto.randomUUID();
  localStorage.setItem("thread_id", threadId);
  resetInteraction();
  questionInput.value = "";
  questionInput.focus();
});

reindexButton.addEventListener("click", async () => {
  reindexButton.disabled = true;
  const res = await fetch("/reindex", { method: "POST" });
  showReindexState(await res.json());
});

// Resume the status display if a reindex is already running (page reload)
refreshReindexState();
