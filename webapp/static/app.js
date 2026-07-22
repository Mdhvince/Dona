const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const button = document.getElementById("ask-button");
const spinner = document.getElementById("spinner");
const answerEl = document.getElementById("answer");
const answerSources = document.getElementById("answer-sources");
const reindexButton = document.getElementById("reindex-button");
const reindexStatus = document.getElementById("reindex-status");
const reindexWarnings = document.getElementById("reindex-warnings");

function resetInteraction() {
  answerEl.hidden = true;
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  answerSources.hidden = true;
  answerSources.textContent = "";
}

function showSources(sources) {
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
  answerSources.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  resetInteraction();
  spinner.hidden = false;
  button.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);

    answerEl.innerHTML = data.response;
    answerEl.hidden = false;
    showSources(data.sources);
  } catch (err) {
    answerEl.textContent = `Une erreur est survenue : ${err.message}`;
    answerEl.classList.add("error");
    answerEl.hidden = false;
  } finally {
    spinner.hidden = true;
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

reindexButton.addEventListener("click", async () => {
  reindexButton.disabled = true;
  const res = await fetch("/reindex", { method: "POST" });
  showReindexState(await res.json());
});

// Resume the status display if a reindex is already running (page reload)
refreshReindexState();
