const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const button = document.getElementById("ask-button");
const spinner = document.getElementById("spinner");
const answerEl = document.getElementById("answer");
const sourceContent = document.getElementById("source-content");
const reindexButton = document.getElementById("reindex-button");
const reindexStatus = document.getElementById("reindex-status");

function resetInteraction() {
  answerEl.hidden = true;
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  sourceContent.textContent = "-";
  sourceContent.classList.add("empty");
}

function showSource(source) {
  sourceContent.textContent = "";
  if (!source) {
    sourceContent.textContent = "Aucune source citée.";
    return;
  }
  sourceContent.classList.remove("empty");

  // #page=N opens the browser PDF viewer directly on the cited page
  const url = source.url + (source.page !== null ? `#page=${source.page}` : "");

  // Non-interactive preview (pointer-events disabled in CSS) wrapped in a
  // link: clicking anywhere opens the document in a new tab
  const link = document.createElement("a");
  link.className = "preview";
  link.href = url;
  link.target = "_blank";
  link.title = "Ouvrir le document";

  const frame = document.createElement("iframe");
  frame.src = source.url + (source.page !== null
    ? `#page=${source.page}&toolbar=0&navpanes=0`
    : "#toolbar=0&navpanes=0");
  link.appendChild(frame);
  sourceContent.appendChild(link);

  const caption = document.createElement("span");
  caption.className = "caption";
  caption.textContent = source.page !== null
    ? `${source.name} - page ${source.page}`
    : source.name;
  sourceContent.appendChild(caption);
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
    showSource(data.source);
  } catch (err) {
    answerEl.textContent = `Une erreur est survenue : ${err.message}`;
    answerEl.classList.add("error");
    answerEl.hidden = false;
  } finally {
    spinner.hidden = true;
    button.disabled = false;
  }
});

function showReindexState(state) {
  reindexButton.disabled = state.running;
  if (state.running) {
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
