const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const button = document.getElementById("ask-button");
const spinner = document.getElementById("spinner");
const answerEl = document.getElementById("answer");
const sourceContent = document.getElementById("source-content");

function resetInteraction() {
  answerEl.hidden = true;
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  sourceContent.textContent = "—";
  sourceContent.classList.add("empty");
}

function showSource(source) {
  sourceContent.textContent = "";
  if (!source) {
    sourceContent.textContent = "Aucune source citée.";
    return;
  }
  sourceContent.classList.remove("empty");

  // #page=N ouvre le lecteur PDF du navigateur directement à la bonne page
  const fragment = source.page !== null ? `#page=${source.page}` : "";
  const url = source.url + fragment;

  // Aperçu non interactif (pointer-events désactivés en CSS) enveloppé dans
  // un lien : cliquer n'importe où ouvre le document dans un nouvel onglet
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
    ? `${source.name} — page ${source.page}`
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
