/* Lecteur d'étapes partagé par les simulations.
   La page définit window.SIM = { title, subtitle, page, legend, steps:[...] }
   step = { tag, phase, narr, stage, side }  (stage/side = chaînes HTML) */
(function () {
  const PAGES = [
    { id: "map",       href: "map.html",           lab: "◍ Carte" },
    { id: "ingestion", href: "sim-ingestion.html", lab: "① Ingestion" },
    { id: "retrieval", href: "sim-retrieval.html", lab: "② Retrieval" },
    { id: "agent",     href: "sim-agent.html",     lab: "③ Agent" },
    { id: "reponse",   href: "sim-reponse.html",   lab: "④ Réponse" }
  ];

  window.navHTML = function (current) {
    return '<nav class="nav">' + PAGES.map(p =>
      '<a href="' + p.href + '"' + (p.id === current ? ' class="on"' : '') + '>' + p.lab + '</a>'
    ).join("") + '</nav>';
  };

  function boot() {
    const S = window.SIM;
    if (!S) return;
    document.title = "Assistant — " + S.title;

    document.body.innerHTML =
      '<header class="head">' +
        '<h1>' + S.title + '</h1>' +
        '<p>' + (S.subtitle || "") + '</p>' +
        window.navHTML(S.page) +
      '</header>' +
      '<div class="ctrl">' +
        '<button id="play" class="pri" type="button">▶ Lecture</button>' +
        '<button id="prev" type="button">← Précédent</button>' +
        '<button id="next" type="button">Suivant →</button>' +
        '<button id="reset" type="button">↺ Recommencer</button>' +
        '<label><input type="checkbox" id="fast"> rapide</label>' +
        '<span class="pos" id="pos"></span>' +
      '</div>' +
      '<div class="wrap">' +
        '<aside class="side" id="side"></aside>' +
        '<main class="scene">' +
          '<div class="phase"><span class="tag" id="tag">—</span><span class="ph" id="ph">—</span></div>' +
          '<p class="narr" id="narr"></p>' +
          '<div id="stage"></div>' +
        '</main>' +
      '</div>' +
      '<footer class="foot">' + (S.legend || []).map(t => '<span>' + t + '</span>').join("") + '</footer>';

    const $ = id => document.getElementById(id);
    let i = 0, timer = null, playing = false;

    function render() {
      const s = S.steps[i];
      $("pos").textContent = "étape " + (i + 1) + " / " + S.steps.length;
      $("tag").textContent = s.tag || "";
      $("ph").textContent = s.phase || "";
      $("narr").innerHTML = s.narr || "";
      $("stage").innerHTML = s.stage || "";
      $("side").innerHTML = typeof s.side === "function" ? s.side(i) : (s.side || "");
      $("prev").disabled = i === 0;
      $("next").disabled = i === S.steps.length - 1;
      const st = $("stage").querySelector(".stagger");
      if (st) Array.prototype.forEach.call(st.children, (el, n) => {
        el.style.animationDelay = (n * ($("fast").checked ? 25 : 70)) + "ms";
      });
    }
    function step(d) {
      const n = i + d;
      if (n < 0 || n >= S.steps.length) { stop(); return; }
      i = n; render();
    }
    function tick() {
      if (i >= S.steps.length - 1) { stop(); return; }
      step(1);
      timer = setTimeout(tick, $("fast").checked ? 1600 : 4200);
    }
    function play() { playing = true; $("play").textContent = "❚❚ Pause"; timer = setTimeout(tick, $("fast").checked ? 900 : 2200); }
    function stop() { playing = false; $("play").textContent = "▶ Lecture"; clearTimeout(timer); }

    $("play").addEventListener("click", () => playing ? stop() : play());
    $("next").addEventListener("click", () => { stop(); step(1); });
    $("prev").addEventListener("click", () => { stop(); step(-1); });
    $("reset").addEventListener("click", () => { stop(); i = 0; render(); });
    document.addEventListener("keydown", e => {
      if (e.key === "ArrowRight") { stop(); step(1); }
      if (e.key === "ArrowLeft") { stop(); step(-1); }
      if (e.key === " ") { e.preventDefault(); playing ? stop() : play(); }
    });
    render();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
