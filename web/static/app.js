/* Desi ToT front end: plain JavaScript, no build step. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const LANG = { hi: "Hindi", te: "Telugu", en: "English" };
const KIND_LABEL = { plain: "plain", hedged: "hedged", confident: "sure", context: "context", exclusion: "ruled out", meta: "small talk" };
const NOPOSTER = "/static/no-poster.svg";
const ICON = {
  star: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>',
  prev: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>',
  next: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg>',
};
const img = (src, alt = "", cls = "") =>
  `<img src="${esc(src || NOPOSTER)}" alt="${esc(alt)}"${cls ? ` class="${cls}"` : ""} loading="lazy" onerror="this.onerror=null;this.src='${NOPOSTER}'">`;
const imdbLink = (id) => (/^tt\d+$/.test(id || "")      // ids come from Wikidata (P345)
  ? `<a class="btn-line" href="https://www.imdb.com/title/${id}/" target="_blank" rel="noopener">IMDb ↗</a>` : "");
const initials = (name) => String(name || "?").split(/\s+/).filter((w) => /^[A-Za-z]/.test(w)).map((w) => w[0]).slice(0, 2).join("").toUpperCase() || "?";
const decadeOf = (y) => Math.floor((y || 0) / 10) * 10;

const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: ignore */ } },
};

let lastQuery = "";
let lastResults = null;
const dlg = $("#dialog");
const dlgInner = $("#dialog-inner");

/* ------------------------------------------------------------------ pages, menu, theme */
const TABS = ["search", "browse", "how", "eval"];
function showTab(name, opts = {}) {
  if (!TABS.includes(name)) name = "search";
  $$(".nav-links .tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "browse") loadBrowse(opts.filter);
  if (name === "eval") loadEval();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (!opts.keepScroll) window.scrollTo(0, 0);
}
$$(".nav-links .tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => showTab((location.hash || "#search").slice(1)));

const menuBtn = $("#menu-btn");
const menu = $("#menu");
function closeMenu() { menu.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); }
menuBtn.addEventListener("click", (ev) => {
  ev.stopPropagation();
  const open = menu.hidden;
  menu.hidden = !open;
  menuBtn.setAttribute("aria-expanded", String(open));
  if (open) menu.querySelector("button").focus();
});
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeMenu(); });
$("#menu-help").addEventListener("click", () => { closeMenu(); showHelp(); });

// one delegated handler for navigation ([data-go]) and for opening a film page ([data-film])
document.addEventListener("click", (ev) => {
  if (!menu.hidden && !ev.target.closest(".menu-wrap")) closeMenu();
  const go = ev.target.closest("[data-go]");
  if (go) {
    ev.preventDefault();
    closeMenu();
    showTab(go.dataset.go, { filter: go.dataset.filter });
    return;
  }
  const film = ev.target.closest("[data-film]");
  if (film && !ev.target.closest("dialog")) openFilm(+film.dataset.film, { label: !!film.closest("#results") && !!lastQuery });
});

function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
}
applyTheme(store.get("theme"));
$("#theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light";
  applyTheme(next);
  store.set("theme", next);
});

function toast(msg, ms = 2800) {
  const t = document.createElement("div");
  t.className = "toast";
  t.setAttribute("role", "status");
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/* ------------------------------------------------------------------ home: "Can you name it?" stage */
const TAG_WORDS = { hedge: "hedged", negation: "rules something out", false_memory: "includes a false memory", decade: "mentions a period",
  hinglish: "Hinglish", lang: "names the language", social: "chatty", name: "names people" };
let slides = [];
let slideIdx = 0;
let slideTimer = null;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function showSlide(i) {
  if (!slides.length) return;
  slideIdx = (i + slides.length) % slides.length;
  const s = slides[slideIdx];
  $("#stage-bg").style.backgroundImage = `url("${s.film.poster_url}")`;
  $("#stage-quote").textContent = s.query;
  const pic = $("#stage-img");
  pic.onerror = () => { pic.onerror = null; pic.src = NOPOSTER; };
  pic.src = s.film.poster_url;
  pic.alt = "";
  $("#stage-poster").classList.remove("revealed");
  $("#stage-poster").setAttribute("aria-label", "Reveal the film");
  const words = (s.tags || []).map((t) => TAG_WORDS[t]).filter(Boolean);
  $("#stage-meta").textContent = ["Vague description from the test set", ...words].join(" · ");
  $$("#stage-dots button").forEach((b, j) => b.setAttribute("aria-selected", String(j === slideIdx)));
}
function startAuto() {
  clearInterval(slideTimer);
  if (!reduceMotion && slides.length > 1) slideTimer = setInterval(() => showSlide(slideIdx + 1), 9000);
}
$("#stage-prev").addEventListener("click", () => { showSlide(slideIdx - 1); startAuto(); });
$("#stage-next").addEventListener("click", () => { showSlide(slideIdx + 1); startAuto(); });
$("#stage").addEventListener("mouseenter", () => clearInterval(slideTimer));
$("#stage").addEventListener("mouseleave", startAuto);
$("#stage").addEventListener("focusin", () => clearInterval(slideTimer));
$("#stage-poster").addEventListener("click", () => {
  const s = slides[slideIdx];
  if (!s) return;
  $("#stage-poster").classList.add("revealed");
  $("#stage-img").alt = `Poster of ${s.film.title}`;
  $("#stage-meta").textContent = `It's ${s.film.title} (${s.film.year}, ${LANG[s.film.lang]}). Press play to see if the search finds it.`;
});
$("#stage-play").addEventListener("click", () => { const s = slides[slideIdx]; if (s) runSearch(s.query); });

let examples = [];
function renderExamples(list) {
  // four at a time: three random ones plus the example from the project brief (no fixed answer)
  const own = list.filter((e) => !e.film);
  const rest = list.filter((e) => e.film).sort(() => Math.random() - 0.5).slice(0, 4 - Math.min(1, own.length));
  examples = [...rest, ...own.slice(0, 1)];
  $("#examples").innerHTML = examples.map((e, i) => `<li><button data-ex="${i}">
      ${e.film ? img(e.film.poster_url) : '<span class="ph" aria-hidden="true">?</span>'}
      <span><span class="un-run"><i aria-hidden="true">▶</i> Try this description</span>
        <span class="un-q">${esc(e.query)}</span>
        <span class="un-a">${e.film ? `Answer: ${esc(e.film.title)} (${esc(e.film.year)})` : "From the project brief: no single right answer"}</span></span>
    </button></li>`).join("");
}
$("#examples").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-ex]");
  if (b) runSearch(examples[+b.dataset.ex].query);
});

async function initHome() {
  try {
    const data = await (await fetch("/api/featured")).json();
    slides = data.slides || [];
    renderExamples(data.examples || []);
    $("#stage-dots").innerHTML = slides.map((s, i) =>
      `<button role="tab" aria-label="Description ${i + 1} of ${slides.length}" aria-selected="false" data-i="${i}"></button>`).join("");
    $$("#stage-dots button").forEach((b) => b.addEventListener("click", () => { showSlide(+b.dataset.i); startAuto(); }));
    showSlide(0);
    startAuto();
  } catch { /* the page still works without the featured strip */ }
}

/* ------------------------------------------------------------------ poster cards and shelves */
let films = null;
async function getFilms() {
  if (!films) films = await (await fetch("/api/films")).json();
  return films;
}
function posterCard(f) {
  const actors = [...new Set((f.cast_roles || []).map((r) => r.actor))];      // dual roles list an actor twice
  const cast = actors.slice(0, 2).join(", ") || (f.stars || []).slice(0, 2).join(", ");
  return `<article class="pcard">
      <button class="pimg" data-film="${f.id}" aria-label="${esc(f.title)} (${esc(f.year)}): details">${img(f.poster_url)}</button>
      <div class="pbody">
        <div class="pmeta"><span>${esc(f.year)}</span><span class="chip-lang">${esc(LANG[f.lang] || f.lang)}</span></div>
        <div class="ptitle" title="${esc(f.title)}">${esc(f.title)}</div>
        ${cast ? `<div class="pcast" title="${esc(cast)}">${esc(cast)}</div>` : ""}
        <button class="btn-blue" data-film="${f.id}">Details</button>
      </div>
    </article>`;
}
function fillShelf(sel, list) {
  const shelf = $(sel);
  shelf.innerHTML = list.map(posterCard).join("");
  const wrap = shelf.parentElement;
  if (wrap.querySelector(".shelf-btn")) return;
  wrap.insertAdjacentHTML("beforeend", `<button class="shelf-btn prev" aria-label="Scroll left">${ICON.prev}</button>
    <button class="shelf-btn next" aria-label="Scroll right">${ICON.next}</button>`);
  const prev = $(".shelf-btn.prev", wrap), next = $(".shelf-btn.next", wrap);
  prev.addEventListener("click", () => shelf.scrollBy({ left: -shelf.clientWidth * 0.9 }));
  next.addEventListener("click", () => shelf.scrollBy({ left: shelf.clientWidth * 0.9 }));
  const sync = () => {                                   // like the reference design: no arrow at either end
    prev.hidden = shelf.scrollLeft < 8;
    next.hidden = shelf.scrollLeft + shelf.clientWidth > shelf.scrollWidth - 8;
  };
  shelf.addEventListener("scroll", sync, { passive: true });
  window.addEventListener("resize", sync);
  requestAnimationFrame(sync);
}
async function initShelves() {
  try {
    const list = await getFilms();
    // the 24 curated films with the most English-Wikipedia page views in the 60 days before the crawl
    fillShelf("#shelf-pop", [...list].sort((a, b) => (b.pageviews_60d || 0) - (a.pageviews_60d || 0)).slice(0, 24));
  } catch { /* shelves are optional */ }
}

/* ------------------------------------------------------------------ film page (dialog) */
function castItems(list) {
  return list.map((r) => `<div class="cast-item"><span class="avatar" aria-hidden="true">${esc(initials(r.actor))}</span>
      <div><b>${esc(r.actor)}</b>${r.character ? `<span class="char">${esc(r.character)}</span>` : ""}${r.role ? `<span class="role">${esc(r.role)}</span>` : ""}</div></div>`).join("");
}
async function openFilm(id, opts = {}) {
  let f;
  try { f = await (await fetch(`/api/film/${id}`)).json(); } catch { toast("Could not load the film."); return; }
  const credit = (label, list) => (list && list.length)
    ? `<div><dt>${label}</dt><dd>${esc(list.slice(0, 4).join(" · "))}</dd></div>` : "";
  const roles = (f.cast_roles && f.cast_roles.length) ? f.cast_roles : (f.stars || []).map((s) => ({ actor: s, character: "", role: "" }));
  const plot = f.plot || f.lead || "";
  dlgInner.innerHTML = `<div class="film">
      ${img(f.poster_url, `Poster of ${f.title}`, "film-poster")}
      <div>
        <h2 class="film-title">${esc(f.title)}</h2>
        <div class="film-sub"><span>${esc(f.year)}</span><span class="chip-lang">${esc(LANG[f.lang] || f.lang)}</span>
          ${f.curated ? "" : '<span class="chip-lang extra">background film</span>'}
          ${f.pageviews_60d ? `<span title="Wikipedia page views in the last 60 days">${Number(f.pageviews_60d).toLocaleString()} Wikipedia views (60 days)</span>` : ""}</div>
        ${f.genres && f.genres.length ? `<div class="genre-chips">${f.genres.map((g) => `<span>${esc(g)}</span>`).join("")}</div>` : ""}
        <dl class="credits">${credit("Director", f.directors)}${credit("Writers", f.writers)}${credit("Music", f.composers)}${credit("Stars", f.stars)}</dl>
        <h3 class="sr-only">Plot</h3>
        <div class="plot">${esc(plot.slice(0, 2600))}${plot.length > 2600 ? "…" : ""}</div>
      </div>
    </div>
    <h2 class="section-title cast-title">Top cast <span class="section-sub">${roles.length} credited role${roles.length === 1 ? "" : "s"}, from Wikipedia</span></h2>
    <div class="cast-grid" id="cast-grid">${castItems(roles.slice(0, 12)) || '<p class="muted">No cast list on Wikipedia.</p>'}</div>
    ${roles.length > 12 ? `<p><button type="button" class="btn-blue" id="more-cast">Show all ${roles.length}</button></p>` : ""}
    <div class="row">
      ${opts.label ? `<button type="button" class="btn-line" id="dlg-label">✓ This is my film: add to test set</button>` : ""}
      <a class="btn-line" href="${esc(f.url)}" target="_blank" rel="noopener">Wikipedia ↗</a>
      ${imdbLink(f.imdb_id)}
      <button value="close" class="btn-gold">Close</button>
    </div>`;
  const more = $("#more-cast");
  if (more) more.addEventListener("click", () => { $("#cast-grid").innerHTML = castItems(roles); more.remove(); });
  const lab = $("#dlg-label");
  if (lab) lab.addEventListener("click", () => confirmLabel(f.id, f.title));
  if (!dlg.open) dlg.showModal();
  dlgInner.focus({ preventScroll: true });              // not the scrollable plot box (would get a focus ring)
  dlg.scrollTop = 0;
}

/* ------------------------------------------------------------------ search */
$("#toggle-opts").addEventListener("click", () => {
  const box = $("#opts");
  box.hidden = !box.hidden;
  $("#toggle-opts").setAttribute("aria-expanded", String(!box.hidden));
  $("#toggle-opts").textContent = box.hidden ? "Components ▾" : "Components ▴";
});
$("#search-form").addEventListener("submit", (ev) => { ev.preventDefault(); runSearch(); });
$("#q").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); runSearch(); }
});
$("#nav-search").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const v = $("#nav-q").value.trim();
  if (v) runSearch(v); else { showTab("search"); $("#q").focus(); }
});
$("#nav-lang").addEventListener("change", () => { if (lastQuery) runSearch(lastQuery); });

function readOptions() {
  const o = {};
  $$("[data-opt]").forEach((el) => { o[el.dataset.opt] = el.type === "checkbox" ? el.checked : el.value; });
  return o;
}

async function runSearch(text, extra = {}) {
  if (typeof text === "string") $("#q").value = text;
  const q = $("#q").value.trim();
  const status = $("#status");
  if (!$("#tab-search").classList.contains("active")) showTab("search", { keepScroll: true });
  if (!q) { status.textContent = "Describe the film first, whatever you remember."; $("#q").focus(); return; }
  $("#nav-q").value = q;
  $("#go").disabled = true;
  status.className = "status";
  status.textContent = "Reading your description…";
  const lang = $("#nav-lang").value;
  try {
    const r = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, k: 5, ranker: $("#ranker").value, curated_only: $("#curated").checked, lang, options: readOptions(),
        evidence: $("#evidence-on") ? $("#evidence-on").checked : true, answer: extra.answer || null }),
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
    lastQuery = q;
    lastResults = res;
    status.textContent = "";
    $("#results-wrap").hidden = false;
    $("#results-title").textContent = `Top ${res.results.length || 5} matches${lang ? ` in ${LANG[lang]}` : ""}`;
    $("#results-for").textContent = `for “${q.length > 160 ? q.slice(0, 158) + "…" : q}”`;
    renderAnalysis(res);
    renderEvidence(res);
    renderResults(res);
    if (!extra.keepScroll) $("#results-wrap").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  } catch (err) {
    status.className = "status error";
    status.textContent = `Search failed: ${err.message}. Is the server running?`;
  } finally {
    $("#go").disabled = false;
  }
}

/* ------------------------------------------------------------------ how the description was read */
function cueTitle(k) {
  switch (k.kind) {
    case "hedge": return k.negated ? "hedged negation (I don't think…)" : "hedge: lowers this clause's weight";
    case "confident": return "confidence: raises this clause's weight";
    case "negation": return "negation";
    case "meta": return "small talk / search history: ignored";
    case "context": return "viewing context";
    case "year": return `decade/year cue: ${k.lo}–${k.hi}`;
    case "lang": return `language cue: ${LANG[k.lang] || k.lang}`;
    case "genre": return `genre cue: ${k.genre}`;
    case "entity": return `${k.type}: ${k.entity} (${k.how})`;
    case "correction": return `spelling fixed to “${k.to}”`;
    case "lexicon": return `Hinglish: ${k.gloss}`;
    default: return k.kind;
  }
}

function renderClauses(q) {
  const text = q.text;
  let html = "", pos = 0;
  for (const c of q.clauses) {
    if (c.start > pos) html += esc(text.slice(pos, c.start));
    const cues = c.cues.filter((k) => k.end > k.start).sort((a, b) => a.start - b.start || b.end - a.end);
    let inner = "", p = c.start;
    for (const k of cues) {
      if (k.start < p) continue;
      inner += esc(text.slice(p, k.start));
      inner += `<span class="cue cue-${k.kind}" title="${esc(cueTitle(k))}">${esc(text.slice(k.start, k.end))}</span>`;
      p = k.end;
    }
    inner += esc(text.slice(p, c.end));
    const withdrawn = c.trust === 0;
    const badge = withdrawn ? "✗ withdrawn" : c.kind === "meta" ? "ignored" : c.kind === "exclusion" ? "✗ ruled out" : `×${(+c.weight).toFixed(2).replace(/\.?0+$/, "")}`;
    const tip = withdrawn ? "withdrawn by the clue check: it fits another film" : `${KIND_LABEL[c.kind] || c.kind} clause`;
    html += `<span class="clause k-${c.kind}${withdrawn ? " withdrawn" : ""}" title="${esc(tip)}">${inner}<sup class="w">${badge}</sup></span>`;
    pos = c.end;
  }
  return html + esc(text.slice(pos));
}

function renderChips(q) {
  const chips = [];
  const seen = new Set();
  for (const c of q.clauses) {
    const excl = c.kind === "exclusion";
    for (const k of c.cues) {
      let label = null, val = null, extra = "";
      if (k.kind === "year") { label = "Period"; val = k.lo === k.hi ? `${k.lo}` : `${k.lo}–${k.hi}`; if (c.hedged) extra = "unsure"; }
      else if (k.kind === "lang") { label = "Language"; val = LANG[k.lang] || k.lang; }
      else if (k.kind === "genre") { label = "Genre"; val = k.genre; }
      else if (k.kind === "entity") {
        label = k.type === "title" ? "Title" : k.type === "character" ? "Character" : "Person"; val = k.entity;
        if (k.how !== "exact") extra = `from “${k.text}”`;
      }
      else if (k.kind === "correction") { label = "Spelling"; val = `${k.text} → ${k.to}`; }
      else if (k.kind === "lexicon") { label = "Hinglish"; val = `${k.text} → ${k.gloss}`; }
      if (!label) continue;
      const key = `${label}|${val}|${excl}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const isExcl = excl && ["Period", "Language", "Genre", "Person", "Title", "Character"].includes(label);
      chips.push(`<span class="chip${isExcl ? " excluded" : ""}" title="${isExcl ? "ruled out" : ""}"><b>${esc(isExcl ? "✗ Not " + label : label)}</b>`
        + `<span>${esc(val)}</span>${extra ? ` <small>${esc(extra)}</small>` : ""}</span>`);
    }
  }
  return chips.join("") || '<span class="muted">No explicit period, language, genre or name cues: the plot description does all the work.</span>';
}

function renderNB(nb) {
  const g = nb.genres || [];
  const max = Math.max(0.0001, ...g.map((x) => x[1]));
  $("#nb-genres").innerHTML = g.length ? g.map(([name, p]) =>
    `<div class="row" title="${esc(name)}: ${(100 * p).toFixed(0)}%"><span>${esc(name)}</span><div class="track"><div class="fill" style="width:${(100 * p / max).toFixed(1)}%"></div></div><span class="val">${(100 * p).toFixed(0)}%</span></div>`).join("")
    : '<span class="muted">Naive Bayes is switched off.</span>';
  const d = nb.decades || [];
  const dmax = Math.max(0.0001, ...d.map((x) => x[1]));
  $("#nb-decades").innerHTML = d.length ? d.map(([dec, p]) =>
    `<div class="col" title="${dec}s: ${(100 * p).toFixed(0)}%"><div class="bar" style="height:${(88 * p / dmax).toFixed(1)}%"></div><div class="lab">${String(dec).slice(2)}s</div></div>`).join("")
    : '<span class="muted">Naive Bayes is switched off.</span>';
}

function renderAnalysis(res) {
  $("#clause-view").innerHTML = renderClauses(res.query);
  $("#cue-chips").innerHTML = renderChips(res.query);
  renderNB(res.nb || {});
  const rk = { ltr: "learned ranker", lex: "clause-weighted BM25", bm25: "plain BM25", tfidf: "TF-IDF", dense: "dense", hybrid: "BM25 + dense (RRF)" };
  $("#took").textContent = `${res.took_ms} ms · ${rk[res.ranker] || res.ranker}`;
}

/* ------------------------------------------------------------------ evidence checks (C1-C4) */
const REL = { borrowed: "borrowed memory", conflict: "conflicting clues", other_version: "other version" };
let lastQuestion = null;

function filmLink(f) {
  return `<button class="link" data-film="${f.id}">${esc(f.title)}</button> <span class="muted">(${esc(LANG[f.lang] || f.lang)}, ${esc(f.year)})</span>`;
}

function questionCard(q) {
  const opts = q.options.map((o, i) => `<button type="button" class="qopt" data-qi="${i}">${esc(o.label)}</button>`).join("");
  return `<div class="ev-card ev-question" role="group" aria-label="One question to narrow it down">
      <h3>One question to narrow it down</h3>
      <p class="qtext">${esc(q.text)}</p>
      <div class="qopts">${opts}
        <button type="button" class="qopt ghost" data-qv="none">None of these</button>
        <button type="button" class="qopt ghost" data-qv="dont_know">I don't remember</button></div>
      <p class="note">Chosen because it is the question most likely to separate the films still in play
        (expected gain ${fmt(q.expected_gain, 2)} bits). “I don't remember” is fine: nothing changes.</p>
    </div>`;
}

function answerQuestion(value, films, label) {
  const q = lastQuestion;
  if (!q) return;
  const answer = { kind: q.kind, value, films: films || [], offered: q.options.map((o) => o.value), candidates: q.candidates, label };
  runSearch(lastQuery, { answer, keepScroll: true });
}

function renderEvidence(res) {
  const box = $("#evidence");
  const ev = res.evidence;
  lastQuestion = ev ? ev.question : null;
  if (!ev) { box.innerHTML = ""; return; }
  let html = "";
  if (ev.answer) {
    html += `<p class="ev-note">Re-ranked with your answer: <b>${esc(ev.answer.label || ev.answer.value)}</b>
      <button type="button" class="link" id="undo-answer">Undo</button></p>`;
  }
  const flags = (ev.checks || []).filter((c) => c.flagged);
  if (flags.length) {
    html += `<div class="ev-card"><h3>Clue check</h3><ul class="ev-list">${flags.map((c) =>
      `<li class="ev-${c.relation}"><span class="tag">${esc(REL[c.relation] || c.relation)}</span> ${esc(c.message)}
        <span class="ev-film">${filmLink(c.source)}</span></li>`).join("")}</ul>
      <p class="note">Each clue is checked against the films that support it: an actor against the cast lists, a scene against the plots.</p></div>`;
  } else if ((ev.checks || []).length) {
    html += `<p class="ev-ok">✓ ${ev.checks.length} clue${ev.checks.length === 1 ? "" : "s"} checked against the cast lists and plots: no conflicts.</p>`;
  }
  if (ev.versions) {
    const v = ev.versions;
    html += `<div class="ev-card"><h3>${v.versions.length > 1 ? "Versions of this story" : "Language you heard"}</h3><p>${esc(v.message)}</p>
      <div class="ver-list">${v.versions.map((x) => `<button type="button" class="ver${x.id === v.chosen.id ? " chosen" : ""}" data-film="${x.id}">
        <b>${esc(x.title)}</b><span>${esc(LANG[x.lang] || x.lang)}, ${esc(x.year)} · ${esc(x.relation)}</span>
        ${x.dubbed_into && x.dubbed_into.length ? `<small>dubbed into ${esc(x.dubbed_into.join(", "))}</small>` : ""}
        ${x.evidence && x.evidence.length ? `<small>${esc(x.evidence.join("; "))}</small>` : ""}</button>`).join("")}</div></div>`;
  }
  if (ev.order && ev.order.pairs.length) {
    html += `<div class="ev-card"><h3>Event order</h3><p>You described ${ev.order.pairs.map((p) =>
      `“${esc(p[0])}” <b>and then</b> “${esc(p[1])}”`).join("; ")}. Each result shows where these events happen in its plot.</p>
      ${ev.order.reversed_query ? '<button type="button" class="btn-line" id="reverse-events">⇄ Reverse the events and search again</button>' : ""}</div>`;
  }
  if (ev.question) html += questionCard(ev.question);
  if (ev.changed) html += `<p class="ev-note muted">These checks changed the first result: the ranker alone put <b>${esc(ev.base_top.title)}</b> first.</p>`;
  box.innerHTML = html;
  $$(".qopt", box).forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.qv) answerQuestion(b.dataset.qv, [], b.textContent.trim());
    else { const o = lastQuestion.options[+b.dataset.qi]; answerQuestion(o.value, o.films, o.label); }
  }));
  const undo = $("#undo-answer");
  if (undo) undo.addEventListener("click", () => runSearch(lastQuery, { keepScroll: true }));
  const rev = $("#reverse-events");
  if (rev) rev.addEventListener("click", () => runSearch(ev.order.reversed_query));
}

function posLabel(x) {
  return x < 0.2 ? "near the start" : x < 0.45 ? "early on" : x < 0.7 ? "midway" : x < 0.9 ? "later" : "near the end";
}

function orderBlock(o) {
  if (!o || !o.steps || !o.steps.length) return "";
  return o.steps.map((s) => `<div class="order-box ${s.fits ? "fits" : "reversed"}">
      <span class="src">${s.fits ? "✓ Your events happen in this order" : "⇄ In this plot the events happen the other way round"}</span>
      <ol><li><b>${esc(s.first)}</b><span>${esc(s.text_first)}</span><small>${posLabel(s.pos_first)}</small></li>
        <li><b>${esc(s.second)}</b><span>${esc(s.text_second)}</span><small>${posLabel(s.pos_second)}</small></li></ol></div>`).join("");
}

/* ------------------------------------------------------------------ results */
function highlight(text, spans) {
  const s = [...(spans || [])].sort((a, b) => a.start - b.start);
  let out = "", p = 0;
  for (const h of s) {
    if (h.start < p) continue;
    out += esc(text.slice(p, h.start)) + `<mark class="hl k-${h.kind}">${esc(text.slice(h.start, h.end))}</mark>`;
    p = h.end;
  }
  return out + esc(text.slice(p));
}

function renderContrib(list) {
  if (!list || !list.length) return "";
  const max = Math.max(0.0001, ...list.map((c) => Math.abs(c.value)));
  return `<div class="contrib">${list.map((c) => {
    const w = (50 * Math.abs(c.value) / max).toFixed(1);
    return `<span>${esc(c.label)}</span><div class="bar" title="${esc(c.label)}: ${c.value >= 0 ? "+" : ""}${c.value.toFixed(3)}"><i class="${c.value >= 0 ? "pos" : "neg"}" style="width:${w}%"></i></div><span class="v">${c.value >= 0 ? "+" : "−"}${Math.abs(c.value).toFixed(3)}</span>`;
  }).join("")}</div>`;
}

function castLine(roles) {
  const list = (roles || []).filter((r) => r.actor).slice(0, 3);
  if (!list.length) return "";
  return `<p class="rcast"><span class="as">Cast</span> ${list.map((r) =>
    `<b>${esc(r.actor)}</b>${r.character ? ` <span class="as">as</span> ${esc(r.character)}` : ""}`).join(" · ")}</p>`;
}

function renderResults(res) {
  const box = $("#results");
  if (!res.results.length) { box.innerHTML = '<p class="muted">No films found.</p>'; return; }
  const clauses = res.query.clauses;
  box.innerHTML = res.results.map((r, i) => {
    const hits = r.clauses.filter((c) => c.matched.length || c.excluded.length).map((c) => {
      const cl = clauses[c.clause];
      const words = c.matched.join(", ") + (c.excluded.length ? ` ✗ ${c.excluded.join(", ")}` : "");
      return `<span class="ch k-${c.kind}" title="${esc(cl ? cl.text : "")}">${esc(KIND_LABEL[c.kind] || c.kind)}: ${esc(words)}</span>`;
    }).join("");
    const reasons = r.reasons.map((x) => `<li class="${x.startsWith("⚠") ? "warn" : ""}">${esc(x.replace(/^⚠\s*/, ""))}</li>`).join("");
    const pct = Math.round(100 * r.confidence);
    const roles = (r.cast_roles && r.cast_roles.length) ? r.cast_roles : (r.stars || []).map((s) => ({ actor: s }));
    return `<article class="rcard${i === 0 ? " top" : ""}">
      <button class="rposter" data-film="${r.id}" aria-label="${esc(r.title)}: details">
        ${img(r.poster_url, `Poster of ${r.title}`)}<span class="ribbon${i === 0 ? " gold" : ""}" aria-hidden="true">${i + 1}</span>
      </button>
      <div class="rbody">
        <div class="rhead">${i === 0 ? '<span class="top-label">Best match</span>' : ""}
          <h3><button data-film="${r.id}">${esc(r.title)}</button></h3><span class="yr">${esc(r.year ?? "?")}</span>
          <span class="chip-lang">${esc(LANG[r.lang] || r.lang)}</span>${r.curated ? "" : '<span class="chip-lang extra" title="A background film from Wikipedia, not one of the curated films">background</span>'}</div>
        <div class="rmeta"><span class="star" title="Relative confidence: share of the top-5 score">${ICON.star}${pct}% <small>match</small></span>
          ${r.directors.length ? `<span class="dot">Dir. ${esc(r.directors.join(", "))}</span>` : ""}
          ${r.genres.length ? `<span class="dot">${esc(r.genres.join(", "))}</span>` : ""}</div>
        ${castLine(roles)}
        ${(r.warnings || []).length ? `<ul class="ev-warn">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
        ${reasons ? `<ul class="reasons">${reasons}</ul>` : ""}
        ${(r.versions || []).length ? `<p class="rversions"><span class="as">Other version${r.versions.length > 1 ? "s" : ""}</span> ${r.versions.map((v) =>
          `${filmLink(v)} <span class="muted">${esc(v.relation)}</span>`).join(" · ")}</p>` : ""}
        ${(r.dubbed_into || []).length ? `<p class="rversions"><span class="as">Dubbed into</span> ${esc(r.dubbed_into.join(", "))}</p>` : ""}
        ${orderBlock(r.order)}
        <blockquote class="passage"><span class="src">Best matching passage</span>${highlight(r.passage, r.spans)}</blockquote>
        ${hits ? `<div class="clause-hits">${hits}</div>` : ""}
        ${r.contributions.length ? `<details class="breakdown"><summary>Score breakdown (learned feature contributions)</summary>${renderContrib(r.contributions)}</details>` : ""}
        <div class="ractions">
          <button class="btn-blue" data-film="${r.id}">Details &amp; full cast</button>
          <a class="btn-line" href="${esc(r.url)}" target="_blank" rel="noopener">Wikipedia ↗</a>
          ${imdbLink(r.imdb_id)}
          <button class="btn-line" data-label="${r.id}" data-title="${esc(r.title)}">✓ This is my film</button>
        </div>
      </div>
    </article>`;
  }).join("");
}

$("#results").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-label]");
  if (b) confirmLabel(+b.dataset.label, b.dataset.title);
});
$("#label-other").addEventListener("click", () => pickOtherFilm());

/* ------------------------------------------------------------------ labelling (human test set) */
function authorField() {
  return `<label class="field" style="margin-top:.8rem">Your name (optional)
    <input type="text" id="author" maxlength="40" value="${esc(store.get("author", ""))}" placeholder="team member"></label>`;
}

async function saveLabel(id) {
  const author = ($("#author")?.value || "").trim();
  if (author) store.set("author", author);
  const r = await fetch("/api/label", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: lastQuery, id, author }),
  });
  const res = await r.json();
  if (!r.ok) throw new Error(res.error || r.statusText);
  dlg.close();
  toast(`Saved. The human test set now has ${res.total_human_queries} descriptions.`);
}

function confirmLabel(id, title) {
  if (!lastQuery) { toast("Search with your own description first."); return; }
  dlgInner.innerHTML = `<h2>Add this description to the human test set?</h2>
    <p class="muted">Description: “${esc(lastQuery)}”<br>Correct film: <b>${esc(title)}</b></p>
    <p class="muted">Only do this if you wrote the description yourself from memory: it becomes evaluation data for
      comparing human-written and LLM-written queries.</p>${authorField()}
    <div class="row"><button value="cancel" class="btn-ghost">Cancel</button><button type="button" class="btn-gold" id="do-save">Save</button></div>`;
  if (!dlg.open) dlg.showModal();
  $("#do-save").addEventListener("click", () => saveLabel(id).catch((e) => toast(`Could not save: ${e.message}`)));
}

function pickOtherFilm() {
  dlgInner.innerHTML = `<h2>Which film did you mean?</h2>
    <p class="muted">Type part of the title. Your description (“${esc(lastQuery.slice(0, 120))}${lastQuery.length > 120 ? "…" : ""}”) will be saved with it.</p>
    <input type="search" id="title-q" placeholder="e.g. sholay" style="width:100%">
    <ul class="suggest" id="suggest"></ul>${authorField()}
    <div class="row"><button value="cancel" class="btn-ghost">Cancel</button><button type="button" class="btn-gold" id="do-save" disabled>Save</button></div>`;
  dlg.showModal();
  let chosen = null, timer = null;
  const input = $("#title-q");
  input.focus();
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const r = await fetch(`/api/titles?q=${encodeURIComponent(input.value)}`);
      const list = await r.json();
      $("#suggest").innerHTML = list.map((f) =>
        `<li><button type="button" data-id="${f.id}">${esc(f.title)} <span class="muted">(${esc(f.year)}, ${esc(LANG[f.lang] || f.lang)})</span></button></li>`).join("")
        || '<li class="muted">No match in the index.</li>';
    }, 150);
  });
  $("#suggest").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-id]");
    if (!b) return;
    $$("#suggest button").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel");
    chosen = +b.dataset.id;
    $("#do-save").disabled = false;
  });
  $("#do-save").addEventListener("click", () => chosen !== null && saveLabel(chosen).catch((e) => toast(`Could not save: ${e.message}`)));
}

function showHelp() {
  dlgInner.innerHTML = `<h2>Add your own test descriptions</h2>
    <p class="muted">The evaluation compares system rankings on LLM-written and human-written descriptions (Kendall's τ),
      as the TREC 2025 Tip-of-the-Tongue track did. To contribute, think of a film you know, describe it <b>from memory</b>
      without looking anything up, and search for it.</p>
    <ol class="muted">
      <li>If it shows up, press <b>✓ This is my film</b> on the right result.</li>
      <li>If not, use <b>Tell us which film you meant</b> under the results and pick it by title.</li>
      <li>Descriptions are saved to <code>data/queries/human_queries.jsonl</code>; run <code>python scripts/evaluate.py</code> to score them.</li>
    </ol>
    <div class="row"><button value="close" class="btn-gold">Got it</button></div>`;
  dlg.showModal();
}

/* ------------------------------------------------------------------ browse */
let browseFilter = "";                                      // "" (all) or a language code
let browseReady = false;
async function loadBrowse(filter) {
  if (filter !== undefined) {
    browseFilter = filter || "";
    $$("#lang-filter button").forEach((b) => b.classList.toggle("on", b.dataset.f === browseFilter));
  }
  if (!browseReady) {
    $("#grid").innerHTML = '<p class="muted">Loading…</p>';
    const list = await getFilms();
    const count = (l) => list.filter((f) => f.lang === l).length;
    $("#browse-title").textContent = `Browse the ${list.length} curated films`;
    $("#browse-counts").textContent = `${count("hi")} Hindi, ${count("te")} Telugu and ${count("en")} English films`;
    const decades = [...new Set(list.map((f) => decadeOf(f.year)))].filter(Boolean).sort();
    $("#decade-filter").innerHTML = '<option value="">Any decade</option>' + decades.map((d) => `<option value="${d}">${d}s</option>`).join("");
    fetch("/api/stats").then((r) => r.json()).then((s) => { $("#n-distractors").textContent = s.distractors; }).catch(() => {});
    browseReady = true;
  }
  drawGrid();
}
function drawGrid() {
  if (!films) return;
  const dec = $("#decade-filter").value;
  const tq = $("#title-filter").value.trim().toLowerCase();
  const sort = $("#sort-filter").value;
  const pass = (f) => !browseFilter || f.lang === browseFilter;
  const list = films.filter((f) => pass(f) && (!dec || decadeOf(f.year) === +dec) && (!tq || f.title.toLowerCase().includes(tq)));
  const order = { hi: 0, te: 1, en: 2 };
  const cmp = {
    lang: (a, b) => (order[a.lang] - order[b.lang]) || (a.year - b.year),
    new: (a, b) => b.year - a.year, old: (a, b) => a.year - b.year,
    views: (a, b) => (b.pageviews_60d || 0) - (a.pageviews_60d || 0), title: (a, b) => a.title.localeCompare(b.title),
  }[sort];
  list.sort(cmp);
  $("#browse-count").textContent = `${list.length} film${list.length === 1 ? "" : "s"}`;
  $("#grid").innerHTML = list.map(posterCard).join("") || '<p class="muted">No films match these filters.</p>';
}
$("#lang-filter").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (b) loadBrowse(b.dataset.f);
});
$("#decade-filter").addEventListener("change", drawGrid);
$("#sort-filter").addEventListener("change", drawGrid);
$("#title-filter").addEventListener("input", drawGrid);

/* ------------------------------------------------------------------ evaluation */
let evalLoaded = false;
const PRETTY = { detailed: "detailed queries", vague: "vague queries (primary)", "vague+noise": "vague + injected false memories", human: "human queries" };

function table(cols, rows, opts = {}) {
  const best = {};
  (opts.bestCols || []).forEach((c) => { best[c] = Math.max(...rows.map((r) => +r[c] || -1)); });
  return `<div class="table-wrap"><table><thead><tr>${cols.map((c) => `<th scope="col">${esc(c[1])}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr class="${opts.ours && opts.ours(r) ? "ours" : ""}">${cols.map(([k, , f]) => {
      const v = r[k];
      const cls = best[k] !== undefined && +v === best[k] ? "best" : "";
      return `<td class="${cls}">${f ? f(v, r) : esc(v)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const isOurs = (r) => /ours|Clause|^\+ /.test(r.system || "");

async function loadEval() {
  if (evalLoaded) return;
  const s = await (await fetch("/api/stats")).json();
  evalLoaded = true;
  const e = s.evaluation;
  const intro = $("#eval-intro");
  if (!e) {
    intro.innerHTML = "No evaluation results yet. Run <code>python scripts/evaluate.py</code> and then <code>python scripts/make_figures.py</code>, then reload this page.";
    return;
  }
  const conds = Object.keys(e.main || {});
  intro.innerHTML = `${s.films} films in the index (${s.curated} curated + ${s.distractors} distractors), ${s.terms.toLocaleString()} index terms,
    ${s.passages.toLocaleString()} embedded plot passages. All learned components are scored with 5-fold cross-validation.
    <span class="muted">★ marks our systems; the best value in each column is bold.</span>`;
  const prim = e.main.vague || e.main[conds[0]];
  const get = (rows, name) => rows.find((r) => r.system === name) || {};
  const bm = get(prim, "BM25F (baseline)"), full = get(prim, "Full LTR: coordinate ascent (ours)");
  let html = `<div class="kpis">
      <div class="kpi"><div class="n">${fmt(full.rr)}</div><div class="l">MRR of the full model on vague queries</div></div>
      <div class="kpi"><div class="n">${fmt(bm.rr)}</div><div class="l">MRR of plain BM25 (baseline)</div></div>
      <div class="kpi"><div class="n">${(100 * (full["s@1"] || 0)).toFixed(0)}%</div><div class="l">right film at rank 1 (full model)</div></div>
      <div class="kpi"><div class="n">${(100 * (full["s@10"] || 0)).toFixed(0)}%</div><div class="l">right film in the top 10</div></div>
    </div>`;

  html += `<section class="eval-section"><h2>Main results across query conditions (MRR)</h2>
    <p class="note">Detailed queries mention names, songs and dialogue; vague queries read like real tip-of-the-tongue posts;
    “+ noise” adds one hedged false memory to every vague query.</p>`;
  const ov = Object.entries(e.overview || {}).map(([system, d]) => ({ system, ...Object.fromEntries(conds.map((c) => [c, d[c] ? d[c].mrr : null])) }));
  html += table([["system", "System"], ...conds.map((c) => [c, PRETTY[c] || c, (v) => fmt(v)])], ov, { bestCols: conds, ours: isOurs }) + "</section>";

  if (s.novelty && s.novelty.c1) {
    const nv = s.novelty;
    const c1rows = Object.entries(nv.c1).map(([set, r]) => ({ set, ...r }));
    html += `<section class="eval-section"><h2>Evidence checks C1: memories borrowed from another film</h2>
      <p class="note">The vague queries with one injected false memory from another film (hedged, plain or certain), correct-actor
      controls and clean queries (false alarms). Base = the deployed ranker; “+C1” adds the clue check. P/R/F1 count flagged clues
      against the injected ones; “source” = the film named as the clue's source is the film it was copied from.</p>`;
    html += table([["set", "Query set"], ["n", "n"], ["base_mrr", "MRR base", (v) => fmt(v)], ["c1_mrr", "MRR +C1", (v) => fmt(v)],
      ["base_s1", "S@1 base", (v) => fmt(v)], ["c1_s1", "S@1 +C1", (v) => fmt(v)], ["queries_with_flag", "Flagged queries", (v) => fmt(v)],
      ["precision", "Precision", (v) => fmt(v)], ["recall", "Recall", (v) => fmt(v)], ["f1", "F1", (v) => fmt(v)],
      ["attribution", "Source correct", (v) => fmt(v)]], c1rows) + "</section>";
    if (nv.c2 && nv.c2.table) {
      const rows = [];
      for (const [variant, by] of Object.entries(nv.c2.table)) for (const [sys, r] of Object.entries(by)) rows.push({ variant, sys, ...r });
      html += `<section class="eval-section"><h2>Evidence checks C2: remakes and dubs</h2>
        <p class="note">${nv.c2.n_items} curated films whose story has another version in the index (${nv.c2.n_stories} stories), described
        with their vague query, the plot alone, or the plot plus the language heard or the lead actor. Credit counts a description that fits
        both versions as solved when both versions are in the top 2.</p>`;
      html += table([["variant", "Description"], ["sys", "System"], ["n", "n"], ["story@1", "Story@1", (v) => fmt(v)],
        ["exact@1", "Exact version@1", (v) => fmt(v)], ["credit", "Credit", (v) => fmt(v)], ["stated_ambiguous", "Says ambiguous", (v) => fmt(v)],
        ["mrr", "MRR", (v) => fmt(v)]], rows) + "</section>";
    }
    if (nv.c3) {
      const c = nv.c3;
      html += `<section class="eval-section"><h2>Evidence checks C3: events in the remembered order</h2>
        <p class="note">${c.n_pairs} pairs of descriptions built from each curated plot: an early and a late event (names removed) joined by
        “and then”, in plot order and reversed. Pairwise accuracy = share of pairs where the target does better with the plot order.</p>`;
      html += table([["m", "Measure"], ["b", "Base"], ["o", "+C3"]], [
        { m: "Pairwise order accuracy (target score)", b: fmt(c.pairwise_score_base), o: fmt(c.pairwise_score_c3) },
        { m: "Pairwise order accuracy (target rank, ties = ½)", b: fmt(c.pairwise_rank_base), o: fmt(c.pairwise_rank_c3) },
        { m: "MRR, events in plot order", b: fmt(c.mrr_correct_base), o: fmt(c.mrr_correct_c3) },
        { m: "MRR, events reversed", b: fmt(c.mrr_reversed_base), o: fmt(c.mrr_reversed_c3) },
        { m: "MRR, ordinary vague queries", b: fmt(c.vague_mrr_base), o: fmt(c.vague_mrr_c3) }]) + "</section>";
    }
    if (nv.c4) {
      const rows = [];
      for (const [set, r] of Object.entries(nv.c4)) {
        rows.push({ set, policy: "no question", world: "—", ask: 0, ans: null, s1: r.s1_no_question, mrr: r.mrr_no_question });
        for (const key of ["fixed_decade|p_prior", "targeted|p_prior", "fixed_decade|p_equal", "targeted|p_equal"]) {
          const [policy, world] = key.split("|");
          const x = r[key];
          rows.push({ set, policy: policy === "targeted" ? "targeted (ours)" : "fixed “Which decade?”", world: world === "p_prior" ? "priors" : "all 0.6",
            ask: x.ask_rate, ans: x.answer_rate, s1: x.s1, mrr: x.mrr });
        }
      }
      html += `<section class="eval-section"><h2>Evidence checks C4: one question</h2>
        <p class="note">Simulated viewers who know each kind of detail with probability P(answer); “I don't remember” leaves the ranking unchanged.
        Expected Success@1 after one question.</p>`;
      html += table([["set", "Query set"], ["policy", "Question"], ["world", "Viewer knowledge"], ["ask", "Ask rate", (v) => fmt(v)],
        ["ans", "Answer rate", (v) => fmt(v)], ["s1", "S@1", (v) => fmt(v)], ["mrr", "MRR", (v) => fmt(v)]], rows) + "</section>";
    }
  }

  for (const c of conds) {
    html += `<section class="eval-section"><h2>Detailed metrics: ${esc(PRETTY[c] || c)}</h2>
      <p class="note">P@10 = Success@10 / 10 and R@10 = Success@10 because each query has exactly one relevant film; p-values: paired randomisation test against BM25.</p>`;
    html += table([["system", "System"], ["rr", "MRR", (v, r) => `${fmt(v)} <span class="muted">(${fmt(r.mrr_ci_low, 2)}–${fmt(r.mrr_ci_high, 2)})</span>`],
      ["s@1", "S@1 (accuracy)", (v) => fmt(v)], ["s@5", "S@5", (v) => fmt(v)], ["s@10", "S@10 = R@10", (v) => fmt(v)],
      ["ndcg10", "nDCG@10", (v) => fmt(v)], ["ndcg1000", "nDCG@all", (v) => fmt(v)], ["p@10", "P@10", (v) => fmt(v)],
      ["f1@10", "F1@10", (v) => fmt(v)], ["p_vs_bm25", "p vs BM25", (v) => (v === null || Number.isNaN(+v)) ? "—" : (+v < 0.001 ? "<0.001" : fmt(v))]],
      e.main[c], { bestCols: ["rr", "s@1", "s@10", "ndcg10"], ours: isOurs }) + "</section>";
  }

  for (const [c, rows] of Object.entries(e.ablation || {})) {
    html += `<section class="eval-section"><h2>Ablation study: ${esc(PRETTY[c] || c)}</h2>
      <p class="note">The full learning-to-rank model is retrained (cross-validated) with one component removed at a time.</p>`;
    html += table([["variant", "Variant"], ["mrr", "MRR", (v) => fmt(v)], ["delta_mrr", "Δ MRR", (v) => (v > 0 ? "+" : "") + fmt(v)],
      ["s@1", "S@1", (v) => fmt(v)], ["s@10", "S@10", (v) => fmt(v)], ["p_vs_full", "p vs full", (v) => (v === null || Number.isNaN(+v)) ? "—" : fmt(v)]], rows) + "</section>";
  }

  for (const [c, rows] of Object.entries(e.negation_modes || {})) {
    html += `<section class="eval-section"><h2>Negation handling variants: ${esc(PRETTY[c] || c)}</h2>
      <p class="note">“none” counts negated words as positive evidence (plain bag of words); “drop” ignores them; “penalize_all” penalises every negation;
      “scope” (ours) penalises only exclusions about the film itself.</p>`;
    html += table([["mode", "Mode"], ["mrr_all", "MRR (all)", (v) => fmt(v)], ["mrr_negation_queries", "MRR (queries with exclusions)", (v) => fmt(v)],
      ["s@1_negation_queries", "S@1 (exclusion queries)", (v) => fmt(v)], ["n_negation_queries", "n"], ["learned_mu", "learned μ"]], rows) + "</section>";
  }

  for (const [c, rows] of Object.entries(e.phenomena || {})) {
    const cols = Object.keys(rows[0] || {}).filter((k) => !["phenomenon", "n"].includes(k));
    html += `<section class="eval-section"><h2>MRR by tip-of-the-tongue phenomenon: ${esc(PRETTY[c] || c)}</h2>`;
    html += table([["phenomenon", "Queries containing…"], ["n", "n"], ...cols.map((k) => [k, k.replace(/ \(.*\)/, ""), (v) => fmt(v)])], rows) + "</section>";
  }

  if (e.nb) {
    const d = e.nb.decade_docs_cv, g = e.nb.genre_docs_cv, mj = e.nb.decade_docs_majority_baseline;
    html += `<section class="eval-section"><h2>Naive Bayes classifiers (5-fold CV on film plots)</h2>`;
    html += table([["task", "Task"], ["acc", "Accuracy"], ["p", "Macro precision"], ["r", "Macro recall"], ["f1", "Macro F1"]], [
      { task: "Decade (8 classes)", acc: fmt(d.accuracy), p: fmt(d.macro_precision), r: fmt(d.macro_recall), f1: fmt(d.macro_f1) },
      { task: "Decade: majority-class baseline", acc: fmt(mj.accuracy), p: "—", r: "—", f1: fmt(mj.macro_f1) },
      { task: "Genre (top-1 accuracy; P/R/F1 for the top-2 predicted genres)", acc: fmt(g.top1_accuracy), p: fmt(g["macro_precision@2"]), r: fmt(g["macro_recall@2"]), f1: fmt(g["macro_f1@2"]) },
    ]) + "</section>";
  }
  if (e.cue_detection) {
    const cd = e.cue_detection;
    html += `<section class="eval-section"><h2>Cue detection (query level, against the annotation tags)</h2>`;
    html += table([["cue", "Detector"], ["precision", "Precision", (v) => fmt(v)], ["recall", "Recall", (v) => fmt(v)], ["f1", "F1", (v) => fmt(v)], ["accuracy", "Accuracy", (v) => fmt(v)]],
      [{ cue: "Hedged statement", ...cd.hedge }, { cue: "Exclusion (ruling something out)", ...cd.exclusion }]) + "</section>";
  }
  if (e.trec) {
    html += `<section class="eval-section"><h2>Comparison with the TREC 2025 Tip-of-the-Tongue track</h2>
      <p class="note">TREC numbers come from Table 1 of the track overview (Arguello et al., 2026). The settings differ: TREC searches 6.4 million
      Wikipedia articles of every kind, so absolute scores are not comparable; the useful comparison is how much each approach gains over its own BM25 baseline.</p>`;
    html += table([["system", "System"], ["queries", "Queries"], ["corpus", "Corpus"], ["ndcg10", "nDCG@10", (v) => fmt(v)], ["mrr", "MRR", (v) => fmt(v)], ["ndcg1000", "nDCG@1000", (v) => fmt(v)]],
      e.trec, { ours: (r) => r.system.startsWith("Desi") }) + "</section>";
  }
  if (e.final_model) {
    const w = Object.entries(e.final_model.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    html += `<section class="eval-section"><h2>The deployed model</h2>
      <p class="note">Coordinate-ascent ranker trained on all ${e.final_model.trained_on} LLM queries. Learned clause weights:
      c<sub>h</sub> = ${e.final_model.clause_weights.hedged}, c<sub>conf</sub> = ${e.final_model.clause_weights.confident},
      c<sub>ctx</sub> = ${e.final_model.clause_weights.context}, μ = ${e.final_model.clause_weights.mu}. Feature weights (standardised features):</p>`;
    html += table([["f", "Feature"], ["w", "Weight", (v) => fmt(v)]], w.map(([f, v]) => ({ f, w: v }))) + "</section>";
  }
  if (e.human) {
    html += `<section class="eval-section"><h2>Human-written queries</h2><p class="note">${esc(e.human.status || "")}</p>`;
    if (e.human.rows) html += table([["system", "System"], ["rr", "MRR", (v) => fmt(v)], ["s@1", "S@1", (v) => fmt(v)], ["s@10", "S@10", (v) => fmt(v)]], e.human.rows, { ours: isOurs });
    if (e.human.kendall && e.human.kendall.tau !== undefined) html += `<p class="note">Kendall's τ between system rankings on LLM-written (vague) and human-written queries: <b>${fmt(e.human.kendall.tau)}</b> (p = ${fmt(e.human.kendall.p)}).</p>`;
    else if (e.human.kendall) html += `<p class="note">${esc(e.human.kendall.status)}</p>`;
    html += "</section>";
  }
  if (s.figures && s.figures.length) {
    html += `<section class="eval-section"><h2>Figures</h2><div class="figs">${s.figures.map((f) =>
      `<figure><img src="/figures/${encodeURIComponent(f)}" alt="${esc(f)}" loading="lazy"><figcaption>${esc(f.replace(/[-_]/g, " ").replace(/\.png$/, ""))}</figcaption></figure>`).join("")}</div></section>`;
  }
  $("#eval-body").innerHTML = html;
}

/* ------------------------------------------------------------------ start */
showTab((location.hash || "#search").slice(1) || "search", { keepScroll: true });
initHome();
initShelves();
const params = new URLSearchParams(location.search);              // shareable links: /?q=...&lang=hi
if (params.get("lang") && LANG[params.get("lang")]) $("#nav-lang").value = params.get("lang");
if (params.get("q")) runSearch(params.get("q"));
if (params.get("film") && /^\d+$/.test(params.get("film"))) openFilm(+params.get("film"));   // /?film=12
