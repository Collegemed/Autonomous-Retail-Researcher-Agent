/**
 * Autonomous Retail Researcher — Frontend Script
 * Handles: API calls, UI state, pipeline animation, history, clipboard
 */

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = (window.location.port === "80" || window.location.port === "") ? "/api" : "http://127.0.0.1:8000";
// ── State ────────────────────────────────────────────────────────────────────
let isLoading = false;
let timerInterval = null;
let currentResult = null;

// ── Pipeline steps & their simulated timing ──────────────────────────────────
const PIPELINE_STEPS = [
  { id: "step-research",  label: "Searching web...",          delay: 0    },
  { id: "step-analysis",  label: "Analyzing results...",      delay: 4000 },
  { id: "step-summary",   label: "Generating insights...",    delay: 8000 },
  { id: "step-storage",   label: "Storing in memory...",      delay: 12000},
];

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setupInputCounter();
  setupNavigation();
  setupKeyboard();
});

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      dot.className  = "status-dot online";
      text.textContent = "API online";
    } else {
      throw new Error("Non-OK response");
    }
  } catch {
    dot.className  = "status-dot error";
    text.textContent = "API offline";
  }
}

// ── Navigation ───────────────────────────────────────────────────────────────
function setupNavigation() {
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".view").forEach(v => v.style.display = "none");
      document.getElementById(`view-${view}`).style.display = "block";

      if (view === "history") loadHistory();
    });
  });
}

// ── Input counter ────────────────────────────────────────────────────────────
function setupInputCounter() {
  const input = document.getElementById("queryInput");
  const count = document.getElementById("inputCount");
  input.addEventListener("input", () => {
    const len = input.value.length;
    count.textContent = `${len} / 500`;
    count.style.color = len > 450 ? "var(--red)" : "";
  });
}

// ── Keyboard shortcut (Ctrl+Enter) ───────────────────────────────────────────
function setupKeyboard() {
  document.getElementById("queryInput").addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") startResearch();
  });
}

// ── Set query from example chips ─────────────────────────────────────────────
function setQuery(q) {
  const input = document.getElementById("queryInput");
  input.value = q;
  input.dispatchEvent(new Event("input"));
  input.focus();
}

// ── Main research flow ───────────────────────────────────────────────────────
async function startResearch() {
  const query = document.getElementById("queryInput").value.trim();
  if (!query) { shake("#queryInput"); return; }
  if (isLoading) return;

  const useCache = document.getElementById("useCache").checked;

  setLoading(true);
  hideError();
  hideResults();
  resetPipeline();

  const startTime = Date.now();
  startTimer();
  animatePipeline();

  try {
    const res = await fetch(`${API_BASE}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, use_cache: useCache }),
      signal: AbortSignal.timeout(90000), // 90s timeout
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    // Complete all pipeline steps
    completePipeline();
    currentResult = data;
    renderResults(data);
    showToast(data.cached ? "⚡ Retrieved from cache" : `✓ Research complete in ${data.execution_time_seconds}s`);

  } catch (err) {
    failPipeline();
    showError(err.name === "TimeoutError" ? "Request timed out. Try again." : (err.message || "Unknown error"));
  } finally {
    setLoading(false);
    stopTimer();
  }
}

// ── Pipeline animation ────────────────────────────────────────────────────────
function resetPipeline() {
  PIPELINE_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) {
      el.className = "pipeline-step";
      el.querySelector(".step-status").textContent = "idle";
    }
  });
}

let pipelineTimeouts = [];

function animatePipeline() {
  pipelineTimeouts.forEach(t => clearTimeout(t));
  pipelineTimeouts = [];

  PIPELINE_STEPS.forEach((step, i) => {
    const t1 = setTimeout(() => {
      const el = document.getElementById(step.id);
      if (el) {
        el.classList.add("active");
        el.querySelector(".step-status").textContent = "running";
      }
      document.getElementById("loadingText").textContent = step.label;
    }, step.delay);

    // Mark previous step as done when next starts
    if (i > 0) {
      const t2 = setTimeout(() => {
        const prevEl = document.getElementById(PIPELINE_STEPS[i - 1].id);
        if (prevEl) {
          prevEl.classList.remove("active");
          prevEl.classList.add("done");
          prevEl.querySelector(".step-status").textContent = "done";
        }
      }, step.delay);
      pipelineTimeouts.push(t2);
    }
    pipelineTimeouts.push(t1);
  });
}

function completePipeline() {
  pipelineTimeouts.forEach(t => clearTimeout(t));
  PIPELINE_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) {
      el.className = "pipeline-step done";
      el.querySelector(".step-status").textContent = "done";
    }
  });
}

function failPipeline() {
  pipelineTimeouts.forEach(t => clearTimeout(t));
  PIPELINE_STEPS.forEach(s => {
    const el = document.getElementById(s.id);
    if (el && el.classList.contains("active")) {
      el.classList.remove("active");
      el.querySelector(".step-status").textContent = "failed";
      el.style.borderColor = "rgba(240,84,84,0.3)";
    }
  });
}

// ── Timer ────────────────────────────────────────────────────────────────────
let timerStart = 0;
function startTimer() {
  timerStart = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - timerStart) / 1000).toFixed(1);
    document.getElementById("loadingTimer").textContent = `${elapsed}s`;
  }, 100);
}
function stopTimer() {
  clearInterval(timerInterval);
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const panel = document.getElementById("resultsPanel");
  const result = data.result || {};

  // Query title
  document.getElementById("resultQuery").textContent = data.query;

  // Badges
  const badgesEl = document.getElementById("resultBadges");
  badgesEl.innerHTML = "";
  if (data.cached)                  addBadge(badgesEl, "⚡ Cached", "amber");
  if (result.rag_context_used)      addBadge(badgesEl, "◈ RAG Memory", "blue");
  if (result.source_count)          addBadge(badgesEl, `${result.source_count} sources`, "green");
  if (data.execution_time_seconds)  addBadge(badgesEl, `${data.execution_time_seconds}s`, "amber");

  // Sections
  const grid = document.getElementById("sectionsGrid");
  grid.innerHTML = "";
  const sections = result.sections || {};

  const sectionOrder = [
    "Executive Summary",
    "Key Findings",
    "Market Trends",
    "Competitive Intelligence",
    "Strategic Recommendations",
    "Overview",
  ];

  let cardIdx = 0;
  for (const key of sectionOrder) {
    if (sections[key]) {
      renderSectionCard(grid, key, sections[key], cardIdx++);
    }
  }

  // Any remaining sections not in order
  for (const [key, val] of Object.entries(sections)) {
    if (!sectionOrder.includes(key) && val) {
      renderSectionCard(grid, key, val, cardIdx++);
    }
  }

  // If no sections, show raw summary
  if (cardIdx === 0 && result.summary) {
    renderSectionCard(grid, "Research Summary", result.summary, 0, true);
  }

  // Key themes
  const themes = result.key_themes || [];
  const themesList = document.getElementById("themesList");
  themesList.innerHTML = themes.map(t =>
    `<span class="theme-tag">${escHtml(t)}</span>`
  ).join("");
  document.getElementById("themesBlock").style.display = themes.length ? "block" : "none";

  // Sources
  const sources = result.sources || [];
  document.getElementById("sourceCount").textContent = sources.length ? `(${sources.length})` : "";
  const sourcesList = document.getElementById("sourcesList");
  sourcesList.innerHTML = sources.map(s => `
    <div class="source-item">
      <span class="source-score">${(s.score * 10).toFixed(0)}/10</span>
      <span class="source-title" title="${escHtml(s.title)}">${escHtml(s.title)}</span>
      ${s.url ? `<a class="source-link" href="${escHtml(s.url)}" target="_blank" rel="noopener">↗ visit</a>` : ""}
    </div>
  `).join("");
  document.getElementById("sourcesBlock").style.display = sources.length ? "block" : "none";

  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSectionCard(grid, title, content, idx, fullWidth = false) {
  const card = document.createElement("div");
  card.className = "section-card" + (fullWidth ? " full-width" : "");
  card.style.animationDelay = `${idx * 60}ms`;

  const lines = content.split("\n").filter(l => l.trim());
  const formatted = lines.map(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith("•") || trimmed.startsWith("-") || trimmed.startsWith("*")) {
      return `<div class="bullet">${escHtml(trimmed.replace(/^[•\-*]\s*/, ""))}</div>`;
    }
    return `<p>${escHtml(trimmed)}</p>`;
  }).join("");

  card.innerHTML = `
    <div class="section-label">${escHtml(title)}</div>
    <div class="section-content">${formatted}</div>
  `;
  grid.appendChild(card);
}

function addBadge(container, text, color) {
  const span = document.createElement("span");
  span.className = `badge badge-${color}`;
  span.textContent = text;
  container.appendChild(span);
}

// ── History ──────────────────────────────────────────────────────────────────
async function loadHistory() {
  const list = document.getElementById("historyList");
  list.innerHTML = `<div class="history-empty">Loading...</div>`;

  try {
    const res = await fetch(`${API_BASE}/history?limit=30`);
    const data = await res.json();

    if (!data.history || data.history.length === 0) {
      list.innerHTML = `<div class="history-empty">No research history yet.</div>`;
      return;
    }

    list.innerHTML = data.history.map((item, i) => {
      const date = new Date(item.created_at * 1000);
      const timeStr = date.toLocaleString();
      const summary = item.result?.summary || item.result?.sections?.["Executive Summary"] || "No summary available.";
      const preview = summary.replace(/\n/g, " ").slice(0, 140);

      return `
        <div class="history-card" style="animation-delay:${i * 40}ms"
             onclick="loadHistoryItem(${i})"
             data-idx="${i}" data-query="${escHtml(item.query)}">
          <div class="hcard-header">
            <span class="hcard-query">${escHtml(item.query)}</span>
            <span class="hcard-time">${timeStr}</span>
          </div>
          <div class="hcard-preview">${escHtml(preview)}${preview.length === 140 ? "..." : ""}</div>
        </div>
      `;
    }).join("");

    // Store history data for click loading
    window._historyData = data.history;

  } catch (err) {
    list.innerHTML = `<div class="history-empty">Failed to load history: ${err.message}</div>`;
  }
}

function loadHistoryItem(idx) {
  const item = window._historyData?.[idx];
  if (!item) return;

  // Switch to research view
  document.querySelector('.nav-btn[data-view="research"]').click();

  // Set query
  document.getElementById("queryInput").value = item.query;
  document.getElementById("queryInput").dispatchEvent(new Event("input"));

  // Render results
  renderResults({ query: item.query, result: item.result, cached: true, execution_time_seconds: 0 });
  showToast("📂 Loaded from history");
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function setLoading(state) {
  isLoading = state;
  document.getElementById("loadingPanel").style.display = state ? "block" : "none";
  const btn = document.getElementById("researchBtn");
  btn.disabled = state;
  btn.querySelector(".btn-text").textContent = state ? "Researching..." : "Run Research";
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("errorPanel").style.display = "flex";
}
function hideError()   { document.getElementById("errorPanel").style.display = "none"; }
function hideResults() { document.getElementById("resultsPanel").style.display = "none"; }
function clearResults() { hideResults(); currentResult = null; }

function copyResult() {
  if (!currentResult) return;
  const result = currentResult.result || {};
  const text = [
    `Query: ${currentResult.query}`,
    "",
    result.summary || "No summary available.",
    "",
    result.key_themes?.length ? `Themes: ${result.key_themes.join(", ")}` : "",
  ].filter(Boolean).join("\n");

  navigator.clipboard.writeText(text).then(() => showToast("📋 Copied to clipboard"));
}

function showToast(msg, duration = 3000) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), duration);
}

function shake(selector) {
  const el = document.querySelector(selector);
  if (!el) return;
  el.style.animation = "none";
  el.offsetHeight; // reflow
  el.style.animation = "shake 0.4s ease-out";
  setTimeout(() => el.style.animation = "", 400);
}

function escHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Add shake animation dynamically
const style = document.createElement("style");
style.textContent = `@keyframes shake {
  0%,100%{transform:translateX(0)}
  20%{transform:translateX(-8px)}
  40%{transform:translateX(8px)}
  60%{transform:translateX(-5px)}
  80%{transform:translateX(5px)}
}`;
document.head.appendChild(style);