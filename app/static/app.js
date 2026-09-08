/* ============================================================================
   Cardiac Risk Model Explorer - front-end
   ----------------------------------------------------------------------------
   Every value rendered here arrives from the API, which reads it from the
   artefacts written by `python -m src.train`. There are no hard-coded metrics,
   probabilities or example results anywhere in this file.
   ========================================================================== */
(() => {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = { schema: null, model: null, metrics: null, busy: false, lastResult: null };

  /* ------------------------------------------------------------- helpers */
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
    ? "—" : Number(v).toFixed(d);
  const pct = (v, d = 1) => (v === null || v === undefined) ? "—"
    : `${(Number(v) * 100).toFixed(d)}%`;

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch {
      const err = new Error(
        "Could not reach the prediction service. Check that the server is " +
        "running, then try again.");
      err.code = "network_error";
      err.fields = {};
      throw err;
    }
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) {
      const err = new Error((body && body.message) || `Request failed (${res.status})`);
      err.fields = (body && body.fields) || {};
      err.code = (body && body.error) || "network_error";
      throw err;
    }
    return body;
  }

  const icon = (id, cls = "") =>
    `<svg aria-hidden="true" ${cls ? `class="${cls}"` : ""}><use href="#${id}"/></svg>`;

  /* --------------------------------------------------------------- theme */
  function initTheme() {
    const btn = $("#theme-toggle");
    const apply = (mode) => {
      if (mode) document.documentElement.setAttribute("data-theme", mode);
      else document.documentElement.removeAttribute("data-theme");
      const dark = mode === "dark" ||
        (!mode && window.matchMedia("(prefers-color-scheme: dark)").matches);
      $("#theme-icon").setAttribute("href", dark ? "#i-sun" : "#i-moon");
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    };
    let stored = null;
    try { stored = localStorage.getItem("crme-theme"); } catch { /* private mode */ }
    apply(stored);
    btn.addEventListener("click", () => {
      const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.hasAttribute("data-theme") &&
          window.matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      apply(next);
      try { localStorage.setItem("crme-theme", next); } catch { /* ignore */ }
      redrawCharts();
    });
    window.matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => { apply(stored); redrawCharts(); });
  }

  /* ----------------------------------------------------------------- nav */
  function initNav() {
    const toggle = $("#nav-toggle");
    const nav = $("#site-nav");
    const isMobile = () => window.matchMedia("(max-width: 860px)").matches;

    const setOpen = (open) => {
      nav.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      $("#nav-icon").setAttribute("href", open ? "#i-close" : "#i-menu");
      toggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
    };
    // Only react when the breakpoint actually changes. Mobile browsers fire
    // resize when the URL bar hides, and closing an open menu on that is a bug.
    let wasMobile = isMobile();
    const sync = (force) => {
      if (isMobile()) { if (force) setOpen(false); }
      else { nav.hidden = false; toggle.setAttribute("aria-expanded", "false"); }
    };
    sync(true);
    window.addEventListener("resize", () => {
      const now = isMobile();
      if (now !== wasMobile) { wasMobile = now; sync(true); }
    });
    toggle.addEventListener("click", () => setOpen(nav.hidden));
    nav.addEventListener("click", (e) => { if (e.target.closest("a") && isMobile()) setOpen(false); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isMobile() && !nav.hidden) { setOpen(false); toggle.focus(); }
    });

    // scroll spy -> aria-current
    const links = $$("#site-nav a");
    const sections = links.map((a) => $(a.getAttribute("href"))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => {
          const on = a.getAttribute("href") === `#${entry.target.id}`;
          if (on) a.setAttribute("aria-current", "true");
          else a.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => observer.observe(s));
  }

  /* ================================================================ CHARTS
     Hand-rolled inline SVG: no chart library, so the page has no runtime
     dependency and the marks follow the project's own spec - 2px lines,
     hairline recessive grid, colours read from CSS custom properties so both
     themes use their own validated step.
     ==================================================================== */
  function redrawCharts() {
    // Charts read their colours from CSS custom properties at build time, so a
    // theme change means re-rendering them with the new theme's validated steps.
    if (state.metrics && state.model && state.schema) renderPerformanceSection();
    if (state.lastResult) renderResult(state.lastResult);
  }

  const cssVar = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function lineChart({ series, refLine, xLabel, yLabel, ticks = 5, height = 250 }) {
    const W = 420, H = height, m = { t: 12, r: 14, b: 40, l: 46 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const sx = (x) => m.l + x * iw;
    const sy = (y) => m.t + (1 - y) * ih;
    const grid = cssVar("--grid"), axis = cssVar("--axis"), muted = cssVar("--muted");

    let g = "";
    for (let i = 0; i <= ticks; i++) {
      const f = i / ticks;
      g += `<line x1="${sx(0)}" y1="${sy(f)}" x2="${sx(1)}" y2="${sy(f)}" stroke="${grid}" stroke-width="1"/>`;
      g += `<text x="${m.l - 8}" y="${sy(f) + 4}" text-anchor="end" font-size="10" fill="${muted}">${f.toFixed(1)}</text>`;
      g += `<text x="${sx(f)}" y="${H - m.b + 16}" text-anchor="middle" font-size="10" fill="${muted}">${f.toFixed(1)}</text>`;
    }
    g += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(0)}" stroke="${axis}" stroke-width="1"/>`;
    g += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(0)}" y2="${sy(1)}" stroke="${axis}" stroke-width="1"/>`;

    if (refLine && refLine.points) {
      const pts = refLine.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ");
      g += `<polyline points="${pts}" fill="none" stroke="${axis}" stroke-width="1.5"
             stroke-dasharray="5 4" stroke-linecap="round"/>`;
    }
    series.forEach((s) => {
      const colour = cssVar(s.colorVar);
      // sklearn returns the precision-recall curve with recall descending; an
      // area polygon built from that order self-intersects, so sort by x first.
      const ordered = [...s.points].sort((a, b) => a[0] - b[0]);
      const pts = ordered.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ");
      if (s.fill) {
        g += `<polygon points="${sx(0)},${sy(0)} ${pts} ${sx(1)},${sy(0)}"
               fill="${colour}" opacity="0.10"/>`;
      }
      g += `<polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
      if (s.markers) {
        ordered.forEach(([x, y]) => {
          g += `<circle cx="${sx(x)}" cy="${sy(y)}" r="4" fill="${colour}"
                 stroke="${cssVar("--surface")}" stroke-width="2"/>`;
        });
      }
    });
    if (refLine && refLine.marker !== undefined) {
      g += `<line x1="${sx(refLine.marker)}" y1="${sy(0)}" x2="${sx(refLine.marker)}" y2="${sy(1)}"
             stroke="${cssVar("--ink-2")}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    }
    g += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle" font-size="10.5" fill="${muted}">${esc(xLabel)}</text>`;
    g += `<text x="12" y="${m.t + ih / 2}" text-anchor="middle" font-size="10.5" fill="${muted}"
           transform="rotate(-90 12 ${m.t + ih / 2})">${esc(yLabel)}</text>`;

    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
  }

  function barChart({ rows, colorVar = "--series-1", valueFormat, signed = false }) {
    const W = 560, rowH = 30, m = { t: 8, r: signed ? 12 : 66, b: 8, l: 214 };
    const H = m.t + rows.length * rowH + m.b;
    const iw = W - m.l - m.r;
    const max = Math.max(...rows.map((r) => Math.abs(r.value))) || 1;
    const ink = cssVar("--ink-2"), muted = cssVar("--muted"), axis = cssVar("--axis");
    const pos = cssVar("--series-2"), neg = cssVar("--series-1");
    const zero = signed ? m.l + iw / 2 : m.l;
    const span = signed ? iw / 2 : iw;

    let g = "";
    if (signed) {
      g += `<line x1="${zero}" y1="${m.t}" x2="${zero}" y2="${H - m.b}"
             stroke="${axis}" stroke-width="1"/>`;
    }
    rows.forEach((r, i) => {
      const y = m.t + i * rowH;
      const w = Math.max(2, (Math.abs(r.value) / max) * span * 0.94);
      const negative = r.value < 0;
      const x = signed && negative ? zero - w : zero;
      const colour = signed ? (negative ? neg : pos) : cssVar(colorVar);
      const label = r.label.length > 30 ? `${r.label.slice(0, 28)}…` : r.label;
      const vx = signed ? (negative ? x - 6 : x + w + 6) : m.l + w + 8;
      const anchor = signed && negative ? "end" : "start";
      g += `<text x="${m.l - 12}" y="${y + 19}" text-anchor="end" font-size="12.5" fill="${ink}">${esc(label)}</text>`;
      g += `<rect x="${x}" y="${y + 8}" width="${w}" height="13" rx="4" fill="${colour}"/>`;
      g += `<text x="${vx}" y="${y + 19}" text-anchor="${anchor}" font-size="11.5" fill="${muted}">${esc(valueFormat(r.value))}</text>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
  }

  function chartCard({ title, sub, svg, legend, ariaLabel }) {
    return `<div class="chart-card">
      <h4>${esc(title)}</h4>
      <p class="sub">${esc(sub)}</p>
      <div role="img" aria-label="${esc(ariaLabel)}" data-chart="${esc(title)}">${svg}</div>
      ${legend ? `<div class="chart-legend">${legend}</div>` : ""}
    </div>`;
  }

  const legendItem = (label, varName, dashed = false) =>
    `<span><span class="swatch${dashed ? " dashed" : ""}"${dashed ? "" : ` style="background:var(${varName})"`}></span>${esc(label)}</span>`;

  /* =============================================================== FORM */
  function buildForm(schema) {
    const host = $("#fieldsets");
    host.innerHTML = schema.groups.map((group) => {
      const feats = schema.features.filter((f) => f.group === group);
      if (!feats.length) return "";
      return `<fieldset>
        <legend>${esc(group)}</legend>
        <div class="field-grid">${feats.map(fieldMarkup).join("")}</div>
      </fieldset>`;
    }).join("");
    host.setAttribute("aria-busy", "false");

    $$("#fieldsets .control").forEach((el) => {
      el.addEventListener("blur", () => validateField(el.name, true));
      el.addEventListener("input", () => {
        if ($(`.field[data-name="${el.name}"]`).classList.contains("has-error")) {
          validateField(el.name, true);
        }
      });
    });
    $("#form-subtitle").textContent =
      `${schema.features.length} clinical inputs, grouped by where they come from`;
  }

  function fieldMarkup(f) {
    const unit = f.unit && f.unit !== "count" ? ` <span class="unit">(${esc(f.unit)})</span>` : "";
    const describedBy = `h-${f.name} e-${f.name}`;
    const control = f.role === "numeric"
      ? `<input class="control" type="number" id="f-${f.name}" name="${f.name}"
                min="${f.min}" max="${f.max}" step="${f.step}"
                inputmode="${f.step === 1 ? "numeric" : "decimal"}"
                aria-describedby="${describedBy}" autocomplete="off">`
      : `<select class="control" id="f-${f.name}" name="${f.name}" aria-describedby="${describedBy}">
           <option value="">Select…</option>
           ${f.choices.map((c) => `<option value="${c.value}">${esc(c.label)}</option>`).join("")}
         </select>`;
    const hint = f.role === "numeric"
      ? (f.help || `Allowed range ${f.min}–${f.max}.`)
      : (f.help || f.description);
    return `<div class="field" data-name="${f.name}">
      <label for="f-${f.name}">${esc(f.label)}${unit}</label>
      ${control}
      <p class="hint" id="h-${f.name}">${esc(hint)}</p>
      <p class="field-error-text" id="e-${f.name}"></p>
    </div>`;
  }

  /* Mirrors the server-side rules in src/inference.validate_record. The server
     re-checks everything; this only shortens the feedback loop. */
  function fieldError(spec, raw) {
    if (raw === "" || raw === null || raw === undefined) return `${spec.label} is required.`;
    const value = Number(raw);
    if (!Number.isFinite(value)) return `${spec.label} must be a number.`;
    if (spec.role === "numeric") {
      if (value < spec.min || value > spec.max) {
        const unit = spec.unit && spec.unit !== "count" ? ` ${spec.unit}` : "";
        return `${spec.label} must be between ${spec.min} and ${spec.max}${unit}.`;
      }
      if (spec.step === 1 && Math.abs(value - Math.round(value)) > 1e-9) {
        return `${spec.label} must be a whole number.`;
      }
    } else if (!spec.choices.some((c) => Number(c.value) === value)) {
      return `${spec.label} must be one of the listed options.`;
    }
    return null;
  }

  function validateField(name, show) {
    const spec = state.schema.features.find((f) => f.name === name);
    const input = $(`#f-${name}`);
    const message = fieldError(spec, input.value.trim());
    if (show) setFieldError(name, message);
    return message;
  }

  function setFieldError(name, message) {
    const wrap = $(`.field[data-name="${name}"]`);
    const input = $(`#f-${name}`);
    const slot = $(`#e-${name}`);
    if (message) {
      wrap.classList.add("has-error");
      input.setAttribute("aria-invalid", "true");
      slot.textContent = message;
    } else {
      wrap.classList.remove("has-error");
      input.removeAttribute("aria-invalid");
      slot.textContent = "";
    }
  }

  function showErrorSummary(errors) {
    const box = $("#error-summary");
    const list = $("#error-summary-list");
    const entries = Object.entries(errors);
    if (!entries.length) { box.hidden = true; return; }
    $("#error-summary-title").textContent = entries.length === 1
      ? "There is 1 problem with this record"
      : `There are ${entries.length} problems with this record`;
    list.innerHTML = entries.map(([name, msg]) =>
      `<li><a href="#f-${esc(name)}">${esc(msg)}</a></li>`).join("");
    box.hidden = false;
    list.querySelectorAll("a").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const target = $(a.getAttribute("href"));
        if (target) { target.focus(); target.scrollIntoView({ block: "center" }); }
      });
    });
    box.focus();
  }

  function collect() {
    const out = {};
    state.schema.features.forEach((f) => {
      const raw = $(`#f-${f.name}`).value.trim();
      out[f.name] = raw === "" ? null : Number(raw);
    });
    return out;
  }

  /* ============================================================== RESULT */
  function renderLoading() {
    $("#result-region").innerHTML = `
      <div class="result-card" aria-busy="true">
        <div class="result-head">
          <div class="loading-row"><span class="spinner"></span>
            <span>Analysing patient data…</span></div>
          <div class="skeleton-line w60"></div>
          <div class="skeleton-line tall w80"></div>
        </div>
        <div class="meter-block">
          <div class="skeleton-line w40"></div>
          <div class="skeleton-line tall"></div>
        </div>
        <div class="influence-block">
          <div class="skeleton-line w60"></div>
          <div class="skeleton-line"></div><div class="skeleton-line w80"></div>
        </div>
      </div>`;
  }

  /* The meter is HTML rather than SVG: a non-uniformly scaled SVG squashes its
     own text, which is exactly what happened to the threshold label. */
  function probabilityMeter(p, t) {
    const positive = p >= t;
    const left = Math.min(96, Math.max(4, t * 100));
    return `<div class="meter-wrap" role="img"
      aria-label="Model-estimated probability ${pct(p)}, against a decision threshold of ${pct(t, 0)}.">
      <div class="meter-track">
        <div class="meter-fill" data-class="${positive ? "Positive" : "Negative"}"
             style="width:${(p * 100).toFixed(1)}%"></div>
        <div class="meter-thresh" style="left:${left}%"></div>
      </div>
      <div class="meter-thresh-label" style="left:${left}%">threshold ${pct(t, 0)}</div>
    </div>`;
  }

  function renderResult(r) {
    state.lastResult = r;
    const positive = r.classification === "Positive";
    const facts = [
      ["Decision threshold", `${pct(r.threshold, 0)}`],
      ["Distance from threshold", `${r.margin_from_threshold >= 0 ? "+" : ""}${(r.margin_from_threshold * 100).toFixed(1)} pts`],
      ["Model", r.model.name],
      ["Response time", `${r.latency_ms} ms`],
    ];
    const maxShare = Math.max(...r.explanation.items.map((i) => i.share)) || 1;

    $("#result-region").innerHTML = `
      <div class="result-card revealing">
        <div class="result-head">
          <span class="result-kicker">Model result</span>
          <div class="verdict" data-class="${esc(r.classification)}">
            <span class="verdict-icon">${icon(positive ? "i-flag" : "i-check")}</span>
            <span class="verdict-text">
              <strong>${esc(r.classification)}</strong>
              <span>Model classification at the ${pct(r.threshold, 0)} decision threshold</span>
            </span>
          </div>
        </div>

        <div class="meter-block">
          <div class="meter-row">
            <span class="label">Model-estimated probability</span>
            <span class="meter-value">${pct(r.probability)}</span>
          </div>
          <div class="meter">${probabilityMeter(r.probability, r.threshold)}</div>
          <div class="meter-legend"><span>0%</span><span>100%</span></div>
          <p class="meter-caption">
            This is the model's own estimate for this record, not a measured
            likelihood of disease. The result sits ${esc(r.confidence_band)}.
          </p>
        </div>

        <dl class="result-facts">
          ${facts.map(([k, v]) => `<div class="fact"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}
        </dl>

        <div class="influence-block">
          <h4>Why did the model produce this result?</h4>
          <p class="note">
            Ranked by how far each input moved this model's own output
            (${esc(r.explanation.units)}). Influence on the model is not medical causation.
          </p>
          ${r.explanation.items.map((it) => `
            <div class="infl-row">
              <div class="infl-top">
                <span class="infl-name">${esc(it.label)}</span>
                <span class="infl-val">${esc(it.display_value)}</span>
              </div>
              <div class="infl-track">
                <div class="infl-fill" data-dir="${esc(it.direction)}"
                     style="width:${(it.share / maxShare * 100).toFixed(1)}%"></div>
              </div>
              <span class="infl-dir" data-dir="${esc(it.direction)}">
                ${icon(it.direction === "increases" ? "i-up" : "i-down")}
                ${it.direction === "increases" ? "Raised" : "Lowered"} the estimate
                · ${(it.share * 100).toFixed(0)}% of total influence
              </span>
            </div>`).join("")}
        </div>

        <p class="result-foot">${esc(r.disclaimer)}</p>
      </div>`;
  }

  function renderResultError(message, code) {
    $("#result-region").innerHTML = `
      <div class="alert" role="alert" style="margin:0">
        ${icon("i-alert")}
        <div>
          <h4>${code === "model_unavailable" ? "Model unavailable" : "The prediction could not be completed"}</h4>
          <p style="margin:0">${esc(message)}</p>
          <p style="margin:8px 0 0">
            ${code === "model_unavailable"
              ? "Run <code>python -m src.train</code> and restart the server."
              : "Check the highlighted fields, then select Analyse risk again."}
          </p>
        </div>
      </div>`;
  }

  /* ============================================================== SUBMIT */
  async function onSubmit(event) {
    event.preventDefault();
    if (state.busy) return;

    const errors = {};
    state.schema.features.forEach((f) => {
      const msg = validateField(f.name, true);
      if (msg) errors[f.name] = msg;
    });
    if (Object.keys(errors).length) { showErrorSummary(errors); return; }
    $("#error-summary").hidden = true;

    state.busy = true;
    const btn = $("#submit-btn");
    btn.disabled = true;
    $("#reset-btn").disabled = true;
    $("#submit-label").textContent = "Analysing…";
    renderLoading();

    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collect()),
      });
      renderResult(result);
    } catch (err) {
      if (err.fields && Object.keys(err.fields).length) {
        Object.entries(err.fields).forEach(([k, v]) => setFieldError(k, v));
        showErrorSummary(err.fields);
      }
      renderResultError(err.message, err.code);
    } finally {
      state.busy = false;
      btn.disabled = false;
      $("#reset-btn").disabled = false;
      $("#submit-label").textContent = "Analyse risk";
    }
  }

  function onReset() {
    window.setTimeout(() => {
      state.schema.features.forEach((f) => setFieldError(f.name, null));
      $("#error-summary").hidden = true;
      state.lastResult = null;
      $("#result-region").innerHTML = `
        <div class="result-empty" id="result-empty">
          ${icon("i-scan")}
          <h3>No result yet</h3>
          <p>Complete the record and select <strong>Analyse risk</strong>. The model output,
             its estimated probability, the decision threshold and the most influential
             inputs will appear here.</p>
        </div>`;
    }, 0);
  }

  async function loadExample() {
    const btn = $("#load-example");
    btn.disabled = true;
    try {
      const ex = await api("/api/example");
      Object.entries(ex.record).forEach(([k, v]) => {
        const el = $(`#f-${k}`);
        if (!el) return;
        const spec = state.schema.features.find((f) => f.name === k);
        el.value = spec.role === "numeric" ? String(v) : String(Number(v));
        setFieldError(k, null);
      });
      $("#error-summary").hidden = true;
      $("#threshold-hint").textContent = "Record loaded from the held-out test split.";
    } catch (err) {
      $("#threshold-hint").textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  }

  /* ======================================================= MODEL SECTION */
  const roleCount = (role) =>
    state.schema.features.filter((f) => f.role === role).length;

  function renderModelSection() {
    const card = state.model.card;
    const meta = state.model.metadata;
    const sel = state.model.selection;
    const m = state.metrics;
    const dict = state.schema.dictionary;

    $("#model-content").innerHTML = `
      <div class="grid grid-2" style="margin-bottom:var(--s-5)">
        <div class="card">
          <h3>Selected model: ${esc(card.name)}</h3>
          <p style="color:var(--ink-2)">${esc(sel.rule)}</p>
          <p style="color:var(--ink-2);margin-bottom:0">
            Every candidate landed inside the one-standard-error band on ROC-AUC
            (cut-off ${num(sel.roc_auc_one_se_cutoff, 4)}), so discrimination alone could not
            separate them. ${esc(card.name)} was selected on the tie-breakers.
          </p>
        </div>
        <div class="card">
          <h3>Preprocessing</h3>
          <dl class="kv">
            <dt>Numeric</dt><dd>Median imputation, then standardisation — ${esc(roleCount("numeric"))} continuous inputs.</dd>
            <dt>Binary</dt><dd>Mode imputation, passed through as 0/1 — ${esc(roleCount("binary"))} inputs.</dd>
            <dt>Categorical</dt><dd>Mode imputation, then one-hot encoding — ${esc(roleCount("nominal"))} inputs.</dd>
            <dt>Design matrix</dt><dd>${esc(card.encoded_features)} encoded columns from ${esc(card.input_features)} raw inputs.</dd>
            <dt>Leakage control</dt><dd>Fitted inside the pipeline, so it re-fits on the training part of every fold. The test split is transformed, never fitted.</dd>
          </dl>
        </div>
      </div>

      <h3>Selection trace</h3>
      <div class="table-wrap" style="margin-bottom:var(--s-5)">
        <table>
          <caption>How each shortlisted model ranked on the tie-breakers, in the order the rule applies them.</caption>
          <thead><tr>
            <th scope="col">Model</th><th scope="col">CV ROC-AUC</th><th scope="col">CV PR-AUC</th>
            <th scope="col">Brier</th><th scope="col">Calibration group</th>
            <th scope="col">Interpretability tier</th><th scope="col">Train–test gap</th>
          </tr></thead>
          <tbody>
            ${sel.ranking_within_shortlist.map((r) => `
              <tr class="${r.model === sel.selected ? "is-final" : ""}">
                <th scope="row">${esc(r.model)}${r.model === sel.selected ? '<span class="pill">Selected</span>' : ""}</th>
                <td>${num(r.cv_roc_auc, 4)}</td><td>${num(r.cv_pr_auc, 4)}</td>
                <td>${num(r.cv_brier, 4)}</td><td>${esc(r.calibration_tie_group)}</td>
                <td>${esc(r.interpretability_tier)}</td><td>${num(r.generalisation_gap, 4)}</td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>

      <h3>Input features</h3>
      <div class="table-wrap" style="margin-bottom:var(--s-5)">
        <table>
          <caption>The thirteen dataset columns, their meaning and their allowed values. The form is generated from this same schema.</caption>
          <thead><tr>
            <th scope="col">Interface label</th><th scope="col">Column</th>
            <th scope="col">Type</th><th scope="col">Units</th>
            <th scope="col">Allowed values</th><th scope="col">Meaning</th>
          </tr></thead>
          <tbody>
            ${dict.map((d) => `<tr>
              <th scope="row">${esc(d.ui_label)}</th>
              <td style="text-align:left"><code>${esc(d.feature)}</code></td>
              <td style="text-align:left">${esc(d.type)}</td>
              <td style="text-align:left">${esc(d.units)}</td>
              <td style="text-align:left;white-space:normal">${esc(d.domain)}</td>
              <td style="text-align:left;white-space:normal">${esc(d.meaning)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>

      <details class="tech">
        <summary>Reproducibility and environment</summary>
        <div class="tech-body">
          <dl class="kv">
            <dt>Dataset</dt><dd>${esc(m.dataset.name)} — <a href="${esc(m.dataset.page)}" rel="noopener noreferrer" target="_blank">UCI repository entry</a></dd>
            <dt>Records</dt><dd>${esc(m.dataset.rows)} (${esc(m.dataset.class_counts["0"])} negative, ${esc(m.dataset.class_counts["1"])} positive)</dd>
            <dt>Target</dt><dd>${esc(m.dataset.target_description)}</dd>
            <dt>Split</dt><dd>${esc(m.split.strategy)}, test size ${esc(m.split.test_size)} — ${esc(m.split.train_rows)} train / ${esc(m.split.test_rows)} test</dd>
            <dt>Cross-validation</dt><dd>${esc(m.split.cv)}</dd>
            <dt>Random seed</dt><dd><code>${esc(m.split.random_seed)}</code></dd>
            <dt>Hyper-parameters</dt><dd><code>${esc(JSON.stringify(meta.final_model.hyperparameters))}</code></dd>
            <dt>Search</dt><dd>${esc(meta.final_model.search_strategy)}, scored on ROC-AUC</dd>
            <dt>Threshold</dt><dd>${esc(meta.final_model.decision_threshold)} — ${esc(meta.final_model.threshold_rationale)}</dd>
            <dt>Calibration</dt><dd>${esc(meta.final_model.calibration)}</dd>
            <dt>Explanation</dt><dd><code>${esc(meta.final_model.local_explanation_method)}</code></dd>
            <dt>Environment</dt><dd>Python ${esc(m.environment.python)}, scikit-learn ${esc(m.environment.scikit_learn)}, XGBoost ${esc(m.environment.xgboost || "n/a")}, SHAP ${esc(m.environment.shap_available ? "available" : "unavailable")}</dd>
            <dt>Trained</dt><dd>${esc(card.trained_utc)} UTC</dd>
          </dl>
        </div>
      </details>`;
  }

  /* ================================================= PERFORMANCE SECTION */
  function renderPerformanceSection() {
    const m = state.metrics;
    const chosen = m.test_chosen, dflt = m.test_default;
    const cm = chosen.confusion_matrix;
    const rows = m.comparison_table;
    const ta = m.threshold_analysis;

    const tiles = [
      ["Test ROC-AUC", num(chosen.roc_auc, 3), "Ranking quality across all thresholds"],
      ["Test PR-AUC", num(chosen.pr_auc, 3), `Base rate ${pct(m.curves.positive_rate, 0)}`],
      ["Sensitivity", pct(chosen.sensitivity), "Positive records correctly flagged"],
      ["Specificity", pct(chosen.specificity), "Negative records correctly cleared"],
    ];

    const rocSvg = lineChart({
      series: [{ points: m.curves.roc.fpr.map((x, i) => [x, m.curves.roc.tpr[i]]), colorVar: "--series-1", fill: true }],
      refLine: { points: [[0, 0], [1, 1]] },
      xLabel: "False positive rate (1 − specificity)", yLabel: "Sensitivity",
    });
    const prSvg = lineChart({
      series: [{ points: m.curves.pr.recall.map((x, i) => [x, m.curves.pr.precision[i]]), colorVar: "--series-2", fill: true }],
      refLine: { points: [[0, m.curves.positive_rate], [1, m.curves.positive_rate]] },
      xLabel: "Recall (sensitivity)", yLabel: "Precision",
    });

    const cal = m.calibration.raw;
    const calSvg = lineChart({
      series: [{
        points: cal.mean_predicted.map((x, i) => [x, cal.observed_frequency[i]]),
        colorVar: "--series-1", markers: true,
      }],
      refLine: { points: [[0, 0], [1, 1]] },
      xLabel: "Mean model-estimated probability", yLabel: "Observed positive rate",
    });

    const sweep = m.threshold_sweep;
    const sweepSvg = lineChart({
      series: [
        { points: sweep.map((r) => [r.threshold, r.sensitivity]), colorVar: "--series-2" },
        { points: sweep.map((r) => [r.threshold, r.specificity]), colorVar: "--series-1" },
      ],
      refLine: { marker: ta.chosen_threshold },
      xLabel: "Decision threshold", yLabel: "Metric value",
    });

    const perm = m.explainability.permutation_importance.slice(0, 8);
    const permSvg = barChart({
      rows: perm.map((p) => ({ label: p.label, value: p.mean_drop_in_roc_auc })),
      valueFormat: (v) => v.toFixed(3),
    });

    let coefBlock = "";
    if (m.explainability.coefficients.length) {
      const coefs = m.explainability.coefficients.slice(0, 10);
      coefBlock = chartCard({
        title: "Model coefficients",
        sub: "Log-odds weight per encoded column; numeric inputs are standardised, so these are comparable.",
        svg: barChart({
          rows: coefs.map((c) => ({ label: c.label, value: c.coefficient })),
          signed: true,
          valueFormat: (v) => (v > 0 ? "+" : "") + v.toFixed(2),
        }),
        legend: legendItem("Raises the estimate", "--series-2") +
                legendItem("Lowers the estimate", "--series-1"),
        ariaLabel: `Coefficient magnitudes. Largest: ${coefs[0].label} at ${coefs[0].coefficient}.`,
      });
    }

    const shap = m.explainability.shap_summary || {};
    const shapBlock = shap.available ? chartCard({
      title: `SHAP importance — ${shap.model}`,
      sub: `Mean absolute TreeSHAP value across ${shap.rows_explained} training records, in log-odds.`,
      svg: barChart({
        rows: shap.by_feature.slice(0, 8).map((s) => ({ label: s.label, value: s.value })),
        valueFormat: (v) => v.toFixed(3),
      }),
      ariaLabel: `SHAP importance for the ${shap.model} model. Highest: ${shap.by_feature[0].label}.`,
    }) : "";

    const err = m.error_analysis;
    const cmMax = Math.max(cm.tn, cm.fp, cm.fn, cm.tp) || 1;
    const shade = (n) => {
      const f = n / cmMax;
      return f > 0.75 ? "--seq-700" : f > 0.45 ? "--seq-500" : f > 0.15 ? "--seq-300" : "--seq-100";
    };
    // Every step of the ramp clears 4.5:1 against both ink tokens, so the text
    // colour is constant - no white-on-mid-blue, which failed contrast.
    const cell = (n, key, desc) => `
      <div class="cm-cell" style="background:var(${shade(n)})">
        <span class="n">${n}</span>
        <span class="k">${key}</span>
        <span class="d">${desc}</span>
      </div>`;

    $("#performance-content").innerHTML = `
      <div class="grid grid-4" style="margin-bottom:var(--s-6)">
        ${tiles.map(([k, v, note]) => `<div class="stat">
          <div class="stat-label">${esc(k)}</div>
          <div class="stat-value">${esc(v)}</div>
          <div class="stat-note">${esc(note)}</div>
        </div>`).join("")}
      </div>

      <h3>Model comparison</h3>
      <div class="table-wrap" style="margin-bottom:var(--s-6)">
        <table>
          <caption>
            Cross-validation columns are the mean over ${esc(rows[0].cv_n_fits)} fits
            (${esc(m.split.cv)}) on the training split; test columns come from the
            ${esc(m.split.test_rows)} held-out records at the default 0.50 threshold, so the
            models are compared on equal terms.
          </caption>
          <thead><tr>
            <th scope="col">Model</th>
            <th scope="col">CV ROC-AUC</th><th scope="col">CV PR-AUC</th>
            <th scope="col">Accuracy</th><th scope="col">Precision</th>
            <th scope="col">Sensitivity</th><th scope="col">Specificity</th>
            <th scope="col">F1</th><th scope="col">ROC-AUC</th><th scope="col">PR-AUC</th>
          </tr></thead>
          <tbody>
            ${rows.map((r) => `<tr class="${r.model === m.final_model ? "is-final" : ""}">
              <th scope="row">${esc(r.model)}${r.model === m.final_model ? '<span class="pill">Final</span>' : ""}</th>
              <td>${num(r.cv_roc_auc_mean, 3)} ± ${num(r.cv_roc_auc_std, 3)}</td>
              <td>${num(r.cv_pr_auc_mean, 3)} ± ${num(r.cv_pr_auc_std, 3)}</td>
              <td>${num(r.test_accuracy, 3)}</td><td>${num(r.test_precision, 3)}</td>
              <td>${num(r.test_sensitivity, 3)}</td><td>${num(r.test_specificity, 3)}</td>
              <td>${num(r.test_f1, 3)}</td><td>${num(r.test_roc_auc, 3)}</td>
              <td>${num(r.test_pr_auc, 3)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>

      <h3>Discrimination</h3>
      <div class="grid grid-2" style="margin-bottom:var(--s-6)">
        ${chartCard({
          title: "ROC curve", sub: `${m.final_model} on the held-out test set — AUC ${num(chosen.roc_auc, 3)}.`,
          svg: rocSvg, ariaLabel: `ROC curve for ${m.final_model} on the held-out test set, area under the curve ${num(chosen.roc_auc, 3)}, compared with a chance diagonal at 0.5.`,
          legend: legendItem(m.final_model, "--series-1") + legendItem("Chance (0.500)", "", true),
        })}
        ${chartCard({
          title: "Precision-recall curve", sub: `Average precision ${num(chosen.pr_auc, 3)} against a ${pct(m.curves.positive_rate, 0)} base rate.`,
          svg: prSvg, ariaLabel: `Precision-recall curve, average precision ${num(chosen.pr_auc, 3)}, against a base rate of ${pct(m.curves.positive_rate, 0)}.`,
          legend: legendItem(m.final_model, "--series-2") + legendItem("Base rate", "", true),
        })}
      </div>

      <h3>Confusion matrix</h3>
      <div class="grid grid-2" style="margin-bottom:var(--s-6)">
        <div class="chart-card">
          <h4>Test-set outcomes at the ${pct(chosen.threshold, 0)} threshold</h4>
          <p class="sub">${esc(m.split.test_rows)} records the model never saw.</p>
          <div class="cm-grid">
            <div></div>
            <div class="cm-head">Predicted negative</div>
            <div class="cm-head">Predicted positive</div>
            <div class="cm-side">Actually<br>negative</div>
            ${cell(cm.tn, "True negative", "correctly cleared")}
            ${cell(cm.fp, "False positive", "flagged in error")}
            <div class="cm-side">Actually<br>positive</div>
            ${cell(cm.fn, "False negative", "missed case")}
            ${cell(cm.tp, "True positive", "correctly flagged")}
          </div>
        </div>
        <div class="card">
          <h3>Reading it</h3>
          <p style="color:var(--ink-2)">
            Of ${esc(cm.tp + cm.fn)} records that were actually positive, the model flagged
            ${esc(cm.tp)} and missed ${esc(cm.fn)}. Of ${esc(cm.tn + cm.fp)} that were actually
            negative, it cleared ${esc(cm.tn)} and flagged ${esc(cm.fp)} in error.
          </p>
          <p style="color:var(--ink-2);margin-bottom:0">${esc(err.cost_note)}</p>
        </div>
      </div>

      <h3>Threshold and calibration</h3>
      <div class="grid grid-2" style="margin-bottom:var(--s-6)">
        ${chartCard({
          title: "Threshold sweep",
          sub: `On cross-validated training predictions. Marker: the selected ${num(ta.chosen_threshold, 2)} threshold.`,
          svg: sweepSvg,
          ariaLabel: `Sensitivity and specificity across decision thresholds. The selected threshold ${num(ta.chosen_threshold, 2)} gives sensitivity ${num(ta.chosen_metrics.sensitivity, 3)} and specificity ${num(ta.chosen_metrics.specificity, 3)} on training folds.`,
          legend: legendItem("Sensitivity", "--series-2") + legendItem("Specificity", "--series-1") + legendItem("Selected threshold", "", true),
        })}
        ${chartCard({
          title: "Calibration curve",
          sub: `Brier ${num(m.calibration.raw.brier, 3)}, expected calibration error ${num(m.calibration.raw.expected_calibration_error, 3)}.`,
          svg: calSvg,
          ariaLabel: `Reliability diagram of cross-validated training predictions against the diagonal of perfect calibration. Brier score ${num(m.calibration.raw.brier, 3)}.`,
          legend: legendItem("Observed vs estimated", "--series-1") + legendItem("Perfect calibration", "", true),
        })}
      </div>

      <div class="table-wrap" style="margin-bottom:var(--s-6)">
        <table>
          <caption>
            What moving the threshold costs and buys, measured on cross-validated training
            predictions. The held-out test set was never used to choose it.
          </caption>
          <thead><tr>
            <th scope="col">Operating point</th><th scope="col">Threshold</th>
            <th scope="col">Sensitivity</th><th scope="col">Specificity</th>
            <th scope="col">Precision</th><th scope="col">F1</th>
          </tr></thead>
          <tbody>
            <tr><th scope="row">Default</th><td>0.50</td>
              <td>${num(ta.default_metrics.sensitivity, 3)}</td><td>${num(ta.default_metrics.specificity, 3)}</td>
              <td>${num(ta.default_metrics.precision, 3)}</td><td>${num(ta.default_metrics.f1, 3)}</td></tr>
            <tr><th scope="row">Best F1</th><td>${num(ta.best_f1_threshold, 2)}</td>
              <td>${num(ta.best_f1_metrics.sensitivity, 3)}</td><td>${num(ta.best_f1_metrics.specificity, 3)}</td>
              <td>${num(ta.best_f1_metrics.precision, 3)}</td><td>${num(ta.best_f1_metrics.f1, 3)}</td></tr>
            <tr class="is-final"><th scope="row">Selected — highest specificity at ≥${pct(ta.min_sensitivity_target, 0)} sensitivity<span class="pill">In use</span></th>
              <td>${num(ta.chosen_threshold, 2)}</td>
              <td>${num(ta.chosen_metrics.sensitivity, 3)}</td><td>${num(ta.chosen_metrics.specificity, 3)}</td>
              <td>${num(ta.chosen_metrics.precision, 3)}</td><td>${num(ta.chosen_metrics.f1, 3)}</td></tr>
          </tbody>
        </table>
      </div>

      <h3>What the model relies on</h3>
      <div class="grid grid-2" style="margin-bottom:var(--s-6)">
        ${chartCard({
          title: "Permutation importance",
          sub: "Drop in test ROC-AUC when one input column is shuffled. Model-agnostic.",
          svg: permSvg,
          ariaLabel: `Permutation importance. Largest drop: ${perm[0].label} at ${num(perm[0].mean_drop_in_roc_auc, 3)} ROC-AUC.`,
        })}
        ${coefBlock || shapBlock}
      </div>
      ${coefBlock && shapBlock ? `<div class="grid grid-2" style="margin-bottom:var(--s-6)">
        ${shapBlock}
        <div class="card">
          <h3>Two views, one dataset</h3>
          <p style="color:var(--ink-2)">
            The coefficients describe the model that is actually deployed here. The SHAP panel
            describes ${esc(shap.model)}, the strongest tree model in the comparison, and is
            included because gradient-boosted trees have no readable coefficients of their own.
          </p>
          <p style="color:var(--ink-2);margin-bottom:0">${esc(shap.note)}</p>
        </div>
      </div>` : ""}
      <p style="color:var(--ink-2);max-width:70ch">
        ${esc(m.explainability.permutation_note)} These rankings describe the model's own
        behaviour. They are not evidence that a feature causes disease.
      </p>

      <h3>Error analysis</h3>
      <div class="grid grid-2 grid-top" style="margin-bottom:var(--s-5)">
        <div class="card">
          <h3>Patterns in the mistakes</h3>
          <ul class="checklist">${err.patterns.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
        </div>
        <div class="card">
          <h3>Individual misclassified records</h3>
          <p style="color:var(--ink-2);font-size:var(--fs-md)">
            ${esc(err.counts.false_negative || 0)} missed positives and
            ${esc(err.counts.false_positive || 0)} false alarms out of ${esc(m.split.test_rows)} test records.
          </p>
          <div class="table-wrap">
            <table>
              <thead><tr><th scope="col">Type</th><th scope="col">Actual</th><th scope="col">Estimated</th><th scope="col">Strongest influence</th></tr></thead>
              <tbody>
                ${err.cases.slice(0, 6).map((c) => `<tr>
                  <th scope="row" style="white-space:nowrap">${esc(c.type.replace("_", " "))}</th>
                  <td>${esc(c.true_label)}</td>
                  <td>${pct(c.model_probability)}</td>
                  <td style="text-align:left;white-space:normal">${esc(c.top_features[0].label)} (${esc(c.top_features[0].value)})</td>
                </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <details class="tech">
        <summary>Class imbalance and early-stopping investigations</summary>
        <div class="tech-body">
          <p style="color:var(--ink-2)">${esc(m.imbalance_study.decision)}</p>
          <div class="table-wrap" style="margin-bottom:var(--s-4)">
            <table>
              <caption>Logistic-regression baseline under three sampling strategies, 5-fold CV on the training split. SMOTE was applied inside each training fold only.</caption>
              <thead><tr><th scope="col">Strategy</th><th scope="col">ROC-AUC</th><th scope="col">Recall</th><th scope="col">F1</th></tr></thead>
              <tbody>
                ${Object.entries(m.imbalance_study.comparison).map(([k, v]) => `<tr>
                  <th scope="row">${esc(k.replace(/_/g, " "))}</th>
                  <td>${num(v.roc_auc.mean, 4)} ± ${num(v.roc_auc.std, 4)}</td>
                  <td>${num(v.recall.mean, 4)} ± ${num(v.recall.std, 4)}</td>
                  <td>${num(v.f1.mean, 4)} ± ${num(v.f1.std, 4)}</td>
                </tr>`).join("")}
              </tbody>
            </table>
          </div>
          ${m.early_stopping.available ? `<dl class="kv">
            <dt>XGBoost early stopping</dt>
            <dd>Best iteration ${esc(m.early_stopping.best_iteration)} of ${esc(m.early_stopping.max_rounds_offered)} offered,
                validation log-loss ${num(m.early_stopping.best_validation_logloss, 4)},
                patience ${esc(m.early_stopping.early_stopping_rounds)} rounds
                (inner split: ${esc(m.early_stopping.inner_split.fit_rows)} fit / ${esc(m.early_stopping.inner_split.validation_rows)} validation).</dd>
            <dt>Why it matters</dt><dd>${esc(m.early_stopping.note)}</dd>
          </dl>` : `<p>XGBoost was unavailable in this environment: ${esc(m.early_stopping.error || "unknown")}</p>`}
        </div>
      </details>`;
  }

  /* ================================================== SMALL TEXT SECTIONS */
  function renderNarrative() {
    const m = state.metrics, meta = state.model.metadata, ta = m.threshold_analysis;
    const chosen = m.test_chosen;

    $("#hero-stats").innerHTML = [
      ["Records", String(m.dataset.rows), `${m.dataset.class_counts["1"]} positive · ${m.dataset.class_counts["0"]} negative`],
      ["Clinical inputs", String(m.dataset.features), "Symptoms, vitals, blood tests, ECG, imaging"],
      ["Models compared", String(m.comparison_table.length), "Logistic regression · linear and RBF SVM · random forest · XGBoost"],
      ["Test ROC-AUC", num(chosen.roc_auc, 3), `${state.model.card.name}, ${m.split.test_rows} held-out records`],
    ].map(([k, v, n]) => `<div class="stat">
      <div class="stat-label">${esc(k)}</div>
      <div class="stat-value">${esc(v)}</div>
      <div class="stat-note">${esc(n)}</div>
    </div>`).join("");

    $("#how-threshold").textContent =
      `A classifier outputs a probability; something has to turn it into a yes or a no. ` +
      `Leaving that cut at 0.50 is a choice, not a default. Measured on cross-validated ` +
      `training predictions, 0.50 gives sensitivity ${num(ta.default_metrics.sensitivity, 3)}; ` +
      `moving it to ${num(ta.chosen_threshold, 2)} raises sensitivity to ` +
      `${num(ta.chosen_metrics.sensitivity, 3)} and lowers specificity to ` +
      `${num(ta.chosen_metrics.specificity, 3)}. The held-out test set played no part in that choice.`;

    $("#how-errors").textContent = m.error_analysis.cost_note;

    $("#threshold-hint").textContent =
      `Model in use: ${state.model.card.name}, threshold ${num(state.model.card.threshold, 2)}.`;

    $("#about-dataset").innerHTML = `
      <dl class="kv">
        <dt>Name</dt><dd>${esc(m.dataset.name)}</dd>
        <dt>Source</dt><dd><a href="${esc(m.dataset.page)}" rel="noopener noreferrer" target="_blank">UCI Machine Learning Repository, dataset 45</a></dd>
        <dt>Records</dt><dd>${esc(m.dataset.rows)}</dd>
        <dt>Features</dt><dd>${esc(m.dataset.features)} clinical inputs</dd>
        <dt>Classes</dt><dd>2 — ${esc(m.dataset.class_counts["0"])} negative, ${esc(m.dataset.class_counts["1"])} positive</dd>
        <dt>Label</dt><dd>${esc(m.dataset.target_description)}</dd>
      </dl>`;

    $("#about-tech").innerHTML = `
      <dl class="kv">
        <dt>Modelling</dt><dd>scikit-learn ${esc(m.environment.scikit_learn)}, XGBoost ${esc(m.environment.xgboost || "n/a")}, SHAP, imbalanced-learn</dd>
        <dt>Backend</dt><dd>FastAPI on Uvicorn; the model is deserialised once at start-up</dd>
        <dt>Frontend</dt><dd>Semantic HTML, CSS custom properties, vanilla JavaScript, hand-drawn inline SVG charts — no UI framework and no chart library</dd>
        <dt>Runtime</dt><dd>Python ${esc(m.environment.python)}</dd>
      </dl>`;

    $("#tech-body").innerHTML = `
      <dl class="kv">
        <dt>Model</dt><dd>${esc(state.model.card.name)} (<code>${esc(state.model.card.estimator)}</code>)</dd>
        <dt>Dataset</dt><dd>${esc(state.model.card.dataset)}, ${esc(state.model.card.dataset_rows)} records</dd>
        <dt>Features</dt><dd>${esc(state.model.card.input_features)} inputs → ${esc(state.model.card.encoded_features)} encoded columns</dd>
        <dt>Classes</dt><dd>2 (Negative / Positive)</dd>
        <dt>Preprocessing</dt><dd>Median / mode imputation, standardisation, one-hot encoding — all fitted on training data only</dd>
        <dt>Threshold</dt><dd>${esc(state.model.card.threshold)}</dd>
        <dt>Calibration</dt><dd>${esc(state.model.card.calibration)}</dd>
        <dt>Explanation</dt><dd><code>${esc(state.model.card.explanation_method)}</code></dd>
        <dt>Test ROC-AUC</dt><dd>${num(state.model.card.test_roc_auc, 4)}</dd>
        <dt>Test PR-AUC</dt><dd>${num(state.model.card.test_pr_auc, 4)}</dd>
        <dt>Test accuracy</dt><dd>${num(state.model.card.test_accuracy, 4)}</dd>
        <dt>Test sensitivity</dt><dd>${num(state.model.card.test_sensitivity, 4)}</dd>
        <dt>Test specificity</dt><dd>${num(state.model.card.test_specificity, 4)}</dd>
      </dl>`;

    $("#footer-meta").textContent =
      `${state.model.card.name} · trained ${meta.generated_utc.slice(0, 10)} · seed ${m.split.random_seed}`;
  }

  function renderFatal(message) {
    ["#model-content", "#performance-content"].forEach((sel) => {
      $(sel).innerHTML = `<div class="alert" role="alert" style="margin:0">
        ${icon("i-alert")}
        <div><h4>Model artefacts unavailable</h4><p style="margin:0">${esc(message)}</p>
        <p style="margin:8px 0 0">Run <code>python -m src.train</code>, then restart the server.</p></div>
      </div>`;
    });
    $("#hero-stats").innerHTML =
      `<p style="grid-column:1/-1;color:var(--ink-2);margin:0">
         Model statistics are unavailable while the service cannot be reached.
       </p>`;
    $("#tech-body").innerHTML =
      `<p style="margin:0;color:var(--ink-2)">${esc(message)}</p>`;
    renderResultError(message, "model_unavailable");
  }

  /* ================================================================ BOOT */
  async function boot() {
    initTheme();
    initNav();

    try {
      state.schema = await api("/api/schema");
      buildForm(state.schema);
      $("#predict-form").addEventListener("submit", onSubmit);
      $("#reset-btn").addEventListener("click", onReset);
      $("#load-example").addEventListener("click", loadExample);
    } catch (err) {
      $("#fieldsets").innerHTML =
        `<div class="alert" role="alert" style="margin:0">${icon("i-alert")}
         <div><h4>The form could not be loaded</h4><p style="margin:0">${esc(err.message)}</p></div></div>`;
    }

    try {
      const [model, metrics] = await Promise.all([api("/api/model"), api("/api/metrics")]);
      state.model = model;
      state.metrics = metrics;
      renderNarrative();
      if (state.schema) {
        renderModelSection();
        renderPerformanceSection();
      }
    } catch (err) {
      renderFatal(err.message);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
