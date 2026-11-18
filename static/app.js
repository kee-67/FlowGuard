/**
 * FlowGuard Penang: Frontend Controller
 * Connects the 5-view civic dashboard to server.py JSON endpoints.
 */

(function () {
  "use strict";

  const state = {
    samples: [],
    zones: {},
    hazards: [],
    metrics: null,
    qaLog: [],
    sourceStages: {},
    selectedSampleId: "clear_gurney",
    uploadedImageBase64: null,
    uploadedFilename: null,
    applyShadowFix: true,
    latestInference: null,
    activeZoneFilter: "ALL",
    activeStatusFilter: "ALL",
    activeStageKey: "frontend",
    activeSlideIndex: 0,
    timerSecondsLeft: 480,
    timerIntervalId: null,
  };

  // DOM helpers
  const $ = (id) => document.getElementById(id);

  function setHtmlText(el, text) {
    if (el) el.textContent = text;
  }

  // ---------------------------------------------------------------------------
  // Theme Persistence (Light / Dark Mode)
  // ---------------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem("flowguard_theme");
    const theme = saved === "dark" ? "dark" : "light";
    applyTheme(theme);

    $("theme-toggle").addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      applyTheme(next);
      localStorage.setItem("flowguard_theme", next);
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const label = theme === "dark" ? "Light Mode" : "Dark Mode";
    const aria = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
    setHtmlText($("theme-label"), label);
    $("theme-toggle").setAttribute("aria-label", aria);
  }

  // ---------------------------------------------------------------------------
  // Navigation Tabs & Mobile Drawer
  // ---------------------------------------------------------------------------
  function initNavigation() {
    const mobileBtn = $("mobile-menu-btn");
    const nav = $("primary-nav");
    const tabs = Array.from(nav.querySelectorAll(".nav-tab"));
    const panels = Array.from(document.querySelectorAll(".view-panel"));

    function activateView(viewName) {
      tabs.forEach((tab) => {
        const match = tab.dataset.view === viewName;
        tab.setAttribute("aria-selected", match ? "true" : "false");
      });
      panels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === "view-" + viewName);
      });
      if (nav.classList.contains("open")) {
        nav.classList.remove("open");
        mobileBtn.setAttribute("aria-expanded", "false");
      }
      if (viewName === "dispatch") {
        fetchHazards();
      }
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => activateView(tab.dataset.view));
    });

    mobileBtn.addEventListener("click", () => {
      const isOpen = nav.classList.toggle("open");
      mobileBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });

    document.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape" && nav.classList.contains("open")) {
        nav.classList.remove("open");
        mobileBtn.setAttribute("aria-expanded", "false");
        mobileBtn.focus();
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Initial Bootstrap Fetches
  // ---------------------------------------------------------------------------
  async function loadSamplesAndZones() {
    try {
      const resp = await fetch("/api/samples");
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || "Failed to load samples");

      state.samples = data.samples || [];
      state.zones = data.zones || {};

      populateZoneSelects();
      renderSampleGrid();
    } catch (err) {
      showInspectError("Unable to load Penang corridor samples: " + err.message);
    }
  }

  async function loadMetricsAndSource() {
    try {
      const resp = await fetch("/api/metrics");
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || "Failed to load metrics");

      state.metrics = data.metrics || {};
      state.qaLog = data.qa_log || [];
      state.sourceStages = data.source_stages || {};

      setHtmlText($("strip-engine"), data.backend_mode || "FlowGuard Engine");
      setHtmlText(
        $("strip-acc"),
        state.metrics.week10_validation_accuracy != null ? state.metrics.week10_validation_accuracy + "%" : "N/A"
      );
      setHtmlText(
        $("strip-lat"),
        state.metrics.avg_latency_ms != null ? state.metrics.avg_latency_ms + " ms" : "N/A"
      );

      renderQADashboard();
      renderArchitectureStage(state.activeStageKey);
      renderCurrentSlide();
    } catch (err) {
      setHtmlText($("strip-engine"), "Connection Error");
    }
  }

  // ---------------------------------------------------------------------------
  // View 1: Inspect Studio Controls & Inference
  // ---------------------------------------------------------------------------
  function populateZoneSelects() {
    const zoneSelect = $("zone-select");
    const filterZone = $("filter-zone");
    zoneSelect.innerHTML = "";
    filterZone.innerHTML = '<option value="ALL">All Corridors</option>';

    Object.keys(state.zones).forEach((zoneName) => {
      const opt1 = document.createElement("option");
      opt1.value = zoneName;
      opt1.textContent = zoneName;
      zoneSelect.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = zoneName;
      opt2.textContent = zoneName;
      filterZone.appendChild(opt2);
    });

    updateZoneDetail();
    zoneSelect.addEventListener("change", updateZoneDetail);
  }

  function updateZoneDetail() {
    const z = $("zone-select").value;
    const meta = state.zones[z];
    if (meta) {
      setHtmlText($("zone-detail"), meta.corridor + " | " + meta.risk_tier + " | " + meta.drain_type);
    }
  }

  function renderSampleGrid() {
    const grid = $("sample-grid");
    grid.innerHTML = "";

    state.samples.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-btn" + (state.selectedSampleId === item.id && !state.uploadedImageBase64 ? " selected" : "");
      btn.dataset.sampleId = item.id;
      btn.setAttribute("aria-label", item.title + ", expected " + item.expected);

      const img = document.createElement("img");
      img.className = "sample-thumb";
      img.src = "/sample_drains/" + item.filename;
      img.alt = item.title;

      const meta = document.createElement("div");
      meta.className = "sample-meta";

      const titleSpan = document.createElement("span");
      titleSpan.className = "sample-name";
      titleSpan.textContent = item.title;

      const subSpan = document.createElement("span");
      subSpan.className = "sample-sub";
      subSpan.textContent = item.zone + " | " + item.expected;

      meta.appendChild(titleSpan);
      meta.appendChild(subSpan);
      btn.appendChild(img);
      btn.appendChild(meta);

      btn.addEventListener("click", () => {
        state.selectedSampleId = item.id;
        state.uploadedImageBase64 = null;
        state.uploadedFilename = null;
        setHtmlText($("upload-filename"), "Selected field sample: " + item.filename);
        $("zone-select").value = item.zone;
        updateZoneDetail();
        updateSampleSelectionUI();
      });

      grid.appendChild(btn);
    });
  }

  function updateSampleSelectionUI() {
    const buttons = Array.from($("sample-grid").querySelectorAll(".sample-btn"));
    buttons.forEach((btn) => {
      const active = !state.uploadedImageBase64 && btn.dataset.sampleId === state.selectedSampleId;
      btn.classList.toggle("selected", active);
    });
  }

  function initInspectControls() {
    const fileInput = $("photo-upload");
    const dropzone = $("upload-dropzone");
    const switchBtn = $("shadow-fix-switch");

    function handleFileSelection(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        state.uploadedImageBase64 = String(reader.result || "");
        state.uploadedFilename = file.name;
        setHtmlText($("upload-filename"), "Ready to inspect upload: " + file.name);
        updateSampleSelectionUI();
      };
      reader.onerror = () => {
        showInspectError("Failed to read selected file from disk. Try another JPG or PNG photo.");
      };
      reader.readAsDataURL(file);
    }

    fileInput.addEventListener("change", (evt) => {
      handleFileSelection(evt.target.files && evt.target.files[0]);
    });

    dropzone.addEventListener("dragover", (evt) => {
      evt.preventDefault();
      dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("drag-over");
    });
    dropzone.addEventListener("drop", (evt) => {
      evt.preventDefault();
      dropzone.classList.remove("drag-over");
      const file = evt.dataTransfer && evt.dataTransfer.files && evt.dataTransfer.files[0];
      handleFileSelection(file);
    });

    switchBtn.addEventListener("click", () => {
      state.applyShadowFix = !state.applyShadowFix;
      switchBtn.setAttribute("aria-checked", state.applyShadowFix ? "true" : "false");
      switchBtn.textContent = state.applyShadowFix ? "ON" : "OFF";
    });

    $("analyze-btn").addEventListener("click", runInspection);
    $("dispatch-btn").addEventListener("click", dispatchLatestToLog);
  }

  function setInspectPanelState(mode) {
    $("inspect-empty").hidden = mode !== "empty";
    $("inspect-loading").hidden = mode !== "loading";
    $("inspect-error").hidden = mode !== "error";
    $("inspect-result").hidden = mode !== "result";
  }

  function showInspectError(message) {
    setHtmlText($("inspect-error-msg"), message);
    setInspectPanelState("error");
  }

  async function runInspection() {
    const analyzeBtn = $("analyze-btn");
    analyzeBtn.disabled = true;
    setHtmlText($("dispatch-feedback"), "");
    setInspectPanelState("loading");

    const payload = {
      zone: $("zone-select").value || "Jalan P. Ramlee",
      apply_shadow_fix: state.applyShadowFix,
    };
    if (state.uploadedImageBase64) {
      payload.image_base64 = state.uploadedImageBase64;
      payload.filename = state.uploadedFilename || "uploaded_drain.jpg";
    } else {
      payload.sample_id = state.selectedSampleId;
    }

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || "Inference failed on the selected photo.");
      }

      state.latestInference = data;
      renderInspectionResult(data);
      updateLiveTensorSnapshot(data);
      setInspectPanelState("result");
    } catch (err) {
      showInspectError(err.message);
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  function severityIconPrefix(label) {
    if (label === "Critical") return "[CRITICAL] ";
    if (label === "Moderate") return "[MODERATE] ";
    return "[CLEAR] ";
  }

  function renderInspectionResult(res) {
    const badge = $("verdict-badge");
    badge.className = "severity-badge severity-" + res.label;
    badge.textContent = severityIconPrefix(res.label) + res.label;

    setHtmlText($("verdict-conf"), res.confidence + "%");
    setHtmlText($("verdict-lat"), res.latency_ms + " ms");
    setHtmlText(
      $("verdict-privacy"),
      res.faces_redacted > 0 ? res.faces_redacted + " face(s) blurred" : "0 faces detected"
    );

    $("img-preprocessed").src = res.preprocessed_b64;
    $("img-heatmap").src = res.heatmap_b64;
    setHtmlText($("shadow-mode-tag"), res.shadow_fix_applied ? "LAB CLAHE ON" : "RAW PRE-FIX");

    ["Clear", "Moderate", "Critical"].forEach((clsName) => {
      const pct = (res.probabilities && res.probabilities[clsName]) || 0;
      $("prob-bar-" + clsName).style.width = Math.max(0, Math.min(100, pct)) + "%";
      setHtmlText($("prob-val-" + clsName), pct + "%");
    });

    const tel = res.telemetry || {};
    setHtmlText($("tel-texture"), (tel.debris_texture_score ?? "--") + " / 100");
    setHtmlText($("tel-grate"), (tel.grate_visibility_score ?? "--") + " / 100");
    setHtmlText($("tel-shadow"), (tel.shadow_index ?? "--") + "%");
    setHtmlText($("tel-chroma"), (tel.chromatic_trash_pct ?? "--") + "%");

    setHtmlText($("recommended-action"), res.recommended_action);
  }

  async function dispatchLatestToLog() {
    if (!state.latestInference) return;
    const btn = $("dispatch-btn");
    btn.disabled = true;

    const r = state.latestInference;
    const notesVal = $("field-notes").value.trim();

    try {
      const resp = await fetch("/api/hazards", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zone: r.zone,
          severity: r.label,
          confidence: r.confidence,
          latency_ms: r.latency_ms,
          source_photo: r.source_photo || "field_photo.jpg",
          notes: notesVal || r.recommended_action,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || "Could not log hazard record.");
      }

      state.hazards = data.hazards || [];
      setHtmlText(
        $("dispatch-feedback"),
        "Logged ticket " + data.record.id + " (" + data.record.zone + ": " + data.record.severity + ") to MBPP Dispatch Map."
      );
      renderHazardsView();
    } catch (err) {
      setHtmlText($("dispatch-feedback"), "Dispatch failed: " + err.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // View 2: MBPP Municipal Dispatch Map & Table
  // ---------------------------------------------------------------------------
  function initDispatchControls() {
    $("filter-zone").addEventListener("change", (evt) => {
      state.activeZoneFilter = evt.target.value;
      renderHazardsView();
    });

    $("filter-status").addEventListener("change", (evt) => {
      state.activeStatusFilter = evt.target.value;
      renderHazardsView();
    });

    $("clear-map-filter").addEventListener("click", () => {
      state.activeZoneFilter = "ALL";
      state.activeStatusFilter = "ALL";
      $("filter-zone").value = "ALL";
      $("filter-status").value = "ALL";
      renderHazardsView();
    });

    $("export-csv-btn").addEventListener("click", exportHazardsCsv);
    fetchHazards();
  }

  async function fetchHazards() {
    $("hazards-loading").hidden = false;
    $("hazards-error").hidden = true;
    $("hazards-empty").hidden = true;

    try {
      const resp = await fetch("/api/hazards");
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "Failed to fetch hazard log.");

      state.hazards = data.hazards || [];
      if (data.zones) state.zones = data.zones;
      $("hazards-loading").hidden = true;
      renderHazardsView();
    } catch (err) {
      $("hazards-loading").hidden = true;
      $("hazards-error").hidden = false;
      setHtmlText($("hazards-error-msg"), err.message);
    }
  }

  function worstSeverityForZone(zoneName) {
    const active = state.hazards.filter((h) => h.zone === zoneName && h.status !== "Cleared");
    if (active.some((h) => h.severity === "Critical")) return "Critical";
    if (active.some((h) => h.severity === "Moderate")) return "Moderate";
    return "Clear";
  }

  function renderPenangMapPins() {
    const layer = $("map-pins-layer");
    if (!layer) return;
    layer.innerHTML = "";

    const svgNS = "http://www.w3.org/2000/svg";
    Object.entries(state.zones).forEach(([zoneName, meta]) => {
      const sev = worstSeverityForZone(zoneName);
      const cx = (meta.map_x / 100) * 420;
      const cy = (meta.map_y / 100) * 460;
      const isSelected = state.activeZoneFilter === zoneName;

      const fillMap = {
        Critical: "#991B1B",
        Moderate: "#B6450D",
        Clear: "#166534",
      };
      const codeMap = {
        Critical: "C",
        Moderate: "M",
        Clear: "OK",
      };

      const g = document.createElementNS(svgNS, "g");
      g.setAttribute("class", "map-pin-btn");
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", zoneName + " corridor status: " + sev + ". Activate to filter table.");

      const ring = document.createElementNS(svgNS, "circle");
      ring.setAttribute("class", "pin-ring");
      ring.setAttribute("cx", String(cx));
      ring.setAttribute("cy", String(cy));
      ring.setAttribute("r", isSelected ? "19" : "15");
      ring.setAttribute("fill", fillMap[sev]);
      ring.setAttribute("stroke", isSelected ? "#FFFFFF" : "#182229");
      ring.setAttribute("stroke-width", isSelected ? "3" : "1.5");

      const codeText = document.createElementNS(svgNS, "text");
      codeText.setAttribute("x", String(cx));
      codeText.setAttribute("y", String(cy + 4));
      codeText.setAttribute("text-anchor", "middle");
      codeText.setAttribute("fill", "#FFFFFF");
      codeText.setAttribute("font-size", "10");
      codeText.setAttribute("font-weight", "700");
      codeText.setAttribute("font-family", "var(--font-mono)");
      codeText.textContent = codeMap[sev];

      const labelBg = document.createElementNS(svgNS, "rect");
      const labelW = zoneName.length * 6.2 + 12;
      labelBg.setAttribute("x", String(cx - labelW / 2));
      labelBg.setAttribute("y", String(cy + 19));
      labelBg.setAttribute("width", String(labelW));
      labelBg.setAttribute("height", "16");
      labelBg.setAttribute("rx", "3");
      labelBg.setAttribute("fill", "var(--surface)");
      labelBg.setAttribute("stroke", "var(--border-subtle)");

      const labelText = document.createElementNS(svgNS, "text");
      labelText.setAttribute("x", String(cx));
      labelText.setAttribute("y", String(cy + 30));
      labelText.setAttribute("text-anchor", "middle");
      labelText.setAttribute("fill", "var(--ink)");
      labelText.setAttribute("font-size", "9.5");
      labelText.setAttribute("font-weight", "700");
      labelText.textContent = zoneName;

      g.appendChild(ring);
      g.appendChild(codeText);
      g.appendChild(labelBg);
      g.appendChild(labelText);

      const selectPin = () => {
        state.activeZoneFilter = state.activeZoneFilter === zoneName ? "ALL" : zoneName;
        $("filter-zone").value = state.activeZoneFilter;
        renderHazardsView();
      };
      g.addEventListener("click", selectPin);
      g.addEventListener("keydown", (evt) => {
        if (evt.key === "Enter" || evt.key === " ") {
          evt.preventDefault();
          selectPin();
        }
      });

      layer.appendChild(g);
    });
  }

  function renderHazardsView() {
    renderPenangMapPins();
    const tbody = $("hazards-tbody");
    tbody.innerHTML = "";

    const filtered = state.hazards.filter((row) => {
      const zoneMatch = state.activeZoneFilter === "ALL" || row.zone === state.activeZoneFilter;
      const statusMatch = state.activeStatusFilter === "ALL" || row.status === state.activeStatusFilter;
      return zoneMatch && statusMatch;
    });

    if (filtered.length === 0) {
      $("hazards-empty").hidden = false;
      $("hazards-table-wrap").hidden = true;
      return;
    }

    $("hazards-empty").hidden = true;
    $("hazards-table-wrap").hidden = false;

    filtered.forEach((row) => {
      const tr = document.createElement("tr");

      const tdId = document.createElement("td");
      tdId.innerHTML = '<strong class="mono"></strong><div class="help-text"></div>';
      tdId.querySelector("strong").textContent = row.id;
      tdId.querySelector(".help-text").textContent = row.timestamp;

      const tdZone = document.createElement("td");
      tdZone.innerHTML = "<strong></strong><div class=\"help-text\"></div>";
      tdZone.querySelector("strong").textContent = row.zone;
      tdZone.querySelector(".help-text").textContent = row.source_photo || "";

      const tdSev = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "severity-badge severity-" + row.severity;
      badge.textContent = row.severity;
      tdSev.appendChild(badge);

      const tdSpeed = document.createElement("td");
      tdSpeed.className = "mono";
      tdSpeed.textContent = row.confidence + "% | " + row.latency_ms + " ms";

      const tdStatus = document.createElement("td");
      const sel = document.createElement("select");
      sel.className = "status-select";
      sel.setAttribute("aria-label", "Update status for " + row.id);
      ["Pending Dispatch", "Crew Dispatched", "Cleared"].forEach((optVal) => {
        const opt = document.createElement("option");
        opt.value = optVal;
        opt.textContent = optVal;
        if (row.status === optVal) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", () => updateHazardStatus(row.id, sel.value));
      tdStatus.appendChild(sel);

      const tdNotes = document.createElement("td");
      tdNotes.textContent = row.notes || "";

      tr.appendChild(tdId);
      tr.appendChild(tdZone);
      tr.appendChild(tdSev);
      tr.appendChild(tdSpeed);
      tr.appendChild(tdStatus);
      tr.appendChild(tdNotes);
      tbody.appendChild(tr);
    });
  }

  async function updateHazardStatus(hazardId, newStatus) {
    try {
      const resp = await fetch("/api/hazards/" + encodeURIComponent(hazardId), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "Update failed");
      state.hazards = data.hazards || [];
      renderHazardsView();
    } catch (err) {
      $("hazards-error").hidden = false;
      setHtmlText($("hazards-error-msg"), "Could not update ticket " + hazardId + ": " + err.message);
    }
  }

  function exportHazardsCsv() {
    const headers = ["id", "timestamp", "zone", "severity", "confidence", "latency_ms", "status", "source_photo", "notes"];
    const escapeCell = (v) => '"' + String(v ?? "").replace(/"/g, '""') + '"';
    const lines = [headers.join(",")];
    state.hazards.forEach((row) => {
      lines.push(headers.map((h) => escapeCell(row[h])).join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "flowguard_penang_hazard_log.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // ---------------------------------------------------------------------------
  // View 3: Anti-Black-Box Architecture Inspector
  // ---------------------------------------------------------------------------
  function initArchitectureInspector() {
    const nodes = Array.from(document.querySelectorAll(".pipe-node"));
    nodes.forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeStageKey = btn.dataset.stage;
        nodes.forEach((n) => n.classList.toggle("active", n === btn));
        renderArchitectureStage(state.activeStageKey);
      });
    });
  }

  function renderArchitectureStage(stageKey) {
    const info = state.sourceStages[stageKey];
    if (!info) return;
    setHtmlText($("stage-title"), info.title);
    setHtmlText($("stage-file"), info.file);
    setHtmlText($("stage-symbol"), info.symbol);
    setHtmlText($("stage-beginner"), info.beginner_note);
    setHtmlText($("stage-expert"), info.expert_note);
    setHtmlText($("stage-code"), info.code);
  }

  function updateLiveTensorSnapshot(res) {
    const ts = res.tensor_stats || {};
    const text =
      "Source: " + (res.source_photo || "sample") +
      " | Orig: " + (ts.original_resolution || "?") +
      " -> Tensor: " + (ts.tensor_shape || "1x224x224x3") +
      " (" + (ts.dtype || "float32") + ")" +
      " | Range: [" + (ts.value_range ? ts.value_range.join(", ") : "-1, 1") + "]" +
      " | Verdict: " + res.label + " (" + res.confidence + "% in " + res.latency_ms + " ms)";
    setHtmlText($("live-tensor-snapshot"), text);
  }

  // ---------------------------------------------------------------------------
  // View 4: Week 10 QA Debug Log & Live Benchmark
  // ---------------------------------------------------------------------------
  function initQAView() {
    $("run-bench-btn").addEventListener("click", runLiveBenchmark);
  }

  function renderQADashboard() {
    const m = state.metrics || {};
    setHtmlText($("qa-pre-acc"), (m.pre_week10_accuracy ?? "--") + "%");
    setHtmlText(
      $("qa-pre-shadow"),
      "Harsh shadow false alarms: " + (m.pre_week10_shadow_false_alarms ?? "--") + " / " + (m.clear_shadow_subset_count ?? "--")
    );
    setHtmlText($("qa-wk10-acc"), (m.week10_validation_accuracy ?? "--") + "%");
    setHtmlText(
      $("qa-wk10-shadow"),
      "Harsh shadow false alarms: " + (m.week10_shadow_false_alarms ?? "--") + " / " + (m.clear_shadow_subset_count ?? "--")
    );
    setHtmlText($("qa-avg-lat"), (m.avg_latency_ms ?? "--") + " ms");
    setHtmlText(
      $("qa-lat-range"),
      "Min / Max across samples: " + (m.min_latency_ms ?? "--") + " ms / " + (m.max_latency_ms ?? "--") + " ms"
    );

    const cm = m.week10_confusion_matrix || [];
    const classes = m.classes || ["Clear", "Moderate", "Critical"];
    const cmBody = $("cm-tbody");
    cmBody.innerHTML = "";
    cm.forEach((rowVals, rIdx) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = classes[rIdx] || "Class " + rIdx;
      tr.appendChild(th);
      rowVals.forEach((val, cIdx) => {
        const td = document.createElement("td");
        td.textContent = String(val);
        if (rIdx === cIdx) td.className = "cm-diag";
        tr.appendChild(td);
      });
      cmBody.appendChild(tr);
    });

    const qaBody = $("qa-log-tbody");
    qaBody.innerHTML = "";
    state.qaLog.forEach((item) => {
      const tr = document.createElement("tr");
      ["bug_id", "module", "symptom", "resolution", "status"].forEach((col) => {
        const td = document.createElement("td");
        if (col === "bug_id") td.className = "mono";
        td.textContent = item[col] || "";
        tr.appendChild(td);
      });
      qaBody.appendChild(tr);
    });
  }

  async function runLiveBenchmark() {
    const btn = $("run-bench-btn");
    btn.disabled = true;
    btn.textContent = "Running 12-Pass Benchmark...";
    try {
      const resp = await fetch("/api/benchmark", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "Benchmark failed");
      $("bench-live-banner").hidden = false;
      setHtmlText(
        $("bench-summary-text"),
        "Executed " +
          data.runs_count +
          " live inferences across all 6 Penang samples | Mean: " +
          data.mean_ms +
          " ms | Min: " +
          data.min_ms +
          " ms | Max: " +
          data.max_ms +
          " ms"
      );
    } catch (err) {
      $("bench-live-banner").hidden = false;
      setHtmlText($("bench-summary-text"), "Benchmark error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run 12-Pass Live Latency Benchmark";
    }
  }

  // ---------------------------------------------------------------------------
  // View 5: 6-Slide Pitch Deck & 8-Minute Viva-Voce Stage Timer
  // ---------------------------------------------------------------------------
  function getPitchSlides() {
    const acc = state.metrics && state.metrics.week10_validation_accuracy != null ? state.metrics.week10_validation_accuracy : "99.2";
    const lat = state.metrics && state.metrics.avg_latency_ms != null ? state.metrics.avg_latency_ms : "20.3";

    return [
      {
        title: "Slide 1: FlowGuard Penang",
        subtitle: "Proactive Storm Drain Blockage Detection Before the Rain Falls",
        speaker: "Speaker: [Name 5 - Beginner] + [Name 1 - Expert]",
        bullets: [
          "Han Chiang High School Makers Club | Pepstein Islanders",
          "Mission: Detect clogged roadside storm drains before heavy monsoon rain triggers Penang flash floods.",
        ],
        layoutNote: "Canva Layout: Bold centered title with dark overlay photo of a Penang street waterway.",
      },
      {
        title: "Slide 2: Why Penang Floods Escalate Fast",
        subtitle: "The Problem: Reactive Flood Cleanup Happens Too Late",
        speaker: "Speaker: [Name 5 - Beginner]",
        bullets: [
          "Blocked Infrastructure: Plastic waste and leaves choke roadside storm drains, cutting water capacity before heavy downpours hit.",
          "Reactive Response: Municipal teams rely on manual patrols or resident complaints after roads are already underwater.",
          "Who Suffers: Daily commuters, shop owners, and emergency responders in flood-prone zones like Jalan P. Ramlee and Jelutong.",
        ],
        layoutNote: "Canva Layout: Two-column split (Left: stacked Penang longkang and flood photos; Right: 3 bullets).",
      },
      {
        title: "Slide 3: How FlowGuard Works in 3 Steps",
        subtitle: "Our Solution & Municipal User Flow",
        speaker: "Speaker: [Name 6 - Beginner]",
        bullets: [
          "01 Snap & Upload: Residents or inspectors take a quick photo of a clogged drain and select their Penang corridor zone.",
          "02 AI Classification: Our TensorFlow MobileNetV2 engine grades blockage severity (Clear, Moderate, or Critical) in milliseconds.",
          "03 Priority Dispatch: Critical hazards pin directly onto the live municipal dashboard for rapid pre-storm clearing.",
        ],
        layoutNote: "Canva Layout: 3 horizontal cards with numbered badges (01, 02, 03) and Camera, Chip, Map Pin icons.",
      },
      {
        title: "Slide 4: Under the Hood: System Architecture",
        subtitle: "Anti-'Black Box' Technical Explanation",
        speaker: "Speaker: [Name 3 - Intermediate] + [Name 1 - Expert]",
        bullets: [
          "Frontend & Bridge: Python Streamlit UI cached with @st.cache_resource (plus zero-lag local web console) for instant live inference.",
          "AI Model: TensorFlow MobileNetV2 transfer learning on 224x224x3 input tensors, chosen for fast edge-ready performance on standard hardware.",
          "Why CNN Over Color Filters: Simple brightness thresholds mistake dark tree shadows or wet asphalt for trash; our CNN inspects debris textures and grate periodicity.",
        ],
        layoutNote: "Canva Layout: Left-to-right flow [Streamlit UI] -> [Preprocessing 224x224] -> [MobileNetV2 CNN] -> [Live Hazard Log].",
      },
      {
        title: "Slide 5: Week 10 QA Testing & Responsible AI",
        subtitle: "Empirical Validation, Bug Fixes & Privacy Safeguards",
        speaker: "Speaker: [Name 4 - Intermediate]",
        bullets: [
          "Performance: " + acc + "% validation accuracy (n=120 held-out test set) with " + lat + " ms average CPU inference latency.",
          "Key Bug Solved: Harsh afternoon shadows initially triggered 14/14 false alarms (76.7% baseline); fixed via LAB CLAHE normalization and brightness/contrast augmentation.",
          "Privacy & Ethics: Zero faces or private properties captured. Automated OpenCV Haar face-blur shield runs before every tensor readout.",
        ],
        layoutNote: "Canva Layout: Left Week 10 QA Debug Log screenshot & confusion matrix; Right 3 concise callout cards.",
      },
      {
        title: "Slide 6: Commercial Viability & 2027 Roadmap",
        subtitle: "B2G Municipal SaaS & Next Steps",
        speaker: "Speaker: [Name 2 - Expert]",
        bullets: [
          "Business Model: B2G SaaS dashboard licensing for local councils (MBPP) and private township developers to cut flood cleanup costs.",
          "Phase 1 (Current): Web & Streamlit prototype + Penang corridor hazard logging.",
          "Phase 2 (Q1 2027): Auto-GPS exif tagging and offline-capable mobile PWA field app.",
          "Phase 3 (Mid 2027): Live rainfall API integration to auto-prioritize high-risk corridors before storms arrive. Transitioning to Live Demo...",
        ],
        layoutNote: "Canva Layout: Top Business Model banner + 3-phase bottom timeline.",
      },
    ];
  }

  function renderCurrentSlide() {
    const slides = getPitchSlides();
    const idx = Math.max(0, Math.min(slides.length - 1, state.activeSlideIndex));
    const slide = slides[idx];

    setHtmlText($("slide-counter"), "Slide " + (idx + 1) + " of " + slides.length);
    setHtmlText($("slide-speaker"), slide.speaker);
    setHtmlText($("slide-title"), slide.title);
    setHtmlText($("slide-subtitle"), slide.subtitle);
    setHtmlText($("slide-layout-note"), slide.layoutNote);

    const ul = $("slide-bullets");
    ul.innerHTML = "";
    slide.bullets.forEach((b) => {
      const li = document.createElement("li");
      li.textContent = b;
      ul.appendChild(li);
    });

    const dots = $("slide-dots");
    dots.innerHTML = "";
    slides.forEach((_, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "slide-dot-btn" + (i === idx ? " active" : "");
      btn.textContent = String(i + 1);
      btn.setAttribute("aria-label", "Go to Slide " + (i + 1));
      btn.addEventListener("click", () => {
        state.activeSlideIndex = i;
        renderCurrentSlide();
      });
      dots.appendChild(btn);
    });
  }

  function updateTimerDisplay() {
    const secs = Math.max(0, state.timerSecondsLeft);
    const mm = String(Math.floor(secs / 60)).padStart(2, "0");
    const ss = String(secs % 60).padStart(2, "0");
    setHtmlText($("stage-clock"), mm + ":" + ss);

    const phaseEl = $("stage-phase");
    const elapsed = 480 - secs;
    if (secs === 0) {
      phaseEl.className = "severity-badge severity-Critical";
      phaseEl.textContent = "Time Up (8:00 Limit Reached)";
    } else if (elapsed < 180) {
      phaseEl.className = "severity-badge severity-Clear";
      phaseEl.textContent = "Phase 1: The Pitch (3:00)";
    } else if (elapsed < 360) {
      phaseEl.className = "severity-badge severity-Moderate";
      phaseEl.textContent = "Phase 2: The Live Demo (3:00)";
    } else {
      phaseEl.className = "severity-badge severity-Critical";
      phaseEl.textContent = "Phase 3: Rapid Q&A Defense (2:00)";
    }
  }

  function initPitchStage() {
    $("slide-prev-btn").addEventListener("click", () => {
      state.activeSlideIndex = Math.max(0, state.activeSlideIndex - 1);
      renderCurrentSlide();
    });
    $("slide-next-btn").addEventListener("click", () => {
      state.activeSlideIndex = Math.min(5, state.activeSlideIndex + 1);
      renderCurrentSlide();
    });

    $("timer-start-btn").addEventListener("click", () => {
      if (state.timerIntervalId !== null) return;
      state.timerIntervalId = window.setInterval(() => {
        if (state.timerSecondsLeft > 0) {
          state.timerSecondsLeft -= 1;
          updateTimerDisplay();
        } else {
          window.clearInterval(state.timerIntervalId);
          state.timerIntervalId = null;
        }
      }, 1000);
    });

    $("timer-pause-btn").addEventListener("click", () => {
      if (state.timerIntervalId !== null) {
        window.clearInterval(state.timerIntervalId);
        state.timerIntervalId = null;
      }
    });

    $("timer-reset-btn").addEventListener("click", () => {
      if (state.timerIntervalId !== null) {
        window.clearInterval(state.timerIntervalId);
        state.timerIntervalId = null;
      }
      state.timerSecondsLeft = 480;
      updateTimerDisplay();
    });

    updateTimerDisplay();
    renderCurrentSlide();
  }

  // ---------------------------------------------------------------------------
  // Init on DOM Ready
  // ---------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initNavigation();
    initInspectControls();
    initDispatchControls();
    initArchitectureInspector();
    initQAView();
    initPitchStage();

    loadSamplesAndZones();
    loadMetricsAndSource();
  });
})();
