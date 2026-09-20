(() => {
  const RING_CIRCUMFERENCE = 452; // 2 * pi * r, r=72, must match style.css

  const PENALTY_META = [
    { key: "missing", label: "Missing values", max: 25 },
    { key: "duplicates", label: "Duplicate rows", max: 15 },
    { key: "outliers", label: "Outliers", max: 15 },
    { key: "suspicious_zeros", label: "Suspicious zeros", max: 10 },
    { key: "imbalance", label: "Class imbalance", max: 15 },
    { key: "useless_columns", label: "Low-value columns", max: 10 },
    { key: "correlation", label: "Correlated features", max: 10 },
  ];

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  const uploadPanel = document.getElementById("upload-panel");
  const loadingPanel = document.getElementById("loading-panel");
  const errorPanel = document.getElementById("error-panel");
  const reportPanel = document.getElementById("report-panel");

  const loadingFilename = document.getElementById("loading-filename");
  const errorText = document.getElementById("error-text");
  const errorRetry = document.getElementById("error-retry");
  const analyzeAnother = document.getElementById("analyze-another");

  const reportFilename = document.getElementById("report-filename");
  const gradeCircle = document.getElementById("grade-circle");
  const gradeLetter = document.getElementById("grade-letter");
  const gradeRingFill = document.getElementById("grade-ring-fill");
  const scoreValue = document.getElementById("score-value");
  const scoreSource = document.getElementById("score-source");
  const statsEl = document.getElementById("stats");
  const penaltyList = document.getElementById("penalty-list");
  const issueList = document.getElementById("issue-list");

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function showPanel(panel) {
    [uploadPanel, loadingPanel, errorPanel, reportPanel].forEach((p) => {
      p.classList.toggle("hidden", p !== panel);
    });
  }

  function resetToUpload() {
    fileInput.value = "";
    gradeRingFill.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    showPanel(uploadPanel);
  }

  function bandForGrade(grade) {
    if (grade === "A" || grade === "B") return "good";
    if (grade === "C") return "fair";
    return "poor";
  }

  function renderStats(shape, target) {
    statsEl.innerHTML = "";
    const rows = [
      ["rows", shape.rows],
      ["columns", shape.columns],
      ["target", target || "none detected"],
    ];
    for (const [label, value] of rows) {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      wrap.append(dt, dd);
      statsEl.appendChild(wrap);
    }
  }

  function renderPenalties(penalties) {
    penaltyList.innerHTML = "";
    for (const { key, label, max } of PENALTY_META) {
      const value = penalties[key] ?? 0;
      const pct = Math.min(100, (value / max) * 100);

      const li = document.createElement("li");

      const labelEl = document.createElement("span");
      labelEl.className = "penalty-label";
      labelEl.textContent = label;

      const track = document.createElement("span");
      track.className = "penalty-track";
      const fill = document.createElement("span");
      fill.className = "penalty-fill";
      fill.style.width = pct + "%";
      track.appendChild(fill);

      const valueEl = document.createElement("span");
      valueEl.className = "penalty-value";
      valueEl.textContent = "-" + value;

      li.append(labelEl, track, valueEl);
      penaltyList.appendChild(li);
    }
  }

  function renderIssues(issues) {
    issueList.innerHTML = "";
    if (!issues || issues.length === 0) {
      const li = document.createElement("li");
      li.className = "no-issues";
      li.textContent = "No issues found. This dataset is clean.";
      issueList.appendChild(li);
      return;
    }
    for (const issue of issues) {
      const li = document.createElement("li");
      li.textContent = issue;
      issueList.appendChild(li);
    }
  }

  function renderReport(filename, data) {
    reportFilename.textContent = filename;

    gradeLetter.textContent = data.grade;
    const band = bandForGrade(data.grade);
    gradeCircle.setAttribute("data-band", band);

    scoreValue.textContent = data.score;
    scoreSource.textContent =
      data.score_source === "model"
        ? "scored by the trained quality model"
        : "scored by the rule-based formula (model unavailable)";

    renderStats(data.shape, data.target);
    renderPenalties(data.penalties || {});
    renderIssues(data.issues || []);

    showPanel(reportPanel);

    // draw the grade ring in from empty, one orchestrated reveal
    const offset = RING_CIRCUMFERENCE * (1 - Math.max(0, Math.min(100, data.score)) / 100);
    gradeRingFill.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
    if (prefersReducedMotion) {
      gradeRingFill.style.strokeDashoffset = String(offset);
    } else {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          gradeRingFill.style.strokeDashoffset = String(offset);
        });
      });
    }
  }

  async function analyzeFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      errorText.textContent = `"${file.name}" doesn't look like a CSV file. Choose a .csv file and try again.`;
      showPanel(errorPanel);
      return;
    }

    loadingFilename.textContent = file.name;
    showPanel(loadingPanel);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });

      if (!res.ok) {
        let message = `The server returned an error (status ${res.status}).`;
        try {
          const body = await res.json();
          if (body && body.error) message = body.error;
        } catch (_) {
          /* response wasn't JSON, keep default message */
        }
        errorText.textContent = message;
        showPanel(errorPanel);
        return;
      }

      const data = await res.json();
      renderReport(file.name, data);
    } catch (err) {
      errorText.textContent =
        "Couldn't reach the grading service. Make sure the app server is running and try again.";
      showPanel(errorPanel);
    }
  }

  // ---- dropzone interactions ----
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) analyzeFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) analyzeFile(file);
  });

  errorRetry.addEventListener("click", resetToUpload);
  analyzeAnother.addEventListener("click", resetToUpload);
})();
