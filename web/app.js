const videoInput = document.querySelector("#video");
const modelInput = document.querySelector("#model");
const skipCut = document.querySelector("#skip-cut");
const runButton = document.querySelector("#run");
const pickButton = document.querySelector("#pick");
const message = document.querySelector("#message");
const health = document.querySelector("#health");
const progress = document.querySelector("#progress");
const logEl = document.querySelector("#log");
const result = document.querySelector("#result");
const summary = document.querySelector("#summary");
const flagged = document.querySelector("#flagged");
const output = document.querySelector("#output");

let timer = null;

function showError(text) {
  message.hidden = !text;
  message.textContent = text || "";
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function setSteps(step, status, reviewOnly) {
  const order = ["transcribe", "analyze", "cut"];
  const active = step === "done" ? order.length : order.indexOf(step);
  document.querySelectorAll(".steps li").forEach((item, index) => {
    item.classList.remove("on", "ok");
    if (reviewOnly && item.dataset.step === "cut") {
      item.textContent = "Cut skipped";
      if (status === "done") item.classList.add("ok");
      return;
    }
    item.textContent = ["Transcribe", "Review", "Cut"][index];
    if (status === "done" || index < active) item.classList.add("ok");
    else if (index === active && status === "running") item.classList.add("on");
  });
}

function render(job) {
  if (!job || job.status === "idle") return;

  progress.hidden = false;
  setSteps(job.step, job.status, job.skip_cut);
  logEl.textContent = (job.log || []).join("\n");
  logEl.scrollTop = logEl.scrollHeight;

  runButton.disabled = job.status === "running";
  runButton.textContent = job.status === "running" ? "Working…" : "Clean this film";

  if (job.status === "error") showError(job.error || "The pipeline stopped.");
  else showError("");

  if (!job.report) {
    result.hidden = true;
    return;
  }

  const count = job.report.flagged_count ?? (job.report.flagged || []).length;
  const total = job.report.total_scenes ?? 0;
  summary.textContent = count
    ? `${count} of ${total} scenes flagged.`
    : `No scenes flagged out of ${total}.`;
  if (job.status !== "error" && job.report.errors && job.report.errors.length) {
    showError(job.report.errors.join(" "));
  }
  flagged.replaceChildren();
  for (const item of job.report.flagged || []) {
    const li = document.createElement("li");
    const scene = document.createElement("p");
    scene.className = "scene";
    scene.textContent = `Scene ${item.scene_number} · ${item.start} – ${item.end}`;
    const reason = document.createElement("p");
    reason.className = "reason";
    reason.textContent = item.reason || "";
    const preview = document.createElement("p");
    preview.className = "preview";
    preview.textContent = item.text_preview || "";
    li.append(scene, reason, preview);
    flagged.append(li);
  }

  output.replaceChildren();
  if (job.output && job.status === "done") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "link";
    button.textContent = "Show cleaned video";
    button.addEventListener("click", () => {
      api("/api/reveal", { path: job.output }).catch((error) => showError(error.message));
    });
    output.append(button);
  }
  result.hidden = false;
}

async function poll() {
  try {
    const job = await api("/api/job");
    render(job);
    if (job.status !== "running" && timer) {
      clearInterval(timer);
      timer = null;
    }
  } catch (error) {
    showError(error.message);
  }
}

function watch() {
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 1000);
  poll();
}

pickButton.addEventListener("click", async () => {
  pickButton.disabled = true;
  showError("");
  try {
    const data = await api("/api/pick", {});
    if (data.path) videoInput.value = data.path;
  } catch (error) {
    showError(error.message);
  } finally {
    pickButton.disabled = false;
  }
});

runButton.addEventListener("click", async () => {
  showError("");
  runButton.disabled = true;
  try {
    const job = await api("/api/job", {
      video: videoInput.value.trim(),
      model: modelInput.value.trim(),
      skip_cut: skipCut.checked,
    });
    render(job);
    watch();
  } catch (error) {
    showError(error.message);
    runButton.disabled = false;
  }
});

api("/api/health")
  .then((info) => {
    const missing = [];
    if (!info.ffmpeg) missing.push("ffmpeg");
    if (!info.ollama) missing.push("Ollama");
    health.classList.toggle("bad", missing.length > 0);
    health.textContent = missing.length
      ? `Missing: ${missing.join(", ")}. The film will not finish until those are available.`
      : `Ready. Speech engine: ${info.whisper}.`;
  })
  .catch(() => {
    health.textContent = "Could not check this machine.";
  });

poll();
