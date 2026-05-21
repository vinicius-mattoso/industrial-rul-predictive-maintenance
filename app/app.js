const state = {
  cycleOffset: 0,
  autoRun: false,
  selectedEngine: "",
  selectedSensors: ["sensor_4", "sensor_7", "sensor_11"],
  acknowledgedAlerts: new Set(),
  snapshot: null
};

const sensorColors = {
  sensor_4: "#1268d8",
  sensor_7: "#17a398",
  sensor_11: "#d98b16",
  sensor_12: "#8a63d2",
  sensor_15: "#d84444",
  sensor_21: "#2e8b57"
};

const els = {
  themeToggle: document.querySelector("#themeToggle"),
  simulateButton: document.querySelector("#simulateButton"),
  singleCycleButton: document.querySelector("#singleCycleButton"),
  autoRunButton: document.querySelector("#autoRunButton"),
  cycleStep: document.querySelector("#cycleStep"),
  riskFilter: document.querySelector("#riskFilter"),
  engineFilter: document.querySelector("#engineFilter"),
  operationClock: document.querySelector("#operationClock"),
  fleetCount: document.querySelector("#fleetCount"),
  warningCount: document.querySelector("#warningCount"),
  criticalCount: document.querySelector("#criticalCount"),
  avgRul: document.querySelector("#avgRul"),
  equipmentGrid: document.querySelector("#equipmentGrid"),
  alertList: document.querySelector("#alertList"),
  priorityTable: document.querySelector("#priorityTable"),
  ackWarningsButton: document.querySelector("#ackWarningsButton"),
  modelStatus: document.querySelector("#modelStatus"),
  sensorSelector: document.querySelector("#sensorSelector"),
  rulChart: document.querySelector("#rulChart"),
  sensorChart: document.querySelector("#sensorChart")
};

function normalizeStatus(status) {
  return status.toLowerCase() === "degrading" ? "warning" : status.toLowerCase();
}

function statusLabel(status) {
  const normalized = normalizeStatus(status);
  return {
    healthy: "Healthy",
    warning: "Warning",
    critical: "Critical"
  }[normalized] || status;
}

function suggestedAction(asset) {
  const status = normalizeStatus(asset.status);
  if (status === "critical") return "Abrir OS e reduzir carga";
  if (asset.rul <= 55) return "Planejar janela em ate 48h";
  return `Inspecionar ${asset.criticalSensor}`;
}

async function loadSnapshot() {
  const params = new URLSearchParams({ offset: String(state.cycleOffset) });
  if (state.selectedEngine) params.set("engine", state.selectedEngine);

  const response = await fetch(`/api/snapshot?${params.toString()}`);
  if (!response.ok) throw new Error(`API error ${response.status}`);

  state.snapshot = await response.json();
  if (!state.selectedEngine) {
    state.selectedEngine = String(state.snapshot.selectedEngine);
  }

  render();
}

function advanceCycles(cycles) {
  state.cycleOffset += cycles;
  loadSnapshot().catch(showLoadError);
}

function showLoadError(error) {
  els.equipmentGrid.innerHTML = `<p class="empty-state">Nao foi possivel carregar a API local: ${error.message}</p>`;
}

function render() {
  if (!state.snapshot) return;
  renderEngineFilter();
  renderSensorSelector();
  renderMetrics();
  renderEquipment();
  renderAlerts();
  renderTable();
  renderRulChart();
  renderSensorChart();
}

function renderEngineFilter() {
  const current = state.selectedEngine || String(state.snapshot.selectedEngine);
  els.engineFilter.innerHTML = state.snapshot.engines
    .map((engineId) => `<option value="${engineId}" ${String(engineId) === current ? "selected" : ""}>Engine ${String(engineId).padStart(3, "0")}</option>`)
    .join("");
}

function renderSensorSelector() {
  els.sensorSelector.innerHTML = state.snapshot.featureSensors
    .map((sensor) => `
      <label class="sensor-chip">
        <input type="checkbox" value="${sensor}" ${state.selectedSensors.includes(sensor) ? "checked" : ""} />
        <span style="--chip-color: ${sensorColors[sensor]}">${sensor}</span>
      </label>
    `)
    .join("");
}

function renderMetrics() {
  const equipment = state.snapshot.equipment;
  const statuses = equipment.map((asset) => normalizeStatus(asset.status));
  const avg = Math.round(equipment.reduce((sum, asset) => sum + asset.rul, 0) / equipment.length);

  els.fleetCount.textContent = equipment.length;
  els.warningCount.textContent = statuses.filter((status) => status === "warning").length;
  els.criticalCount.textContent = statuses.filter((status) => status === "critical").length;
  els.avgRul.textContent = avg;
  els.operationClock.textContent = `Ciclo operacional ${state.snapshot.operationCycle}`;
  els.modelStatus.textContent = state.snapshot.model.available ? "Modelo LightGBM ativo" : "Fallback por RUL";
  els.modelStatus.title = state.snapshot.model.message;
  els.modelStatus.classList.toggle("warning-pill", !state.snapshot.model.available);
}

function renderEquipment() {
  const filter = els.riskFilter.value;
  const cards = state.snapshot.equipment
    .map((asset) => ({ ...asset, statusKey: normalizeStatus(asset.status) }))
    .filter((asset) => filter === "all" || asset.statusKey === filter)
    .sort((a, b) => a.rul - b.rul)
    .map((asset) => {
      const healthPercent = Math.max(4, Math.min(100, Math.round((asset.rul / 125) * 100)));
      const confidence = asset.confidence === null ? "Regra RUL" : `${Math.round(asset.confidence * 100)}% conf.`;
      return `
        <article class="equipment-card ${asset.statusKey}">
          <header>
            <h3>${asset.id}</h3>
            <span class="health-badge ${asset.statusKey}">${statusLabel(asset.status)}</span>
          </header>
          <div class="equipment-meta">
            <span>${asset.area} | ciclo ${asset.cycle}</span>
            <strong>${Math.round(asset.rul)} ciclos RUL</strong>
          </div>
          <div class="progress" aria-label="Health score ${healthPercent}%">
            <span style="width: ${healthPercent}%"></span>
          </div>
          <div class="sensor-row">
            <span>${asset.criticalSensor}</span>
            <span>${confidence}</span>
          </div>
        </article>
      `;
    })
    .join("");

  els.equipmentGrid.innerHTML = cards || `<p class="empty-state">Nenhum equipamento para o filtro selecionado.</p>`;
}

function buildAlerts() {
  return state.snapshot.equipment
    .map((asset) => ({ ...asset, statusKey: normalizeStatus(asset.status) }))
    .filter((asset) => asset.statusKey !== "healthy")
    .sort((a, b) => a.rul - b.rul)
    .map((asset) => ({
      id: `${asset.id}-${asset.statusKey}-${asset.cycle}`,
      title: `${asset.id} em ${statusLabel(asset.status)}`,
      message: `${asset.area}: ${asset.criticalSensor} indica degradacao. RUL estimado em ${Math.round(asset.rul)} ciclos. ${suggestedAction(asset)}.`,
      severity: asset.statusKey
    }));
}

function renderAlerts() {
  const alerts = buildAlerts();
  const visibleAlerts = alerts.filter((alert) => !state.acknowledgedAlerts.has(alert.id) || alert.severity === "critical");

  els.alertList.innerHTML = visibleAlerts.length
    ? visibleAlerts
      .map((alert) => `
        <article class="alert-item ${alert.severity}">
          <span class="alert-severity" aria-hidden="true"></span>
          <div>
            <strong>${alert.title}</strong>
            <p>${alert.message}</p>
          </div>
        </article>
      `)
      .join("")
    : `<p class="empty-state">Sem warnings ativos apos o reconhecimento.</p>`;
}

function renderTable() {
  els.priorityTable.innerHTML = state.snapshot.equipment
    .map((asset) => ({ ...asset, statusKey: normalizeStatus(asset.status) }))
    .sort((a, b) => a.rul - b.rul)
    .map((asset) => `
      <tr>
        <td><strong>${asset.id}</strong></td>
        <td><span class="health-badge ${asset.statusKey}">${statusLabel(asset.status)}</span></td>
        <td>${Math.round(asset.rul)} ciclos</td>
        <td>${asset.criticalSensor}</td>
        <td>${suggestedAction(asset)}</td>
      </tr>
    `)
    .join("");
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function renderRulChart() {
  const { ctx, width, height } = setupCanvas(els.rulChart);
  const styles = getComputedStyle(document.body);
  const padding = 42;
  const values = state.snapshot.rulHistory;
  const min = Math.max(0, Math.min(...values.map((item) => item.avg)) - 10);
  const max = Math.max(...values.map((item) => item.avg), 125) + 5;

  drawChartBase(ctx, width, height, styles, padding);
  drawLine(ctx, pointsFor(values, "avg", width, height, padding, min, max), styles.getPropertyValue("--brand").trim(), 4);

  ctx.fillStyle = styles.getPropertyValue("--muted").trim();
  ctx.font = "14px Inter, sans-serif";
  ctx.fillText(`RUL medio: ${values.at(-1).avg}`, padding, 24);
  ctx.fillText(`Ciclo ${values.at(-1).cycle}`, width - 130, height - 16);
}

function renderSensorChart() {
  const { ctx, width, height } = setupCanvas(els.sensorChart);
  const styles = getComputedStyle(document.body);
  const padding = 48;
  const rows = state.snapshot.sensorSeries.rows;

  drawChartBase(ctx, width, height, styles, padding);

  if (!rows.length) return;

  const selected = state.selectedSensors;
  const allValues = selected.flatMap((sensor) => rows.map((row) => row[sensor]));
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  selected.forEach((sensor) => {
    const points = pointsFor(rows, sensor, width, height, padding, min, max);
    drawLine(ctx, points, sensorColors[sensor] || styles.getPropertyValue("--brand").trim(), 2.5);

    const envelope = state.snapshot.sensorSeries.envelope[sensor];
    if (envelope) {
      drawEnvelopeLine(ctx, rows, envelope.p05, width, height, padding, min, max, sensorColors[sensor], 0.22);
      drawEnvelopeLine(ctx, rows, envelope.p95, width, height, padding, min, max, sensorColors[sensor], 0.22);
    }
  });

  ctx.fillStyle = styles.getPropertyValue("--muted").trim();
  ctx.font = "14px Inter, sans-serif";
  ctx.fillText(`Engine ${String(state.snapshot.sensorSeries.engineId).padStart(3, "0")} | ciclo ${state.snapshot.sensorSeries.currentCycle}`, padding, 24);
}

function drawChartBase(ctx, width, height, styles, padding) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = styles.getPropertyValue("--surface-2").trim();
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = styles.getPropertyValue("--line").trim();
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = padding + ((height - padding * 2) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }
}

function pointsFor(rows, key, width, height, padding, min, max) {
  const span = Math.max(0.0001, max - min);
  return rows.map((row, index) => {
    const x = padding + ((width - padding * 2) / Math.max(1, rows.length - 1)) * index;
    const y = height - padding - ((row[key] - min) / span) * (height - padding * 2);
    return { x, y };
  });
}

function drawLine(ctx, points, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();
}

function drawEnvelopeLine(ctx, rows, value, width, height, padding, min, max, color, alpha) {
  const span = Math.max(0.0001, max - min);
  const y = height - padding - ((value - min) / span) * (height - padding * 2);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(padding, y);
  ctx.lineTo(width - padding, y);
  ctx.stroke();
  ctx.restore();
}

els.simulateButton.addEventListener("click", () => advanceCycles(Number(els.cycleStep.value)));
els.singleCycleButton.addEventListener("click", () => advanceCycles(1));
els.riskFilter.addEventListener("change", renderEquipment);
els.engineFilter.addEventListener("change", () => {
  state.selectedEngine = els.engineFilter.value;
  loadSnapshot().catch(showLoadError);
});
els.themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  localStorage.setItem("rul-theme", document.body.classList.contains("dark") ? "dark" : "light");
  renderRulChart();
  renderSensorChart();
});
els.autoRunButton.addEventListener("click", () => {
  state.autoRun = !state.autoRun;
  els.autoRunButton.classList.toggle("active", state.autoRun);
  els.autoRunButton.setAttribute("aria-pressed", String(state.autoRun));
  els.autoRunButton.textContent = state.autoRun ? "Pausar auto-run" : "Auto-run";
});
els.ackWarningsButton.addEventListener("click", () => {
  buildAlerts()
    .filter((alert) => alert.severity === "warning")
    .forEach((alert) => state.acknowledgedAlerts.add(alert.id));
  renderAlerts();
});
els.sensorSelector.addEventListener("change", (event) => {
  if (event.target.tagName !== "INPUT") return;
  const selected = [...els.sensorSelector.querySelectorAll("input:checked")].map((input) => input.value);
  state.selectedSensors = selected.length ? selected : [event.target.value];
  renderSensorSelector();
  renderSensorChart();
});
window.addEventListener("resize", () => {
  renderRulChart();
  renderSensorChart();
});

setInterval(() => {
  if (state.autoRun) advanceCycles(1);
}, 1400);

if (localStorage.getItem("rul-theme") === "dark") {
  document.body.classList.add("dark");
}

loadSnapshot().catch(showLoadError);
