"use strict";

/* ================= Helpers ================= */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 3) => (v == null || Number.isNaN(+v) ? "–" : (+v).toFixed(d));
const fmtBytes = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const debounce = (fn, ms = 350) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
};

const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  leaf: '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10Z"/><path d="M2 21c0-3 1.9-5.4 5.1-6"/>',
  sprout: '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 6-6"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7l10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
  flame: '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.4-.5-2-1-3-1.1-2.1-.2-4 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.2.4-2.3 1-3.2.4 1.5 1.4 2.7 2.5 2.7Z"/>',
  wind: '<path d="M17.7 7.7A2.5 2.5 0 1 1 19.5 12H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>',
  scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><rect x="7" y="7" width="10" height="10" rx="1"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/>',
  table: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
  map: '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3Z"/><path d="M9 3v15M15 6v15"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
  alert: '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  check: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  pin: '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
};
const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;

const KIND_LABEL = { raster: "GeoTIFF", image: "Image", table: "CSV", vector: "Vector" };
const KIND_ICON = { raster: "layers", image: "image", table: "table", vector: "map" };

const TOOLS = [
  { id: "ndvi", group: "Vegetation", title: "NDVI Viewer", icon: "leaf", tone: "green", needs: ["raster", "image"],
    blurb: "Vegetation index maps from Sentinel-2, Landsat, MODIS or RGB imagery, with class breakdown and GeoTIFF export." },
  { id: "crop", group: "Vegetation", title: "Crop Monitoring", icon: "sprout", tone: "green", needs: ["table", "raster", "image"],
    blurb: "Build an NDVI time series from a CSV or from imagery, smooth it and flag crop stress." },
  { id: "landuse", group: "Vegetation", title: "Land Use Classifier", icon: "layers", tone: "green", needs: ["raster", "image"],
    blurb: "Classify imagery with NDVI thresholds, spectral clustering or a deep-learning model." },
  { id: "gps", group: "Mapping", title: "GPS Heatmapper", icon: "flame", tone: "amber", needs: ["table"],
    blurb: "Interactive heatmaps from GPS points in a CSV, or draw your own points on the map." },
  { id: "pollution", group: "Mapping", title: "Pollution Visualizer", icon: "wind", tone: "amber", needs: ["table", "raster"],
    blurb: "Map NO₂, PM2.5 and other pollutants from station CSVs or satellite rasters." },
  { id: "georef", group: "Mapping", title: "Georeference", icon: "pin", tone: "amber", needs: ["raster", "image"],
    blurb: "Pin an image to the map with ground control points and export a georeferenced GeoTIFF." },
  { id: "detect", group: "AI Detection", title: "Object Detection", icon: "scan", tone: "", needs: ["raster", "image"],
    blurb: "YOLOv8 object detection with an annotated image and exportable detections." },
];

const RDYLGN = ["#a50026", "#f46d43", "#fee08b", "#d9ef8b", "#66bd63", "#006837"];
const INFERNO = ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"];
const gradient = (stops) => `linear-gradient(90deg, ${stops.join(", ")})`;
function rampColor(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
  const a = stops[i].match(/\w\w/g).map((h) => parseInt(h, 16)), b = stops[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}

/* ================= State ================= */
const state = { files: [], config: null, charts: [], maps: [], crop: store.get("geoai-crop-points", []) };

async function api(path, body, method) {
  const res = await fetch(path, {
    method: method || (body ? "POST" : "GET"),
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

function toast(msg, type = "error") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `${icon(type === "ok" ? "check" : "alert")}<div>${esc(msg)}</div>`;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), type === "ok" ? 3000 : 6500);
}

async function busy(btn, fn) {
  btn.classList.add("loading");
  btn.disabled = true;
  try { return await fn(); } catch (e) { toast(e.message); } finally { btn.classList.remove("loading"); btn.disabled = false; }
}

async function refreshFiles() {
  try { state.files = await api("/api/files"); } catch (e) { state.files = []; }
  const n = state.files.length;
  $("#files-pill").innerHTML = `${icon("database")}<span><b>${n}</b> file${n === 1 ? "" : "s"} loaded</span>`;
}
const filesOf = (kinds) => state.files.filter((f) => kinds.includes(f.kind));
const fileById = (id) => state.files.find((f) => f.id === id);

/* ================= UI building blocks ================= */
function pageHead(tool) {
  return `<div class="page-head"><div>
    <h1><span class="badge-ico">${icon(tool.icon)}</span>${esc(tool.title)}</h1>
    <p>${esc(tool.blurb)}</p></div></div>`;
}

function emptyFiles(kinds) {
  const names = kinds.map((k) => KIND_LABEL[k]).join(", ");
  return `<div class="card empty">
    <div class="big">${icon("upload")}</div>
    <h3>No suitable data yet</h3>
    <p>This tool works with ${esc(names)} files. Add some to your library and come back.</p>
    <a class="btn primary" href="#/data">${icon("upload")} Upload data</a></div>`;
}

function fileSelect(id, kinds, label = "Dataset") {
  const files = filesOf(kinds);
  return `<div class="field"><label for="${id}">${esc(label)}</label>
    <select id="${id}">${files.map((f) => `<option value="${f.id}">${esc(f.name)} · ${KIND_LABEL[f.kind]}</option>`).join("")}</select></div>`;
}

function slider(id, label, min, max, step, value, digits = 2) {
  return `<div class="field"><label for="${id}">${esc(label)} <span class="val" id="${id}-v">${(+value).toFixed(digits)}</span></label>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" data-digits="${digits}"></div>`;
}
function bindSliders(root) {
  $$('input[type="range"]', root).forEach((r) => r.addEventListener("input", () => {
    const v = $(`#${r.id}-v`, root);
    if (v) v.textContent = (+r.value).toFixed(+r.dataset.digits);
  }));
}

function segmented(id, options, value) {
  return `<div class="segmented" id="${id}" role="tablist">${options.map((o) =>
    `<button type="button" data-v="${esc(o)}" class="${o === value ? "on" : ""}">${esc(o)}</button>`).join("")}</div>`;
}
function bindSegmented(el, onChange) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  });
}

const notices = (list, type = "warn") => (list || []).map((w) => `<div class="notice ${type}">${icon(type === "warn" ? "alert" : "info")}<div>${esc(w)}</div></div>`).join("");
const statTile = (k, v, unit = "") => `<div class="card stat"><div class="k">${esc(k)}</div><div class="v">${v}${unit ? `<small>${esc(unit)}</small>` : ""}</div></div>`;
const downloadsHtml = (list) => `<div class="downloads">${(list || []).map((d) =>
  `<a class="btn sm" href="${d.url}" download>${icon("download")}${esc(d.label)}</a>`).join("")}</div>`;
const skeleton = (msg) => `<div class="skeleton">${esc(msg)}</div>`;

function placeholder(title, text, ico = "arrow") {
  return `<div class="card empty"><div class="big">${icon(ico)}</div><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
}

/* Charts */
function chart(canvas, cfg) {
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--border");
  const c = new Chart(canvas, cfg);
  state.charts.push(c);
  return c;
}
function histogramChart(canvas, hist, colorFor, label = "Pixels") {
  const centers = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2);
  return chart(canvas, {
    type: "bar",
    data: { labels: centers.map((c) => c.toFixed(2)), datasets: [{ label, data: hist.counts, backgroundColor: centers.map(colorFor), borderRadius: 3, barPercentage: 1, categoryPercentage: 0.95 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 9 } }, y: { beginAtZero: true, ticks: { maxTicksLimit: 5 } } } },
  });
}

/* Maps */
function makeMap(el, center = [20.59, 78.96], zoom = 4, startBase = 0) {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const esri = (svc) => `https://server.arcgisonline.com/ArcGIS/rest/services/${svc}/MapServer/tile/{z}/{y}/{x}`;
  const base = {
    [dark ? "Dark Gray" : "Light Gray"]: L.tileLayer(esri(dark ? "Canvas/World_Dark_Gray_Base" : "Canvas/World_Light_Gray_Base"), { attribution: "Tiles © Esri", maxZoom: 16 }),
    OpenStreetMap: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors", maxZoom: 19 }),
    Satellite: L.tileLayer(esri("World_Imagery"), { attribution: "Imagery © Esri", maxZoom: 19 }),
  };
  const map = L.map(el, { center, zoom, layers: [Object.values(base)[startBase]], zoomControl: true });
  map.layersControl = L.control.layers(base, {}, { position: "topright" }).addTo(map);
  state.maps.push(map);
  setTimeout(() => map.invalidateSize(), 60);
  return map;
}

function mapLegend(map, title, stops, lo, hi, digits = 2) {
  const ctl = L.control({ position: "bottomleft" });
  ctl.onAdd = () => {
    const d = L.DomUtil.create("div", "map-legend");
    d.innerHTML = `<b>${esc(title)}</b><div class="legend-bar" style="background:${gradient(stops)}"></div>
      <div class="legend-scale"><span>${num(lo, digits)}</span><span>${num(hi, digits)}</span></div>`;
    return d;
  };
  return ctl.addTo(map);
}

function boundaryControl() {
  const vectors = filesOf(["vector"]);
  if (!vectors.length) return "";
  return `<div class="field"><label for="boundary">Boundary overlay</label>
    <select id="boundary"><option value="">None</option>${vectors.map((f) => `<option value="${f.id}">${esc(f.name)}</option>`).join("")}</select></div>`;
}
function bindBoundary(root, getMap) {
  const sel = $("#boundary", root);
  if (!sel) return;
  let layer = null;
  sel.addEventListener("change", async () => {
    const map = getMap();
    if (layer && map) { map.removeLayer(layer); layer = null; }
    if (!sel.value || !map) return;
    try {
      const gj = await (await fetch(`/api/files/${sel.value}/geojson`)).json();
      layer = L.geoJSON(gj, { style: { color: cssVar("--primary"), weight: 2, fillOpacity: 0.04 } }).addTo(map);
      map.fitBounds(layer.getBounds(), { padding: [20, 20] });
    } catch (e) { toast("Could not load boundary file"); }
  });
}

function sensorControls(prefix, file) {
  const sats = state.config.satellites;
  const rgb = state.config.rgb_mode;
  const dataBands = file && file.kind === "raster" ? file.meta.bands - (file.meta.alpha ? 1 : 0) : 0;
  const def = dataBands >= 8 ? "Sentinel-2" : dataBands >= 4 ? "Custom" : rgb;
  const options = (file && file.kind === "raster" ? Object.keys(sats) : []).concat([rgb]);
  return `<div class="field"><label for="${prefix}-sat">Sensor profile</label>
    <select id="${prefix}-sat">${options.map((o) => `<option ${o === def ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>
    <span class="hint" id="${prefix}-sat-hint"></span></div>
    <div class="row" id="${prefix}-bands" hidden>
      <div class="field"><label for="${prefix}-red">Red band</label><input type="number" id="${prefix}-red" min="1" value="3"></div>
      <div class="field"><label for="${prefix}-nir">NIR band</label><input type="number" id="${prefix}-nir" min="1" value="4"></div>
    </div>`;
}
function bindSensor(root, prefix) {
  const sel = $(`#${prefix}-sat`, root);
  const update = () => {
    const s = sel.value, prof = state.config.satellites[s];
    $(`#${prefix}-bands`, root).hidden = s !== "Custom";
    $(`#${prefix}-sat-hint`, root).textContent = s === state.config.rgb_mode
      ? "Red = R channel, NIR ≈ G channel (proxy, no NIR needed)."
      : prof && prof.red ? `${prof.description}: red = band ${prof.red}, NIR = band ${prof.nir}.` : "Pick the band numbers in your file.";
  };
  sel.addEventListener("change", update);
  update();
  return () => ({
    satellite: sel.value,
    red: sel.value === "Custom" ? +$(`#${prefix}-red`, root).value : null,
    nir: sel.value === "Custom" ? +$(`#${prefix}-nir`, root).value : null,
  });
}

/* ================= Pages ================= */
function renderHome(view) {
  const n = state.files.length;
  const groups = [...new Set(TOOLS.map((t) => t.group))];
  view.innerHTML = `
    <div class="hero">
      <h1>Earth observation insights, <span>without the GIS setup</span>.</h1>
      <p>Upload satellite rasters, imagery or GPS tables once, then analyse vegetation, land use, crop health, pollution and objects, all in your browser.</p>
      <div class="actions">
        <a class="btn primary" href="#/data">${icon("upload")} ${n ? "Manage data" : "Upload data"}</a>
        <a class="btn" href="#/ndvi">${icon("leaf")} Try NDVI Viewer</a>
      </div>
      <div class="hero-stats">
        <div><b>${n}</b>files loaded</div>
        <div><b>${TOOLS.length}</b>analysis tools</div>
        <div><b>GeoTIFF</b>export with georeferencing</div>
      </div>
    </div>
    ${groups.map((g) => `<div class="section-title">${esc(g)}</div><div class="grid">${TOOLS.filter((t) => t.group === g).map((t) => {
      const ready = filesOf(t.needs).length;
      return `<a class="card tool-card ${t.tone}" href="#/${t.id}">
        <div class="ico">${icon(t.icon)}</div><h3>${esc(t.title)}</h3><p>${esc(t.blurb)}</p>
        <div class="chips">${t.needs.map((k) => `<span class="chip">${KIND_LABEL[k]}</span>`).join("")}
          ${ready ? `<span class="chip ok">${ready} ready</span>` : ""}</div></a>`;
    }).join("")}</div>`).join("")}`;
}

function renderData(view) {
  const card = (f) => {
    const m = f.meta || {};
    const chips = f.kind === "raster" ? [`${m.width}×${m.height}`, `${m.bands} band${m.bands > 1 ? "s" : ""}`, m.dtype]
      : f.kind === "image" ? [`${m.width}×${m.height}`, m.mode]
      : f.kind === "table" ? [`${m.rows} rows`, `${(m.columns || []).length} columns`] : [`${m.features} features`];
    const geo = f.kind === "raster" ? (m.georeferenced ? `<span class="chip ok">${esc(m.crs)}</span>` : `<span class="chip warn">Not georeferenced</span>`) : "";
    const thumb = ["raster", "image"].includes(f.kind)
      ? `<div class="thumb" style="background-image:url('/api/files/${f.id}/thumb')"></div>`
      : `<div class="thumb">${icon(KIND_ICON[f.kind])}</div>`;
    return `<div class="card file-card">${thumb}<div class="body">
      <div class="top"><span class="name" title="${esc(f.name)}">${esc(f.name)}</span>
        <button class="icon-btn" data-del="${f.id}" aria-label="Remove ${esc(f.name)}" title="Remove">${icon("trash")}</button></div>
      <div class="chips"><span class="chip">${KIND_LABEL[f.kind]}</span><span class="chip">${fmtBytes(f.size)}</span>
        ${chips.filter(Boolean).map((c) => `<span class="chip">${esc(c)}</span>`).join("")}${geo}</div></div></div>`;
  };
  view.innerHTML = `
    <div class="page-head"><div><h1><span class="badge-ico">${icon("database")}</span>Data library</h1>
      <p>Files stay in this server's memory for the session and are available to every tool.</p></div></div>
    <label class="dropzone" id="drop">
      <input type="file" id="picker" multiple hidden accept=".tif,.tiff,.geotiff,.jpg,.jpeg,.png,.csv,.geojson,.json,.kml,.gpkg,.zip">
      <div class="big">${icon("upload")}</div>
      <h3 id="drop-title">Drop files here or click to browse</h3>
      <p id="drop-sub">GeoTIFF · JPG/PNG · CSV · GeoJSON/KML/GeoPackage · zipped Shapefile</p>
    </label>
    ${state.files.length ? `<div class="file-grid">${state.files.map(card).join("")}</div>`
      : `<div class="hint" style="margin-top:14px">No files yet. Satellite scenes up to a few hundred MB work fine.</div>`}`;

  const drop = $("#drop", view), picker = $("#picker", view);
  const upload = (files) => {
    if (!files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/files");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) $("#drop-title", view).textContent = `Uploading… ${Math.round((100 * e.loaded) / e.total)}%`;
    };
    xhr.upload.onload = () => { $("#drop-title", view).textContent = "Reading files…"; };
    xhr.onload = async () => {
      let res = [];
      try { res = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      if (xhr.status >= 400 || !Array.isArray(res)) toast("Upload failed");
      res.filter((r) => r.error).forEach((r) => toast(`${r.name}: ${r.error}`));
      const ok = res.filter((r) => !r.error).length;
      if (ok) toast(`Added ${ok} file${ok > 1 ? "s" : ""}`, "ok");
      await refreshFiles();
      renderData(view);
    };
    xhr.onerror = () => toast("Upload failed: server unreachable");
    xhr.send(fd);
  };
  picker.addEventListener("change", () => upload(picker.files));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));
  $$("[data-del]", view).forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/files/${b.dataset.del}`, null, "DELETE");
    await refreshFiles();
    renderData(view);
  }));
}

/* ---------- NDVI ---------- */
function renderNdvi(view, tool) {
  if (!filesOf(tool.needs).length) { view.innerHTML = pageHead(tool) + emptyFiles(tool.needs); return; }
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Input</h3>${fileSelect("nd-file", tool.needs)}
      <div id="nd-sensor"></div>
      <button class="btn primary block" id="nd-run">${icon("leaf")} Calculate NDVI</button>
    </div>
    <div class="results" id="nd-out">${placeholder("Ready when you are", "Pick a dataset and sensor profile, then calculate NDVI.")}</div></div>`;
  let getSensor;
  const fileSel = $("#nd-file", view);
  const drawSensor = () => {
    $("#nd-sensor", view).innerHTML = sensorControls("nd", fileById(fileSel.value));
    getSensor = bindSensor(view, "nd");
  };
  fileSel.addEventListener("change", drawSensor);
  drawSensor();

  $("#nd-run", view).addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const out = $("#nd-out", view);
    out.innerHTML = skeleton("Computing NDVI…");
    const r = await api("/api/ndvi", { file_id: fileSel.value, ...getSensor() }).catch((err) => {
      out.innerHTML = placeholder("Could not calculate NDVI", err.message, "alert"); throw err;
    });
    out.innerHTML = `${notices(r.warnings)}
      <div class="stats">${statTile("Mean NDVI", num(r.stats.mean))}${statTile("Min", num(r.stats.min))}${statTile("Max", num(r.stats.max))}${statTile("Valid pixels", num(r.stats.valid_pct, 1), "%")}</div>
      <div class="card"><div class="card-head"><div><h3>${esc(r.mode)}</h3><div class="sub">${r.width.toLocaleString()} × ${r.height.toLocaleString()} px</div></div>${downloadsHtml(r.downloads)}</div>
        <div class="figure"><img src="${r.image}" alt="NDVI map"></div>
        <div class="card-pad" style="padding-top:14px"><div class="legend-bar" style="background:${gradient(RDYLGN)}"></div>
          <div class="legend-scale"><span>−1 water / bare</span><span>0</span><span>+1 dense vegetation</span></div></div></div>
      <div class="two">
        <div class="card"><div class="card-head"><h3>Vegetation classes</h3></div><div class="card-pad">
          <div class="stack" style="margin-bottom:18px">${r.classes.map((c) => `<i style="width:${c.pct}%;background:${c.color}" title="${esc(c.label)} ${num(c.pct, 1)}%"></i>`).join("")}</div>
          <div class="legend">${r.classes.map((c) => `<div class="legend-row"><span class="sw" style="background:${c.color}"></span>
            <span>${esc(c.label)} <span class="hint">(${esc(c.range)})</span></span><span class="pct">${num(c.pct, 1)}%</span></div>`).join("")}</div></div></div>
        <div class="card"><div class="card-head"><h3>NDVI distribution</h3></div><div class="chart-box"><canvas id="nd-hist"></canvas></div></div>
      </div>`;
    histogramChart($("#nd-hist", view), r.histogram, (c) => rampColor(RDYLGN, (c + 1) / 2));
  }));
}

/* ---------- Crop monitoring ---------- */
function renderCrop(view, tool) {
  const tables = filesOf(["table"]), images = filesOf(["raster", "image"]);
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Add data</h3>
      ${segmented("cr-src", ["From CSV", "From imagery"], tables.length || !images.length ? "From CSV" : "From imagery")}
      <div id="cr-src-body"></div>
      <div class="divider"></div>
      <h3 class="section">Series <span class="hint" id="cr-count"></span></h3>
      <div class="series-list" id="cr-list"></div>
      <button class="btn sm ghost" id="cr-clear">${icon("trash")} Clear series</button>
      <div class="divider"></div>
      <h3 class="section">Analysis</h3>
      ${slider("cr-thr", "Stress threshold", 0, 1, 0.01, 0.5)}
      ${slider("cr-smooth", "Smoothing window (days)", 0, 30, 1, 7, 0)}
    </div>
    <div class="results" id="cr-out"></div></div>`;
  bindSliders(view);

  const save = () => store.set("geoai-crop-points", state.crop);
  const body = $("#cr-src-body", view);
  const drawSource = (src) => {
    if (src === "From CSV") {
      body.innerHTML = tables.length ? `${fileSelect("cr-csv", ["table"], "NDVI table (date, ndvi)")}<div id="cr-cols"></div>
        <button class="btn primary block" id="cr-load">${icon("table")} Load series</button>` : emptyMini("CSV");
      const load = $("#cr-load", body);
      if (load) load.addEventListener("click", () => busy(load, async () => {
        const cols = $("#cr-cols", body);
        const req = { file_id: $("#cr-csv", body).value };
        if ($("#cr-date", cols)) Object.assign(req, { date_col: $("#cr-date", cols).value, ndvi_col: $("#cr-ndvi", cols).value });
        const r = await api("/api/crop/csv", req);
        if (!r.points.length) {
          const opts = (sel) => r.columns.map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
          cols.innerHTML = `<div class="notice info">${icon("info")}<div>Choose which columns hold the date and NDVI values.</div></div><div style="height:12px"></div>
            <div class="row"><div class="field"><label>Date column</label><select id="cr-date">${opts(r.date_col)}</select></div>
            <div class="field"><label>NDVI column</label><select id="cr-ndvi">${opts(r.ndvi_col)}</select></div></div>`;
          return;
        }
        state.crop = r.points; save(); refresh();
        toast(`Loaded ${r.points.length} observations`, "ok");
      }));
    } else {
      body.innerHTML = images.length ? `${fileSelect("cr-img", ["raster", "image"], "Image")}<div id="cr-sensor"></div>
        <div class="field"><label for="cr-date-in">Acquisition date</label><input type="date" id="cr-date-in" value="${new Date().toISOString().slice(0, 10)}"></div>
        <button class="btn primary block" id="cr-add">${icon("plus")} Add mean NDVI to series</button>` : emptyMini("image");
      if (!images.length) return;
      let getSensor;
      const imgSel = $("#cr-img", body);
      const drawSensor = () => { $("#cr-sensor", body).innerHTML = sensorControls("cr", fileById(imgSel.value)); getSensor = bindSensor(body, "cr"); };
      imgSel.addEventListener("change", drawSensor);
      drawSensor();
      const add = $("#cr-add", body);
      add.addEventListener("click", () => busy(add, async () => {
        const r = await api("/api/crop/image", { file_id: imgSel.value, date: $("#cr-date-in", body).value, ...getSensor() });
        state.crop = state.crop.filter((p) => p.date !== r.date).concat([{ date: r.date, ndvi: r.ndvi }]);
        save(); refresh();
        toast(`${r.date}: mean ${r.mode} ${num(r.ndvi)}`, "ok");
      }));
    }
  };
  const emptyMini = (what) => `<div class="notice info">${icon("info")}<div>No ${what} files yet. <a href="#/data">Upload data</a>.</div></div>`;
  bindSegmented($("#cr-src", view), drawSource);
  drawSource($("#cr-src .on", view).dataset.v);

  $("#cr-clear", view).addEventListener("click", () => { state.crop = []; save(); refresh(); });
  $("#cr-list", view).addEventListener("click", (e) => {
    const b = e.target.closest("[data-rm]");
    if (b) { state.crop = state.crop.filter((p) => p.date !== b.dataset.rm); save(); refresh(); }
  });

  let lineChart = null;
  const out = $("#cr-out", view);
  const refresh = debounce(async () => {
    const pts = [...state.crop].sort((a, b) => a.date.localeCompare(b.date));
    $("#cr-count", view).textContent = `(${pts.length})`;
    $("#cr-list", view).innerHTML = pts.length ? pts.slice().reverse().map((p) =>
      `<div class="series-item"><span>${esc(p.date)}</span><span>${num(p.ndvi)} <button class="icon-btn" style="width:24px;height:24px" data-rm="${esc(p.date)}" aria-label="Remove">${icon("x")}</button></span></div>`).join("")
      : `<div class="hint">No observations yet.</div>`;
    if (!pts.length) {
      lineChart = null;
      out.innerHTML = placeholder("Build a time series", "Load an NDVI CSV, or add mean NDVI from images taken on different dates.", "sprout");
      return;
    }
    const thr = +$("#cr-thr", view).value;
    let r;
    try { r = await api("/api/crop/analyze", { points: pts, threshold: thr, smooth_days: +$("#cr-smooth", view).value }); }
    catch (err) { toast(err.message); return; }
    const vals = r.raw.map((p) => p.ndvi), latest = r.raw[r.raw.length - 1];
    out.innerHTML = `
      <div class="stats">${statTile("Observations", r.raw.length)}${statTile("Latest NDVI", num(latest.ndvi))}
        ${statTile("Mean NDVI", num(vals.reduce((a, b) => a + b, 0) / vals.length))}${statTile("Stressed dates", r.stressed.length)}</div>
      ${r.stressed.length ? `<div class="notice warn">${icon("alert")}<div>NDVI fell below ${thr.toFixed(2)} on ${r.stressed.length} of ${r.raw.length} observed dates.</div></div>`
        : `<div class="notice ok">${icon("check")}<div>No stress detected: every observation is above ${thr.toFixed(2)}.</div></div>`}
      <div class="card"><div class="card-head"><h3>Crop health time series</h3><span class="sub">${esc(r.raw[0].date)} → ${esc(latest.date)}</span></div>
        <div class="chart-box tall"><canvas id="cr-chart"></canvas></div></div>
      ${r.stressed.length ? `<div class="card"><div class="card-head"><h3>Stressed observations</h3></div><div class="table-scroll"><table class="data">
        <thead><tr><th>Date</th><th>Observed NDVI</th><th>Smoothed</th></tr></thead><tbody>
        ${r.stressed.map((s) => `<tr><td>${esc(s.date)}</td><td>${num(s.ndvi)}</td><td>${num(s.value)}</td></tr>`).join("")}</tbody></table></div></div>` : ""}`;
    const t = (d) => Date.parse(d);
    const xs = r.raw.map((p) => t(p.date)).concat(r.smooth.map((p) => t(p.date)));
    const DAY = 864e5;
    let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    if (x1 - x0 < 2 * DAY) { x0 -= 15 * DAY; x1 += 15 * DAY; }  // one observation: show a month around it
    const stressedSet = new Set(r.stressed.map((s) => s.date));
    lineChart = chart($("#cr-chart", view), {
      type: "line",
      data: { datasets: [
        { label: "Observed", data: r.raw.map((p) => ({ x: t(p.date), y: p.ndvi })), showLine: r.smooth.length === 0, borderColor: cssVar("--primary"),
          backgroundColor: r.raw.map((p) => (stressedSet.has(p.date) ? cssVar("--danger") : cssVar("--primary"))), pointRadius: 4, pointHoverRadius: 6 },
        { label: "Smoothed", data: r.smooth.map((p) => ({ x: t(p.date), y: p.ndvi })), borderColor: cssVar("--accent"), borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
        { label: "Stress threshold", data: [{ x: x0, y: thr }, { x: x1, y: thr }], borderColor: cssVar("--danger"), borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: { maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false },
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8 } },
          tooltip: { callbacks: { title: (items) => new Date(items[0].parsed.x).toISOString().slice(0, 10) } } },
        scales: { x: { type: "linear", min: x0, max: x1, ticks: { callback: (v) => new Date(v).toISOString().slice(0, 10), maxTicksLimit: 8 }, grid: { display: false } },
          y: { suggestedMin: 0, suggestedMax: 1, title: { display: true, text: "NDVI" } } } },
    });
  }, 120);
  $("#cr-thr", view).addEventListener("input", refresh);
  $("#cr-smooth", view).addEventListener("input", refresh);
  refresh();
}

/* ---------- Land use ---------- */
function renderLanduse(view, tool) {
  if (!filesOf(tool.needs).length) { view.innerHTML = pageHead(tool) + emptyFiles(tool.needs); return; }
  const methods = state.config.landuse_methods;
  const label = { "Simple NDVI-based": "NDVI", "Spectral Clustering": "Clustering", "DeepLabV3+ (ML Model)": "DeepLab" };
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Input</h3>${fileSelect("lu-file", tool.needs)}
      <div class="label" style="margin-bottom:6px">Method</div>
      ${segmented("lu-method", methods.map((m) => label[m]), "Clustering")}
      <div id="lu-params"></div>
      ${methods.length < 3 ? `<p class="hint">DeepLabV3+ appears once <code>deeplabv3_finetuned_RS_openearthmap_v2.pth</code> is in <code>landuse_classifier/</code>.</p>` : ""}
      <button class="btn primary block" id="lu-run">${icon("layers")} Classify land use</button>
    </div>
    <div class="results" id="lu-out">${placeholder("Classify an image", "Clustering works on any imagery; NDVI thresholds need a near-infrared band for meaningful classes.")}</div></div>`;
  let method = "Spectral Clustering";
  const params = $("#lu-params", view);
  const drawParams = () => {
    params.innerHTML = method === "Simple NDVI-based"
      ? slider("lu-water", "Water threshold (NDVI <)", -1, 0, 0.05, -0.3) + slider("lu-veg", "Vegetation threshold (NDVI >)", 0, 1, 0.05, 0.3)
      : method === "Spectral Clustering" ? slider("lu-k", "Number of clusters", 2, 10, 1, 6, 0)
      : `<p class="hint">Segments into OpenEarthMap classes (bareland, rangeland, trees, buildings, roads, water, agriculture).</p>`;
    bindSliders(params);
  };
  bindSegmented($("#lu-method", view), (v) => { method = Object.keys(label).find((k) => label[k] === v); drawParams(); });
  drawParams();

  $("#lu-run", view).addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const out = $("#lu-out", view);
    out.innerHTML = skeleton("Classifying… large scenes take a few seconds");
    const req = { file_id: $("#lu-file", view).value, method };
    if ($("#lu-water", view)) Object.assign(req, { water_threshold: +$("#lu-water", view).value, veg_threshold: +$("#lu-veg", view).value });
    if ($("#lu-k", view)) req.n_clusters = +$("#lu-k", view).value;
    const r = await api("/api/landuse", req).catch((err) => { out.innerHTML = placeholder("Classification failed", err.message, "alert"); throw err; });
    const shown = r.classes.filter((c) => c.pixels > 0);
    out.innerHTML = `${notices(r.warnings)}
      <div class="card"><div class="card-head"><div><h3>${esc(method)}</h3><div class="sub">${shown.length} classes</div></div>
        ${segmented("lu-view", ["Overlay", "Classes", "Original"], "Overlay")}</div>
        <div class="figure"><img id="lu-img" src="${r.overlay}" alt="Classification result"></div></div>
      <div class="two">
        <div class="card"><div class="card-head"><h3>Class breakdown</h3></div><div class="card-pad legend">
          ${shown.map((c) => `<div class="legend-row"><span class="sw" style="background:${c.color}"></span><span>${esc(c.name)}</span>
            <span class="pct">${num(c.pct, 1)}%</span><div class="bar"><i style="width:${c.pct}%;background:${c.color}"></i></div></div>`).join("")}</div></div>
        <div class="card"><div class="card-head"><h3>Share of area</h3></div><div class="chart-box"><canvas id="lu-chart"></canvas></div></div>
      </div>
      <div class="card card-pad"><div class="label" style="margin-bottom:10px">Export</div>${downloadsHtml(r.downloads)}</div>`;
    const seg = $("#lu-view", out);
    $(".segmented", out).style.margin = "0";
    bindSegmented(seg, (v) => { $("#lu-img", out).src = v === "Overlay" ? r.overlay : v === "Classes" ? r.classified : r.original; });
    chart($("#lu-chart", view), {
      type: "doughnut",
      data: { labels: shown.map((c) => c.name), datasets: [{ data: shown.map((c) => +c.pct.toFixed(2)), backgroundColor: shown.map((c) => c.color), borderColor: cssVar("--surface"), borderWidth: 2 }] },
      options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right", labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.parsed}%` } } } },
    });
  }));
}

/* ---------- GPS heatmapper ---------- */
function renderGps(view, tool) {
  const tables = filesOf(["table"]);
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Source</h3>
      ${segmented("gp-src", ["From CSV", "Draw on map"], tables.length ? "From CSV" : "Draw on map")}
      <div id="gp-body"></div>
      <div class="divider"></div>
      <h3 class="section">Heatmap style</h3>
      ${slider("gp-radius", "Radius", 5, 50, 1, 18, 0)}${slider("gp-blur", "Blur", 5, 50, 1, 15, 0)}
      <label class="check"><input type="checkbox" id="gp-markers"> Show individual points</label>
    </div>
    <div class="results"><div class="stats" id="gp-stats"></div>
      <div class="card"><div class="card-head"><h3 id="gp-title">Heatmap</h3><div id="gp-actions"></div></div><div class="map" id="gp-map"></div></div></div></div>`;
  bindSliders(view);
  const map = makeMap($("#gp-map", view));
  let heat = null, markers = null, points = [], drawn = null, drawCtl = null;

  const render = () => {
    if (heat) map.removeLayer(heat);
    if (markers) map.removeLayer(markers);
    heat = markers = null;
    if (!points.length) return;
    const ws = points.map((p) => (p[2] == null ? 1 : +p[2]));
    const wmax = Math.max(...ws) || 1;
    heat = L.heatLayer(points.map((p, i) => [p[0], p[1], ws[i] / wmax]), { radius: +$("#gp-radius", view).value, blur: +$("#gp-blur", view).value, max: 1 }).addTo(map);
    if ($("#gp-markers", view).checked) {
      markers = L.layerGroup(points.slice(0, 3000).map((p) => L.circleMarker([p[0], p[1]], { radius: 3, color: cssVar("--primary"), weight: 1, fillOpacity: 0.8 })
        .bindPopup(`${num(p[0], 5)}, ${num(p[1], 5)}${p[2] != null ? `<br>weight ${num(p[2], 2)}` : ""}`))).addTo(map);
    }
  };
  ["gp-radius", "gp-blur", "gp-markers"].forEach((id) => $(`#${id}`, view).addEventListener("input", render));
  const setStats = (count, total) => {
    if (!points.length) { $("#gp-stats", view).innerHTML = ""; return; }
    const lats = points.map((p) => p[0]), lons = points.map((p) => p[1]);
    $("#gp-stats", view).innerHTML = statTile("Points", count.toLocaleString()) + (total > count ? statTile("In file", total.toLocaleString()) : "")
      + statTile("Latitude span", `${num(Math.min(...lats), 3)} → ${num(Math.max(...lats), 3)}`) + statTile("Longitude span", `${num(Math.min(...lons), 3)} → ${num(Math.max(...lons), 3)}`);
  };
  const download = (name, text, type) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const body = $("#gp-body", view);
  const drawSource = (src) => {
    points = []; render(); setStats(0, 0);
    if (drawCtl) { map.removeControl(drawCtl); map.removeLayer(drawn); drawCtl = drawn = null; }
    $("#gp-actions", view).innerHTML = "";
    if (src === "From CSV") {
      $("#gp-title", view).textContent = "Heatmap";
      body.innerHTML = tables.length ? `${fileSelect("gp-file", ["table"], "GPS table")}<div id="gp-cols"></div>
        <button class="btn primary block" id="gp-load">${icon("flame")} Generate heatmap</button>`
        : `<div class="notice info">${icon("info")}<div>No CSV files yet. <a href="#/data">Upload data</a> or draw points instead.</div></div>`;
      const load = $("#gp-load", body);
      if (!load) return;
      $("#gp-file", body).addEventListener("change", () => { $("#gp-cols", body).innerHTML = ""; });
      load.addEventListener("click", () => busy(load, async () => {
        const req = { file_id: $("#gp-file", body).value };
        if ($("#gp-lat", body)) Object.assign(req, { lat_col: $("#gp-lat", body).value, lon_col: $("#gp-lon", body).value, value_col: $("#gp-w", body).value || null });
        const r = await api("/api/gps", req);
        const opts = (list, sel, none) => (none ? `<option value="">None</option>` : "") + list.map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
        $("#gp-cols", body).innerHTML = `<div class="row"><div class="field"><label>Latitude</label><select id="gp-lat">${opts(r.columns, r.lat_col)}</select></div>
          <div class="field"><label>Longitude</label><select id="gp-lon">${opts(r.columns, r.lon_col)}</select></div></div>
          <div class="field"><label>Weight (optional)</label><select id="gp-w">${opts(r.numeric_columns.filter((c) => c !== r.lat_col && c !== r.lon_col), r.value_col, true)}</select></div>`;
        if (!r.points.length) { toast("Couldn't detect latitude/longitude columns. Pick them and generate again.", "error"); return; }
        points = r.points; render(); setStats(points.length, r.total);
        map.fitBounds(L.latLngBounds(points.map((p) => [p[0], p[1]])), { padding: [30, 30], maxZoom: 15 });
      }));
    } else {
      $("#gp-title", view).textContent = "Draw points";
      body.innerHTML = `<div class="notice info">${icon("info")}<div>Use the marker tool on the map to place points. Edit or delete them with the toolbar.</div></div>`;
      drawn = new L.FeatureGroup().addTo(map);
      drawCtl = new L.Control.Draw({ position: "topleft", edit: { featureGroup: drawn },
        draw: { polyline: false, polygon: false, rectangle: false, circle: false, circlemarker: false, marker: true } });
      map.addControl(drawCtl);
      const sync = () => { points = drawn.getLayers().map((l) => [l.getLatLng().lat, l.getLatLng().lng, 1]); render(); setStats(points.length, 0); };
      map.on(L.Draw.Event.CREATED, (e) => { drawn.addLayer(e.layer); sync(); });
      map.on(L.Draw.Event.EDITED, sync);
      map.on(L.Draw.Event.DELETED, sync);
      $("#gp-actions", view).innerHTML = `<div class="downloads"><button class="btn sm" id="gp-csv">${icon("download")}CSV</button><button class="btn sm" id="gp-geojson">${icon("download")}GeoJSON</button></div>`;
      $("#gp-csv", view).addEventListener("click", () => download("points.csv", "lat,lon,weight\n" + points.map((p) => p.join(",")).join("\n"), "text/csv"));
      $("#gp-geojson", view).addEventListener("click", () => download("points.geojson", JSON.stringify({ type: "FeatureCollection",
        features: points.map((p) => ({ type: "Feature", properties: { weight: p[2] }, geometry: { type: "Point", coordinates: [p[1], p[0]] } })) }), "application/geo+json"));
    }
  };
  bindSegmented($("#gp-src", view), drawSource);
  drawSource($("#gp-src .on", view).dataset.v);
}

/* ---------- Pollution ---------- */
function renderPollution(view, tool) {
  if (!filesOf(tool.needs).length) { view.innerHTML = pageHead(tool) + emptyFiles(tool.needs); return; }
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Input</h3>${fileSelect("po-file", tool.needs, "Station CSV or raster")}
      <div id="po-opts"></div>
      <button class="btn primary block" id="po-run">${icon("wind")} Visualize</button>
      <div id="po-live"></div>
      ${boundaryControl()}
    </div>
    <div class="results" id="po-out">${placeholder("Map pollution", "CSV station data becomes a heatmap with markers; rasters are reprojected and draped on the map.", "wind")}</div></div>`;
  let map = null;
  bindBoundary(view, () => map);
  const fileSel = $("#po-file", view);
  const out = $("#po-out", view), live = $("#po-live", view);
  fileSel.addEventListener("change", () => { $("#po-opts", view).innerHTML = ""; live.innerHTML = ""; });

  const mapCard = (title, extra = "") => `<div class="card"><div class="card-head"><h3>${esc(title)}</h3>${extra}</div><div class="map" id="po-map"></div></div>`;
  const newMap = () => {
    state.maps = state.maps.filter((m) => m !== map);
    if (map) map.remove();
    map = makeMap($("#po-map", out));
    if ($("#boundary", view) && $("#boundary", view).value) $("#boundary", view).dispatchEvent(new Event("change"));
    return map;
  };

  const runCsv = async (btn) => {
    const req = { file_id: fileSel.value };
    if ($("#po-pol", view)) Object.assign(req, { lat_col: $("#po-lat", view).value, lon_col: $("#po-lon", view).value, value_col: $("#po-pol", view).value });
    const r = await api("/api/pollution/csv", req);
    const opts = (list, sel) => list.map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
    $("#po-opts", view).innerHTML = `<div class="field"><label>Pollutant</label><select id="po-pol">${opts(r.pollutants, r.value_col)}</select></div>
      <div class="row"><div class="field"><label>Latitude</label><select id="po-lat">${opts(r.columns, r.lat_col)}</select></div>
      <div class="field"><label>Longitude</label><select id="po-lon">${opts(r.columns, r.lon_col)}</select></div></div>`;
    if (!r.points.length) { toast("Pick the latitude, longitude and pollutant columns, then visualize again."); return; }
    const { min, max } = r.stats, digits = max - min < 10 ? 2 : 1;
    live.innerHTML = `<div class="divider"></div><h3 class="section">Filter</h3>
      ${slider("po-lo", "Minimum", min, max, (max - min) / 200 || 1, min, digits)}${slider("po-hi", "Maximum", min, max, (max - min) / 200 || 1, max, digits)}`;
    bindSliders(live);
    out.innerHTML = `<div class="stats" id="po-stats"></div>${mapCard(`${r.value_col} concentration`)}
      <div class="card"><div class="card-head"><h3>Distribution</h3></div><div class="chart-box"><canvas id="po-hist"></canvas></div></div>`;
    newMap();
    let layers = [];
    const draw = () => {
      layers.forEach((l) => map.removeLayer(l));
      const lo = +$("#po-lo", live).value, hi = +$("#po-hi", live).value;
      const pts = r.points.filter((p) => p[2] >= lo && p[2] <= hi);
      const span = (max - min) || 1;
      layers = [L.heatLayer(pts.map((p) => [p[0], p[1], (p[2] - min) / span]), { radius: 22, blur: 20, max: 1,
        gradient: { 0.2: INFERNO[1], 0.45: INFERNO[2], 0.7: INFERNO[3], 0.9: INFERNO[4], 1: INFERNO[5] } }).addTo(map),
        L.layerGroup(pts.slice(0, 3000).map((p) => L.circleMarker([p[0], p[1]], { radius: 5, weight: 1, color: "#fff", fillColor: rampColor(INFERNO, (p[2] - min) / span), fillOpacity: 0.95 })
          .bindPopup(`<b>${esc(r.value_col)}</b>: ${num(p[2], 2)}<br>${num(p[0], 4)}, ${num(p[1], 4)}`))).addTo(map)];
      const vals = pts.map((p) => p[2]);
      $("#po-stats", out).innerHTML = vals.length ? statTile("Stations", vals.length) + statTile("Mean", num(vals.reduce((a, b) => a + b, 0) / vals.length, 2))
        + statTile("Max", num(Math.max(...vals), 2)) + statTile("Min", num(Math.min(...vals), 2)) : statTile("Stations", 0);
    };
    $("#po-lo", live).addEventListener("input", draw);
    $("#po-hi", live).addEventListener("input", draw);
    draw();
    mapLegend(map, r.value_col, INFERNO.slice(1), min, max);
    map.fitBounds(L.latLngBounds(r.points.map((p) => [p[0], p[1]])), { padding: [30, 30], maxZoom: 13 });
    histogramChart($("#po-hist", out), r.histogram, (c) => rampColor(INFERNO, (c - min) / ((max - min) || 1)), "Stations");
  };

  const runRaster = async () => {
    let r = await api("/api/pollution/raster", { file_id: fileSel.value });
    const [min, max] = r.range, label = r.pollutant || "Concentration", digits = max - min < 10 ? 3 : 1;
    live.innerHTML = `<div class="divider"></div><h3 class="section">Display</h3>
      ${slider("po-op", "Opacity", 0, 1, 0.05, 0.75)}${slider("po-lo", "Minimum", min, max, (max - min) / 200 || 1, min, digits)}${slider("po-hi", "Maximum", min, max, (max - min) / 200 || 1, max, digits)}`;
    bindSliders(live);
    out.innerHTML = `${notices(r.warnings)}<div class="stats">${statTile("Mean", num(r.stats.mean, digits))}${statTile("Min", num(r.stats.min, digits))}${statTile("Max", num(r.stats.max, digits))}${statTile("Std dev", num(r.stats.std, digits))}</div>
      ${mapCard(label, downloadsHtml(r.downloads))}
      <div class="card"><div class="card-head"><h3>Value distribution</h3></div><div class="chart-box"><canvas id="po-hist"></canvas></div></div>`;
    newMap();
    let overlay = L.imageOverlay(r.image, r.bounds, { opacity: 0.75 }).addTo(map);
    let legend = mapLegend(map, label, INFERNO, min, max, digits);
    map.fitBounds(r.bounds);
    histogramChart($("#po-hist", out), r.histogram, (c) => rampColor(INFERNO, (c - min) / ((max - min) || 1)));
    $("#po-op", live).addEventListener("input", (e) => overlay.setOpacity(+e.target.value));
    const rerange = debounce(async () => {
      const lo = +$("#po-lo", live).value, hi = +$("#po-hi", live).value;
      if (lo >= hi) return;
      try { r = await api("/api/pollution/raster", { file_id: fileSel.value, vmin: lo, vmax: hi }); } catch (err) { toast(err.message); return; }
      overlay.setUrl(r.image);
      map.removeControl(legend);
      legend = mapLegend(map, label, INFERNO, lo, hi, digits);
    }, 400);
    $("#po-lo", live).addEventListener("input", rerange);
    $("#po-hi", live).addEventListener("input", rerange);
  };

  $("#po-run", view).addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const f = fileById(fileSel.value);
    out.innerHTML = skeleton(f.kind === "raster" ? "Reprojecting raster…" : "Reading stations…");
    try { await (f.kind === "raster" ? runRaster() : runCsv()); }
    catch (err) { out.innerHTML = placeholder("Could not visualize this file", err.message, "alert"); throw err; }
  }));
}

/* ---------- Georeference ---------- */
function renderGeoref(view, tool) {
  if (!filesOf(tool.needs).length) { view.innerHTML = pageHead(tool) + emptyFiles(tool.needs); return; }
  view.innerHTML = pageHead(tool) + `
    <div class="card card-pad georef-bar">
      <div class="georef-controls">
        ${fileSelect("gr-file", tool.needs, "Image to georeference")}
        <div class="field"><label for="gr-search">Find a place on the world map</label>
          <form class="search" id="gr-find"><input type="text" id="gr-search" placeholder="e.g. Kanpur, India">
          <button class="btn" type="submit" aria-label="Search">${icon("search")}</button></form></div>
      </div>
      <div class="notice info" id="gr-status"></div>
    </div>
    <div class="panes">
      <div class="card"><div class="card-head"><h3>1 · Your image</h3><span class="sub">Click a recognisable feature</span></div><div class="map" id="gr-img"></div></div>
      <div class="card"><div class="card-head"><h3>2 · World map</h3><span class="sub">Click the same feature</span></div><div class="map" id="gr-world"></div></div>
    </div>
    <div class="two" style="margin-top:20px">
      <div class="card"><div class="card-head"><h3>Control points</h3><button class="btn sm ghost" id="gr-clear">${icon("trash")} Clear all</button></div>
        <div class="table-scroll"><table class="data" id="gr-table"></table></div></div>
      <div class="card" id="gr-result"></div>
    </div>`;

  const fileSel = $("#gr-file", view);
  const world = makeMap($("#gr-world", view), [20.59, 78.96], 4, 2);  // satellite basemap
  const imgMap = L.map($("#gr-img", view), { crs: L.CRS.Simple, minZoom: -6, zoomSnap: 0.25, attributionControl: false });
  state.maps.push(imgMap);
  const imgLayer = L.layerGroup().addTo(imgMap), imgPins = L.layerGroup().addTo(imgMap), worldPins = L.layerGroup().addTo(world);
  let file, sx = 1, sy = 1, gcps = [], residuals = [], fitted = null, overlay = null;
  const key = () => `geoai-gcps-${file.name}`;
  const complete = () => gcps.filter((g) => g.lat != null && g.lon != null);
  const pending = () => gcps.findIndex((g) => g.lat == null || g.lon == null);
  const pinIcon = (i, wait) => L.divIcon({ className: `gcp-pin${wait ? " pending" : ""}`, html: `<span>${i + 1}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] });

  const status = () => {
    const n = complete().length, p = pending();
    $("#gr-status", view).innerHTML = icon("info") + "<div>" + (p >= 0
      ? `<b>Point ${p + 1}:</b> now click the same spot on the world map.`
      : !gcps.length ? "Click a sharp, recognisable feature on your image (bridge, road junction, river bend), then the same feature on the world map."
      : n < 3 ? `${n} point${n > 1 ? "s" : ""} placed. Add at least ${3 - n} more.`
      : `Fitted with ${n} points. Points near the corners improve accuracy. Save when the overlay lines up.`) + "</div>";
  };

  const table = () => {
    $("#gr-table", view).innerHTML = gcps.length ? `<thead><tr><th>#</th><th>Pixel (col, row)</th><th>Latitude</th><th>Longitude</th><th>Error</th><th></th></tr></thead><tbody>
      ${gcps.map((g, i) => `<tr><td><span class="gcp-dot ${g.lat == null ? "pending" : ""}">${i + 1}</span></td><td>${Math.round(g.col)}, ${Math.round(g.row)}</td>
        <td><input type="number" step="any" data-i="${i}" data-k="lat" value="${g.lat ?? ""}" placeholder="lat"></td>
        <td><input type="number" step="any" data-i="${i}" data-k="lon" value="${g.lon ?? ""}" placeholder="lon"></td>
        <td>${residuals[i] != null ? `${num(residuals[i], 1)} m` : "–"}</td>
        <td><button class="icon-btn" data-rm="${i}" aria-label="Remove point ${i + 1}">${icon("x")}</button></td></tr>`).join("")}</tbody>`
      : `<tbody><tr><td class="hint" style="padding:20px">No control points yet.</td></tr></tbody>`;
  };

  const draw = () => {
    imgPins.clearLayers(); worldPins.clearLayers();
    gcps.forEach((g, i) => {
      const wait = g.lat == null || g.lon == null;
      L.marker([-g.row / sy, g.col / sx], { icon: pinIcon(i, wait), draggable: true }).addTo(imgPins).on("dragend", (e) => {
        const ll = e.target.getLatLng();
        Object.assign(g, { col: ll.lng * sx, row: -ll.lat * sy }); changed();
      });
      if (!wait) L.marker([g.lat, g.lon], { icon: pinIcon(i), draggable: true }).addTo(worldPins).on("dragend", (e) => {
        const ll = e.target.getLatLng();
        Object.assign(g, { lat: +ll.lat.toFixed(6), lon: +ll.lng.toFixed(6) }); changed();
      });
    });
    table(); status();
  };

  const result = (html) => { $("#gr-result", view).innerHTML = html; };
  const resultIdle = () => result(placeholder("Add 3+ control points", "The fit, its accuracy and a preview overlay appear here.", "pin"));

  const fit = debounce(async () => {
    const pts = complete();
    if (pts.length < 3) { residuals = []; fitted = null; if (overlay) world.removeLayer(overlay); overlay = null; resultIdle(); draw(); return; }
    let r;
    try { r = await api("/api/georef", { file_id: file.id, gcps: pts }); }
    catch (err) { residuals = []; result(placeholder("Can't fit yet", err.message, "alert")); draw(); return; }
    fitted = r;
    const byPoint = new Map(pts.map((g, i) => [g, r.residuals_m[i]]));
    residuals = gcps.map((g) => byPoint.get(g));
    if (overlay) world.removeLayer(overlay);
    overlay = L.imageOverlay(r.preview.image, r.preview.bounds, { opacity: +($("#gr-op", view)?.value ?? 0.6) }).addTo(world);
    const px = r.pixel_size_m, q = r.rmse_m <= 2 * px ? ["ok", "Good fit"] : r.rmse_m <= 6 * px ? ["warn", "Fair fit"] : ["warn", "Check your points"];
    result(`<div class="card-head"><h3>Fit</h3><span class="chip ${q[0]}">${q[1]}</span></div><div class="card-pad">
      <div class="stats" style="margin-bottom:16px">
        <div class="stat" style="padding:0"><div class="k">RMS error</div><div class="v">${num(r.rmse_m, 1)}<small>m</small></div></div>
        <div class="stat" style="padding:0"><div class="k">Pixel size</div><div class="v">${num(px, 2)}<small>m</small></div></div>
        <div class="stat" style="padding:0"><div class="k">Output CRS</div><div class="v" style="font-size:18px">${esc(r.crs)}</div></div></div>
      ${slider("gr-op", "Overlay opacity", 0, 1, 0.05, overlay.options.opacity)}
      <div class="downloads" id="gr-save-box"><button class="btn primary" id="gr-save">${icon("save")} Save georeferenced GeoTIFF</button></div></div>`);
    bindSliders(view);
    $("#gr-op", view).addEventListener("input", (e) => overlay && overlay.setOpacity(+e.target.value));
    $("#gr-save", view).addEventListener("click", (e) => busy(e.currentTarget, async () => {
      const s = await api("/api/georef", { file_id: file.id, gcps: complete(), save: true });
      await refreshFiles();
      $("#gr-save-box", view).innerHTML = `${downloadsHtml(s.downloads)}<a class="btn sm" href="#/pollution">${icon("map")} View on map</a><a class="btn sm ghost" href="#/data">Open library</a>`;
      toast(`Saved ${s.file.name} to your library`, "ok");
    }));
    draw();
  }, 250);

  const changed = () => { store.set(key(), gcps); draw(); fit(); };

  const loadFile = () => {
    file = fileById(fileSel.value);
    imgLayer.clearLayers();
    const img = new Image();
    img.onload = () => {
      const [pw, ph] = [img.naturalWidth, img.naturalHeight];
      sx = file.meta.width / pw; sy = file.meta.height / ph;
      const bounds = [[-ph, 0], [0, pw]];
      L.imageOverlay(img.src, bounds).addTo(imgLayer);
      imgMap.setMaxBounds(L.latLngBounds(bounds).pad(0.5));
      imgMap.invalidateSize();
      imgMap.fitBounds(bounds);
      gcps = store.get(key(), []);
      residuals = [];
      changed();
    };
    img.src = `/api/files/${file.id}/thumb?size=2048`;
  };

  imgMap.on("click", (e) => {
    const col = e.latlng.lng * sx, row = -e.latlng.lat * sy;
    if (col < 0 || row < 0 || col > file.meta.width || row > file.meta.height) return;
    const p = pending();
    if (p >= 0) Object.assign(gcps[p], { col, row });  // re-place the point still waiting for its map location
    else gcps.push({ col, row, lat: null, lon: null });
    changed();
  });
  world.on("click", (e) => {
    const p = pending();
    if (p < 0) { toast("Click the feature on your image first, then here.", "error"); return; }
    Object.assign(gcps[p], { lat: +e.latlng.lat.toFixed(6), lon: +e.latlng.lng.toFixed(6) });
    changed();
  });
  $("#gr-table", view).addEventListener("change", (e) => {
    const t = e.target;
    if (t.dataset.k) { gcps[+t.dataset.i][t.dataset.k] = t.value === "" ? null : +t.value; changed(); }
  });
  $("#gr-table", view).addEventListener("click", (e) => {
    const b = e.target.closest("[data-rm]");
    if (b) { gcps.splice(+b.dataset.rm, 1); changed(); }
  });
  $("#gr-clear", view).addEventListener("click", () => { gcps = []; changed(); });
  $("#gr-find", view).addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#gr-search", view).value.trim();
    if (!q) return;
    try {
      const res = await (await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`)).json();
      if (!res.length) { toast(`No place found for “${q}”`); return; }
      const [s, n, w, ea] = res[0].boundingbox.map(Number);
      world.fitBounds([[s, w], [n, ea]]);
    } catch (err) { toast("Place search unavailable; pan the map manually"); }
  });
  fileSel.addEventListener("change", loadFile);
  resultIdle();
  loadFile();
}

/* ---------- Object detection ---------- */
function renderDetect(view, tool) {
  if (!filesOf(tool.needs).length) { view.innerHTML = pageHead(tool) + emptyFiles(tool.needs); return; }
  view.innerHTML = pageHead(tool) + `<div class="tool">
    <div class="panel card card-pad">
      <h3 class="section">Input</h3>${fileSelect("de-file", tool.needs, "Image")}
      ${slider("de-conf", "Confidence threshold", 0.05, 0.95, 0.05, 0.25)}${slider("de-iou", "IoU threshold", 0.1, 0.9, 0.05, 0.45)}
      <button class="btn primary block" id="de-run">${icon("scan")} Detect objects</button>
      <p class="hint" style="margin:14px 0 0">Uses the stock YOLOv8n COCO model (vehicles, boats, planes…). A satellite-trained model gives better aerial results.</p>
    </div>
    <div class="results" id="de-out">${placeholder("Find objects", "Run detection to get an annotated image and a table of detections.", "scan")}</div></div>`;
  bindSliders(view);
  $("#de-run", view).addEventListener("click", (e) => busy(e.currentTarget, async () => {
    const out = $("#de-out", view);
    out.innerHTML = skeleton("Running YOLOv8…");
    const r = await api("/api/detect", { file_id: $("#de-file", view).value, conf: +$("#de-conf", view).value, iou: +$("#de-iou", view).value })
      .catch((err) => { out.innerHTML = placeholder("Detection failed", err.message, "alert"); throw err; });
    out.innerHTML = `
      <div class="stats">${statTile("Objects", r.detections.length)}${r.counts.slice(0, 5).map((c) => statTile(c.name, c.count)).join("")}</div>
      ${r.detections.length ? "" : `<div class="notice info">${icon("info")}<div>No objects found at this confidence. Lower the threshold, or use a model trained on aerial imagery.</div></div>`}
      <div class="card"><div class="card-head"><h3>Annotated image</h3>${downloadsHtml(r.downloads)}</div><div class="figure"><img src="${r.image}" alt="Detections"></div></div>
      ${r.detections.length ? `<div class="card"><div class="card-head"><h3>Detections</h3></div><div class="table-scroll"><table class="data">
        <thead><tr><th>Class</th><th>Confidence</th><th>Box (x1, y1, x2, y2)</th></tr></thead><tbody>
        ${r.detections.map((d) => `<tr><td>${esc(d.class_name)}</td><td>${num(d.confidence, 2)}</td><td>${d.bbox.map((v) => Math.round(v)).join(", ")}</td></tr>`).join("")}
        </tbody></table></div></div>` : ""}`;
  }));
}

/* ================= Router & shell ================= */
const PAGES = { "": renderHome, data: renderData, ndvi: renderNdvi, crop: renderCrop, landuse: renderLanduse, gps: renderGps, pollution: renderPollution, georef: renderGeoref, detect: renderDetect };

function renderNav(active) {
  const link = (href, ico, label, id) => `<a class="nav-link ${active === id ? "active" : ""}" href="${href}">${icon(ico)}<span>${esc(label)}</span></a>`;
  const groups = [...new Set(TOOLS.map((t) => t.group))];
  $("#nav").innerHTML = `<div class="nav-group">${link("#/", "home", "Home", "")}${link("#/data", "database", "Data library", "data")}</div>`
    + groups.map((g) => `<div class="nav-group"><div class="nav-label">${esc(g)}</div>${TOOLS.filter((t) => t.group === g).map((t) => link(`#/${t.id}`, t.icon, t.title, t.id)).join("")}</div>`).join("");
}

async function route() {
  const id = location.hash.replace(/^#\/?/, "").split("?")[0];
  const page = PAGES[id] ? id : "";
  state.charts.forEach((c) => c.destroy());
  state.maps.forEach((m) => m.remove());
  state.charts = []; state.maps = [];
  renderNav(page);
  const tool = TOOLS.find((t) => t.id === page);
  $("#crumb").textContent = tool ? tool.title : page === "data" ? "Data library" : "GeoAI Tools";
  document.title = tool ? `${tool.title} · GeoAI Tools` : "GeoAI Tools";
  $(".app").classList.remove("nav-open");
  const view = $("#view");
  await PAGES[page](view, tool);
  window.scrollTo(0, 0);
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("geoai-theme", t); } catch (e) { /* ignore */ }
  $("#theme-toggle").innerHTML = icon(t === "dark" ? "sun" : "moon");
}

async function init() {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("#theme-toggle").innerHTML = icon(dark ? "sun" : "moon");
  $("#theme-toggle").addEventListener("click", () => {
    const now = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    setTheme(now ? "light" : "dark");
    route(); // re-render so charts/maps pick up theme colours
  });
  $("#menu-btn").innerHTML = icon("menu");
  $("#menu-btn").addEventListener("click", () => $(".app").classList.add("nav-open"));
  $("#scrim").addEventListener("click", () => $(".app").classList.remove("nav-open"));
  try { state.config = await api("/api/config"); } catch (e) { toast("Cannot reach the GeoAI server. Is it running?"); return; }
  await refreshFiles();
  window.addEventListener("hashchange", route);
  route();
}
init();
