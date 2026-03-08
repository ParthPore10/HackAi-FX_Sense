const liveTable = document.querySelector("#live-table tbody");
const signalsTable = document.querySelector("#signals-table tbody");
const pulseGrid = document.getElementById("pulse");

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed ${url}`);
  return res.json();
}

function renderPulse(rows) {
  const counts = { USD: 0, EUR: 0, GBP: 0, JPY: 0, AUD: 0, INR: 0, CNY: 0 };
  rows.forEach((r) => {
    const bias = r.currency_bias || "";
    const events = (r.events || "").toString().trim();
    if (!events) return;
    Object.keys(counts).forEach((ccy) => {
      if (bias.includes(ccy)) counts[ccy] += 1;
    });
  });
  pulseGrid.innerHTML = "";
  Object.keys(counts).forEach((ccy) => {
    const val = counts[ccy];
    const cls = val > 0 ? "pill-pos" : "pill-neu";
    const div = document.createElement("div");
    div.className = `pulse-pill ${cls}`;
    div.innerHTML = `${ccy} ${val}`;
    pulseGrid.appendChild(div);
  });
}

function renderLive(rows, ts) {
  if (liveTable) liveTable.innerHTML = "";
  const tickerTrack = document.getElementById("ticker-track");
  let tickerHtml = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.pair}</td>
      <td>${r.last ?? "-"}</td>
      <td>${r.change ?? "-"}</td>
      <td>${r.change_pct ?? "-"}</td>
      <td>${r.timestamp ?? ts}</td>
    `;
    if (liveTable) liveTable.appendChild(tr);

    const cls = r.change_pct > 0 ? "up" : r.change_pct < 0 ? "down" : "";
    tickerHtml += `<span class="ticker-item">${r.pair} ${r.last ?? "-"} <span class="${cls}">${r.change_pct ?? "-"}%</span></span>`;
  });
  const tsEl = document.getElementById("live-updated");
  if (tsEl) tsEl.textContent = `Updated ${ts}`;
  if (tickerTrack) {
    // duplicate for seamless loop
    tickerTrack.innerHTML = tickerHtml + tickerHtml;
    // restart animation
    tickerTrack.style.animation = "none";
    tickerTrack.offsetHeight;
    tickerTrack.style.animation = "";
  }
}

function renderSignals(rows) {
  signalsTable.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const sent = (r.sentiment || "").toLowerCase();
    const sentClass = sent === "positive" ? "pill-green" : sent === "negative" ? "pill-red" : "pill-yellow";
    const trade = r.trade_suggestion || "";
    const tradeClass = trade.toLowerCase().includes("long") ? "pill-green" : trade.toLowerCase().includes("short") ? "pill-red" : "pill-yellow";
    tr.innerHTML = `
      <td>${r.source ?? ""}</td>
      <td>${r.headline ?? ""}</td>
      <td>${r.topic ?? ""}</td>
      <td><span class="badge-pill ${sentClass}">${r.sentiment ?? ""}</span></td>
      <td>${r.events ?? ""}</td>
      <td>${r.macro_interpretation ?? ""}</td>
      <td><span class="badge-pill pill-blue">${r.currency_bias ?? ""}</span></td>
      <td><span class="badge-pill ${tradeClass}">${r.trade_suggestion ?? ""}</span></td>
      <td>${r.signal_confidence ?? ""}</td>
    `;
    signalsTable.appendChild(tr);
  });
  renderPulse(rows);
  renderTradeStrip(rows);
}

function drawLine(canvas, rows) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(260, rect.width);
  const height = 180;
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);

  ctx.clearRect(0, 0, width, height);
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, "#0b0b0b");
  grad.addColorStop(1, "#0a0a0a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  if (!rows || rows.length === 0) {
    ctx.fillStyle = "#9fb0c8";
    ctx.font = "12px Space Grotesk";
    ctx.fillText("No data", 10, 20);
    return;
  }

  const values = rows.map((r) => r.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad;
  const hi = max + pad;

  ctx.strokeStyle = "#141414";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = (height / 4) * i;
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#8de1ff";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = (i / (rows.length - 1)) * (width - 10) + 5;
    const y = height - ((r.v - lo) / (hi - lo)) * (height - 10) - 5;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "#9a9a9a";
  ctx.font = "10px JetBrains Mono";
  ctx.textAlign = "left";
  ctx.fillText(rows[0]?.t?.slice(0, 10) || "", 6, height - 6);
  ctx.textAlign = "right";
  ctx.fillText(rows[rows.length - 1]?.t?.slice(0, 10) || "", width - 6, height - 6);

  ctx.textAlign = "left";
  ctx.fillText(hi.toFixed(3), 6, 12);
  ctx.fillText(lo.toFixed(3), 6, height - 18);
}

let allPairs = [];
let extraPair = null;

function renderChartsStack(pairs) {
  const container = document.getElementById("charts-stack");
  if (!container) return;
  container.innerHTML = "";
  pairs.forEach((pair) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    const title = document.createElement("div");
    title.className = "chart-title";
    title.textContent = pair;
    const canvas = document.createElement("canvas");
    const id = `chart-${pair.replace(/[^a-zA-Z0-9]/g, "")}`;
    canvas.id = id;
    card.appendChild(title);
    card.appendChild(canvas);
    container.appendChild(card);
  });
}

async function initCharts() {
  const { pairs } = await fetchJSON("/api/pairs");
  allPairs = pairs || [];
  const base = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"];
  const selected = base.filter((p) => allPairs.includes(p));
  if (extraPair && !selected.includes(extraPair)) selected.push(extraPair);
  renderChartsStack(selected);

  for (const pair of selected) {
    const id = `chart-${pair.replace(/[^a-zA-Z0-9]/g, "")}`;
    const canvas = document.getElementById(id);
    if (!canvas) continue;
    const hist = await fetchJSON(`/api/history?pair=${encodeURIComponent(pair)}`);
    drawLine(canvas, hist.rows || []);
  }

  const sel = document.getElementById("chart-pair-select");
  if (sel) {
    sel.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = "Select pair...";
    sel.appendChild(opt0);
    allPairs.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
    sel.onchange = async (e) => {
      const value = e.target.value;
      if (!value) return;
      extraPair = value;
      await initCharts();
    };
  }
}

async function refreshLive() {
  try {
    const data = await fetchJSON("/api/live?limit=20");
    renderLive(data.rows, data.last_update);
  } catch (e) {
    console.error(e);
  }
}

async function refreshSignals() {
  try {
    const data = await fetchJSON("/api/signals");
    renderSignals(data.rows || []);
  } catch (e) {
    console.error(e);
  }
}

// banner rotation is driven by country news now

async function start() {
  await initCharts();
  await refreshLive();
  await refreshSignals();
  await refreshCountryNews();
  setInterval(refreshLive, 60000);
  setInterval(refreshSignals, 90000);
  setInterval(refreshCountryNews, 300000);
}

start();

function renderTradeStrip(rows) {
  const strip = document.getElementById("trade-strip");
  if (!strip) return;
  strip.innerHTML = "";
  if (!rows || rows.length === 0) return;
  rows
    .filter((r) => {
      const t = (r.trade_suggestion || "").toLowerCase();
      return t && t !== "no trade";
    })
    .slice(0, 6)
    .forEach((r) => {
    const action = (r.trade_suggestion || "").toLowerCase();
    const isBuy = action.includes("long") || action.includes("buy");
    const isSell = action.includes("short") || action.includes("sell");
    const cls = isBuy ? "buy" : isSell ? "sell" : "";
    const card = document.createElement("div");
    card.className = `trade-card ${cls}`;
    card.innerHTML = `
      <div class="label">${r.topic || "Signal"}</div>
      <div class="action">${r.trade_suggestion || "No trade"}</div>
      <div class="muted">${r.currency_bias || ""}</div>
    `;
    strip.appendChild(card);
  });
}

async function refreshCountryNews() {
  const container = document.getElementById("country-news");
  if (!container) return;
  try {
    const data = await fetchJSON("/api/country_news?limit=4");
    const news = data.news || {};
    container.innerHTML = "";
    const bannerItems = [];
    Object.keys(news).forEach((ccy) => {
      const card = document.createElement("div");
      card.className = "news-card";
      const items = news[ccy] || [];
      if (items[0]) bannerItems.push({ ccy, title: items[0].title });
      const list = items
        .map(
          (n) =>
            `<div class="news-item"><a href="${n.link}" target="_blank" rel="noopener">${n.title}</a></div>`
        )
        .join("");
      card.innerHTML = `<div class="label">${ccy} Headlines</div>${list || "<div class='news-item'>No feed configured.</div>"}`;
      container.appendChild(card);
    });

    const banner = document.getElementById("alert-banner");
    if (banner && bannerItems.length > 0) {
      let idx = 0;
      const bubble = document.createElement("div");
      bubble.className = "news-bubble";
      const ccy = document.createElement("span");
      ccy.className = "ccy";
      const text = document.createElement("span");
      text.className = "text";
      bubble.appendChild(ccy);
      bubble.appendChild(text);
      banner.innerHTML = "";
      banner.appendChild(bubble);

      const update = () => {
        const item = bannerItems[idx % bannerItems.length];
        bubble.classList.add("fade");
        setTimeout(() => {
          ccy.textContent = item.ccy;
          text.textContent = item.title || "";
          bubble.classList.remove("fade");
        }, 200);
        idx += 1;
      };
      update();
      if (!window.__bannerTimer) {
        window.__bannerTimer = setInterval(update, 5000);
      }
    }
  } catch (e) {
    console.error(e);
  }
}

// Map
async function initMap() {
  const mapEl = document.getElementById("fx-map");
  if (!mapEl || typeof L === "undefined") return;
  const map = L.map("fx-map", { zoomControl: false, attributionControl: false }).setView([20, 0], 2);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 6,
  }).addTo(map);

  try {
    const data = await fetchJSON("/api/map_points");
    (data.points || []).forEach((p) => {
      const marker = L.circleMarker([p.lat, p.lon], {
        radius: p.size || 6,
        color: p.color || "#2ee6a6",
        weight: 2,
        fillColor: p.color || "#2ee6a6",
        fillOpacity: 0.6,
      }).addTo(map);
      if (p.label) marker.bindTooltip(p.label, { direction: "top" });
    });
  } catch (e) {
    console.error(e);
  }

  // Tankers layer (AISStream)
  try {
    const data = await fetchJSON("/api/ais_tankers?limit=300");
    (data.tankers || []).forEach((t) => {
      const marker = L.circleMarker([t.lat, t.lon], {
        radius: 4,
        color: "#ff9f1c",
        weight: 1,
        fillColor: "#ff9f1c",
        fillOpacity: 0.6,
      }).addTo(map);
      marker.bindTooltip(`${t.name || t.mmsi} · ${t.speed || 0} kn`, { direction: "top" });
    });
  } catch (e) {
    console.error(e);
  }

}

initMap();

document.addEventListener("DOMContentLoaded", () => {
  // Live news tabs
  const liveTabs = document.getElementById("live-tabs");
  const liveIframe = document.getElementById("live-iframe");
  const liveFallback = document.getElementById("live-fallback");
  const liveLink = document.getElementById("live-link");
  const toggleBtn = document.getElementById("toggle-play");
  const fullscreenBtn = document.getElementById("fullscreen");
  let paused = false;

  async function setLiveChannel(channel) {
    if (!liveIframe) return;
    try {
      const data = await fetchJSON(`/api/youtube_live?channel=${channel}`);
      if (data.embed_url) {
        liveIframe.src = data.embed_url;
        if (liveFallback) liveFallback.style.display = "none";
      } else {
        if (liveFallback) liveFallback.style.display = "grid";
      }
      if (liveLink && data.live_url) liveLink.href = data.live_url;
    } catch (e) {
      if (liveFallback) liveFallback.style.display = "grid";
    }
  }

  if (liveTabs && liveIframe) {
    liveTabs.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab");
      if (!btn) return;
      liveTabs.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      const channel = btn.getAttribute("data-channel");
    if (channel) setLiveChannel(channel);
    });
    setLiveChannel("cnbc");
  }

  if (toggleBtn && liveIframe) {
    toggleBtn.addEventListener("click", () => {
      paused = !paused;
      if (paused) {
        liveIframe.src = "about:blank";
        document.querySelector(".live-status").innerHTML = '<span class="dot"></span>PAUSED';
      } else {
        const active = liveTabs?.querySelector(".tab.active");
        const channel = active?.getAttribute("data-channel") || "cnbc";
        setLiveChannel(channel);
        document.querySelector(".live-status").innerHTML = '<span class="dot"></span>LIVE';
      }
    });
  }

  if (fullscreenBtn && liveIframe) {
    fullscreenBtn.addEventListener("click", () => {
      if (liveIframe.requestFullscreen) liveIframe.requestFullscreen();
    });
  }
});
