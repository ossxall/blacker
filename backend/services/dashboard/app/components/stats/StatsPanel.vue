<script setup lang="ts">
// BLACKER
// Copyright (C) 2026 Juan José Caballero Rey
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

import Chart from "chart.js/auto";
import { onBeforeUnmount, onMounted, ref } from "vue";

// -----------------------------------------------------------------------------
// Formatadores
// -----------------------------------------------------------------------------

const money = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const fm = (v: number) => (v >= 0 ? "+" : "") + money.format(v);
const fp = (v: number) => money.format(v);
const dt = (v: number) =>
  new Intl.DateTimeFormat("es-ES", {
    timeZone: "UTC",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(v));

interface NormalizedTrade {
  n: number;
  side: string;
  entry: number;
  exit: number;
  entryTime: number;
  exitTime: number;
  pnl: number;
  qty: number;
  cumPnl: number;
  durationMin: number;
}

interface Dashboard {
  symbol: string;
  trades: NormalizedTrade[];
  summary: {
    realizedPnl: number;
    cash: number;
    initialCash: number;
    winRate: number;
    profitFactor: number;
    maxDrawdown: number;
    maxWin: number;
    maxLoss: number;
    avgPnl: number;
  };
  openPosition: {
    side: string;
    quantity: number;
    avg_price: number;
  } | null;
}

interface Props {
  /** URL del JSON (por defecto: state.json junto al componente). Se ignora si se pasa `dashboard`. */
  src?: string;
  /** Payload bruto en memoria (`MasterState`) en lugar de cargar `state.json`. */
  dashboard?: any;
}

const props = withDefaults(defineProps<Props>(), { src: "state.json" });

// -----------------------------------------------------------------------------
// Refs de elementos del DOM
// -----------------------------------------------------------------------------

const subEl = ref<HTMLElement>();
const badgeEl = ref<HTMLElement>();
const pnlEl = ref<HTMLElement>();
const cashEl = ref<HTMLElement>();
const wrEl = ref<HTMLElement>();
const pfEl = ref<HTMLElement>();
const ddEl = ref<HTMLElement>();
const bestEl = ref<HTMLElement>();
const worstEl = ref<HTMLElement>();
const avgEl = ref<HTMLElement>();
const wrSmallEl = ref<HTMLElement>();
const longShortEl = ref<HTMLElement>();
const barEl = ref<HTMLCanvasElement>();
const cumEl = ref<HTMLCanvasElement>();
const drawdownEl = ref<HTMLCanvasElement>();
const priceEl = ref<HTMLCanvasElement>();
const bodyEl = ref<HTMLElement>();

const errorMsg = ref<string>("");

let charts: Chart[] = [];

// -----------------------------------------------------------------------------
// Normalización del JSON
// -----------------------------------------------------------------------------

/**
 * Acepta tanto el JSON ya simplificado (`{ trades, summary }`) como la
 * estructura completa de backend (`{ master: { engine_state: { portfolio } } }`).
 */
function normalize(raw: any): Dashboard {
  if (raw && Array.isArray(raw.trades) && raw.summary) {
    if (raw.summary.initialCash === undefined) {
      // fallback si el JSON ya viene simplificado
      raw.summary.initialCash = raw.summary.cash - raw.summary.realizedPnl;
    }
    return raw;
  }
  const master = raw && (raw.master || raw);
  const engine = master && master.engine_state;
  const portfolio = engine && engine.portfolio;
  if (!portfolio || !Array.isArray(portfolio.trades)) {
    throw new Error(
      "estructura de JSON no reconocida (claves top-level: " +
        Object.keys(raw || {}).join(", ") +
        ")",
    );
  }
  const symbol = master.symbol || raw.symbol || "";
  let cum = 0,
    peak = 0,
    maxDD = 0;
  const trades: NormalizedTrade[] = portfolio.trades.map((t: any, i: number) => {
    cum += t.pnl;
    peak = Math.max(peak, cum);
    maxDD = Math.min(maxDD, cum - peak);
    return {
      n: i + 1,
      side: t.side,
      entry: t.entry_price,
      exit: t.exit_price,
      entryTime: t.entry_time,
      exitTime: t.exit_time,
      pnl: t.pnl,
      qty: t.quantity,
      cumPnl: cum,
      durationMin: (t.exit_time - t.entry_time) / 60000,
    };
  });
  const pnls = trades.map((t) => t.pnl);
  const wins = pnls.filter((p) => p >= 0),
    losses = pnls.filter((p) => p < 0);
  const grossProfit = wins.reduce((a, b) => a + b, 0),
    grossLoss = Math.abs(losses.reduce((a, b) => a + b, 0));
  const summary: Dashboard["summary"] = {
    realizedPnl: portfolio.realized_pnl,
    cash: portfolio.cash,
    initialCash: portfolio.initial_cash,
    winRate: trades.length ? (wins.length / trades.length) * 100 : 0,
    profitFactor: grossLoss
      ? grossProfit / grossLoss
      : grossProfit > 0
        ? Infinity
        : 0,
    maxDrawdown: maxDD,
    maxWin: pnls.length ? Math.max(...pnls) : 0,
    maxLoss: pnls.length ? Math.min(...pnls) : 0,
    avgPnl: pnls.length ? pnls.reduce((a, b) => a + b, 0) / pnls.length : 0,
  };
  return {
    symbol,
    trades,
    summary,
    openPosition: portfolio.position || null,
  };
}

// -----------------------------------------------------------------------------
// Render
// -----------------------------------------------------------------------------

function render(D: Dashboard) {
  if (D.symbol) {
    const extra =
      D.openPosition === null || !D.openPosition
        ? ""
        : " · posición abierta: " +
          D.openPosition.side +
          " " +
          D.openPosition.quantity +
          " @ " +
          money.format(D.openPosition.avg_price);
    subEl.value!.innerHTML = "<b>" + D.symbol + "</b> · solo operaciones cerradas del portfolio" + extra;
  }

  badgeEl.value!.textContent = D.trades.length + " trades";
  pnlEl.value!.textContent = fm(D.summary.realizedPnl);
  cashEl.value!.textContent = money.format(D.summary.cash);
  wrEl.value!.textContent = D.summary.winRate.toFixed(1) + "%";
  pfEl.value!.textContent = D.summary.profitFactor.toFixed(2);
  ddEl.value!.textContent = fm(D.summary.maxDrawdown);
  bestEl.value!.textContent = fm(D.summary.maxWin);
  worstEl.value!.textContent = fm(D.summary.maxLoss);
  avgEl.value!.textContent = fm(D.summary.avgPnl);

  const winners = D.trades.filter((t) => t.pnl >= 0).length,
    losers = D.trades.length - winners;
  wrSmallEl.value!.textContent = winners + " ganadores / " + losers + " perdedores";

  const longs = D.trades.filter((t) => t.side === "BUY").length,
    shorts = D.trades.length - longs;
  longShortEl.value!.textContent = longs + " / " + shorts;

  const labels = D.trades.map((t) => "#" + t.n);
  const pnl = D.trades.map((t) => t.pnl);
  const cumSeries = D.trades.map((t) => t.cumPnl);

  // ---------------------------------------------------------------------------
  // Gráfico 1 · PnL por trade (barras)
  // ---------------------------------------------------------------------------
  const barOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#cbd5e1" } },
      tooltip: { callbacks: { label: (c: any) => " " + fm(c.parsed.y) } },
    },
    scales: {
      x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.05)" } },
      y: {
        ticks: { color: "#94a3b8", callback: (v: number) => money.format(v) },
        grid: { color: "rgba(255,255,255,.07)" },
      },
    },
  };
  charts.push(
    new Chart(barEl.value!, {
      type: "bar",
      data: { labels, datasets: [{ label: "PnL", data: pnl }] },
      options: barOpts,
    }),
  );

  // ---------------------------------------------------------------------------
  // Gráfico 2 · PnL acumulado y curva de equity (doble eje)
  // ---------------------------------------------------------------------------
  const equity = D.trades.map((t) => D.summary.initialCash + t.cumPnl);
  const cumOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#cbd5e1" } },
      tooltip: {
        callbacks: {
          label: (c: any) =>
            " " + c.dataset.label + ": " + (c.dataset.yAxisID === "y1" ? money.format(c.parsed.y) : fm(c.parsed.y)),
        },
      },
    },
    scales: {
      x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.05)" } },
      y: {
        position: "left",
        title: { display: true, text: "PnL acumulado", color: "#94a3b8" },
        ticks: { color: "#94a3b8", callback: (v: number) => money.format(v) },
        grid: { color: "rgba(255,255,255,.07)" },
      },
      y1: {
        position: "right",
        title: { display: true, text: "Equity", color: "#94a3b8" },
        ticks: { color: "#94a3b8", callback: (v: number) => money.format(v) },
        grid: { display: false },
      },
    },
  };
  charts.push(
    new Chart(cumEl.value!, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "PnL acumulado", data: cumSeries, yAxisID: "y", tension: 0.25, pointRadius: 4, borderColor: "#6ea8fe" },
          { label: "Equity", data: equity, yAxisID: "y1", tension: 0.25, pointRadius: 4, borderColor: "#35d07f" },
        ],
      },
      options: cumOpts,
    }),
  );

  // ---------------------------------------------------------------------------
  // Gráfico 3 · Entrada → salida de cada trade (scatter)
  // ---------------------------------------------------------------------------
  const sets = D.trades.map((t) => ({
    label: "Trade #" + t.n,
    data: [
      { x: t.entryTime, y: t.entry },
      { x: t.exitTime, y: t.exit },
    ],
    showLine: true,
    tension: 0,
    pointRadius: 5,
    borderWidth: 2,
  }));
  charts.push(
    new Chart(priceEl.value!, {
      type: "scatter",
      data: { datasets: sets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#94a3b8", callback: (v: any) => dt(v) },
            grid: { color: "rgba(255,255,255,.05)" },
          },
          y: {
            ticks: { color: "#94a3b8", callback: (v: any) => money.format(v) },
            grid: { color: "rgba(255,255,255,.07)" },
          },
        },
      },
    }),
  );

  // ---------------------------------------------------------------------------
  // Gráfico 4 · Curva de equity con drawdown
  // ---------------------------------------------------------------------------
  const initial = D.summary.initialCash;
  const eq = D.trades.map((t) => initial + t.cumPnl);
  const isDown = (i: number) => eq[i] < initial;
  let pk = Math.max(eq[0] || initial, initial),
    mddPct = 0;
  eq.forEach((v) => {
    pk = Math.max(pk, v);
    if (v < pk) mddPct = Math.min(mddPct, (v - pk) / pk);
  });
  const mddTxt = "Máx. drawdown: " + Math.abs(mddPct * 100).toFixed(2) + "%";
  charts.push(
    new Chart(drawdownEl.value!, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Breakeven (capital inicial)",
            data: eq.map(() => initial),
            borderColor: "rgba(148,163,184,.8)",
            borderDash: [6, 4],
            borderWidth: 1,
            pointRadius: 0,
          },
          {
            label: "Equity · " + mddTxt,
            data: eq,
            tension: 0.25,
            pointRadius: 3,
            fill: false,
            borderColor: "#6ea8fe",
            pointBackgroundColor: (ctx: any) => (isDown(ctx.dataIndex) ? "#ff5d73" : "#6ea8fe"),
            pointBorderColor: (ctx: any) => (isDown(ctx.dataIndex) ? "#ff5d73" : "#6ea8fe"),
            segment: {
              borderColor: (ctx: any) =>
                isDown(ctx.p0DataIndex) || isDown(ctx.p1DataIndex) ? "#ff5d73" : "#6ea8fe",
            },
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#cbd5e1" }, align: "end" },
          tooltip: {
            callbacks: {
              label: (c: any) =>
                " " + c.dataset.label + ": " + money.format(c.parsed.y) + (isDown(c.dataIndex) ? " (en drawdown)" : ""),
            },
          },
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.05)" } },
          y: {
            ticks: { color: "#94a3b8", callback: (v: any) => money.format(v) },
            grid: { color: "rgba(255,255,255,.07)" },
          },
        },
      },
    }),
  );

  // ---------------------------------------------------------------------------
  // Tabla · Detalle de operaciones
  // ---------------------------------------------------------------------------
  bodyEl.value!.innerHTML = D.trades
    .map((t) => {
      const pc = t.pnl >= 0 ? "win" : "loss",
        sc = t.side === "BUY" ? "buy" : "sell",
        dur =
          t.durationMin < 60
            ? t.durationMin.toFixed(1) + " min"
            : (t.durationMin / 60).toFixed(1) + " h";
      return (
        "<td>" +
        t.n +
        "</td><td class=\"" +
        sc +
        "\">" +
        t.side +
        "</td><td>" +
        fp(t.entry) +
        "</td><td>" +
        fp(t.exit) +
        "</td><td>" +
        dur +
        "</td><td>" +
        t.qty +
        "</td><td class=\"" +
        pc +
        "\">" +
        fm(t.pnl) +
        "</td><td class=\"" +
        (t.cumPnl >= 0 ? "win" : "loss") +
        "\">" +
        fm(t.cumPnl) +
        "</td><td>" +
        dt(t.entryTime) +
        "</td><td>" +
        dt(t.exitTime) +
        "</td>"
      );
    })
    .map((cells) => "<tr>" + cells + "</tr>")
    .join("");
}

// -----------------------------------------------------------------------------
// Ciclo de vida
// -----------------------------------------------------------------------------

function load() {
  try {
    if (props.dashboard) {
      render(normalize(props.dashboard));
      return;
    }
  } catch (err: any) {
    errorMsg.value = "No se pudieron interpretar los datos del dashboard: " + err.message;
    return;
  }
  fetch(props.src)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
    .then((json) => render(normalize(json)))
    .catch((err: any) => {
      errorMsg.value =
        "No se pudo cargar " +
        props.src +
        ": " +
        err.message +
        " — asegúrate de servirlo por HTTP y de que " +
        props.src +
        " esté en la misma carpeta.";
    });
}

onMounted(load);
onBeforeUnmount(() => {
  charts.forEach((c) => c.destroy());
  charts = [];
});
</script>

<template>
  <div class="wrap">
    <div v-if="!errorMsg" class="dash">
      <header>
        <div>
          <h1>Backtesting · Trades ejecutados</h1>
          <div ref="subEl" class="sub"></div>
        </div>
        <div ref="badgeEl" class="badge"></div>
      </header>

      <section class="grid">
        <div class="card">
          <div class="label">PnL realizado</div>
          <div ref="pnlEl" class="value"></div>
          <div class="small">Cash final: <span ref="cashEl"></span></div>
        </div>
        <div class="card">
          <div class="label">Win rate</div>
          <div ref="wrEl" class="value"></div>
          <div class="small" ref="wrSmallEl"></div>
        </div>
        <div class="card">
          <div class="label">Profit factor</div>
          <div ref="pfEl" class="value"></div>
          <div class="small">Ganancia bruta / pérdida bruta</div>
        </div>
        <div class="card">
          <div class="label">Máximo drawdown</div>
          <div ref="ddEl" class="value"></div>
          <div class="small">Sobre PnL acumulado</div>
        </div>
      </section>

      <div class="charts">
        <section class="panel">
          <h2>PnL por trade</h2>
          <div class="chartbox">
            <canvas ref="barEl"></canvas>
          </div>
        </section>
        <section class="panel">
          <h2>PnL acumulado y curva de equity</h2>
          <div class="chartbox">
            <canvas ref="cumEl"></canvas>
          </div>
        </section>
      </div>

      <section class="panel">
        <h2>Entrada → salida de cada trade</h2>
        <div class="chartbox">
          <canvas ref="priceEl"></canvas>
        </div>
      </section>

      <section class="panel">
        <h2>Curva de equity</h2>
        <div class="chartbox">
          <canvas ref="drawdownEl"></canvas>
        </div>
      </section>

      <section class="stats">
        <div class="stat">
          <div class="label">Mejor trade</div>
          <strong ref="bestEl"></strong>
        </div>
        <div class="stat">
          <div class="label">Peor trade</div>
          <strong ref="worstEl"></strong>
        </div>
        <div class="stat">
          <div class="label">Pnl medio</div>
          <strong ref="avgEl"></strong>
        </div>
        <div class="stat">
          <div class="label">Long / Short</div>
          <strong ref="longShortEl"></strong>
        </div>
      </section>

      <section class="panel">
        <h2>Detalle de operaciones</h2>
        <div class="tablewrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Lado</th>
                <th>Entrada</th>
                <th>Salida</th>
                <th>Duración</th>
                <th>Qty</th>
                <th>PnL</th>
                <th>Acumulado</th>
                <th>Entrada UTC</th>
                <th>Salida UTC</th>
              </tr>
            </thead>
            <tbody ref="bodyEl"></tbody>
          </table>
        </div>
      </section>
    </div>

    <div v-else class="errmsg">{{ errorMsg }}</div>
  </div>
</template>

<style scoped>
.wrap {
  padding: 28px;
  max-width: 1500px;
  margin: auto;
}
header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 20px;
}
h1 {
  margin: 0;
  font-size: 28px;
}
.sub {
  color: #94a3b8;
  margin-top: 6px;
}
.badge {
  border: 1px solid #26324b;
  background: var(--p, #11182b);
  border-radius: 99px;
  padding: 8px 12px;
  white-space: nowrap;
}
.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}
.card {
  background: var(--p, #11182b);
  border: 1px solid #26324b;
  border-radius: 14px;
  padding: 16px;
}
.label {
  color: #94a3b8;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.value {
  font-size: 24px;
  font-weight: 750;
  margin-top: 7px;
}
.small {
  color: #94a3b8;
  font-size: 12px;
  margin-top: 5px;
}
.charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 16px;
}
.panel {
  background: var(--p, #11182b);
  border: 1px solid #26324b;
  border-radius: 14px;
  padding: 16px;
  margin-top: 16px;
}
.charts .panel {
  margin-top: 0;
}
.panel h2 {
  font-size: 16px;
  margin: 0 0 12px;
}
.chartbox {
  height: 330px;
  position: relative;
}
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 16px;
}
.stat {
  background: var(--p, #11182b);
  border: 1px solid #26324b;
  border-radius: 14px;
  padding: 13px;
}
.stat strong {
  display: block;
  font-size: 17px;
  margin-top: 5px;
}
:deep(.tablewrap) {
  overflow: auto;
}
:deep(table) {
  width: 100%;
  min-width: 950px;
  border-collapse: collapse;
}
:deep(th),
:deep(td) {
  padding: 10px 11px;
  border-bottom: 1px solid #26324b;
  text-align: right;
  white-space: nowrap;
}
:deep(th:first-child),
:deep(td:first-child),
:deep(th:nth-child(2)),
:deep(td:nth-child(2)) {
  text-align: left;
}
:deep(th) {
  color: #94a3b8;
  font-size: 11px;
  text-transform: uppercase;
}
:deep(.win) {
  color: #35d07f;
  font-weight: 700;
}
:deep(.loss) {
  color: #ff5d73;
  font-weight: 700;
}
:deep(.buy) {
  color: #6ea8fe;
  font-weight: 700;
}
:deep(.sell) {
  color: #f6c453;
  font-weight: 700;
}
.errmsg {
  padding: 40px;
  text-align: center;
  color: #ff5d73;
}
@media (max-width: 900px) {
  .grid,
  .stats {
    grid-template-columns: repeat(2, 1fr);
  }
  .charts {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 600px) {
  .wrap {
    padding: 16px;
  }
  header {
    flex-direction: column;
  }
  .grid,
  .stats {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 23px;
  }
}
</style>
