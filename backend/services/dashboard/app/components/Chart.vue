<script setup lang="ts">
// BLACKER
// Copyright (C) 2026 Juan José Caballero Rey
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

import { onBeforeUnmount, onMounted, ref } from "vue";
import { createChart } from "@/packages/src/index";
import type { ChartOptions } from "~/packages/src/core/config";
import type {
  AnyChartSeries,
  ChartEngine,
  ChartEvent,
} from "~/packages/src/core/types";
import { seriesRegistry, type SeriesId, type SeriesKind } from "~/stores/tabs";

const props = defineProps<{
  timeframeId: string;
}>();

export interface Series {
  id: string;
  kind: string;
  level: number;
  primary: boolean;
  overlay: boolean;
  params: Record<string, unknown>;
  [key: string]: any;
}

export interface ChartTimeframe {
  id: string;
  series: Record<string, Series>;
  timeframe_ms: number;
  [key: string]: any;
}

interface RuntimeSeries extends Series {
  chart: ChartEngine;
  serie: AnyChartSeries;
  kind: SeriesKind;
}

/**
 * -------------------------------------------------------------------------
 * Runtime series belonging to this Chart.vue instance.
 * -------------------------------------------------------------------------
 */
const allSeries = new Map<SeriesId, RuntimeSeries>();

/**
 * -------------------------------------------------------------------------
 * The single ChartEngine owned by the primary series.
 *
 * All overlay series use this same ChartEngine.
 * -------------------------------------------------------------------------
 */
let primaryChart: ChartEngine | null = null;

/**
 * -------------------------------------------------------------------------
 * Main Chart container.
 * -------------------------------------------------------------------------
 */
const container = ref<HTMLDivElement | null>(null);

/**
 * -------------------------------------------------------------------------
 * All ChartEngines created by this component.
 *
 * There can be:
 *   - one primary ChartEngine
 *   - zero or more standalone ChartEngines
 *
 * Overlay series NEVER create their own ChartEngine.
 * -------------------------------------------------------------------------
 */
const charts = new Set<ChartEngine>();

/**
 * -------------------------------------------------------------------------
 * Subscribes to every chart event of a ChartEngine.
 * -------------------------------------------------------------------------
 */
const emit = defineEmits<{
  chart: [event: ChartEvent];
}>();

function _subscribeChart(chart: ChartEngine) {
  chart.subscribe((event) => {
    emit("chart", event);
  });
}

function cleanAllSeries() {
  const uniqueCharts = new Set<ChartEngine>();

  for (const runtime of allSeries.values()) {
    uniqueCharts.add(runtime.chart);
  }

  for (const chart of charts) {
    uniqueCharts.add(chart);
  }

  for (const runtime of allSeries.values()) {
    try {
      runtime.serie.destroy({ silent: true });
    } catch {}
  }

  for (const chart of uniqueCharts) {
    try {
      chart.api.destroy();
    } catch {}
  }

  allSeries.clear();
  charts.clear();
  primaryChart = null;

  container.value?.replaceChildren();
}

/**
 * -------------------------------------------------------------------------
 * Creates the DOM element used by a chart.
 * -------------------------------------------------------------------------
 */
function addChartContainer(
  seriesId: SeriesId,
  width: string,
  height: string,
): HTMLDivElement {
  if (!container.value) {
    throw new Error("Chart container is not mounted.");
  }

  const element = document.createElement("div");

  element.className = "chart-area";
  element.dataset.seriesId = seriesId;

  element.style.width = width;
  element.style.height = height;

  container.value.appendChild(element);

  return element;
}

/**
 * -------------------------------------------------------------------------
 * Builds the chart series configuration from a backend definition.
 * -------------------------------------------------------------------------
 */
function buildSeries(seriesId: SeriesId, seriesValue: Series) {
  const seriesFactory = seriesRegistry[seriesValue.kind as SeriesKind];

  if (!seriesFactory) {
    throw new Error(
      `Unknown series kind "${seriesValue.kind}" for series "${seriesId}".`,
    );
  }

  return seriesFactory({
    id: seriesId,
    label: seriesValue.params?.label as string,
    color: seriesValue.params?.color as string,
    layer: seriesValue.params?.layer as "background" | "foreground",
    priceTagColor: seriesValue.params?.priceTagColor as string,
    params: seriesValue.params as any,
  });
}

/**
 * -------------------------------------------------------------------------
 * Validates an individual series.
 *
 * The topology is:
 *
 *   primary: true
 *     -> primary chart
 *
 *   primary: false + overlay: true
 *     -> overlay on primary chart
 *
 *   primary: false + overlay: false
 *     -> standalone chart
 *
 * Therefore primary=false and overlay=false is VALID.
 * -------------------------------------------------------------------------
 */
function validateSeries(series: Series) {
  if (typeof series.primary !== "boolean") {
    throw new Error(`Series "${series.id}" must define primary as a boolean.`);
  }

  if (typeof series.overlay !== "boolean") {
    throw new Error(`Series "${series.id}" must define overlay as a boolean.`);
  }
}

/**
 * -------------------------------------------------------------------------
 * Validates the complete timeframe topology.
 *
 * There can be at most one primary.
 *
 * Zero primaries is a legitimate transitional state: the primary
 * is added by the user after the UI has started.
 * -------------------------------------------------------------------------
 */
function validateLayout(definitions: Record<string, Series>) {
  let primaryCount = 0;

  for (const series of Object.values(definitions)) {
    validateSeries(series);

    if (series.primary) {
      primaryCount++;
    }
  }

  if (primaryCount > 1) {
    throw new Error("At most one primary series is required.");
  }
}

/**
 * -------------------------------------------------------------------------
 * Creates the runtime primary series.
 *
 * The primary owns the main ChartEngine.
 * -------------------------------------------------------------------------
 */
function createPrimarySeries(seriesId: SeriesId, seriesValue: Series) {
  if (primaryChart) {
    throw new Error(
      `Primary series already exists. Cannot create "${seriesId}".`,
    );
  }

  const build = buildSeries(seriesId, seriesValue);

  const chart = createChart(
    addChartContainer(seriesId, build.width, build.height),
  );

  _subscribeChart(chart);

  const serie = chart.api.addSeries(build);

  primaryChart = chart;

  charts.add(chart);

  allSeries.set(seriesId, {
    ...seriesValue,
    chart,
    serie,
    kind: seriesValue.kind as SeriesKind,
  });
}

/**
 * -------------------------------------------------------------------------
 * Creates the runtime overlay series.
 *
 * Overlays NEVER create their own ChartEngine.
 *
 * They are always added to the primary ChartEngine.
 * -------------------------------------------------------------------------
 */
function createOverlaySeries(seriesId: SeriesId, seriesValue: Series) {
  if (!primaryChart) {
    console.warn(
      `Cannot create overlay "${seriesId}" because no primary series exists.`,
    );
    return;
  }

  const build = buildSeries(seriesId, seriesValue);

  const serie = primaryChart.api.addSeries(build);

  allSeries.set(seriesId, {
    ...seriesValue,
    chart: primaryChart,
    serie,
    kind: seriesValue.kind as SeriesKind,
  });
}

/**
 * -------------------------------------------------------------------------
 * Creates a standalone series.
 *
 * This is the important third topology:
 *
 *   primary: false
 *   overlay: false
 *
 * It gets its OWN ChartEngine.
 * -------------------------------------------------------------------------
 */
function createStandaloneSeries(seriesId: SeriesId, seriesValue: Series) {
  const build = buildSeries(seriesId, seriesValue);

  const chart = createChart(
    addChartContainer(seriesId, build.width, build.height),
  );

  _subscribeChart(chart);

  const serie = chart.api.addSeries(build);

  charts.add(chart);

  allSeries.set(seriesId, {
    ...seriesValue,
    chart,
    serie,
    kind: seriesValue.kind as SeriesKind,
  });
}

/**
 * -------------------------------------------------------------------------
 * Creates a runtime series according to its topology.
 *
 * 1. primary=true
 *      -> primary ChartEngine
 *
 * 2. primary=false + overlay=true
 *      -> primary ChartEngine
 *
 * 3. primary=false + overlay=false
 *      -> standalone ChartEngine
 * -------------------------------------------------------------------------
 */
function _createRuntimeSeries(seriesId: SeriesId, seriesValue: Series) {
  validateSeries(seriesValue);

  // Primary
  if (seriesValue.primary) {
    createPrimarySeries(seriesId, seriesValue);
    return;
  }

  // Overlay on primary
  if (seriesValue.overlay) {
    if (!primaryChart) {
      // The primary has not been added yet.
      // It is safe to defer; the overlay will be resolved
      // on the next layout application once a primary exists.
      return;
    }

    createOverlaySeries(seriesId, seriesValue);
    return;
  }

  // Standalone chart
  createStandaloneSeries(seriesId, seriesValue);
}

/**
 * -------------------------------------------------------------------------
 * Creates a runtime series without letting one bad definition abort the
 * whole layout.
 *
 * An unregistered `kind` (a series added in the UI with no implementation
 * in `seriesRegistry`) used to throw out of `applyLayout`, which silently
 * left the chart empty: no series, no legend, no error. It is reported and
 * skipped so the remaining series still render.
 * -------------------------------------------------------------------------
 */
function _createRuntimeSeriesSafe(seriesId: SeriesId, seriesValue: Series) {
  try {
    _createRuntimeSeries(seriesId, seriesValue);
  } catch (err) {
    console.error(
      `[Chart] Skipped series "${seriesId}" (kind "${seriesValue.kind}"):`,
      err instanceof Error ? err.message : err,
    );
  }
}

/**
 * -------------------------------------------------------------------------
 * Resolves the creation order.
 *
 * Required order:
 *
 *   1. Primary
 *   2. Overlays
 *   3. Standalone charts
 *
 * The primary must exist before overlays because overlays
 * share the primary ChartEngine.
 *
 * level is only used to preserve ordering between series
 * of the same topology.
 * -------------------------------------------------------------------------
 */
function resolveSeriesOrder(series: Record<string, Series>): Series[] {
  return Object.values(series).sort((a, b) => {
    // Primary first
    if (a.primary !== b.primary) {
      return a.primary ? -1 : 1;
    }

    // Overlays before standalone series
    if (a.overlay !== b.overlay) {
      return a.overlay ? -1 : 1;
    }

    // Preserve existing ordering
    return a.level - b.level;
  });
}

/**
 * -------------------------------------------------------------------------
 * Removes the DOM element owned by a ChartEngine.
 *
 * Overlay series share the primary ChartEngine, so only the primary
 * and standalone charts own their own `.chart-area` element.
 * -------------------------------------------------------------------------
 */
function _removeChartArea(runtime: RuntimeSeries) {
  if (runtime.overlay) {
    return;
  }

  const area = runtime.chart?.area;

  area?.parentElement?.removeChild(area);
}

/**
 * -------------------------------------------------------------------------
 * Removes a single runtime series.
 *
 * When `silent` is true the `series:removed` event is suppressed,
 * so no DELETE request is triggered on the backend. Used for internal
 * reconciliation (series not present in the incoming backend state) and
 * component unmount cleanup.
 * -------------------------------------------------------------------------
 */
function _destroySeries(seriesId: SeriesId, silent = false) {
  const runtime = allSeries.get(seriesId);

  if (!runtime) {
    return;
  }

  try {
    runtime.serie.destroy({ silent });
  } catch {}

  allSeries.delete(seriesId);

  /**
   * The primary/standalone series owns its ChartEngine and its
   * `.chart-area` DOM element. Remove both so no empty chart
   * containers are left behind.
   */
  _removeChartArea(runtime);

  /**
   * If the primary itself was destroyed, its ChartEngine
   * must no longer be considered available.
   */
  if (runtime.primary) {
    primaryChart = null;
  }
}

/**
 * -------------------------------------------------------------------------
 * Order-insensitive deep equality.
 *
 * Compares JSON-like structures by content, ignoring object key order.
 * The backend serializes series params through HashMaps whose iteration
 * order is not stable across messages, so string comparison would
 * wrongly trigger series recreation on every live-update.
 * -------------------------------------------------------------------------
 */
function _deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) {
    return true;
  }

  if (a == null || b == null) {
    return false;
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false;
    }

    return a.every((value, index) => _deepEqual(value, b[index]));
  }

  if (typeof a === "object" && typeof b === "object") {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);

    if (aKeys.length !== bKeys.length) {
      return false;
    }

    return aKeys.every((key) =>
      _deepEqual(
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key],
      ),
    );
  }

  return false;
}

/**
 * -------------------------------------------------------------------------
 * Determines whether an existing runtime series
 * can be reused.
 *
 * A series must be recreated if its kind or topology changes.
 * Params are compared by content, ignoring key order.
 * -------------------------------------------------------------------------
 */
function requiresRecreation(
  runtime: RuntimeSeries,
  seriesValue: Series,
): boolean {
  if (runtime.kind !== seriesValue.kind) {
    return true;
  }

  if (runtime.primary !== seriesValue.primary) {
    return true;
  }

  if (runtime.overlay !== seriesValue.overlay) {
    return true;
  }

  if (!_deepEqual(runtime.params, seriesValue.params)) {
    return true;
  }

  return false;
}

/**
 * -------------------------------------------------------------------------
 * Removes ChartEngines that no longer own active series.
 * -------------------------------------------------------------------------
 */
function _cleanupEmptyCharts() {
  const activeCharts = new Set<ChartEngine>();

  for (const runtime of allSeries.values()) {
    activeCharts.add(runtime.chart);
  }

  for (const chart of charts) {
    if (activeCharts.has(chart)) {
      continue;
    }

    try {
      chart.api.destroy();
    } catch {}

    try {
      chart.area?.parentElement?.removeChild(chart.area);
    } catch {}

    charts.delete(chart);

    if (primaryChart === chart) {
      primaryChart = null;
    }
  }
}

/**
 * -------------------------------------------------------------------------
 * Applies a timeframe layout.
 *
 * Reconciliation is incremental:
 *
 * - Existing compatible series are reused.
 * - New series are created.
 * - Removed series are destroyed.
 * - Series whose kind/topology changed are recreated.
 *
 * Data is intentionally NOT updated here.
 * -------------------------------------------------------------------------
 */
function applyLayout(timeframe: ChartTimeframe) {
  const definitions = timeframe.series;

  validateLayout(definitions);

  /**
   * -----------------------------------------------------------------------
   * Remove series that no longer exist.
   * -----------------------------------------------------------------------
   */
  const nextSeriesIds = new Set(Object.keys(definitions));

  const seriesToRemove: SeriesId[] = [];

  for (const seriesId of allSeries.keys()) {
    if (!nextSeriesIds.has(seriesId)) {
      seriesToRemove.push(seriesId);
    }
  }

  /**
   * -----------------------------------------------------------------------
   * Destroy overlays/standalone series before the primary.
   *
   * The primary owns the overlay ChartEngine.
   * -----------------------------------------------------------------------
   */
  seriesToRemove.sort((a, b) => {
    const seriesA = definitions[a];
    const seriesB = definitions[b];

    if (!seriesA || !seriesB) {
      return 0;
    }

    // Non-primary series before primary
    if (seriesA.primary !== seriesB.primary) {
      return seriesA.primary ? 1 : -1;
    }

    // For non-primary series, destroy overlays before standalone.
    if (seriesA.overlay !== seriesB.overlay) {
      return seriesA.overlay ? -1 : 1;
    }

    return seriesB.level - seriesA.level;
  });

  for (const seriesId of seriesToRemove) {
    _destroySeries(seriesId, true);
  }

  /**
   * -----------------------------------------------------------------------
   * Resolve primary -> overlay -> standalone creation order.
   * -----------------------------------------------------------------------
   */
  const orderedSeries = resolveSeriesOrder(definitions);

  /**
   * -----------------------------------------------------------------------
   * Create new series or reuse existing ones.
   * -----------------------------------------------------------------------
   */
  for (const seriesValue of orderedSeries) {
    const seriesId = seriesValue.id;

    const existing = allSeries.get(seriesId);

    if (!existing) {
      _createRuntimeSeriesSafe(seriesId, seriesValue);

      continue;
    }

    /**
     * ---------------------------------------------------------------------
     * Existing series whose runtime topology changed.
     * ---------------------------------------------------------------------
     */
    if (requiresRecreation(existing, seriesValue)) {
      _destroySeries(seriesId, true);

      _createRuntimeSeriesSafe(seriesId, seriesValue);

      continue;
    }
  }

  _cleanupEmptyCharts();
}

/**
 * -------------------------------------------------------------------------
 * Applies options to the ChartEngine owning the series.
 *
 * Therefore:
 *
 * - primary -> primary ChartEngine
 * - overlay -> primary ChartEngine
 * - standalone -> its own ChartEngine
 * -------------------------------------------------------------------------
 */
function applyOptions(seriesId: SeriesId, config: Partial<ChartOptions>) {
  const runtime = allSeries.get(seriesId);

  if (!runtime) {
    console.warn(`Cannot apply options: series "${seriesId}" not found.`);

    return;
  }

  runtime.chart.api.applyOptions(config);
}

/**
 * -------------------------------------------------------------------------
 * Sets the complete data of a series.
 * -------------------------------------------------------------------------
 */
function setData(seriesId: SeriesId, data: any) {
  allSeries.get(seriesId)?.serie.setData(data);
}

/**
 * -------------------------------------------------------------------------
 * Patches existing series data.
 * -------------------------------------------------------------------------
 */
function patchData(seriesId: SeriesId, data: any) {
  allSeries.get(seriesId)?.serie.patchData(data);
}

/**
 * Patches existing series data lazily over time.
 *
 * The incoming data is appended in chunks so the chart visibly replays
 * the bars. Returns a cancel function.
 */
function patchDataLazy(seriesId: SeriesId, data: any, intervalMs = 25) {
  return allSeries.get(seriesId)?.serie.patchDataLazy(data, intervalMs);
}

/**
 * -------------------------------------------------------------------------
 * Updates the latest live candle/tick.
 * -------------------------------------------------------------------------
 */
function updateLive(seriesId: SeriesId, candle: any) {
  allSeries.get(seriesId)?.serie.update(candle);
}

/**
 * -------------------------------------------------------------------------
 * Returns the runtime series.
 * -------------------------------------------------------------------------
 */
function getSeriesById(seriesId: SeriesId) {
  return allSeries.get(seriesId);
}

/**
 * -------------------------------------------------------------------------
 * Public Chart.vue API.
 * -------------------------------------------------------------------------
 */
defineExpose({
  patchData,
  patchDataLazy,
  getSeriesById,
  applyLayout,
  applyOptions,
  setData,
  updateLive,
});

/**
 * -------------------------------------------------------------------------
 * Make sure the component is mounted before charts are created.
 * -------------------------------------------------------------------------
 */
onMounted(() => {
  if (!container.value) {
    throw new Error("Chart container was not mounted.");
  }
});

/**
 * -------------------------------------------------------------------------
 * Completely clean up this Chart.vue instance.
 * -------------------------------------------------------------------------
 */
onBeforeUnmount(() => {
  cleanAllSeries();
});
</script>

<template>
  <div ref="container" class="chart-container" />
</template>

<style>
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  overflow-y: auto;
  overflow-x: hidden;
  box-sizing: border-box;
  scrollbar-gutter: stable;
  border-radius: var(--ui-radius);
}

.chart-area {
  border-radius: var(--ui-radius);
  padding-right: 0.25rem;
}

/* Chrome / Edge / Safari */
.chart-container::-webkit-scrollbar {
  background: var(--ui-bg-accented);
  width: 12px;
}

.chart-container::-webkit-scrollbar-track {
  background: transparent;
}

.chart-container::-webkit-scrollbar-thumb {
  background: rgba(128, 128, 128, 0.45);

  background-clip: padding-box;
  border-radius: var(--ui-radius);
}

.chart-container::-webkit-scrollbar-thumb:hover {
  background: rgba(128, 128, 128, 0.7);
  background-clip: padding-box;
}
</style>
