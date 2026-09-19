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

import { ref, computed } from "vue";
import type { SelectItem } from "@nuxt/ui";
import { useBacktestingTabStore } from "~/stores/tabs/backtesting-tab.store";
import type { Series } from "./Chart.vue";

const props = defineProps({
  tabId: {
    type: String,
    required: true,
  },

  timeframes: {
    type: Array as PropType<string[]>,
    required: true,
  },

  activeTimeframe: {
    type: String,
    required: true,
  },
});

const emit = defineEmits<{
  "update:timeframe": [timeframe: string];
}>();

const toast = useToast();

// -----------------------------------------------------------------------------
// Tab / Store
// -----------------------------------------------------------------------------

const tabsStore = useTabManager();
const tab = tabsStore.getTabById(props.tabId);
const tabStore = useBacktestingTabStore(tab as BacktestingTab);

//----------------------------------------------------------------------------------------------------------------------
// SLAVE STATE
//----------------------------------------------------------------------------------------------------------------------

interface KeyLabelColor {
  key: string;
  label: string;
  color: any;
}

const slavesStatus = computed<KeyLabelColor[]>(() => [
  {
    key: "master",
    label: "Master",
    color: tabStore.globalState.status === "Ready" ? "success" : "error",
  },
  {
    key: "engine",
    label: "Engine",
    color: tabStore.isEngineConnected ? "success" : "error",
  },
]);

//----------------------------------------------------------------------------------------------------------------------
// TIMEFRAMES
//----------------------------------------------------------------------------------------------------------------------

const timeframeModalOpen = ref(false);
const timeframeModalTitle = ref("Add Timeframe");
const timeframeModalItems = ref<SelectItem[]>([
  {
    type: "label",
    label: "Minutes",
  },
  "1m",
  "5m",
  "15m",
  "30m",
  "45m",
  {
    type: "separator",
  },
  {
    type: "label",
    label: "Hours",
  },
  "1h",
  "2h",
  "3h",
  "4h",
  "6h",
]);
const timeframeSelected = ref("1m");

//----------------------------------------------------------------------------------------------------------------------
// SERIES
//----------------------------------------------------------------------------------------------------------------------

const seriesModalOpen = ref(false);
const statsModalOpen = ref(false);
const seriesModalTitle = ref("Add Series");
const seriesData: Series[] = [
  {
    id: "ema",
    kind: "EMA",
    level: 1,
    primary: false,
    overlay: true,
    params: {
      label: "EMA 55",
      layer: "foreground",
      color: "#FF9800",
      priceTagColor: "#FF9800",
      period: { value: 55, affectsCompute: true, min: 1, max: 200, step: 1 },
      lineWidth: 1,
    },
    name: "Exponential Moving Average",
  },
  {
    id: "reversal-trap",
    kind: "ReversalTrap",
    level: 1,
    primary: false,
    overlay: true,
    params: {
      label: "Reversal Trap",
      layer: "foreground",
      color: "#FF9800",
      priceTagColor: "#FF9800",
      lineWidth: 1.5,
      envelope_len: {
        value: 55,
        affectsCompute: true,
        min: 1,
        max: 200,
        step: 1,
      },
      multiplier: {
        value: 4,
        affectsCompute: true,
        min: 0.1,
        max: 20,
        step: 0.1,
      },
      trap_window: {
        value: 10,
        affectsCompute: true,
        min: 1,
        max: 100,
        step: 1,
      },
      signal_gap: {
        value: 10,
        affectsCompute: true,
        min: 0,
        max: 100,
        step: 1,
      },
      rsi_len: { value: 20, affectsCompute: true, min: 1, max: 100, step: 1 },
      stop_mult: {
        value: 0.5,
        affectsCompute: true,
        min: 0.1,
        max: 5,
        step: 0.1,
      },
      max_trades: {
        value: 500,
        affectsCompute: true,
        min: 1,
        max: 5000,
        step: 1,
      },
      atr_len: { value: 100, affectsCompute: true, min: 1, max: 300, step: 1 },
      target_source: {
        value: "Basis Line",
        affectsCompute: true,
        options: ["Basis Line", "Envelope"],
      },
    },
    name: "Reversal Trap",
  },
];

//----------------------------------------------------------------------------------------------------------------------
// HANDLERS
//----------------------------------------------------------------------------------------------------------------------

async function onStartBacktest() {
  try {
    await tabStore.startBacktest();
  } catch (err: any) {
    toast.add({
      title: "Error starting backtest",
      description: err.data.message,
      icon: "i-lucide-circle-x",
      color: "error",
    });
  }
}

async function onStopBacktest() {
  try {
    await tabStore.stopBacktest();
  } catch (err: any) {
    toast.add({
      title: "Error stopping backtest",
      description: err.data.message,
      icon: "i-lucide-circle-x",
      color: "error",
    });
  }
}

async function onTimeframeSelected() {
  try {
    await tabStore.addTimeframe(timeframeSelected.value);
  } catch (err: any) {
    toast.add({
      title: "Error adding timeframe",
      description: err.data.message,
      icon: "i-lucide-circle-x",
      color: "error",
    });
  }

  timeframeModalOpen.value = false;
}

async function onSeriesSelected(series: Series) {
  try {
    await tabStore.addSeries(props.activeTimeframe, series);
  } catch (err: any) {
    toast.add({
      title: "Error adding series",
      description: err.data.message,
      icon: "i-lucide-circle-x",
      color: "error",
    });
  }

  seriesModalOpen.value = false;
}
</script>

<template>
  <div class="backtesting-toolbar">
    <!----------WORKER STATES ------------>

    <div class="backtesting-toolbar-workers">
      <UButton
        v-for="state in slavesStatus"
        :key="state.key"
        variant="outline"
        color="neutral"
        size="xs"
      >
        <UChip standalone inset size="xs" :color="state.color" />
        <span class="label">{{ state.label }}</span>
      </UButton>
    </div>

    <USeparator orientation="vertical" class="h-10 pl-4 pr-4" />

    <!--------- TIMEFRAMES ---------->

    <div class="backtesting-toolbar-controls">
      <UModal
        v-model:open="timeframeModalOpen"
        :title="timeframeModalTitle"
        :close="{
          color: 'neutral',
          variant: 'outline',
          class: 'rounded-full',
        }"
        :overlay="false"
      >
        <UButton color="neutral" variant="outline" icon="lucide:plus" size="xs"
          >Timeframe</UButton
        >

        <template #body>
          <UForm class="space-y-4">
            <UFormField label="Timeframe">
              <USelect
                class="w-full"
                v-model="timeframeSelected"
                size="lg"
                :items="timeframeModalItems"
              />
            </UFormField>
          </UForm>
        </template>

        <template #footer>
          <div class="content w-full flex justify-end gap-2">
            <UButton
              color="neutral"
              size="md"
              variant="outline"
              @click="timeframeModalOpen = false"
              >Cancel</UButton
            >

            <UButton
              color="neutral"
              size="md"
              variant="solid"
              @click="onTimeframeSelected"
            >
              Add
            </UButton>
          </div>
        </template>
      </UModal>

      <USeparator orientation="vertical" class="h-10 pl-4 pr-4" />

      <UModal
        v-model:open="seriesModalOpen"
        :title="seriesModalTitle"
        :close="{
          color: 'neutral',
          variant: 'outline',
          class: 'rounded-full',
        }"
        :overlay="false"
        :ui="{
          content: 'w-fit max-w-none rounded-lg shadow-lg ring ring-default',
        }"
      >
        <UButton color="neutral" variant="outline" icon="lucide:plus" size="xs"
          >Series</UButton
        >

        <template #body>
          <SeriesSearch :data="seriesData" @select="onSeriesSelected" />
        </template>
      </UModal>

      <USeparator orientation="vertical" class="h-10 pl-4 pr-4" />

      <UModal
        v-model:open="statsModalOpen"
        title="Stats"
        :close="{
          color: 'neutral',
          variant: 'outline',
          class: 'rounded-full',
        }"
        :overlay="false"
        :ui="{
          content: 'w-fit max-w-none rounded-lg shadow-lg ring ring-default',
        }"
      >
        <UButton
          title="Stats"
          color="neutral"
          variant="outline"
          icon="lucide:chart-column-big"
          size="xs"
        />

        <template #body>
          <StatsPanel :dashboard="tabStore.globalState" />
        </template>
      </UModal>

      <USeparator orientation="vertical" class="h-10 pl-4 pr-4" />

      <UButton
        v-for="timeframe in props.timeframes"
        :key="timeframe"
        color="neutral"
        variant="outline"
        size="xs"
        :class="{
          'bg-neutral-200 dark:bg-neutral-800':
            timeframe === props.activeTimeframe,
        }"
        @click="emit('update:timeframe', timeframe)"
      >
        {{ timeframe }}
      </UButton>
    </div>
    <!------------- BACKTEST REPLAY ------------>

    <div class="backtesting-toolbar-replay">
      <UButton
        title="Back"
        color="neutral"
        icon="lucide:step-back"
        variant="outline"
        size="sm"
      />
      <UButton
        :disabled="tabStore.isRunning"
        title="Play"
        color="neutral"
        icon="lucide:play"
        @click="onStartBacktest"
        :variant="tabStore.isRunning ? 'solid' : 'outline'"
        :loading="tabStore.isRunning"
        size="sm"
      />

      <UButton
        :disabled="!tabStore.isRunning"
        title="Stop"
        color="neutral"
        icon="lucide:square"
        @click="onStopBacktest"
        :variant="tabStore.isRunning ? 'outline' : 'solid'"
        size="sm"
      />
      <UButton
        title="Next"
        color="neutral"
        icon="lucide:step-forward"
        variant="outline"
        size="sm"
      />
    </div>
  </div>
</template>

<style scoped>
.backtesting-toolbar {
  display: flex;
  overflow: hidden;
  align-items: center;
  padding: 0rem 1rem;
  background: var(--ui-bg);
  border-radius: var(--ui-radius);
  height: var(--header-height);
}

.backtesting-toolbar-workers {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.backtesting-toolbar-workers .label {
  color: var(--ui-text-muted);
  font-weight: 500;
}

.backtesting-toolbar-controls {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.backtesting-toolbar-replay {
  display: flex;
  align-items: center;
  margin-left: auto;
  gap: 0.5rem;
}
</style>
