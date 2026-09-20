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

import { TabKind, type TradingTab } from "~/stores/tabManager.store";

const tabManager = useTabManager();

const onSelect = (e: any) => {
  const newTab: TradingTab = {
    id: crypto.randomUUID(),
    kind: TabKind.Trading,
    title: "tab test",
    subtitle: "tab sub",
    description: "tab description",
    color: "primary",
    symbol: "BTCUSDT",
    source: "binance",
    timeframe: "1m",
  };

  tabManager.addTab(newTab);
  tabManager.symbolSearchModal = false;
};
</script>

<template>
  <div class="tab-add flex items-center h-[inherit]">
    <UButton
      class="mt-[5px]"
      icon="i-lucide-plus"
      size="sm"
      color="neutral"
      @click="tabManager.symbolSearchModal = true"
      variant="ghost"
    />

    <UModal
      v-model:open="tabManager.symbolSearchModal"
      title="Symbol Search"
      :ui="{
        content: 'w-fit max-w-none rounded-lg shadow-lg ring ring-default',
      }"
    >
      <template #body>
        <SymbolSearch
          :data="tabManager.getInstrumentList()"
          @select="onSelect"
        />
      </template>
    </UModal>
  </div>
</template>
