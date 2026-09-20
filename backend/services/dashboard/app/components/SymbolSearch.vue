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

import { h, resolveComponent } from "vue";
import type { TableColumn, TableRow } from "@nuxt/ui";

export type Instrument = {
  id: string;
  source: string;
  symbol: string;
  status: "sync" | "unsync";
  legend: string;
  market: string;
};

const props = defineProps<{
  data: Instrument[];
}>();

const emit = defineEmits<{
  (e: "select", instrument: Instrument): void;
}>();

const UBadge = resolveComponent("UBadge");

const columns: TableColumn<Instrument>[] = [
  {
    accessorKey: "source",
    header: "Source",
    cell: ({ row }) => `${row.getValue("source")}`,
  },
  {
    accessorKey: "symbol",
    header: "Symbol",
    cell: ({ row }) => {
      return row.getValue("symbol");
    },
  },
  {
    accessorKey: "legend",
    header: "Legend",
  },
  {
    accessorKey: "market",
    header: "Market",
    meta: {
      class: {
        th: "text-right",
        td: "text-right font-medium",
      },
    },
    cell: ({ row }) => {
      return row.getValue("market");
    },
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => {
      const color = {
        sync: "success" as const,
        unsync: "error" as const,
      }[row.getValue("status") as string];

      return h(UBadge, { class: "capitalize", variant: "subtle", color }, () =>
        row.getValue("status"),
      );
    },
  },
];

const globalFilter = ref("");

function onSelect(e: Event, row: TableRow<Instrument>) {
  emit("select", row.original);
}
</script>

<template>
  <div class="flex flex-col w-[800px] h-[500px]">
    <div class="py-4 pt-0 border-b border-accented">
      <UInput
        v-model="globalFilter"
        class="w-full"
        placeholder="Search"
        size="lg"
        icon="i-lucide-search"
      />
    </div>

    <div class="flex-1 overflow-y-auto">
      <UTable
        ref="table"
        v-model:global-filter="globalFilter"
        :data="props.data"
        :columns="columns"
        @select="onSelect"
        :ui="{ td: 'select-none cursor-default' }"
      />
    </div>
  </div>
</template>
