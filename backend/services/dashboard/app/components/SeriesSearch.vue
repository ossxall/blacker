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
import type { Series } from "./Chart.vue";

const props = defineProps<{
  data: Series[];
}>();

const emit = defineEmits<{
  (e: "select", instrument: Series): void;
}>();

const columns: TableColumn<Series>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => `${row.getValue("name")}`,
  }
];

const globalFilter = ref("");

function onSelect(e: Event, row: TableRow<Series>) {
  emit("select", row.original);
}
</script>

<template>
  <div class="flex flex-col w-[800px] h-[500px]">
    <div class="py-4 pt-0 border-b border-accented">
      <UInput v-model="globalFilter" class="w-full" placeholder="Search" size="lg" icon="i-lucide-search"/>
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
