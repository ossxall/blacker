import {
  Candlestick,
  type CandlestickConfig,
} from "~/packages/playground/series/Candlestick/Candlestick";
import {
  ADXSeries,
  type ADXConfig,
} from "~/packages/playground/series/ADXSeries/ADXSeries";
import { TabKind, type Tab } from "../tabManager.store";
import { useBacktestingTabStore } from "./backtesting-tab.store";
import { useTradingTabStore } from "./trading-tab.store";
import { EMA, type EMAConfig } from "~/packages/playground/series/EMASeries/EMASeries";
import {
  ReversalTrap,
  type ReversalTrapConfig,
} from "~/packages/playground/series/ReversalTrap/ReversalTrap";

export function useTabContentStore(tab: Tab) {
  if (!tab) return null;

  switch (tab.kind) {
    case TabKind.Backtesting:
      return useBacktestingTabStore(tab);
    case TabKind.Trading:
      return useTradingTabStore(tab);
    //case TabKind.Algo:        return useAlgoTabStore(tab.id)
  }
}

export const seriesRegistry = {
  ADX: (config: ADXConfig) => ADXSeries(config),
  Candlestick: (config: CandlestickConfig) => Candlestick(config),
  EMA: (config: EMAConfig) => EMA(config),
  ReversalTrap: (config: ReversalTrapConfig) => ReversalTrap(config),
} as const;
export type SeriesRegistry = typeof seriesRegistry;
export type SeriesKind = keyof SeriesRegistry;
export type SeriesId = string;

export { useBacktestingTabStore };
