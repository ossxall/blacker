from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def timeframe_tuple(raw: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normaliza uno o varios timeframes.

    Acepta un ``str`` separado por comas (``"1D,30min"``) o una secuencia
    (``["1D", "30min"]``) y devuelve una tupla sin elementos vacíos.
    """
    if isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw]
    else:
        parts = str(raw).split(",")
    return tuple(part.strip() for part in parts if part.strip())


@dataclass
class Config:
    ema_fast: int = 55
    ema_slow: int = 200
    timeframe: str = "30min"
    allow_long: bool = True
    allow_short: bool = True

    initial_capital: float = 10_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    risk_pct: float = 0.01
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.01

    confirm_bars: int = 4
    entry_adx_enabled: bool = True
    entry_adx_len: int = 14
    entry_adx_min: float = 20.0

    # Filtro de tendencia de timeframe superior (HTF). Solo se permite la
    # entrada si la tendencia de `entry_htf_trend_timeframe` coincide con el
    # lado del cruce. La tendencia del HTF se mide con la EMA rápida y lenta de
    # la estrategia (`ema_fast`/`ema_slow`, por defecto EMA55/EMA200) sobre las
    # velas del HTF. Evita operar contra la tendencia mayor, que es el patrón
    # que más entradas falsas genera. Desactivado por defecto.
    entry_htf_trend_enabled: bool = True
    entry_htf_trend_timeframe: str = "1D"

    # Filtro "no perseguir": descarta entradas cuando el precio ya está
    # demasiado extendido respecto a la EMA lenta, en la dirección del cruce
    # (LONG: close por encima; SHORT: close por debajo). `0` desactiva el
    # filtro; p. ej. 0.04 descarta entradas a más de un 4% de la EMA lenta.
    entry_max_dist_slow_pct: float = 0.0

    # Filtro de volatilidad nativo del timeframe de la estrategia (30min):
    # descarta la entrada cuando el ATR relativo (ATR/close) alcanza
    # `entry_atr_max_pct`. Con `stop_loss_pct=0.01` (stop del 1%), un valor de
    # 0.01 evita entrar cuando el ruido de una sola vela ya iguala el stop, que
    # es el patrón de las entradas falsas rápidas (el precio revierte y toca el
    # stop en 1-2 velas). Un valor menor (p. ej. 0.008) filtra más pero recorta
    # más retorno; `0` desactiva el filtro.
    entry_atr_max_pct: float = 0.0
    entry_atr_len: int = 14

    adx_exit_enabled: bool = True
    # Uno o varios timeframes (combinados con OR). Acepta "1D", "1D,30min" o
    # ["1D", "30min"]; la salida ADX se dispara si hay reversión en cualquiera.
    adx_timeframe: str | list[str] | tuple[str, ...] = "1D"
    # Ventana (en velas del timeframe de la estrategia) durante la cual una
    # reversión ADX mantiene activada la salida.
    adx_exit_range: int = 9
    adx_dilen: int = 14
    adx_adxlen: int = 14
    adx_key_level: float = 23.0
    adx_exit_pct: float = 100.0

    ema_weakening_exit_enabled: bool = True
    # Uno o varios timeframes (combinados con OR). Acepta "1D", "1D,30min" o
    # ["1D", "30min"]; la salida weakening se dispara si hay debilitamiento en
    # cualquiera de ellos.
    ema_weakening_timeframe: str | list[str] | tuple[str, ...] = "30min"
    ema_weakening_gap_pct: float = 0.0005
    ema_weakening_exit_pct: float = 25.0

    # Gestión de salida orientada a reducir la racha de pérdidas en whipsaw:
    # al alcanzar `partial_tp_r` veces el riesgo inicial se asegura beneficio
    # cerrando `partial_tp_pct` de la posición y, opcionalmente, se mueve el
    # stop del resto a breakeven. Sube el win rate real (rompe rachas de
    # stop-loss consecutivos) a cambio de recortar parte del retorno.
    partial_tp_enabled: bool = True
    partial_tp_r: float = 1.0
    partial_tp_pct: float = 25.0
    move_stop_to_breakeven: bool = True

    # Salida si el precio toca la EMA slow después de una entrada por cross EMA.
    # `..._pct` es la tolerancia en % alrededor de la EMA (0.10 = 0.10%).
    # `..._after` decide el sentido de la tolerancia: False (default) la aplica
    # ANTES del toque (LONG dispara en EMA*(1+tol), SHORT en EMA*(1-tol)); True
    # la aplica DESPUÉS del toque (LONG en EMA*(1-tol), SHORT en EMA*(1+tol)).
    # `..._exit_pct` es el % de la posición que se cierra al tocar la EMA
    # (100 = cierre total; < 100 = salida parcial, una sola vez por posición).
    close_on_ema_slow_touch: bool = True
    close_on_ema_slow_touch_pct: float = 0.1  # Tolerancia en % (0.10 = 0.10%)
    close_on_ema_slow_touch_after: bool = (
        False  # False = antes del toque; True = después
    )
    close_on_ema_slow_touch_exit_pct: float = (
        25.0  # % de la posición a cerrar en el toque
    )
    # Throttle de riesgo por racha de pérdidas / drawdown de equity.
    # Cuando se cumplen las condiciones, el riesgo por operación se multiplica
    # por el factor correspondiente (0 = no entrar).
    risk_throttle_streak_k: int = 1
    risk_throttle_streak_scale: float = 0.75
    risk_throttle_dd_pct: float = 5.0
    risk_throttle_dd_scale: float = 0.75

    force_close_at_end: bool = True

    @property
    def adx_timeframes(self) -> tuple[str, ...]:
        """Timeframes ADX normalizados (uno o varios); la salida usa OR entre ellos."""
        return timeframe_tuple(self.adx_timeframe)

    @property
    def ema_weakening_timeframes(self) -> tuple[str, ...]:
        """Timeframes de weakening normalizados (uno o varios); la salida usa OR entre ellos."""
        return timeframe_tuple(self.ema_weakening_timeframe)

    def __post_init__(self):
        if min(self.ema_fast, self.ema_slow) <= 0 or self.ema_fast >= self.ema_slow:
            raise ValueError("Se requiere 0 < ema_fast < ema_slow.")
        if pd.Timedelta(self.timeframe) <= pd.Timedelta(0):
            raise ValueError("timeframe debe ser positivo.")
        for timeframe in self.adx_timeframes:
            if pd.Timedelta(timeframe) <= pd.Timedelta(0):
                raise ValueError(f"adx_timeframe inválido: {timeframe!r}.")
        if self.adx_exit_enabled and not self.adx_timeframes:
            raise ValueError("adx_timeframe no puede estar vacío.")
        if self.ema_weakening_exit_enabled and not self.ema_weakening_timeframes:
            raise ValueError("ema_weakening_timeframe no puede estar vacío.")
        for timeframe in self.ema_weakening_timeframes:
            if pd.Timedelta(timeframe) <= pd.Timedelta(0):
                raise ValueError(f"ema_weakening_timeframe inválido: {timeframe!r}.")
        if self.initial_capital <= 0 or self.max_leverage <= 0:
            raise ValueError("Capital y leverage deben ser positivos.")
        if not 0 < self.risk_pct <= 0.2:
            raise ValueError("risk_pct debe estar en (0, 0.2].")
        if min(self.commission_bps, self.slippage_bps, self.stop_loss_pct) < 0:
            raise ValueError("Comisiones, slippage y stop no pueden ser negativos.")
        if self.confirm_bars < 0:
            raise ValueError("confirm_bars no puede ser negativo.")
        if self.entry_adx_enabled and (
            self.entry_adx_len <= 0 or self.entry_adx_min < 0
        ):
            raise ValueError("Configuración de ADX de entrada inválida.")
        if self.entry_htf_trend_enabled:
            if pd.Timedelta(self.entry_htf_trend_timeframe) < pd.Timedelta(
                self.timeframe
            ):
                raise ValueError("entry_htf_trend_timeframe debe ser >= timeframe.")
        if self.entry_max_dist_slow_pct < 0:
            raise ValueError("entry_max_dist_slow_pct no puede ser negativo.")
        if self.entry_atr_max_pct < 0:
            raise ValueError("entry_atr_max_pct no puede ser negativo.")
        if self.entry_atr_max_pct > 0 and self.entry_atr_len <= 0:
            raise ValueError(
                "entry_atr_len debe ser > 0 si entry_atr_max_pct está activo."
            )
        if self.adx_exit_enabled:
            if min(self.adx_dilen, self.adx_adxlen) <= 0 or self.adx_key_level < 0:
                raise ValueError("Configuración de ADX de salida inválida.")
            if not 0 < self.adx_exit_pct <= 100:
                raise ValueError("adx_exit_pct debe estar en (0, 100].")
            if self.adx_exit_range < 1:
                raise ValueError("adx_exit_range debe ser >= 1.")
        if self.ema_weakening_exit_enabled:
            if (
                self.ema_weakening_gap_pct <= 0
                or not 0 < self.ema_weakening_exit_pct <= 100
            ):
                raise ValueError("Configuración de weakening inválida.")
        if self.partial_tp_enabled:
            if self.partial_tp_r <= 0 or not 0 < self.partial_tp_pct <= 100:
                raise ValueError("Configuración de take-profit parcial inválida.")
        if self.close_on_ema_slow_touch:
            if self.close_on_ema_slow_touch_pct < 0:
                raise ValueError("close_on_ema_slow_touch_pct debe ser >= 0.")
            if not 0 < self.close_on_ema_slow_touch_exit_pct <= 100:
                raise ValueError(
                    "close_on_ema_slow_touch_exit_pct debe estar en (0, 100]."
                )
        if not (self.allow_long or self.allow_short):
            raise ValueError("Debe habilitarse LONG, SHORT o ambos.")
        if self.risk_throttle_streak_k < 0:
            raise ValueError("risk_throttle_streak_k debe ser >= 0.")
        if not 0 <= self.risk_throttle_streak_scale <= 1:
            raise ValueError("risk_throttle_streak_scale debe estar en [0, 1].")
        if self.risk_throttle_dd_pct < 0:
            raise ValueError("risk_throttle_dd_pct debe ser >= 0.")
        if not 0 <= self.risk_throttle_dd_scale <= 1:
            raise ValueError("risk_throttle_dd_scale debe estar en [0, 1].")
