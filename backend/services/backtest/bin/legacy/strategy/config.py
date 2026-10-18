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

    adx_exit_enabled: bool = True
    # Uno o varios timeframes (combinados con OR). Acepta "1D", "1D,30min" o
    # ["1D", "30min"]; la salida ADX se dispara si hay reversión en cualquiera.
    adx_timeframe: str | list[str] | tuple[str, ...] = ("1D")
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
    ema_weakening_gap_pct: float = 0.0001
    ema_weakening_exit_pct: float = 25.0

    # Gestión de salida orientada a reducir la racha de pérdidas en whipsaw:
    # al alcanzar `partial_tp_r` veces el riesgo inicial se asegura beneficio
    # cerrando `partial_tp_pct` de la posición y, opcionalmente, se mueve el
    # stop del resto a breakeven. Sube el win rate real (rompe rachas de
    # stop-loss consecutivos) a cambio de recortar parte del retorno.
    partial_tp_enabled: bool = True
    partial_tp_r: float = 1.0
    partial_tp_pct: float = 25.0
    # Second partial take-profit (optional). If enabled (>0), it triggers at
    # partial_tp2_r (R-multiple of initial risk) closing partial_tp2_pct of
    # the position. Partial 2 is intended to be at larger distance than TP1.
    partial_tp2_r: float = 2.0
    partial_tp2_pct: float = 25.0
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
    close_on_ema_slow_touch_after: bool = False  # False = antes del toque; True = después
    close_on_ema_slow_touch_exit_pct: float = 25.0  # % de la posición a cerrar en el toque

    # --- Entrada parcial + confirmación (scale-in) --------------------------
    # Abre `entry_partial_pct` % de la posición en la señal y añade el resto
    # cuando se cumpla `entry_confirm_mode`. Activado por defecto (50%/50%).
    entry_partial_enabled: bool = True
    entry_partial_pct: float = 50.0        # % inicial (0 < pct < 100)
    entry_confirm_mode: str = "bars"       # "bars" | "close_above_ema" | "pullback_retest"
    entry_confirm_bars: int = 2            # Nº de velas tras entrada parcial
    entry_confirm_close_in_dir: bool = True  # Exige cierre en dirección tras confirmación (modo bars)
    entry_confirm_retest_tol_pct: float = 0.10  # Tolerancia % para pullback_retest

    # --- Filtro de régimen -------------------------------------------------
    # Sólo se admiten entradas alineadas con la tendencia del timeframe
    # superior `trend_timeframe`:
    #   "off"   -> sin filtro (comportamiento original).
    #   "cross" -> dirección del cruce EMA rápida/lenta en ese timeframe.
    #   "slow"  -> cierre por encima (long) o por debajo (short) de la EMA slow.
    # El sesgo se recalcula con la última vela superior ya cerrada, así que no
    # hay lookahead.
    trend_filter: str = "off"
    trend_timeframe: str = "1D"

    # --- Filtro anti-chop --------------------------------------------------
    # Exige que el ADX (Wilder, `adx_dilen`/`adx_adxlen`) de `chop_timeframe`
    # esté por encima de `chop_adx_min` para abrir posiciones: en mercados
    # laterales el cruce EMA genera muchas entradas pequeñas en pérdida.
    # Igual que el filtro de régimen, se alinea con la última vela ya cerrada.
    chop_filter: bool = False
    chop_timeframe: str = "1D"
    chop_adx_min: float = 25.0

    # --- Protección mensual ------------------------------------------------
    # Si la equity cae más de `monthly_loss_limit_pct` % por debajo de la
    # equity al inicio del mes calendario, se dejan de abrir posiciones (y, con
    # monthly_loss_limit_close, se cierra la abierta) hasta el siguiente mes
    # o, si `monthly_loss_limit_days` > 0, durante ese número de días.
    # Red de seguridad para el peor mes; con `streak_trend_only` activo apenas
    # llega a dispararse (el peor mes cae a ~-2,7%). 0 = desactivado.
    monthly_loss_limit_pct: float = 4.5
    monthly_loss_limit_close: bool = True
    monthly_loss_limit_days: int = 0

    # --- Pausa tras rachas de pérdidas -------------------------------------
    # Tras `consecutive_loss_pause` posiciones consecutivas en pérdida no se
    # abren nuevas durante `consecutive_loss_pause_bars` velas (96 velas de
    # 30min = 48h). Evita re-entrar en pleno whipsaw. Con `streak_trend_only`
    # activo la pausa no baja más la racha máxima (queda en 5) y cuesta entre
    # 60 y 110 puntos de rendimiento, así que arranca desactivada (0); se
    # conserva como palanca opcional para mercados muy violentos.
    consecutive_loss_pause: int = 0
    consecutive_loss_pause_bars: int = 96

    # Además de la pausa temporal, se ignoran las primeras
    # `consecutive_loss_skip_signals` señales de entrada válidas que aparezcan
    # tras una posición cerrada en pérdida (una victoria las reinicia). Ataca
    # directamente la racha máxima de pérdidas: en whipsaw la siguiente señal
    # suele ser otra trampa. 0 = desactivado.
    consecutive_loss_skip_signals: int = 0

    # Tras una posición cerrada en pérdida, la siguiente entrada sólo se abre
    # si va en la dirección del sesgo de tendencia de `trend_timeframe` (vela
    # superior ya cerrada, sin lookahead); una victoria reinicia el requisito.
    # Ataca la racha de pérdidas: en whipsaw la señal de cruce suele ir contra
    # la tendencia mayor. Sólo exige el sesgo tras perder, no siempre, para no
    # recortar el retorno de los cruces en tendencia. En BTCUSDT 2020-2026
    # baja la racha máxima de 7 a 5 (train y test), sube el PF de 1,76 a 2,07
    # y recorta el peor mes de -4,75% a -2,65%, con el rendimiento total
    # encima del baseline (328% -> 363%). False = desactivado.
    streak_trend_only: bool = True

    # --- Tamaño de posición adaptativo -------------------------------------
    # `risk_pct` se multiplica por `clamp(1 + k * (win_rate últimas n posiciones
    # - 0.5), min, max)`: en rachas de pérdidas se arriesga menos y en rachas
    # ganadoras (mercado en tendencia) se arriesga más. 0 = desactivado.
    adaptive_risk_window: int = 0
    adaptive_risk_k: float = 2.0
    adaptive_risk_min: float = 0.5
    adaptive_risk_max: float = 1.5

    force_close_at_end: bool = True
    timestamp_is_close: bool = False

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
        if self.entry_adx_enabled and (self.entry_adx_len <= 0 or self.entry_adx_min < 0):
            raise ValueError("Configuración de ADX de entrada inválida.")
        if self.adx_exit_enabled:
            if min(self.adx_dilen, self.adx_adxlen) <= 0 or self.adx_key_level < 0:
                raise ValueError("Configuración de ADX de salida inválida.")
            if not 0 < self.adx_exit_pct <= 100:
                raise ValueError("adx_exit_pct debe estar en (0, 100].")
            if self.adx_exit_range < 1:
                raise ValueError("adx_exit_range debe ser >= 1.")
        if self.ema_weakening_exit_enabled:
            if self.ema_weakening_gap_pct <= 0 or not 0 < self.ema_weakening_exit_pct <= 100:
                raise ValueError("Configuración de weakening inválida.")
        if self.partial_tp_enabled:
            if self.partial_tp_r <= 0 or not 0 < self.partial_tp_pct <= 100:
                raise ValueError("Configuración de take-profit parcial inválida.")
        if self.partial_tp2_r > 0 or self.partial_tp2_pct > 0:
            if self.partial_tp2_r <= 0 or not 0 < self.partial_tp2_pct <= 100:
                raise ValueError("Configuración de take-profit parcial 2 inválida.")
            if self.partial_tp2_r <= self.partial_tp_r:
                raise ValueError("partial_tp2_r debe ser mayor que partial_tp_r.")
        if self.close_on_ema_slow_touch:
            if self.close_on_ema_slow_touch_pct < 0:
                raise ValueError("close_on_ema_slow_touch_pct debe ser >= 0.")
            if not 0 < self.close_on_ema_slow_touch_exit_pct <= 100:
                raise ValueError("close_on_ema_slow_touch_exit_pct debe estar en (0, 100].")
        if self.entry_partial_enabled:
            if not (0 < self.entry_partial_pct < 100):
                raise ValueError("entry_partial_pct debe estar en (0, 100).")
            if self.entry_confirm_mode not in ("bars", "close_above_ema", "pullback_retest"):
                raise ValueError("entry_confirm_mode debe ser 'bars', 'close_above_ema' o 'pullback_retest'.")
            if self.entry_confirm_bars < 0:
                raise ValueError("entry_confirm_bars no puede ser negativo.")
            if self.entry_confirm_retest_tol_pct < 0:
                raise ValueError("entry_confirm_retest_tol_pct no puede ser negativo.")
        if not (self.allow_long or self.allow_short):
            raise ValueError("Debe habilitarse LONG, SHORT o ambos.")
        if self.trend_filter not in ("off", "cross", "slow"):
            raise ValueError("trend_filter debe ser 'off', 'cross' o 'slow'.")
        if (self.trend_filter != "off" or self.streak_trend_only) and pd.Timedelta(self.trend_timeframe) <= pd.Timedelta(0):
            raise ValueError("trend_timeframe debe ser positivo.")
        if self.chop_filter:
            if pd.Timedelta(self.chop_timeframe) <= pd.Timedelta(0):
                raise ValueError("chop_timeframe debe ser positivo.")
            if self.chop_adx_min < 0:
                raise ValueError("chop_adx_min no puede ser negativo.")
        if self.monthly_loss_limit_pct < 0:
            raise ValueError("monthly_loss_limit_pct no puede ser negativo.")
        if self.monthly_loss_limit_days < 0:
            raise ValueError("monthly_loss_limit_days no puede ser negativo.")
        if self.consecutive_loss_pause < 0 or self.consecutive_loss_pause_bars < 1:
            raise ValueError("consecutive_loss_pause debe ser >= 0 y ..._bars >= 1.")
        if self.consecutive_loss_skip_signals < 0:
            raise ValueError("consecutive_loss_skip_signals no puede ser negativo.")
        if self.adaptive_risk_window < 0:
            raise ValueError("adaptive_risk_window no puede ser negativo.")
        if self.adaptive_risk_window and not (0 < self.adaptive_risk_min <= self.adaptive_risk_max):
            raise ValueError("Se requiere 0 < adaptive_risk_min <= adaptive_risk_max.")