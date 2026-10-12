#!/usr/bin/env bash
#
# dev.sh — atajos locales (sin Docker) para bot1.
#
# Genera las bases de datos en este mismo directorio:
#   ./feed.db     (feed.py)
#   ./signal.db   (bot.py)
#
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

case "${1:-help}" in
  install)
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    ;;
  feed)  exec "$PY" feed.py ;;
  bot)   exec "$PY" bot.py ;;
  once)  "$PY" bot.py --once ;;
  up)
    "$PY" feed.py > >(sed -u 's/^/[feed] /') 2>&1 & FEED=$!
    "$PY" bot.py  > >(sed -u 's/^/[bot]  /') 2>&1 & BOT=$!
    trap 'kill $FEED $BOT 2>/dev/null; exit 0' INT TERM
    wait
    ;;
  status)
    "$PY" - <<'PY'
import os, sqlite3
for path, sql in [("feed.db", "SELECT COUNT(*) FROM candles"),
                  ("signal.db", "SELECT COUNT(*) FROM signals")]:
    if os.path.exists(path):
        n = sqlite3.connect(path).execute(sql).fetchone()[0]
        print(f"{path}: {n}")
    else:
        print(f"{path}: no existe")
PY
    ;;
  *)
    cat <<'EOF'
Uso: ./dev.sh <comando>

  install   Crea .venv e instala dependencias
  up        Arranca feed y bot y muestra sus logs juntos (Ctrl-C para parar)
  feed      Corre feed.py  -> ./feed.db
  bot       Corre bot.py   -> ./signal.db
  once      Un ciclo del bot y sale
  status    Velas en feed.db y eventos en signal.db
EOF
    ;;
esac
