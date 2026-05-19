#!/usr/bin/env bash
# Supervisor: keep the bot alive across crashes/stalls.
#
# The runtime has internal resilience (decoupled housekeeping, gap-repair),
# but a hard crash or OOM still needs an external restarter. This wraps the
# bot in an until-loop with capped exponential backoff and timestamped logs.
#
# Usage:  ./scripts/run_bot.sh
# Stop:   Ctrl-C (the trap kills the child and exits without restarting).
set -u

cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

backoff=2
max_backoff=60
child_pid=""

cleanup() {
    echo "[supervisor] stopping (signal received)"
    [ -n "$child_pid" ] && kill "$child_pid" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

while true; do
    ts="$(date +%Y%m%d-%H%M%S)"
    log="$LOG_DIR/bot-$ts.log"
    echo "[supervisor] starting bot → $log"

    python -m src.runtime.main 2>&1 | tee "$log" &
    child_pid=$!
    wait "$child_pid"
    code=$?

    echo "[supervisor] bot exited (code=$code); restarting in ${backoff}s"
    sleep "$backoff"
    # Exponential backoff capped at max_backoff; reset is implicit on a
    # long-lived run because the loop only reaches here on exit.
    backoff=$(( backoff * 2 ))
    [ "$backoff" -gt "$max_backoff" ] && backoff="$max_backoff"
done
