#!/bin/bash
WATCH_DIR="$HOME/bot_trigger"
BOT_BIN="$HOME/domains/gg.meaigo.eu.org/clawguandan/target/release/clawguandan"
echo "[watcher] PID=$$ started, watching $WATCH_DIR"
rm -f "$WATCH_DIR"/*.trigger

while true; do
  FILE=$(inotifywait -q -e create,moved_to --format "%f" "$WATCH_DIR" 2>/dev/null)
  case "$FILE" in
    *.trigger)
      TABLE_ID=$(cat "$WATCH_DIR/$FILE" 2>/dev/null)
      rm -f "$WATCH_DIR/$FILE"
      if [ -n "$TABLE_ID" ]; then
        echo "[watcher] Starting bot for table: $TABLE_ID"
        nohup "$BOT_BIN" bot beat-it -t "$TABLE_ID" --hands 100 > /dev/null 2>&1 &
      fi
      ;;
  esac
done