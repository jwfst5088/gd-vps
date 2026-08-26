#!/bin/sh
# 房规：样本自动统计——累计新增≥30局时自动生成真人vsAI习惯对照报告
# 报告：clawguandan/stats_report.txt（私有）+ public_html/stats_report.txt（浏览器可看）
BASE=/home/Cooki/domains/gg.meaigo.eu.org
cd "$BASE/clawguandan" || exit 0
[ -f game_logs.jsonl ] || exit 0
N=$(grep -c . game_logs.jsonl 2>/dev/null | tr -d " ")
[ -z "$N" ] && N=0
LAST=$(cat .stats_last_count 2>/dev/null | tr -d " ")
[ -z "$LAST" ] && LAST=0
DIFF=$((N - LAST))
if [ "$N" -ge 30 ] && [ "$DIFF" -ge 30 ]; then
  {
    echo "=== $(date "+%Y-%m-%d %H:%M") 样本 $LAST -> $N ==="
    ./target/release/clawguandan stats --log-file game_logs.jsonl 2>&1
    echo
  } >> stats_report.txt
  echo "$N" > .stats_last_count
  cp stats_report.txt "$BASE/public_html/stats_report.txt" 2>/dev/null
fi
