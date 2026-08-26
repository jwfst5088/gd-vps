#!/bin/sh
# 房规全自动学习闭环 v2（API 驱动）
#  攒够≥30局新样本 → ①习惯统计报告 ②真人差值有界微调(warm-start)
#  → ③经 /api/v1/learn 训练(有≥5条真人局用 record 真日志模式,否则 selfplay)
#  → ④17键范围校验 ⑤原子换参+备份(留5份) ⑥重启+ping校验(失败回滚)
#  → ⑦发布 public_html/auto_params.json 镜像供 CF 端运行时跟随
BASE=/home/Cooki/domains/gg.meaigo.eu.org
cd "$BASE/clawguandan" || exit 0
LOG=auto_train.log

if [ -f .auto_learn.lock ]; then
  NOW=$(date +%s); MTIME=$(stat -f %m .auto_learn.lock 2>/dev/null || echo 0)
  AGE=$((NOW - MTIME)); [ "$AGE" -lt 13000 ] && exit 0
fi
echo $$ > .auto_learn.lock
trap 'rm -f .auto_learn.lock' EXIT

[ -f game_logs.jsonl ] || exit 0
N=$(grep -c . game_logs.jsonl 2>/dev/null | tr -d " "); [ -z "$N" ] && N=0
LAST=$(cat .learn_last_count 2>/dev/null | tr -d " "); [ -z "$LAST" ] && LAST=0
DIFF=$((N - LAST))
FORCED="$1"
if [ "$FORCED" = "bg" ] || [ "$FORCED" = "forced" ] || [ "$FORCED" = "finalize" ]; then
  :  # 直接进入主流程
else
  { [ "$N" -ge 30 ] && [ "$DIFF" -ge 30 ]; } || exit 0
  LR=$(cat .learn_last_run 2>/dev/null || echo 0); NOW=$(date +%s)
  [ $((NOW - LR)) -lt 259200 ] && exit 0
  nohup /bin/sh "$0" bg >> auto_train.log 2>&1 &
  echo "$(date '+%F %T') 已派发后台学习任务" >> $LOG
  exit 0
fi
echo "$(date '+%F %T') === 触发 N=$N LAST=$LAST ===" >> $LOG

# ②真人差值微调起点（±15%钳制；统计不可用则原参直起）
cp advanced_params.json tune_base.json
python3 - <<'PYEOF' >> $LOG 2>&1 || true
import json,re,subprocess
p=json.load(open('advanced_params.json'))
try:
    out=subprocess.run(['./target/release/clawguandan','stats','--log-file','game_logs.jsonl'],capture_output=True,text=True,timeout=120).stdout
    hs=re.search(r'--- human ---([\s\S]*?)(?:\Z|\n--- )',out); bs=re.search(r'--- bot ---([\s\S]*?)(?:\Z|\n--- )',out)
    def grab(b,k):
        m=re.search(k+r'\s*:\s*([0-9.]+)',b or ''); return float(m.group(1)) if m else None
    if hs and bs:
        h_s,b_s=grab(hs.group(1),'单张出牌均值'),grab(bs.group(1),'单张出牌均值')
        h_l,b_l=grab(hs.group(1),'首出最大点数均值'),grab(bs.group(1),'首出最大点数均值')
        cl=lambda v,lo,hi:max(lo,min(hi,v))
        if h_s and b_s and h_s<b_s: p['low_card_dump_bias']=cl(round(p['low_card_dump_bias']*1.10,4),0.5,3.0)
        if h_l and b_l and h_l>b_l: p['proactive_play_bias']=cl(round(p['proactive_play_bias']*1.08,4),0.5,3.0)
except Exception as e:
    print('nudge-skip:',e)
json.dump(p,open('tune_base.json','w'),indent=2)
PYEOF

# ③训练（服务进程内 API；先暂存原参、以微调点为 warm-start）
if [ "$FORCED" != "finalize" ]; then
cp advanced_params.json advanced_params.pretrain.json
cp tune_base.json advanced_params.json
HUMAN_N=$(grep -c "\"human_seats\":\[[^]]" game_logs.jsonl 2>/dev/null | tr -d " "); [ -z "$HUMAN_N" ] && HUMAN_N=0
if [ "$HUMAN_N" -ge 5 ]; then MODE=record; else MODE=selfplay; fi
[ "$FORCED" = "forced" ] && MODE=selfplay
echo "$(date '+%F %T') start learn mode=$MODE human_entries=$HUMAN_N" >> $LOG
curl -s -m 20 -X POST http://localhost:2230/api/v1/learn -H 'Content-Type: application/json' \
  -d "{\"matchesPerEval\":12,\"iterations\":20,\"mode\":\"$MODE\",\"populationSize\":12}" >> $LOG 2>&1
echo >> $LOG

STATUS=""
i=0
while [ $i -lt 680 ]; do
  sleep 10; i=$((i+1))
  STATUS=$(curl -s -m 10 http://localhost:2230/api/v1/learn/status)
  R=$(echo "$STATUS" | grep -o '"is_running":[a-z]*' | head -1 | cut -d: -f2)
  [ "$R" = "false" ] && break
done
echo "final-status: $(echo "$STATUS" | head -c 300)" >> $LOG

else
  # finalize：daemon 已把新参数写入 advanced_params.json，补建 pretrain 基线
  [ -f advanced_params.pretrain.json ] || cp backups/$(ls -t backups | head -1) advanced_params.pretrain.json 2>/dev/null || true
  [ -f advanced_params.pretrain.json ] || cp advanced_params.json advanced_params.pretrain.json
fi
# ④校验：与 pretrain 比，17键一致、纯数值有限、每项在原值[0.5x,2x]
V=$(python3 - <<'PYEOF' 2>&1
import json,math,sys
try: cur=json.load(open('advanced_params.pretrain.json')); cand=json.load(open('advanced_params.json'))
except Exception as e: print('ERR'); sys.exit(0)
if set(cand.keys())!=set(cur.keys()): print('ERR'); sys.exit(0)
for k,v in cand.items():
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v): print('ERR'); sys.exit(0)
    lo,hi=cur[k]*0.5,cur[k]*2.0
    if not (lo<=v<=hi): print('ERR'); sys.exit(0)
print('OK')
PYEOF
)

ROLLBACK=0
MD5_OLD=$(md5 -q advanced_params.pretrain.json); MD5_NEW=$(md5 -q advanced_params.json)
if [ "$V" = "OK" ] && ! curl -sf -m 5 http://localhost:2230/ping >/dev/null 2>&1; then V="ERR"; fi
if [ "$V" = "OK" ]; then
  TS=$(date +%Y%m%d_%H%M%S); mkdir -p backups
  cp advanced_params.pretrain.json "backups/advanced_${TS}.json"
  ls -t backups/advanced_*.json 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
  echo "$(date '+%F %T') 参数更新生效 old=$MD5_OLD new=$MD5_NEW mode=$MODE" >> $LOG
else
  ROLLBACK=1
  cp advanced_params.pretrain.json advanced_params.json
  echo "$(date '+%F %T') 校验未通过(v=$V) 保持原参 mode=$MODE" >> $LOG
fi
rm -f advanced_params.pretrain.json tune_base.json

# ⑥发布镜像（CF 端每10分钟拉取跟随）
cp advanced_params.json "$BASE/public_html/auto_params.json" 2>/dev/null
echo "$N" > .learn_last_count
date +%s > .learn_last_run
echo "$(date '+%F %T') 完成 rollback=$ROLLBACK 镜像已发布" >> $LOG
exit 0