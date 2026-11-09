#!/usr/bin/env bash
# Train one jduck variant, export it, evaluate it. Results → logs/pipeline/<name>.txt
#   scripts/jduck_pipeline.sh <name> [iters] -- KEY=VAL ...    (JDUCK_* env vars of the run)
# Fixed: firmware K16 FF0 DB0, 0.96 N·m, 4.8 rad/s, 4096 envs, seed from SEED (default 1).
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; iters=${2:-4000}; shift 2; [ "${1:-}" = "--" ] && shift
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs/pipeline
out=logs/pipeline/$name.txt
envs=(JDUCK_FW=16,0,0 JDUCK_NOLOAD=4.8 JDUCK_TORQUE=0.96 WANDB_MODE=offline "$@")   # later JDUCK_FW wins
fw=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_FW=//p' | tail -1)
K=${fw%%,*}
ksweep=$(python3 -c "k=float('$K'); print(f'{k*0.75:g},{k*1.25:g}')")
[ "$fw" = orig ] && ksweep=""
fr=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_FRICTION=//p' | tail -1)
frarg=""; [ -n "$fr" ] && frarg="--friction $fr"
echo "== $name  $(date +%F_%T)  ${envs[*]}  iters=$iters seed=${SEED:-1}" > "$out"
env -u PYTHONPATH "${envs[@]}" uv run --frozen train Mjlab-Velocity-Flat-JDuck --env.scene.num-envs 4096 \
    --agent.max_iterations "$iters" --agent.seed "${SEED:-1}" --agent.run-name "$name" \
    --agent.logger tensorboard > "logs/pipeline/$name.train.log" 2>&1
grep -a -o 'Learning iteration [0-9]*' "logs/pipeline/$name.train.log" | tail -1 >> "$out"
grep -a -E 'Mean episode length|fell_over|track_angular_velocity|stand_action_rate' "logs/pipeline/$name.train.log" | tail -4 >> "$out"
d=$(ls -td logs/rsl_rl/jduck_velocity*/*_"$name"/ 2>/dev/null | head -1)
[ -n "$d" ] || { echo "no run dir" >> "$out"; exit 1; }
echo "run: $d" >> "$out"
env -u PYTHONPATH uv run --frozen python - "$d" >> "$out" 2>/dev/null <<'PY'
import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); ea.Reload(); tags = ea.Tags()["scalars"]
T = [("Train/mean_episode_length", "回合长度"), ("Episode_Termination/fell_over", "摔倒"),
     ("Episode_Reward/air_time", "air_time"), ("Episode_Reward/track_linear_velocity", "线速度跟踪"),
     ("Episode_Reward/track_angular_velocity", "角速度跟踪"), ("Episode_Reward/stand_quiet", "安静站立")]
last = max(e.step for e in ea.Scalars("Train/mean_episode_length"))
parts = []
for t, n in T:
    if t in tags:
        v = [e.value for e in ea.Scalars(t) if e.step > last - 100]
        parts.append(f"{n} {np.mean(v):.3f}")
print("训练末期(最后100次): " + "  ".join(parts) + "   [参考 r9: 回合长度 971 摔倒 0.25 air_time 1.04]")
PY
last=$(( iters - 1 ))
onnx=jduck_$name.onnx
scripts/jduck_export.sh "$d" "$last" "$onnx" 2>&1 | grep -E 'ONNX ok|训练环境|失败|键盘玩' >> "$out"
JDUCK_EVAL_FW=$fw env -u PYTHONPATH uv run --frozen python scripts/jduck_eval.py "$onnx" 2>&1 \
    | grep -v -E 'Warp|mdp\]|Vel cmd' >> "$out"
env -u PYTHONPATH uv run --frozen python ../../jduck_tools/hw_chain_eval.py "$onnx" --fw "$fw" --k-sweep "$ksweep" $frarg 2>&1 | grep -v -E 'Warp|mdp\]' >> "$out"
echo "== done $(date +%F_%T)" >> "$out"
