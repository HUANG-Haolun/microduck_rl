#!/usr/bin/env bash
# Export a jduck training run to ONNX, reproducing the training env (JDUCK_*)
# from the experiment directory name.
#
#   scripts/jduck_export.sh                                  # newest jduck run, newest checkpoint
#   scripts/jduck_export.sh <run_dir>                        # that run, newest checkpoint
#   scripts/jduck_export.sh <run_dir> 1999                   # that run, model_1999.pt
#   scripts/jduck_export.sh <run_dir> 1999 out.onnx          # choose the output name
#   EVAL=1 scripts/jduck_export.sh ...                       # also run scripts/jduck_eval.py
#
# Run from third_party/microduck_rl. <run_dir> is any directory that contains
# model_*.pt, e.g. logs/rsl_rl/jduck_velocity_ubt12hc_k16ff15db1_t0.96_w4.8_homenew/2026-11-03_17-07-35_full_seed1_cont2
#
# The experiment directory name encodes the env (see jduck_constants.py /
# jduck_velocity_env_cfg.py):
#   jduck_velocity_pd_home{new,old}                       → JDUCK_ACTUATOR=pd
#   jduck_velocity_ubt12hc_k16ff15db1[_t0.96][_w4.8]_homenew
#                                                         → JDUCK_FW=16,15,1 JDUCK_TORQUE=0.96 JDUCK_NOLOAD=4.8
#   jduck_velocity (no suffix, v0)                        → JDUCK_ACTUATOR=pd JDUCK_HOME=old
set -euo pipefail
cd "$(dirname "$0")/.."

run="${1:-}"
if [ -z "$run" ]; then
    run=$(ls -td logs/rsl_rl/jduck_velocity*/*/ 2>/dev/null | head -1)
    [ -n "$run" ] || { echo "没找到 jduck 训练目录（logs/rsl_rl/jduck_velocity*/）" >&2; exit 1; }
fi
run="${run%/}"
[ -d "$run" ] || { echo "目录不存在：$run" >&2; exit 1; }

if [ -n "${2:-}" ]; then
    ckpt="$run/model_$2.pt"
else
    ckpt=$(ls "$run"/model_*.pt 2>/dev/null | sort -V | tail -1)
fi
[ -f "$ckpt" ] || { echo "没有 checkpoint：$ckpt" >&2; exit 1; }
it=$(basename "$ckpt" .pt); it="${it#model_}"

exp=$(basename "$(dirname "$run")")
envs=()
case "$exp" in
    jduck_velocity)          envs=(JDUCK_ACTUATOR=pd JDUCK_HOME=old) ;;
    jduck_velocity_pd_*)     envs=(JDUCK_ACTUATOR=pd) ;;
    jduck_velocity_ubt12hc*) envs=(JDUCK_ACTUATOR=ubt12hc) ;;
    jduck_sitstand_ubt12hc*) envs=(JDUCK_ACTUATOR=ubt12hc) ;;
    *) echo "不认识的实验名：$exp（不是 jduck 训练？）" >&2; exit 1 ;;
esac
[[ "$exp" =~ _home(new|old) ]] && envs+=("JDUCK_HOME=${BASH_REMATCH[1]}")
if [[ "$exp" =~ _k([0-9.]+)ff([0-9.]+)db([0-9.]+)fl([0-9.]+) ]]; then
    envs+=("JDUCK_FW=${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]},${BASH_REMATCH[4]}")
elif [[ "$exp" =~ _k([0-9.]+)ff([0-9.]+)db([0-9.]+) ]]; then
    envs+=("JDUCK_FW=${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]}")
fi
[[ "$exp" =~ _t([0-9.]+) ]] && envs+=("JDUCK_TORQUE=${BASH_REMATCH[1]}")
[[ "$exp" =~ _w([0-9.]+) ]] && envs+=("JDUCK_NOLOAD=${BASH_REMATCH[1]}")
[[ "$exp" == *_noquant* ]] && envs+=("JDUCK_QUANT=0")
if [[ "$exp" == *_stand2* ]]; then envs+=("JDUCK_STAND=v2")
elif [[ "$exp" == *_standpen* ]]; then envs+=("JDUCK_STAND=v1")
else envs+=("JDUCK_STAND=0"); fi
if [[ "$exp" =~ _sar([0-9.]+)from([0-9]+) ]]; then
    envs+=("JDUCK_STAND_ACT_RATE=${BASH_REMATCH[1]}" "JDUCK_SAR_START=${BASH_REMATCH[2]}")
elif [[ "$exp" =~ _sar([0-9.]+) ]]; then
    envs+=("JDUCK_STAND_ACT_RATE=${BASH_REMATCH[1]}" "JDUCK_SAR_START=0")
fi
[[ "$exp" =~ _quiet([0-9.]+) ]] && envs+=("JDUCK_STAND_QUIET=${BASH_REMATCH[1]}")
[[ "$exp" =~ _quiet[0-9.]+s([0-9.]+) ]] && envs+=("JDUCK_STAND_QUIET_S2=${BASH_REMATCH[1]}")
[[ "$exp" =~ _term([0-9.]+) ]] && envs+=("JDUCK_TERM=${BASH_REMATCH[1]}")
[[ "$exp" == *_adcobs* ]] && envs+=("JDUCK_OBS_ADC=1")
[[ "$exp" =~ _qdema([0-9.]+) ]] && envs+=("JDUCK_QD_EMA=${BASH_REMATCH[1]}")
[[ "$exp" =~ _lag([0-9]+) ]] && envs+=("JDUCK_OBS_LAG=${BASH_REMATCH[1]}")
if [[ "$exp" =~ _capsS([0-9.]+)T([0-9.]+) ]]; then envs+=("JDUCK_CAPS_S=${BASH_REMATCH[1]}" "JDUCK_CAPS_T=${BASH_REMATCH[2]}"); fi
[[ "$exp" =~ _capsS[0-9.]+T[0-9.]+(b[0-9.]+)?w([0-9.]+) ]] && envs+=("JDUCK_CAPS_WALK=${BASH_REMATCH[2]}")
[[ "$exp" =~ _capsS[0-9.]+T[0-9.]+b([0-9.]+) ]] && envs+=("JDUCK_CAPS_STAND=${BASH_REMATCH[1]}")
[[ "$exp" =~ _capsS[0-9.]+T[0-9.]+[bw0-9.]*c([0-9.]+) ]] && envs+=("JDUCK_CAPS_CALM=${BASH_REMATCH[1]}")
[[ "$exp" =~ _capsS[0-9.]+T[0-9.]+[bwc0-9.]*s([0-9]+) ]] && envs+=("JDUCK_CAPS_START=${BASH_REMATCH[1]}")
[[ "$exp" =~ _cd([0-9]+) ]] && envs+=("JDUCK_CURR_DELAY=${BASH_REMATCH[1]}")
[[ "$exp" =~ _fr([0-9.]+)(_|$) ]] && envs+=("JDUCK_FRICTION=${BASH_REMATCH[1]}")
[[ "$exp" =~ _frdr([0-9.]+)-([0-9.]+) ]] && envs+=("JDUCK_FRICTION_DR=${BASH_REMATCH[1]},${BASH_REMATCH[2]}")
[[ "$exp" =~ _kpdr([0-9.]+)-([0-9.]+) ]] && envs+=("JDUCK_KP_DR=${BASH_REMATCH[1]},${BASH_REMATCH[2]}")
[[ "$exp" =~ _angema([0-9.]+) ]] && envs+=("JDUCK_ANG_EMA=${BASH_REMATCH[1]}")

out="${3:-jduck_${exp#jduck_velocity_}_$(basename "$run" | cut -c1-19)_it${it}.onnx}"
out="${out//__/_}"

echo "checkpoint : $ckpt"
echo "训练环境   : ${envs[*]}"
echo "输出       : $out"
task=Mjlab-Velocity-Flat-JDuck
[[ "$exp" == jduck_sitstand_* ]] && task=Mjlab-SitStand-Flat-JDuck
env "${envs[@]}" WANDB_MODE=offline uv run --frozen scripts/export.py $task \
    --checkpoint-file "$ckpt" --onnx-file "$out" > /tmp/jduck_export.log 2>&1 \
    || { echo "导出失败，日志：/tmp/jduck_export.log" >&2; tail -20 /tmp/jduck_export.log >&2; exit 1; }

uv run --frozen python - "$out" <<'EOF' 2>/dev/null
import sys, onnxruntime as ort
s = ort.InferenceSession(sys.argv[1]); m = s.get_modelmeta().custom_metadata_map
print(f"ONNX ok    : obs {s.get_inputs()[0].shape} -> actions {s.get_outputs()[0].shape}")
print(f"HOME       : {m.get('default_joint_pos')}")
EOF

if [[ "$exp" == jduck_sitstand_* ]]; then
    fw=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_FW=//p')
    play="uv run --frozen scripts/infer_policy.py --robot jduck --sitstand $out --jduck-fw $fw   # Y = 坐下 / 站起"
    evalarg=""
elif [[ "$exp" == jduck_velocity_ubt12hc* ]]; then
    fw=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_FW=//p'); fw=${fw:-orig}
    play="uv run --frozen scripts/infer_policy.py --robot jduck --walking $out --jduck-fw $fw"
    t=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_TORQUE=//p'); [ -n "$t" ] && play+=" --jduck-torque $t"
    w=$(printf '%s\n' "${envs[@]}" | sed -n 's/^JDUCK_NOLOAD=//p'); [ -n "$w" ] && play+=" --jduck-noload $w"
    evalarg="$out"
else
    play="uv run --frozen scripts/infer_policy.py --robot jduck --walking $out --jduck-fw ''"
    evalarg="pd:$out"
fi
echo
echo "键盘玩     : $play"

if [ "${EVAL:-0}" = 1 ] && [ -n "$evalarg" ]; then
    echo; echo "== 测评 =="
    uv run --frozen python scripts/jduck_eval.py "$evalarg" 2>&1 | grep -v -E 'Warp|mdp\]|Vel cmd'
fi
