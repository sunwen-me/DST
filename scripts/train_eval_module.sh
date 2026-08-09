#!/usr/bin/env bash
# =============================================================================
# DST-Calib 评估模块一键训练 (论文 III-D-1, 完整论文形态入口, 契约见 INTERFACES.md 附录 A4)
# 流程: 参数解析/收集 session → 环境与 GPU 检查 → 逐 session 校验帧数与基准外参 →
#       数据摘要 + 时长估算 → conda python -m dst_calib.train (多 session) →
#       打印 best.pt 路径与最优损失, 提示 one_click 下次自动进入 both 模式
# 产物: runs/train_eva/ (best.pt / checkpoint.pt 断点续训 / model_final.pt)
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------ 常量
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DST_ROOT="${DST_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
CONDA_PY="${CONDA_PY:-$(command -v python3 || true)}"
DEFAULT_OUT="$DST_ROOT/runs/train_eva"   # calibrate --eva_ckpt 的默认断点位置

usage() {
  cat <<'EOF'
用法: train_eval_module.sh [session_dir ...] [-- 额外train参数]
  session_dir   采集 session, 两种形式均可:
                  runs/<时间戳>          one_click_calib.sh 的 run 目录 (自动取其 data/)
                  runs/<时间戳>/data     直接含 frame_*.npz 的采集目录
                无参数时自动收集 $DST_ROOT/runs/*/ 下同时满足
                「data/ 内有 frame_*.npz」且「有 extrinsic.yaml (标定成功)」的 run。
  --            其后参数原样透传给 python -m dst_calib.train, 例如:
                  ./scripts/train_eval_module.sh -- --epochs 100 --arch db --num_workers 4
说明:
  每个 session 的 extrinsic.yaml (calibrate.py 输出) 作为该 session 的基准外参;
  训练完成后 one_click_calib.sh 在 CALIB_MODE=auto (默认) 下检测到
  runs/train_eva/best.pt 会自动进入 both 双路径模式, CALIB_MODE=fast 可秒级标定。
  建议先在 3-5 个结构不同的场景各跑一次 one_click_calib.sh 再来训练 (见 README 第 8 节)。
EOF
}

# ------------------------------------------------------------------ 参数
SESSIONS_IN=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --)        shift; EXTRA=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    -*)        echo "[错误] 未知参数: $1 (dst_calib.train 的参数请放在 -- 之后)"; usage; exit 1 ;;
    *)         SESSIONS_IN+=("$1"); shift ;;
  esac
done
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
  if [[ "$a" == "--data_dir" || "$a" == --data_dir=* ]]; then
    echo "[错误] -- 之后不要再传 --data_dir, session 目录请作为位置参数放在 -- 之前"; exit 1
  fi
done

# 从透传参数中取某选项的值 (支持 "--opt VAL" 与 "--opt=VAL"), 取不到时返回默认值
extra_opt() {
  local name="$1" def="$2" i n=${#EXTRA[@]}
  for ((i = 0; i < n; i++)); do
    if [[ "${EXTRA[i]}" == "$name" && $((i + 1)) -lt $n ]]; then
      echo "${EXTRA[i + 1]}"; return
    elif [[ "${EXTRA[i]}" == "$name="* ]]; then
      echo "${EXTRA[i]#*=}"; return
    fi
  done
  echo "$def"
}

# run 目录或采集目录 → 实际含 frame_*.npz 的目录; 无帧时输出空串
resolve_session() {
  local d="${1%/}"
  if compgen -G "$d/frame_*.npz" >/dev/null; then
    echo "$d"
  elif compgen -G "$d/data/frame_*.npz" >/dev/null; then
    echo "$d/data"
  else
    echo ""
  fi
}

# ------------------------------------------------------------------ (1) 收集 session
echo "[1/5] 收集训练 session..."
SESSIONS=()
if [[ ${#SESSIONS_IN[@]} -gt 0 ]]; then
  for s in "${SESSIONS_IN[@]}"; do
    if [[ ! -d "$s" ]]; then
      echo "[错误] session 目录不存在: $s"; exit 1
    fi
    r="$(resolve_session "$s")"
    if [[ -z "$r" ]]; then
      echo "[错误] $s 及其 data/ 子目录中均没有 frame_*.npz —— 不是有效采集 session"; exit 1
    fi
    # 绝对化: 第 (4) 步训练在 (cd $DST_ROOT) 子 shell 中执行, 相对路径若按
    # 调用者 cwd 收集会被重新解析到错误位置 (校验/软链接/训练必须看到同一目录)
    SESSIONS+=("$(readlink -f "$r")")
  done
else
  # 无参数: 收集 runs/*/ 下标定成功的 run (实际布局: 帧在 data/, extrinsic.yaml 在 run 根)
  for run_dir in "$DST_ROOT"/runs/*/; do
    [[ -d "$run_dir" ]] || continue
    run_dir="${run_dir%/}"
    [[ -f "$run_dir/extrinsic.yaml" ]] || continue
    compgen -G "$run_dir/data/frame_*.npz" >/dev/null || continue
    SESSIONS+=("$run_dir/data")
  done
  if [[ ${#SESSIONS[@]} -eq 0 ]]; then
    echo "[错误] $DST_ROOT/runs/ 下没有找到任何合格 session"
    echo "  合格条件: runs/<时间戳>/ 同时含 data/frame_*.npz 与 extrinsic.yaml (即标定成功的 run)。"
    echo "  请先在几个不同场景各运行一次 one_click_calib.sh, 或显式传入 session 目录。"
    exit 1
  fi
fi
# 去重 (显式传参可能重复)
declare -A _seen=()
UNIQ=()
for s in "${SESSIONS[@]}"; do
  if [[ -n "${_seen[$s]:-}" ]]; then
    echo "[警告] session 重复, 忽略: $s"
  else
    _seen[$s]=1; UNIQ+=("$s")
  fi
done
SESSIONS=("${UNIQ[@]}")
echo "  共 ${#SESSIONS[@]} 个 session"

# ------------------------------------------------------------------ (2) 环境与 GPU 检查
echo "[2/5] 环境检查..."
if [[ ! -x "$CONDA_PY" ]]; then
  echo "[错误] 找不到 conda 解释器 $CONDA_PY —— conda env dstcalib 未创建?"; exit 1
fi
if ! GPU_DESC="$("$CONDA_PY" - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"; then
  echo "[错误] $CONDA_PY 无法导入 torch —— conda env dstcalib 损坏?"; exit 1
fi
if [[ -n "$GPU_DESC" ]]; then
  echo "  GPU: $GPU_DESC"
else
  echo "[警告] 未检测到 CUDA GPU —— 将在 CPU 上训练, 慢 10-20 倍;"
  echo "  建议换有 GPU 的机器训练, 再把 best.pt 拷回 $DEFAULT_OUT/。"
fi

# ------------------------------------------------------------------ (3) 数据校验与摘要
echo "[3/5] 逐 session 校验与数据摘要:"
TOTAL_FRAMES=0
for s in "${SESSIONS[@]}"; do
  n="$(find "$s" -maxdepth 1 -name 'frame_*.npz' | wc -l)"
  # 基准外参: dst_calib.train 只认 session 目录内的 extrinsic.yaml; one_click 布局
  # 把它放在 run 根 (runs/<ts>/), 故在此补一个相对软链接桥接两种布局
  ext=""
  if [[ -e "$s/extrinsic.yaml" ]]; then
    ext="$s/extrinsic.yaml"
  elif [[ -f "$(dirname "$s")/extrinsic.yaml" ]]; then
    if ln -s "../extrinsic.yaml" "$s/extrinsic.yaml" 2>/dev/null; then
      ext="$s/extrinsic.yaml (链接 ← 父目录)"
    else
      echo "[警告]   无法在 $s 内创建 extrinsic.yaml 软链接, 该 session 将退回名义外参"
    fi
  fi
  printf '  %-52s %3d 帧  基准外参: %s\n' "$s" "$n" "${ext:-名义外参 (未找到 extrinsic.yaml)}"
  if [[ $n -lt 4 ]]; then
    echo "[警告]   $s 仅 $n 帧, 样本多样性不足 (one_click 默认每 run 12 帧)"
  fi
  TOTAL_FRAMES=$((TOTAL_FRAMES + n))
done
if [[ ${#SESSIONS[@]} -lt 3 ]]; then
  echo "[警告] 仅 ${#SESSIONS[@]} 个场景 —— 评估模块泛化可能不足, 建议 3-5 个结构不同的场景"
fi
EPOCHS="$(extra_opt --epochs 200)"
SPF="$(extra_opt --samples_per_frame 32)"
EST="$(awk -v e="$EPOCHS" -v f="$TOTAL_FRAMES" -v s="$SPF" \
       'BEGIN { n = e * f * s; printf "%.1f-%.1f", n*0.02/3600, n*0.06/3600 }')"
echo "  总帧数 $TOTAL_FRAMES, 每 epoch 样本 $((TOTAL_FRAMES * SPF)), 共 $EPOCHS epochs"
echo "  预计训练时长 (RTX 3060 级 GPU): 约 ${EST} 小时 (估算式见 README 第 8 节; CPU 慢 10-20 倍)"

# ------------------------------------------------------------------ (4) 训练
OUT_DIR="$(extra_opt --out "$DEFAULT_OUT")"
[[ "$OUT_DIR" != /* ]] && OUT_DIR="$DST_ROOT/$OUT_DIR"   # train 在 DST_ROOT 下运行, 相对路径按此解析
CMD=("$CONDA_PY" -m dst_calib.train --data_dir "${SESSIONS[@]}")
if [[ "$(extra_opt --out __none__)" == "__none__" ]]; then
  CMD+=(--out "$OUT_DIR")
fi
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})
echo "[4/5] 训练 (中断后重跑本脚本可从 $OUT_DIR/checkpoint.pt 断点续训):"
echo "  ${CMD[*]}"
if ! ( cd "$DST_ROOT" && PYTHONPATH="$DST_ROOT" "${CMD[@]}" ); then
  echo "[错误] 训练失败 —— 常见原因:"
  echo "  某 session 帧点云/深度点过少 (上方 dst_calib.train 有具体帧号) /"
  echo "  显存不足 (可 -- --batch_size 4) / 断点 arch 与 --arch 不一致 (删 $OUT_DIR/checkpoint.pt 重训)"
  exit 1
fi

# ------------------------------------------------------------------ (5) 结果摘要
BEST="$OUT_DIR/best.pt"
if [[ ! -f "$BEST" ]]; then
  echo "[错误] 训练结束但未生成 $BEST"; exit 1
fi
echo "[5/5] 结果摘要:"
"$CONDA_PY" - "$BEST" <<'PY'
import sys
import torch
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("=" * 62)
print(f"最优断点: {sys.argv[1]}")
print(f"  arch = {st.get('arch')}, epoch = {st.get('epoch')}, "
      f"最优损失 L_eva(式16) = {float(st.get('loss', float('nan'))):.4f}")
vc = st.get("virtual_camera")
if vc:
    print(f"  virtual_camera = {vc}")
print("=" * 62)
PY
if [[ "$BEST" == "$DEFAULT_OUT/best.pt" ]]; then
  echo "下次运行 one_click_calib.sh: CALIB_MODE=auto (默认) 将检测到该断点自动进入 both 双路径模式;"
  echo "  CALIB_MODE=fast 可跳过自监督优化, 秒级标定。"
else
  echo "[提示] 断点不在默认位置 $DEFAULT_OUT/best.pt, calibrate 不会自动使用它;"
  echo "  需显式指定 --eva_ckpt $BEST, 或将其拷贝/链接到默认位置。"
fi
exit 0
