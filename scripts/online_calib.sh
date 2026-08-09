#!/usr/bin/env bash
# DST-Calib 在线动态标定一键入口。
# 默认 fast 滑窗持续更新；ROS2 主进程收流，conda CUDA 子进程异步标定。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DST_ROOT="${DST_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/lyrical/setup.bash}"
LIVOX_INSTALL="${LIVOX_INSTALL:-$DST_ROOT/../Super-LIO/install/setup.bash}"
ORBBEC_INSTALL="${ORBBEC_INSTALL:-$DST_ROOT/ros2_ws/install/setup.bash}"
DEFAULT_CKPT="$DST_ROOT/runs/train_eva/best.pt"
LIDAR_IP="192.168.1.12"

MODE="${ONLINE_MODE:-fast}"
EVA_CKPT="${EVA_CKPT:-$DEFAULT_CKPT}"
SKIP_DRIVERS=0
OUT=""
FORWARD=()

usage() {
  cat <<'EOF'
用法: online_calib.sh [选项]

  --mode fast|both|selfsup  在线推荐 fast；both/selfsup 更新较慢
  --eva-ckpt PATH           评估模块断点，默认 runs/train_eva/best.pt
  --init-yaml PATH          首个滑窗的粗外参先验
  --window-size N           滑窗帧数
  --window-stride N         每新增 N 帧触发一次
  --out DIR                 输出目录，默认 runs/online_<时间戳>
  --save-windows            保留每次提交的原始滑窗
  --skip-drivers            使用已经启动的话题，不启动 Livox/Orbbec 驱动

其余 online_calib_node.py 参数也会原样传递。
运行中服务:
  ros2 service call /dst_calib_online/trigger std_srvs/srv/Trigger '{}'
  ros2 service call /dst_calib_online/reset_filter std_srvs/srv/Trigger '{}'
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; FORWARD+=("$1" "$2"); shift 2 ;;
    --eva-ckpt)
      EVA_CKPT="$2"; FORWARD+=("$1" "$2"); shift 2 ;;
    --out)
      OUT="$2"; shift 2 ;;
    --skip-drivers)
      SKIP_DRIVERS=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      FORWARD+=("$1")
      if [[ $# -ge 2 && "$2" != --* ]]; then
        FORWARD+=("$2"); shift 2
      else
        shift
      fi
      ;;
  esac
done

case "$MODE" in
  fast|both|selfsup|auto) ;;
  *) echo "[错误] --mode 应为 fast|both|selfsup|auto"; exit 2 ;;
esac
if [[ "$MODE" == "fast" || "$MODE" == "both" ]]; then
  if [[ ! -f "$EVA_CKPT" ]]; then
    echo "[错误] 在线 $MODE 模式需要评估模块断点: $EVA_CKPT"
    echo "请先运行 scripts/train_eval_module.sh，或临时使用 --mode selfsup。"
    exit 1
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT:-$DST_ROOT/runs/online_$TS}"
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(readlink -f "$RUN_DIR")"
export LIVOX_CFG="$RUN_DIR/livox_calib_config.json"

LIVOX_PID=""
ORBBEC_PID=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in $LIVOX_PID $ORBBEC_PID; do
    [[ -n "$pid" ]] || continue
    kill -INT "$pid" 2>/dev/null || true
  done
  if [[ -n "$LIVOX_PID$ORBBEC_PID" ]]; then
    sleep 2
  fi
  for pid in $LIVOX_PID $ORBBEC_PID; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" 2>/dev/null || true
  done
  exit "$code"
}
trap cleanup EXIT INT TERM

set +u
source "$ROS_SETUP"
source "$LIVOX_INSTALL"
[[ -f "$ORBBEC_INSTALL" ]] && source "$ORBBEC_INSTALL"
set -u

if [[ $SKIP_DRIVERS -eq 0 ]]; then
  echo "[1/3] 检查 Mid-360 网络..."
  if ! ping -c1 -W1 "$LIDAR_IP" >/dev/null 2>&1; then
    echo "[错误] 无法访问 $LIDAR_IP；请确认主机已配置 192.168.1.x 地址。"
    exit 1
  fi
  echo "[2/3] 启动 Mid-360 与 Gemini 335 驱动..."
  ros2 launch "$DST_ROOT/scripts/livox_mid360_calib.launch.py" \
    >"$RUN_DIR/logs/livox.log" 2>&1 &
  LIVOX_PID=$!
  ros2 launch orbbec_camera gemini_330_series.launch.py \
    depth_registration:=true enable_frame_sync:=true \
    >"$RUN_DIR/logs/orbbec.log" 2>&1 &
  ORBBEC_PID=$!
  sleep 4
  kill -0 "$LIVOX_PID" 2>/dev/null || {
    echo "[错误] Livox 驱动退出，见 $RUN_DIR/logs/livox.log"; exit 1; }
  kill -0 "$ORBBEC_PID" 2>/dev/null || {
    echo "[错误] Orbbec 驱动退出，见 $RUN_DIR/logs/orbbec.log"; exit 1; }
else
  echo "[1/3] 使用现有 ROS2 话题，不启动驱动。"
fi

echo "[3/3] 启动在线动态标定: mode=$MODE, out=$RUN_DIR"
NODE_CMD=(
  python3 "$DST_ROOT/scripts/online_calib_node.py"
  --config "$DST_ROOT/config/default.yaml"
  --mode "$MODE"
  --out "$RUN_DIR"
)
if [[ -n "$EVA_CKPT" ]]; then
  NODE_CMD+=(--eva-ckpt "$EVA_CKPT")
fi
NODE_CMD+=("${FORWARD[@]}")
"${NODE_CMD[@]}"
