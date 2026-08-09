#!/usr/bin/env bash
# =============================================================================
# DST-Calib 一键标定 (arXiv 2601.01188 复现)
# 流程: 参数解析 → 网络自检 → 生成 livox JSON → 启动驱动 → 等话题 →
#       采集(系统 python3) → 标定(conda python, 模式由 CALIB_MODE 控制) →
#       结果摘要/可视化 → 清理
# 产物统一进 runs/<时间戳>/ : data/ extrinsic.yaml overlay_*.png vis/ logs/
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------ 常量
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DST_ROOT="${DST_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SCRIPTS_DIR="$DST_ROOT/scripts"
ROS_SETUP="${ROS_SETUP:-/opt/ros/lyrical/setup.bash}"
LIVOX_INSTALL="${LIVOX_INSTALL:-$DST_ROOT/../Super-LIO/install/setup.bash}"
ORBBEC_INSTALL="${ORBBEC_INSTALL:-$DST_ROOT/ros2_ws/install/setup.bash}"
CONDA_PY="${CONDA_PY:-$(command -v python3 || true)}"
LIDAR_IP="192.168.1.12"
EXPECT_HOST_IP="192.168.1.5"
NET_DEV="enp6s0"
CALIB_MODE="${CALIB_MODE:-auto}"   # 标定模式, 传给 dst_calib.calibrate --mode (见 usage)

# ------------------------------------------------------------------ 参数
FRAMES=""
OUT=""
SKIP_DRIVERS=0
INIT_YAML=""

usage() {
  cat <<'EOF'
用法: one_click_calib.sh [--frames N] [--out DIR] [--skip-drivers] [--init 先验外参.yaml]
  --frames N      采集帧数 (默认取 config/default.yaml capture.num_frames=12)
  --out DIR       输出目录 (默认 $DST_ROOT/runs/<时间戳>)
  --skip-drivers  跳过网络自检与驱动启动 (假设 livox/orbbec 话题已在发布)
  --init YAML     含 T_cam_lidar 的先验外参 yaml, 跳过粗搜索直接精化

环境变量:
  CALIB_MODE=auto|selfsup|fast|both  标定模式 (默认 auto), 传给 dst_calib.calibrate --mode:
    selfsup  纯自监督由粗到细优化 —— 无需任何训练, 分钟级 (无断点时的默认路径)
    fast     仅评估模块全监督路径 —— 需已训练断点 runs/train_eva/best.pt, 秒级出结果
    both     双路径全激活 —— fast 结果作先验再自监督精化, 精度最高 (推荐)
    auto     自动选择: 存在断点 runs/train_eva/best.pt → both, 否则 selfsup
  提示: 在几个结构不同的场景各跑一次本脚本后, 执行 scripts/train_eval_module.sh
        一键训练评估模块, 即可解锁秒级标定 (fast/both, 见 README 第 8 节)。
  用例: CALIB_MODE=fast ./scripts/one_click_calib.sh --skip-drivers
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames)        FRAMES="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --skip-drivers)  SKIP_DRIVERS=1; shift ;;
    --init)          INIT_YAML="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[错误] 未知参数: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$INIT_YAML" && ! -f "$INIT_YAML" ]]; then
  echo "[错误] --init 指定的文件不存在: $INIT_YAML"; exit 1
fi
if [[ -n "$INIT_YAML" ]]; then
  INIT_YAML="$(readlink -f "$INIT_YAML")"   # 标定步 cd 到 DST_ROOT, 相对路径须先绝对化
fi
case "$CALIB_MODE" in
  auto|selfsup|fast|both) ;;
  *) echo "[错误] CALIB_MODE=$CALIB_MODE 非法, 应为 auto|selfsup|fast|both"; exit 1 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT:-$DST_ROOT/runs/$TS}"
mkdir -p "$RUN_DIR"
# 绝对化: 第 (6) 步采集按调用者 cwd 运行, 第 (7) 步标定在 (cd $DST_ROOT) 子
# shell 中执行 —— 相对 --out 若不绝对化, 两步会看到两个不同目录
RUN_DIR="$(readlink -f "$RUN_DIR")"
DATA_DIR="$RUN_DIR/data"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"
echo "[1/9] 输出目录: $RUN_DIR  (帧数: ${FRAMES:-默认12}, skip-drivers=$SKIP_DRIVERS)"

# ------------------------------------------------------------------ 清理 (trap)
LIVOX_PID=""
ORBBEC_PID=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM   # 防重入: INT/TERM 触发后末尾 exit 会再触发 EXIT trap
  # 仅当本脚本确实启动过驱动时才清理 —— 避免早期失败/--skip-drivers 时
  # 误杀用户在别处运行的驱动进程
  if [[ -n "$LIVOX_PID$ORBBEC_PID" ]]; then
    echo "[9/9] 清理驱动进程..."
    for pid in $LIVOX_PID $ORBBEC_PID; do
      kill -INT "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $LIVOX_PID $ORBBEC_PID; do
      kill -9 "$pid" 2>/dev/null || true
    done
    # 兜底: launch 子进程可能脱离进程组
    pkill -f livox_ros_driver2_node 2>/dev/null || true
    pkill -f orbbec_camera_node    2>/dev/null || true
  fi
  if [[ $code -ne 0 ]]; then
    echo "[失败] 退出码 $code。驱动日志: $LOG_DIR/livox.log, $LOG_DIR/orbbec.log"
  fi
  exit $code
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ (2) 网络自检
if [[ $SKIP_DRIVERS -eq 0 ]]; then
  echo "[2/9] 网络自检: ping Mid-360 ($LIDAR_IP) ..."
  if ! ping -c1 -W1 "$LIDAR_IP" >/dev/null 2>&1; then
    CUR_IPS="$(ip -4 -o addr show | awk '{print $2": "$4}' | tr '\n' ' ')"
    cat <<EOF
[错误] 无法 ping 通 Mid-360 ($LIDAR_IP)。
  当前主机地址: $CUR_IPS
  Mid-360 要求主机有 192.168.1.x 网段地址 (推荐 $EXPECT_HOST_IP)。
  请手动执行以下命令后重试 (本脚本不自动修改网络配置):
      sudo ip addr add $EXPECT_HOST_IP/24 dev $NET_DEV
  其他排查: 网线插好? 雷达上电 (蓝灯)? 交换机/直连?
  (若驱动已在别处运行且话题正常, 可加 --skip-drivers 跳过本检查)
EOF
    exit 1
  fi
  echo "  雷达可达。"
else
  echo "[2/9] --skip-drivers: 跳过网络自检"
fi

# ------------------------------------------------------------------ (3) 生成 livox JSON
if [[ $SKIP_DRIVERS -eq 0 ]]; then
  HOST_IP="$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 \
             | grep -m1 '^192\.168\.1\.' || true)"
  if [[ -z "$HOST_IP" ]]; then
    echo "[错误] 本机没有 192.168.1.x 地址, 无法作为 livox host_ip。"
    echo "  请执行: sudo ip addr add $EXPECT_HOST_IP/24 dev $NET_DEV"
    exit 1
  fi
  LIVOX_CFG="$RUN_DIR/livox_calib_config.json"
  echo "[3/9] 生成 livox 配置 (host_ip=$HOST_IP): $LIVOX_CFG"
  cat > "$LIVOX_CFG" <<EOF
{
  "lidar_summary_info" : { "lidar_type": 8 },
  "MID360": {
    "lidar_net_info" : {
      "cmd_data_port": 56100, "push_msg_port": 56200, "point_data_port": 56300,
      "imu_data_port": 56400, "log_data_port": 56500
    },
    "host_net_info" : {
      "cmd_data_ip" : "$HOST_IP", "cmd_data_port": 56101,
      "push_msg_ip": "$HOST_IP", "push_msg_port": 56201,
      "point_data_ip": "$HOST_IP", "point_data_port": 56301,
      "imu_data_ip" : "$HOST_IP", "imu_data_port": 56401,
      "log_data_ip" : "", "log_data_port": 56501
    }
  },
  "lidar_configs" : [
    {
      "ip" : "$LIDAR_IP",
      "pcl_data_type" : 1,
      "pattern_mode" : 0,
      "extrinsic_parameter" : { "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                                "x": 0, "y": 0, "z": 0 }
    }
  ]
}
EOF
  export LIVOX_CFG
else
  echo "[3/9] --skip-drivers: 跳过 livox 配置生成"
fi

# ------------------------------------------------------------------ (4) source ROS + 启动驱动
echo "[4/9] source ROS 环境并启动驱动..."
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[错误] 找不到 $ROS_SETUP —— ROS2 lyrical 未安装?"; exit 1
fi
set +u   # ROS setup 脚本引用未定义变量, 临时放宽 nounset
source "$ROS_SETUP"
if [[ -f "$LIVOX_INSTALL" ]]; then
  source "$LIVOX_INSTALL"
else
  echo "[错误] 找不到 $LIVOX_INSTALL —— livox_ros_driver2 未编译 (可设置 LIVOX_INSTALL 指向其 setup.bash)"
  set -u; exit 1
fi
if [[ -f "$ORBBEC_INSTALL" ]]; then
  source "$ORBBEC_INSTALL"
fi
set -u

if [[ $SKIP_DRIVERS -eq 0 ]]; then
  ros2 launch "$SCRIPTS_DIR/livox_mid360_calib.launch.py" \
      >"$LOG_DIR/livox.log" 2>&1 &
  LIVOX_PID=$!
  echo "  livox 驱动 PID=$LIVOX_PID (日志 $LOG_DIR/livox.log)"

  if ! ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
    echo "[错误] 找不到 orbbec_camera 包 —— 请先运行 $SCRIPTS_DIR/setup_drivers.sh"
    exit 1
  fi
  ros2 launch orbbec_camera gemini_330_series.launch.py \
      depth_registration:=true enable_frame_sync:=true \
      >"$LOG_DIR/orbbec.log" 2>&1 &
  ORBBEC_PID=$!
  echo "  orbbec 驱动 PID=$ORBBEC_PID (日志 $LOG_DIR/orbbec.log)"

  sleep 4
  if ! kill -0 "$LIVOX_PID" 2>/dev/null; then
    echo "[错误] livox 驱动启动即退出 —— 常见原因: 端口被占用/配置 JSON 错误。"
    echo "  查看日志: tail -50 $LOG_DIR/livox.log"; exit 1
  fi
  if ! kill -0 "$ORBBEC_PID" 2>/dev/null; then
    echo "[错误] orbbec 驱动启动即退出 —— 常见原因: udev 规则未装/USB 带宽不足。"
    echo "  查看日志: tail -50 $LOG_DIR/orbbec.log"; exit 1
  fi
else
  echo "  --skip-drivers: 不启动驱动, 直接使用现有话题"
fi

# ------------------------------------------------------------------ (5) 等话题就绪
echo "[5/9] 等待话题就绪 (最长 30s)..."
readarray -t TOPICS < <(python3 - "$DST_ROOT/config/default.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
s = c["sensors"]
print(s["lidar"]["topic"])
print(s["camera"]["color_topic"])
print(s["camera"]["depth_topic"])
print(s["camera"]["info_topic"])
PY
)
# 进程替换内 python 的失败不会被 set -e 捕获 (yaml 语法错/键缺失时 TOPICS 为空
# 或不全, 等待循环会因"无缺失项"误报话题就绪) —— 显式校验条数
if [[ ${#TOPICS[@]} -ne 4 ]]; then
  echo "[错误] 从 $DST_ROOT/config/default.yaml 读取话题名失败 (应 4 条, 实得 ${#TOPICS[@]})"
  echo "  请检查该文件 sensors 段的 yaml 语法与 lidar.topic / camera.{color,depth,info}_topic 键"
  exit 1
fi
TOPIC_OK=0
MISS=""
for _i in $(seq 1 30); do
  LIST="$(ros2 topic list 2>/dev/null || true)"
  MISS=""
  for t in "${TOPICS[@]}"; do
    grep -qx "$t" <<<"$LIST" || MISS="$MISS $t"
  done
  if [[ -z "$MISS" ]]; then TOPIC_OK=1; break; fi
  sleep 1
done
if [[ $TOPIC_OK -ne 1 ]]; then
  echo "[错误] 30s 内话题未就绪, 缺失:$MISS"
  echo "  /livox/lidar 缺失 → 检查雷达网络与 $LOG_DIR/livox.log"
  echo "  /camera/* 缺失   → 检查 USB3 连接、udev 规则与 $LOG_DIR/orbbec.log"
  exit 1
fi
echo "  话题全部就绪。"

# ------------------------------------------------------------------ (6) 采集 (系统 python3)
echo "[6/9] 采集数据 → $DATA_DIR"
if ! python3 "$SCRIPTS_DIR/capture_data.py" --out "$DATA_DIR" \
      ${FRAMES:+--num_frames "$FRAMES"}; then
  echo "[错误] 数据采集失败 —— 上方为 capture_data.py 的中文诊断; 常见原因:"
  echo "  话题中途断流 / 深度全 0 (被强光或超出 0.25-6m 量程) / 磁盘写入失败"
  exit 1
fi

# ------------------------------------------------------------------ (7) 标定 (conda python)
echo "[7/9] 标定 (模式: $CALIB_MODE, conda: $CONDA_PY)..."
if [[ ! -x "$CONDA_PY" ]]; then
  echo "[错误] 找不到 conda 解释器 $CONDA_PY —— conda env dstcalib 未创建?"
  exit 1
fi
if ! ( cd "$DST_ROOT" && PYTHONPATH="$DST_ROOT" "$CONDA_PY" -m dst_calib.calibrate \
        --data_dir "$DATA_DIR" --config "$DST_ROOT/config/default.yaml" \
        --output "$RUN_DIR" --mode "$CALIB_MODE" \
        ${INIT_YAML:+--init_yaml "$INIT_YAML"} ); then
  echo "[错误] 标定失败 —— 常见原因:"
  echo "  场景太单一(纯平墙)导致 Chamfer 无梯度 / 深度有效像素太少 /"
  echo "  粗搜索未覆盖真实 yaw (可用 --init 提供先验外参) /"
  echo "  fast/both 模式断点缺失或损坏 (可 CALIB_MODE=selfsup 回退纯自监督)"
  exit 1
fi

# ------------------------------------------------------------------ (8) 摘要 + 可视化
EXT_YAML="$RUN_DIR/extrinsic.yaml"
if [[ ! -f "$EXT_YAML" ]]; then
  echo "[错误] 标定完成但未生成 $EXT_YAML"; exit 1
fi
echo "[8/9] 结果摘要:"
"$CONDA_PY" - "$EXT_YAML" "$CALIB_MODE" <<'PY'
import sys
import numpy as np
import yaml
y = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
T = np.asarray(y["T_cam_lidar"], dtype=float).reshape(4, 4)
print("=" * 62)
# 所用模式以 yaml 为准 (auto 在 calibrate 内解析为实际模式); 旧版结果无该字段
print(f"标定模式: {y.get('mode', sys.argv[2])} (CALIB_MODE={sys.argv[2]}), "
      f"评估模块参与 eva_used: {y.get('eva_used', False)}")
print("T_cam_lidar (p_cam = T @ p_lidar):")
print(np.array_str(T, precision=6, suppress_small=True))
print("欧拉角 ZYX [yaw, pitch, roll] (deg):", y.get("euler_zyx_deg"))
print("平移 t (m):", y.get("translation_m"))
print("最终截断 Chamfer 距离 final_cd:", y.get("final_cd"))
if "per_frame_scores" in y:
    print("多帧评分 s_i (式20/21):", y["per_frame_scores"])
print("=" * 62)
PY
if ! "$CONDA_PY" "$SCRIPTS_DIR/visualize_result.py" --data_dir "$DATA_DIR" \
      --extrinsic_yaml "$EXT_YAML" --out "$RUN_DIR/vis"; then
  echo "[警告] 可视化失败 (不影响标定结果), 可稍后手动执行:"
  echo "  $CONDA_PY $SCRIPTS_DIR/visualize_result.py --data_dir $DATA_DIR \\"
  echo "      --extrinsic_yaml $EXT_YAML --out $RUN_DIR/vis"
fi
echo "全部产物: $RUN_DIR"
echo "  ├── data/           原始采集帧 frame_*.npz"
echo "  ├── extrinsic.yaml  外参结果 (T_cam_lidar / 欧拉角 / 平移 / final_cd)"
echo "  ├── overlay_*.png   calibrate 生成的投影叠加图"
echo "  ├── vis/            visualize_result.py 对比图与统计"
echo "  └── logs/           驱动日志"
# (9) 清理由 trap cleanup 完成
exit 0
