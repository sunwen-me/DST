#!/usr/bin/env bash
# =============================================================================
# DST-Calib 相机驱动安装: 克隆并编译 OrbbecSDK_ROS2 (v2, Gemini 330/335 系列)
# 到仓库下的 ros2_ws。幂等: 已克隆则 git pull; 已编译可重复执行。
# livox_ros_driver2 已在 /home/sw/Super-LIO 编译好, 本脚本不处理。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DST_ROOT="${DST_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/lyrical/setup.bash}"
WS="${WS:-$DST_ROOT/ros2_ws}"
REPO_URL="${REPO_URL:-https://github.com/orbbec/OrbbecSDK_ROS2.git}"
BRANCH="${BRANCH:-v2-main}"        # v2 分支 (Gemini 330 系列须用 v2; v1 为 main)
SRC="$WS/src/OrbbecSDK_ROS2"
COMPAT_PATCH="$DST_ROOT/patches/orbbec-lyrical-compat.patch"
COMPAT_OVERLAY="$DST_ROOT/patches/orbbec-lyrical-compat/orbbec_camera"

# ------------------------------------------------------------------ (1) 克隆/更新
echo "[1/5] 准备工作区 $WS ..."
mkdir -p "$WS/src"
if [[ -d "$SRC/.git" ]]; then
  echo "  已存在, git pull 更新 (分支 $BRANCH)..."
  git -C "$SRC" fetch origin "$BRANCH" 2>/dev/null \
    && git -C "$SRC" checkout "$BRANCH" \
    && git -C "$SRC" pull --ff-only \
    || echo "[警告] git 更新失败 (可能离线), 使用现有代码继续"
else
  echo "  克隆 $REPO_URL (分支 $BRANCH)..."
  if ! git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$SRC"; then
    echo "[错误] 克隆失败 —— 检查网络能否访问 github.com; 或手动下载后放到 $SRC"
    exit 1
  fi
fi

# ------------------------------------------------------------------ (1b) lyrical 兼容补丁
if [[ -f "$COMPAT_PATCH" ]]; then
  if git -C "$SRC" apply --unidiff-zero --check "$COMPAT_PATCH" >/dev/null 2>&1; then
    git -C "$SRC" apply --unidiff-zero "$COMPAT_PATCH"
    echo "  已应用 lyrical 兼容补丁。"
  elif git -C "$SRC" apply --unidiff-zero --reverse --check "$COMPAT_PATCH" >/dev/null 2>&1; then
    echo "  lyrical 兼容补丁已存在，跳过。"
  else
    echo "[错误] lyrical 兼容补丁无法应用，可能是 OrbbecSDK_ROS2 版本不匹配。"
    exit 1
  fi
  cp -a "$COMPAT_OVERLAY/include" "$SRC/orbbec_camera/"
  cp -a "$COMPAT_OVERLAY/cmake/ament_target_dependencies_compat.cmake" \
        "$SRC/orbbec_camera/cmake/"
fi

# ------------------------------------------------------------------ (2) udev 规则
echo "[2/5] 检查 udev 规则..."
if ls /etc/udev/rules.d/99-obsensor*.rules >/dev/null 2>&1; then
  echo "  已安装, 跳过。"
else
  cat <<EOF
  [需要手动操作] udev 规则未安装 (需要 sudo, 本脚本不自动提权)。
  请在终端手动执行:
      cd $SRC/orbbec_camera/scripts
      sudo bash install_udev_rules.sh
      sudo udevadm control --reload-rules && sudo udevadm trigger
  完成后重新插拔 Gemini 335 的 USB 线。未装规则时普通用户无法打开相机。
EOF
fi

# ------------------------------------------------------------------ (3) rosdep (可选)
echo "[3/5] rosdep 安装依赖 (可选)..."
set +u   # ROS setup 脚本引用未定义变量, 临时放宽 nounset
source "$ROS_SETUP"
set -u
if command -v rosdep >/dev/null 2>&1; then
  rosdep install --from-paths "$WS/src" --ignore-src -r -y \
    || echo "[警告] rosdep 失败 (常见于未 rosdep init/网络问题); 若下一步编译通过可忽略"
else
  echo "  未安装 rosdep, 跳过 (缺依赖时编译会报错, 届时按报错 apt 安装)。"
fi

# ------------------------------------------------------------------ (4) 编译
echo "[4/5] colcon build --symlink-install ..."
cd "$WS"
if ! colcon build --symlink-install --event-handlers console_cohesion+ \
      --cmake-args -DCMAKE_BUILD_TYPE=Release; then
  echo "[错误] 编译失败 —— 查看 $WS/log/latest_build/ 内日志;"
  echo "  常见缺依赖: libgflags-dev nlohmann-json3-dev ros-lyrical-image-transport"
  echo "  ros-lyrical-image-publisher ros-lyrical-camera-info-manager libdw-dev"
  exit 1
fi

# ------------------------------------------------------------------ (5) 自检
echo "[5/5] 自检: 查找 gemini_330_series.launch.py ..."
LAUNCH_FILE="$(find "$WS/install" -name 'gemini_330_series.launch.py' 2>/dev/null | head -n1 || true)"
if [[ -n "$LAUNCH_FILE" ]]; then
  echo "  OK: $LAUNCH_FILE"
  echo "驱动就绪。测试相机:"
  echo "  source $ROS_SETUP && source $WS/install/setup.bash"
  echo "  ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true enable_frame_sync:=true"
  echo "随后即可运行一键标定: bash $DST_ROOT/scripts/one_click_calib.sh"
else
  echo "[错误] 编译产物中找不到 gemini_330_series.launch.py —— orbbec_camera 包未编译成功?"
  echo "  检查: ls $WS/install; 以及编译日志 $WS/log/latest_build/orbbec_camera/"
  exit 1
fi
