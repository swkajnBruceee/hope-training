#!/bin/bash
set -e


# 获取当前脚本名称和完整路径
script_name=$(basename "$0")
# script_dir=$(dirname "$0")
script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")

echo "script_name: $script_name"
echo "script_dir: $script_dir"
echo "script_path: $script_path"


# Print usage information and exit
Usage() {
    cat <<EOF
    start_motion_control.sh [options]
    -p: debug port
    -g: run with debug mode
    -c: using custom config file(absolute path)
    -h: help
    e.g.: 1) start_motion_control.sh -c /home/agibot/motion_control_config.yaml
EOF
    exit 1
}

# Parse command line arguments
ARGS=$(getopt -o c:p:gh --long config:,port:,debug,help -n "$0" -- "$@")
if [ $? != 0 ]; then
    Usage
fi

# Get current timestamp for logging
function get_time() {
    local time=$(date "+%H:%M:%S")
    echo "($time)"
}


eval set -- "${ARGS}"
echo $(get_time) INFO: formatted parameters=[$@]

# Default configuration values
PORT=65001

declare -A robot_map

# 初始化映射关系
robot_map["A3_T1D0"]="a3_t2d0"
robot_map["A3_T2D0"]="a3_t2d0"
robot_map["A3_ULTRA_T2D0"]="a3_ultra_t2d0"
robot_map["A3_T2D5"]="a3_t2d5"
robot_map["A3_ULTRA_T2D5"]="a3_ultra_t2d5"

# 检查环境变量是否存在
if [ -z "$AGIBOT_ROBOT_MODEL" ]; then
    echo "ERROR: AGIBOT_ROBOT_MODEL environment variable is not set"

    robot_directory="${script_dir}/../../config/motion_control/configuration/robot"
    directories=($(find "$robot_directory" -maxdepth 1 -mindepth 1 -type d | sort))
    echo "Please select a robot: "
    for i in "${!directories[@]}"; do
      dir_name=$(basename "${directories[$i]}")
      echo -e "\e[1;31m$i\e[0m: \e[1;32m$dir_name\e[0m"
    done
    read -p "Please enter your choice: " choice
    if [[ "$choice" =~ ^[0-9]+$ && "$choice" -ge 0 && "$choice" -lt "${#directories[@]}" ]]; then
      selected_robot_directory="${directories[$choice]}"
      robot_name=$(basename "$selected_robot_directory")
    else
      echo "Invalid choice. Exiting."
      exit 1
    fi
    echo "Selected robot: $robot_name"
else
    # 构造查找键
    key="${AGIBOT_ROBOT_MODEL}"
    # 获取对应的机器人名称
    robot_name="${robot_map[$key]}"
fi

CONFIG_FILE_PATH="../config/motion_control/configuration/robot/${robot_name}/aimrt/default.yaml"


# 启动 iox-roudi 进程
if pgrep -f "./iox-roudi" > /dev/null
then
    echo "iox-roudi 进程已运行"
else
    echo "进程未运行，正在启动..."
    # 根据实际情况替换为你的启动命令（这里是直接执行）
    # 记录当前目录
    echo "当前目录: $(pwd)"
    current_dir=$(pwd)

    cd /${script_dir}/../../bin
    export LD_LIBRARY_PATH=./ && ./iox-roudi &
    echo "iox-roudi 启动完成"

    # 切换回原目录
    cd $current_dir
fi



# Process command line options
while true; do
    case "$1" in
    -g | --debug)
        echo "$(get_time) INFO: Run with debug mode"
        DEBUG_FLAG=true
        shift
        ;;
    -p | --port)
        PORT=${2}
        shift 2
        ;;
    -c | --config)
        echo "$(get_time) INFO: Run with config: $2"
        CONFIG_FILE_PATH=${2}
        shift 2
        ;;
    --)
        shift
        break
        ;;
    -h | --help)
        Usage
        ;;
    *)
        Usage
        ;;
    esac
done

# Set up debug command if debug mode is enabled
DEBUG_COMMAND=""
if [[ $DEBUG_FLAG == true ]]; then
    DEBUG_COMMAND="lldb-server g 0.0.0.0:${PORT} -- "
fi

# Get script directory and change to bin folder
SHELL_FOLDER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SHELL_FOLDER
pushd "$SHELL_FOLDER"/../../bin || exit

# Add signal handler function
cleanup() {
    echo -e "\n$(get_time) INFO: 🛑 Received stop signal, terminating program..."
    if [[ -n $PID ]]; then
        kill -TERM $PID 2>/dev/null || true
        sleep 2
        # Force kill if process is still running
        if kill -0 $PID 2>/dev/null; then
            echo "$(get_time) WARNING: Force killing process $PID"
            kill -KILL $PID 2>/dev/null || true
        fi
    fi
    popd 2>/dev/null || true
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Execute motion_control binary with background execution to capture PID
echo -e "\n$(get_time) INFO: ✊ Start running motion_control ..."
chmod a+x ./motion_control
export LOG_PATH=${LOG_PATH:-"../log"} 
export PNC_IP=${PNC_IP:-"127.0.0.1"}
export MC_PARAM_ROOT_PATH="../config/motion_control/configuration"
export MC_RESOURCE_ROOT_PATH="../share/robot_model/resource"
export MC_ROBOT_NAME="$robot_name"

# 如果存在 AGIBOT_MASTER_IP 环境变量，说明是在 实际机器人上运行的
if [ -z "$AGIBOT_MASTER_IP" ]; then
    IGNORE_PREDEFINED_CFG=" --ignore_predefined_cfg"
fi

# Start program and get PID
RMW_LIBRARY_PATH=$(pwd)/librmw_fastrtps.so LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ${DEBUG_COMMAND} ./motion_control --cfg_file_path=$CONFIG_FILE_PATH $IGNORE_PREDEFINED_CFG &
PID=$!

# Wait for process to finish
wait $PID
EXIT_CODE=$?

# Check execution status and exit accordingly
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n$(get_time) INFO: ✅ motion_control executed successfully"
else
    echo -e "\n$(get_time) ERROR: ❌ motion_control execution failed, exit code: $EXIT_CODE"
    exit 1
fi
popd || exit
