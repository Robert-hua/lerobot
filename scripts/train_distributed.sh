#!/bin/bash
# LeRobot 分布式训练脚本
# 支持单机多卡和多机多卡训练

# ============================================
# 配置区域 - 根据你的环境修改
# ============================================
# nohup bash scripts/train_distributed.sh > logs/act_pick_bottle_max_speed.log 2>&1 &
# 配置文件路径
CONFIG_PATH="train_config/train_config_pick_maxspeed.json"

# 是否恢复训练 (true/false)
RESUME=false

# 输出目录 (如果不恢复训练且目录存在，则删除)
# OUTPUT_DIR="outputs/train/pick_bottle_act"
# max_speed
OUTPUT_DIR="outputs/train/pick_bottle_act_max_speed"

if [ "$RESUME" = false ] && [ -d "$OUTPUT_DIR" ]; then
    echo "Removing existing output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# 单机多卡训练 (推荐先测试)
# ============================================
# 使用所有可用 GPU
# accelerate launch --multi_gpu \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=$CONFIG_PATH

# 指定 GPU 数量
# accelerate launch --num_processes=4 --multi_gpu \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=$CONFIG_PATH

# ============================================
# 多机多卡训练
# ============================================
# 需要在每台机器上运行此脚本，修改 MACHINE_RANK

# 主节点 IP 地址 (修改为你的主节点 IP)
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"

# 总机器数
NUM_MACHINES=1

# 每台机器的 GPU 数量
NUM_GPUS_PER_MACHINE=2

# 当前机器的 rank (主节点为 0，其他节点依次递增)
MACHINE_RANK=0

# 总进程数
NUM_PROCESSES=$((NUM_MACHINES * NUM_GPUS_PER_MACHINE))

# ============================================
# 方式 1: 使用 accelerate launch (推荐)
# ============================================
# 在主节点 (MACHINE_RANK=0) 运行:
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$NUM_MACHINES \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --multi_gpu \
    --mixed_precision=fp16 \
    -m lerobot.scripts.lerobot_train \
    --config_path=$CONFIG_PATH

# ============================================
# 方式 2: 使用 torchrun (备选)
# ============================================
# torchrun \
#     --nnodes=$NUM_MACHINES \
#     --nproc_per_node=$NUM_GPUS_PER_MACHINE \
#     --node_rank=$MACHINE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=$CONFIG_PATH
