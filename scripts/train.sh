#!/bin/bash
# LeRobot 训练脚本

# nohup bash scripts/train.sh > logs/act_pick_bottle_max_speed.log 2>&1 &
# 激活虚拟环境
source .venv/bin/activate

# 配置文件路径
# CONFIG_PATH="train_config/train_config.json"

# max speed
CONFIG_PATH="train_config/train_config_pick_maxspeed.json"

# ============================================
# 单机单卡训练
# ============================================
python -m lerobot.scripts.lerobot_train --config_path=$CONFIG_PATH

# ============================================
# 单机多卡训练 (取消注释使用)
# ============================================
# accelerate launch --multi_gpu \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=$CONFIG_PATH

# 指定 GPU 数量 (例如 4 卡)
# accelerate launch --num_processes=4 --multi_gpu \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=$CONFIG_PATH