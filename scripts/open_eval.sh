CUDA_VISIBLE_DEVICES=1 python3 -m lerobot.scripts.openloop_eval     \
                                --policy.path=outputs/train/pick_bottle_act/checkpoints/001000/pretrained_model     \
                                --dataset.repo_id=lerobot/pick_bottle     \
                                --dataset.root=/mnt/data2/baohua/dataHub/lerobot/pick_bottle     \
                                --resize_size='[480,640]'     \
                                --state_indices='[0,1,2,3,4,5,6,14]'     \
                                --action_indices='[0,1,2,3,4,5,6,14]'


# max speed
# CUDA_VISIBLE_DEVICES=1 python3 -m lerobot.scripts.openloop_eval     \
#                                 --policy.path=outputs/train/pick_bottle_act_max_speed/checkpoints/last/pretrained_model     \
#                                 --dataset.repo_id=lerobot/pick_bottle2     \
#                                 --dataset.root=/mnt/data2/baohua/dataHub/lerobot/pick_bottle2     \
#                                 --resize_size='[480,640]'     \
#                                 --state_indices='[0,1,2,3,4,5,6,14]'     \
#                                 --action_indices='[0,1,2,3,4,5,6,14]'     \
#                                 --eval.episode_ids='[0]'     \
#                                 --eval.steps=500