#!/bin/sh
#PJM -L elapse=24:00:00
#PJM -N train_spiral
#PJM -L node=1
#PJM -L rscgrp=c-batch
#PJM -j

# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module load cuda/12.2.2
module load cudnn/8.9.7
module load gcc-toolset/12
module load nccl/2.22.3

# Activate conda environment
source ~/miniconda3/bin/activate 
cd /home/pj24002027/ku40001342/code/spiral/
conda activate spiral
export WANDB_API_KEY=3ddada10760318e2ddba58ef19546cd83a11a010

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG
export WANDB_API_KEY=3ddada10760318e2ddba58ef19546cd83a11a010
export OPENROUTER_API_KEY=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
# Notes ==========
# Setting `--save_steps 16` to save checkpoints every 16 policy iteration steps.
# Set `--eval_opponent_names google/gemini-2.0-flash-lite-001` if you have OpenRouter access.

python train_spiral.py \
    --env_id KuhnPoker-v1 \
    --use_llm_obs_wrapper \
    --eval_env_ids KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers True \
    --eval_split "" \
    --gamma 1 \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-4B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --save_ckpt \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 2 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --max_context_length 32768 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 16 \
    --save_steps 16 \
    --save_path /home/pj24002027/ku40001342/code/spiral/spiral-qwen3-4b-base-kp-4k-self-play_continue \
    --eval_games 16 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 51200 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name spiral-qwen3-4b-base-kp-4k-self-play_continue \
    --wb_project spiral \
