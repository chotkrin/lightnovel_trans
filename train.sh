#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_MIXED_PRECISION="bf16"

project_name='light'
exp_name="Qwen3-30B-A3B-SFT" 

export RAY_TMPDIR="${HOME}/.tmp"
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/home/xxx/verl"} # modify this
MODEL_PATH="${RAY_DATA_HOME}/models/Qwen3-30B-A3B"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
MAX_ACTOR_CKPT_TO_KEEP=3
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/${project_name}/sft_chunk/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/${project_name}/sft_chunk/test.parquet"}
echo "Start SFT Training..."
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=32 \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    model.partial_pretrain="${MODEL_PATH}" \
    model.fsdp_config.model_dtype="bf16" \
    model.use_liger=true \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]' \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.max_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
