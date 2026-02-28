#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_MIXED_PRECISION="bf16"

echo "Start DPO Training..."

python3 -m verl.trainer.main_dpo \
    data.train_files=['train.parquet'] \
    data.val_files=['test.parquet'] \
    data.prompt_key=prompt \
    data.chosen_key=chosen \
    data.rejected_key=rejected \
    data.max_length=4096 \
    data.max_prompt_length=2048 \
    +data.apply_chat_template_kwargs='{"enable_thinking": false}' \
    model.partial_pretrain="Qwen/Qwen3-30B-A3B" \
    model.use_lora=false \
    trainer.dpo_beta=0.1 \
    trainer.micro_batch_size=4 \
    trainer.gradient_accumulation_steps=2 \
    trainer.lr=5e-7 \
    trainer.num_epochs=3 \
    trainer.gradient_checkpointing=true \
    trainer.fsdp_config.fsdp_cpu_offload=false \
    trainer.logger=['wandb'] \
    trainer.experiment_name="JFF" \
    trainer.default_local_dir=""