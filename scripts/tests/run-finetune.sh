#!/bin/bash
# Change to the repository root (two levels up from scripts/tests)
cd "$(dirname "$0")/../.."

pwd 
# Set the CUDA devices to use
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Run the command
torchrun --nproc_per_node=4 --standalone \
    -m rtfm.finetune \
    --train-task-file "./sampledata/v6.0.3-serialized/train/train-files.txt" \
    --eval-task-file  "./sampledata/v6.0.3-serialized/train/traineval-files.txt" \
    --run_validation "False" \
    --use_wandb "True" \
    --wandb_project "llama_recipes" \
    --wandb_entity "brian-jay-hsu-columbia-university" \
    --warmup_steps 10 \
    --num_workers_dataloader 8 \
    --max_steps 20 \
    --model_name "mlfoundations/tabula-8b" \
    --save_checkpoint_root_dir "/user/bh2976/FoundationalTabularBandits/training/checkpoints" \
    --run_name "tabula_finetune" \
    --save_model \
    --save_optimizer \
    --enable_fsdp \
    --sharding_strategy "FULL_SHARD" \
    --pure_bf16 "True" \
    --batch_size_training 2 \
    --gradient_accumulation_steps 1
