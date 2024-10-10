#!/bin/bash

# SLURM job configurations
#SBATCH --job-name=run_attacks             # Job name
#SBATCH --output=outputs/slurm/run_attacks_%A_%a.log     # Output log (%A for job ID, %a for array task ID)
#SBATCH --time=0-96:00:00                  # Time limit: 24 hours
#SBATCH --partition=gpu_a100               # Partition (queue) to use
#SBATCH --cpus-per-gpu=4                   # CPU cores per GPU
#SBATCH --mem=16G                          # Total memory per node
#SBATCH --gres=gpu:1                       # Number of GPUs to use
#SBATCH --qos=default                      # Quality of service
#SBATCH --array=0-5                        # Array of jobs (one for each model)

# Define the models from your YAML in an array
MODEL_LIST=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Meta-Llama-3.1-8B-Instruct" "qwen/Qwen2-7B-Instruct" "HuggingFaceH4/zephyr-7b-beta" "ContinuousAT/Llama-2-7B-CAT" "ContinuousAT/Zephyr-CAT")
# MODEL_LIST=("mistralai/Mistral-7B-Instruct-v0.3")
# MODEL_LIST=("google/Mistral-7B-Instruct-v0.3")
# MODEL_LIST=("HuggingFaceH4/zephyr-7b-beta")
# Select the model based on the SLURM task ID
MODEL_NAME=${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}

# Activate any required environments
eval "$(conda shell.bash hook)"
conda activate std
export CUDA_LAUNCH_BLOCKING=1

# Run the Python script with Hydra model override
python src/run_attacks.py model_name=$MODEL_NAME attack_name=pgd dataset_name=adv_behaviors