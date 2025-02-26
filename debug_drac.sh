#!/bin/bash

python run_attacks.py  --config-name=config_rf_llama32 \
    ++hf_hub_offline=true \
    ++datasets.adv_behaviors.idx=0 \
    ++datasets.adv_behaviors.batch=2 \
    attacks.gcg.num_steps=15 \
    root_dir=$PWD \
    save_dir=$SCRATCH/llm-quick-check-outputs
