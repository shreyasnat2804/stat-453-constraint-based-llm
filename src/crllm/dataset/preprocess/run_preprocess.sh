#!/bin/bash
"""
Preprocesses the raw data for training. This includes cleaning, deduplication, and balancing the dataset.
Usage-----
$ chmod +x run_preprocess.sh
$ ./run_preprocess.sh
"""
python preprocess.py \
        --input /Users/gayu/github/uwm_gayu/spring2026/stat453/project/stat-453-constraint-based-llm/datasets/RECAST-30K.jsonl \
        --output /Users/gayu/github/uwm_gayu/spring2026/stat453/project/stat-453-constraint-based-llm/datasets/recast_30k_clean.jsonl \
        --min_length 15 \
        --dedup_threshold 0.85 \
        --imbalance_threshold 0.5 \
        --n_jobs 8