#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=..cache
max_seq_length=512
Task=rtd
tag=deberta-base

CUDA_VISIBLE_DEVICES=1 python -m DeBERTa.apps.run --model_config config.json  \
    --data_dir /media/data/huypn10/Vietnamese%20Pretrained%20Model/pre-training/dataset/tmp \
    --tokenizer_dir /media/data/huypn10/Vietnamese%20Pretrained%20Model/pre-training/tokenizer/spm_1e4_5e5 \
	--vocab_path /media/data/huypn10/Vietnamese%20Pretrained%20Model/pre-training/spm.model \
	--model_config ../config/deberta_base.json \
	--tag $tag \
	--do_train \
	--num_training_steps 100000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--vocab_type spm \
	--workers 0 \
	--output_dir /tmp/ttonly/$tag/$task \
	--num_train_epochs 1 \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 16 \
	--fp16 True \