#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
SAVE_PATH=$3
PORT=$((RANDOM + 10000))
NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH}  \
    --resume-from ./work_dirs/test-resnet-w-dcnv3-re/epoch_7.pth \
    --launcher pytorch ${@:4} --deterministic
