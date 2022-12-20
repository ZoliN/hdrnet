#!/bin/bash

cm=1

CUDA_VISIBLE_DEVICES=$1 python ../hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --data_pipeline ImageFilesDataPipeline \
        --model_name HDRNetGaussianPyrNN \
        --nobatch_norm \
        --output_resolution 256 256 \
        --channel_multiplier $cm \
        out \
        data/filelist.txt
