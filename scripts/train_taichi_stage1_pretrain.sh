#!/bin/bash
# ============================================================================
# GenDeF - Stage 1: Pretrain (Image Generation) on TaiChi-HD
# ============================================================================
# This script trains the image generation backbone (single-frame) on the
# TaiChi-HD dataset. The pretrained model will be used as initialization
# for Stage 2 (video generation with deformation field).
#
# Usage:
#   bash scripts/train_taichi_stage1_pretrain.sh
#
# You can adjust NUM_GPUS, BATCH_SIZE, and paths below.
# ============================================================================

set -e

# --- User Configuration ---
DATA_PATH="data/taichi_256.zip"                       # Path to TaiChi dataset zip
OUTPUT_DIR="output/taichi_pretrain"                    # Output directory
NUM_GPUS=8                                             # Number of GPUs
BATCH_SIZE=64                                          # Total batch size across all GPUs

# --- Training ---
python src/infra/launch.py \
    hydra.run.dir=. \
    exp_suffix=taichi_pretrain \
    env=local slurm=false \
    dataset=taichi \
    dataset.path=${DATA_PATH} \
    dataset.resolution=256 \
    training.mirror=false \
    training.batch_size=${BATCH_SIZE} \
    training.num_workers=1 \
    training.aug=ada \
    sampling.num_frames_per_video=1 \
    model.generator.bspline=false \
    model.generator.bs_emb=false \
    model.generator.fmaps=0.5 \
    model.discriminator.fmaps=0.5 \
    model.optim.generator.lr=0.0025 \
    model.generator.fuse_w=concat \
    model.generator.dcn=true \
    model.generator.dcn_min_res=36 \
    model.generator.dcn_max_res=52 \
    model.discriminator.tsm=false \
    model.discriminator.tmean=true \
    model.loss_kwargs.r1_gamma=0.5 \
    model.generator.learnable_motion_mask=false \
    model.generator.init_motion_mask=zeros \
    model.generator.time_enc.min_period_len=16 \
    num_gpus=${NUM_GPUS} \
    training.snap=100 \
    project_release_dir=${OUTPUT_DIR}
