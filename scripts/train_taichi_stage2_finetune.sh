#!/bin/bash
# ============================================================================
# GenDeF - Stage 2: Finetune (Video Generation with Deformation Field) on TaiChi-HD
# ============================================================================
# This script finetunes the pretrained image model from Stage 1 to learn
# the generative deformation field (GenDeF) for video generation.
# It introduces canonical image generation and deformable convolution-based
# deformation field prediction.
#
# Usage:
#   bash scripts/train_taichi_stage2_finetune.sh
#
# Prerequisites:
#   - Stage 1 pretrained checkpoint (best.pkl from Stage 1 output)
#
# You can adjust NUM_GPUS, BATCH_SIZE, and paths below.
# ============================================================================

set -e

# --- User Configuration ---
DATA_PATH="data/taichi_256.zip"                                    # Path to TaiChi dataset zip
PRETRAINED_CKPT="output/taichi_pretrain/output/best.pkl"           # Stage 1 pretrained checkpoint
OUTPUT_DIR="output/taichi_finetune"                                # Output directory
NUM_GPUS=8                                                         # Number of GPUs
BATCH_SIZE=64                                                      # Total batch size across all GPUs

# --- Training ---
python src/infra/launch.py \
    hydra.run.dir=. \
    exp_suffix=taichi_finetune \
    env=local slurm=false \
    dataset=taichi \
    dataset.path=${DATA_PATH} \
    dataset.resolution=256 \
    training.mirror=false \
    training.batch_size=${BATCH_SIZE} \
    training.num_workers=1 \
    training.aug=ada \
    sampling.num_frames_per_video=3 \
    model.generator.bspline=false \
    model.generator.bs_emb=false \
    model.generator.fmaps=0.5 \
    model.discriminator.fmaps=0.5 \
    model.optim.generator.lr=0.0025 \
    model.generator.fuse_w=concat \
    model.generator.dcn=true \
    model.generator.dcn_min_res=36 \
    model.generator.dcn_max_res=52 \
    model.discriminator.tsm=true \
    model.discriminator.tmean=true \
    model.loss_kwargs.r1_gamma=8 \
    model.generator.learnable_motion_mask=true \
    model.generator.init_motion_mask=zeros \
    model.generator.time_enc.min_period_len=16 \
    num_gpus=${NUM_GPUS} \
    training.snap=100 \
    training.resume=${PRETRAINED_CKPT} \
    project_release_dir=${OUTPUT_DIR} \
    model.generator.with_canonical=true \
    model.generator.canonical_cond=concat \
    model.generator.canonical_cond_dim=64 \
    model.generator.canonical_feat=L13_256_64 \
    model.generator.deform_init=zero \
    model.generator.deform_dcn=true \
    model.generator.deform_dcn_min_res=4 \
    model.generator.deform_dcn_max_res=64 \
    model.generator.deform_dcn_torgb=true
