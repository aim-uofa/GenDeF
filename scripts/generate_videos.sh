#!/bin/bash
# ============================================================================
# GenDeF - Video Generation (Sampling)
# ============================================================================
# Generate videos using a trained GenDeF model.
#
# Usage:
#   bash scripts/generate_videos.sh
# ============================================================================

set -e

# --- User Configuration ---
MODEL_PATH="output/taichi_finetune/output/best.pkl"  # Path to trained model checkpoint
OUTDIR="sample/taichi_finetune"                       # Output directory for generated videos
NUM_VIDEOS=100                                        # Number of videos to generate
VIDEO_LEN=128                                         # Number of frames per video
FPS=25                                                # Frames per second
BATCH_SIZE=25                                         # Batch size for generation
SEED=42                                               # Random seed
TRUNCATION_PSI=0.9                                    # Truncation psi (lower = higher quality, less diversity)

# --- Generation ---
python src/scripts/generate_ours.py \
    --network_pkl ${MODEL_PATH} \
    --num_videos ${NUM_VIDEOS} \
    --save_as_mp4 true \
    --fps ${FPS} \
    --video_len ${VIDEO_LEN} \
    --batch_size ${BATCH_SIZE} \
    --outdir ${OUTDIR} \
    --truncation_psi ${TRUNCATION_PSI} \
    --seed ${SEED}
