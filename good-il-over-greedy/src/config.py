import torch

# ── Architecture (these define the checkpoint shape — never change after first train) ──
MAX_N_ULDS    = 6
MAX_N_PKGS    = 300
MAX_SEQ_LEN   = MAX_N_ULDS + MAX_N_PKGS   # 306
N_ULD_CLASSES = MAX_N_ULDS + 1            # 7: ULD 0-5 + NONE

# Raw feature dimensions (excluding tightness — injected separately, see model)
ULD_FEAT_DIM  = 7    # length, width, height, weight_limit, volume, uld_index, reserved(0)
PKG_FEAT_DIM  = 9    # length, width, height, weight, volume, is_priority,
                     # delay_cost, cost_density, reserved(0)
# NOTE: tightness is NOT in these dims. It is added as a learned scalar embedding
# after the Linear projection, keeping checkpoint shapes compatible with RL fine-tuning.

D_MODEL       = 128
N_HEADS       = 8
N_LAYERS      = 4
D_FF          = 512
DROPOUT       = 0.1

# Normalisation constants
MAX_ULD_DIM    = 320
MAX_ULD_WEIGHT = 975
MAX_PKG_DIM    = 100
MAX_PKG_WEIGHT = 100
MAX_DELAY_COST = 100
MAX_TIGHTNESS  = 2000.0   # cap for normalisation; >1 means can't fit in n-1 ULDs

# Training
BATCH_SIZE     = 16
N_EPOCHS       = 300
LR             = 3e-5
PATIENCE       = 15
IGNORE_INDEX   = -100

# ── Capacity-violation penalty weights (see src/losses.py) ────────────────────
# These scale the auxiliary loss that punishes the model for predicting
# assignments that would overflow a ULD's weight or volume limit — the
# behaviour the GreedyLabeller always respects but the IL model wasn't
# explicitly taught to.
LAMBDA_WEIGHT_PENALTY = 0.0025
LAMBDA_VOLUME_PENALTY = 0.0025

# ── Real-world safety limits for inference/training on oversized instances ───
# Instances with more packages/ULDs than the model was trained for are split
# into chunks no larger than these limits and solved chunk-by-chunk (see
# src/data_utils.py and src/inference.py) instead of being truncated or
# crashing.
MAX_SAFE_PKGS = MAX_N_PKGS
MAX_SAFE_ULDS = MAX_N_ULDS

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
