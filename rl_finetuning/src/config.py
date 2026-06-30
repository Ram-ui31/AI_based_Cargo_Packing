import torch

# ── Architecture (must match imitation_cluster_v2.ipynb exactly) ──────────────
MAX_N_ULDS    = 6
MAX_N_PKGS    = 300
MAX_SEQ_LEN   = MAX_N_ULDS + MAX_N_PKGS   # 306
N_ULD_CLASSES = MAX_N_ULDS + 1            # 7: ULD 0-5 + NONE

# Raw feature dims — tightness is NOT counted here (injected separately)
ULD_FEAT_DIM  = 7
PKG_FEAT_DIM  = 9

D_MODEL  = 128
N_HEADS  = 8
N_LAYERS = 4
D_FF     = 512
DROPOUT  = 0.1

# Normalisation constants
MAX_ULD_DIM    = 320.0
MAX_ULD_WEIGHT = 975.0
MAX_PKG_DIM    = 100.0
MAX_PKG_WEIGHT = 100.0
MAX_DELAY_COST = 100.0
MAX_TIGHTNESS  = 2000.0

IGNORE_INDEX   = -100

# ── RL hyperparameters ────────────────────────────────────────────────────────
RL_LR           = 3e-5
RL_EPOCHS       = 80
RL_GRAD_CLIP    = 1.0
RL_ENTROPY_COEF = 0.05
RL_KL_COEF      = 1.5
RL_EVAL_EVERY   = 2
RL_PATIENCE     = 12
RL_TEMPERATURE  = 1.5
MAX_EPS_PER_ULD = 200

# ── RL capacity-violation penalty weights ─────────────────────────────────────
# Penalty is computed from raw (pre-mask) logits so the network receives
# a gradient toward respecting capacity, not just the masking code.
RL_LAMBDA_WEIGHT_PENALTY = 5e-8
RL_LAMBDA_VOLUME_PENALTY = 1e-7

# ── Real-world safety limits for instances bigger than the model ──────────────
# Instances exceeding these limits are chunked rather than truncated.
MAX_SAFE_PKGS = MAX_N_PKGS
MAX_SAFE_ULDS = MAX_N_ULDS

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
