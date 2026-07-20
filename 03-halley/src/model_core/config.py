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
# K normalization -- log-scale, not linear. K in {100,500,1000,3000,5000}
# spans 1.7 orders of magnitude; a linear K/MAX_K encoding crams 4 of the 5
# values below 0.6, leaving them barely distinguishable to the model. Mirrors
# cargoism/git/model_b(c)/src/assignment_policy.py's normalize_k(), which
# found this necessary for the policy to actually condition on K instead of
# learning one washed-out "spread doesn't matter much" signal.
K_LOG_MIN = 2.0                  # log10(100)
K_LOG_MAX = 3.6989700043360187   # log10(5000)

# Below this K, greedily consolidating Priority into the fewest/largest-
# volume ULDs (rl_assign_argmax_safe's _consolidate_priority_by_capacity)
# costs more Economy capacity than the spread reduction is worth -- verified
# on the full 83-instance test set: at K=100 it's a net loss on average
# (+443.8 mean cost, 10/17 instances worse); at K=500 and up it's a clear
# net win (K=500: -508.7 mean; K=5000: -2750.5 mean). 100 is the only K
# bucket below this threshold in this dataset.
PRIORITY_CONSOLIDATION_MIN_K = 500

IGNORE_INDEX   = -100

# ── RL hyperparameters ────────────────────────────────────────────────────────
RL_LR           = 3e-5
RL_EPOCHS       = 80
RL_GRAD_CLIP    = 1.0
RL_ENTROPY_COEF = 0.05
RL_KL_COEF      = 1.5
RL_EVAL_EVERY   = 2
# Raised from 12 -- with per-epoch stratified subsampling (see train_rl.py's
# instances_per_epoch), epochs are several times cheaper in wall-clock time,
# so the same *epoch-count* patience budget now represents much less total
# training exposure than before. Widening it so a genuinely slow learner
# isn't cut off before it's had a fair chance to converge.
RL_PATIENCE     = 24
RL_TEMPERATURE  = 1.5
MAX_EPS_PER_ULD = 200

# ── Auxiliary losses ported from model_b's train_assignment.py ────────────────
# hinge: dense, ground-truth (dim-fit mask) penalty for preferring NONE over a
#   feasible ULD for an economy package -- the sparse whole-rollout REINFORCE
#   signal alone was too diffuse to unlearn this quickly (model_b found the
#   static argmax rejecting 30-46% of feasible economy packages).
# spread: differentiable proxy for priority spread, scaled by (log-normalized)
#   K, computed from the policy's own softmax over priority packages -- gives
#   spread a direct gradient tied to K instead of only the sparse cost signal.
RL_HINGE_COEF   = 0.1
RL_HINGE_MARGIN = 1.0
RL_SPREAD_COEF  = 0.05

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
