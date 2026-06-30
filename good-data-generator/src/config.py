# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  —  all calibrated from input.csv
# ─────────────────────────────────────────────────────────────────────────────

# ULD geometry
ULD_L_MAX    = 240          # hard ceiling per problem spec
ULD_W_MAX    = 320
ULD_H_MAX    = 240
ULD_DIM_MIN  = 65           # minimum on any axis
ULD_DIM_STEP = 5            # all dims multiples of 5

# ULD weight limits  (input.csv: 2500-3500 kg, step 100)
ULD_WT_MIN  = 2500
ULD_WT_MAX  = 4000
ULD_WT_STEP = 100

# ULD count per instance
N_ULDS_MIN = 2
N_ULDS_MAX = 6

# Package dims  (input.csv: 40-110 cm each axis)
PKG_DIM_MIN = 25
PKG_DIM_MAX = 100           # further bounded per-instance to min ULD dim on each axis

# Package weight  (input.csv: 7-271 kg)
PKG_WT_MIN = 7
PKG_WT_MAX = 150

# Economy delay cost  (input.csv: 60-176)
DELAY_MIN = 60
DELAY_MAX = 176

# Priority ratio  (input.csv: ~0.257, spec: 0.25-0.30)
PRIO_RATIO_MIN = 0.25
PRIO_RATIO_MAX = 0.30

# Target aggregate ratios from input.csv  (both with +/-5% noise for variety)
TARGET_VOL_RATIO = 1.2   # sum(pkg_vol) / sum(uld_vol)
TARGET_WT_RATIO  = 1.3   # sum(pkg_wt)  / sum(uld_wt)
RATIO_NOISE      = 0.05  # +/-5%

# ── Priority-vs-ULD share ratios ──────────────────────────────────────────────
# In input.csv:
#   sum(priority_vol) / sum(uld_vol) ~= 0.4808
#   sum(priority_wt)  / sum(uld_wt)  ~= 0.4380
# These two numbers are close to each other (both ~0.44-0.48). We keep that
# property in the synthetic data: per instance we draw ONE base "priority
# share" near the midpoint of those two calibration values, then jitter the
# volume-side and weight-side targets independently by a small amount so
# there's variety between instances, but the two ratios stay close together
# (mirroring the real data) rather than drifting apart.
PRIO_SHARE_VOL_CAL = 0.4808   # sum(priority_vol) / sum(uld_vol), from input.csv
PRIO_SHARE_WT_CAL  = 0.4380   # sum(priority_wt)  / sum(uld_wt),  from input.csv
PRIO_SHARE_BASE    = 0.25
PRIO_SHARE_NOISE   = 0.06     # +/-6% jitter applied independently to vol & wt targets
PRIO_SHARE_SPREAD  = 0.03     # extra +/- half-spread between the vol- and wt- targets

# Dataset sizes
N_TRAIN         = 1000
N_TEST          = 100
TRAIN_BASE_SEED = 1000
TEST_BASE_SEED  = 9999      # fixed — never change
