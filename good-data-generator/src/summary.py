from .config import (
    PRIO_RATIO_MIN, PRIO_RATIO_MAX,
    TARGET_VOL_RATIO, TARGET_WT_RATIO, RATIO_NOISE,
    PRIO_SHARE_BASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summarise(meta_df, name):
    print(f'\n=== {name} ({len(meta_df)} instances) ===')
    for col, label, target in [
        ('n_ulds',              'n_ulds             ', None),
        ('n_packages',          'n_packages         ', None),
        ('priority_ratio',      'priority_ratio     ', f'target {PRIO_RATIO_MIN}–{PRIO_RATIO_MAX}'),
        ('vol_ratio',           'vol_ratio          ', f'target {TARGET_VOL_RATIO}±{RATIO_NOISE*100:.0f}%'),
        ('wt_ratio',            'wt_ratio           ', f'target {TARGET_WT_RATIO}±{RATIO_NOISE*100:.0f}%'),
        ('prio_vol_uld_ratio',  'prio_vol/uld_ratio ', f'target ≈{PRIO_SHARE_BASE:.3f} (≈ wt ratio)'),
        ('prio_wt_uld_ratio',   'prio_wt/uld_ratio  ', f'target ≈{PRIO_SHARE_BASE:.3f} (≈ vol ratio)'),
    ]:
        s = meta_df[col]
        t = f'  ({target})' if target else ''
        print(f'  {label}: min={s.min():.3f}  max={s.max():.3f}  '
              f'mean={s.mean():.3f}  median={s.median():.3f}{t}')

    # How close are the two priority-share ratios to each other, on average?
    gap = (meta_df['prio_vol_uld_ratio'] - meta_df['prio_wt_uld_ratio']).abs()
    print(f'  |prio_vol/uld - prio_wt/uld| gap: mean={gap.mean():.3f}  max={gap.max():.3f}')
