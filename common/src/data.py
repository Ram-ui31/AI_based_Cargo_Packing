"""Loading of good_data ULD-packing instances."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

DATA_ROOT = os.path.expanduser("~/Desktop/good_data")
TRAIN_DIR = os.path.join(DATA_ROOT, "synthetic_train")
TEST_DIR = os.path.join(DATA_ROOT, "synthetic_test")


@dataclass
class Instance:
    instance_id: str
    packages: pd.DataFrame  # Package_ID, Length, Width, Height, Weight, Type, Delay_Cost
    ulds: pd.DataFrame  # ULD_ID, Length, Width, Height, Weight_Limit
    K: int


def _instance_ids(split_dir: str) -> list[str]:
    files = glob.glob(os.path.join(split_dir, "instance_*_packages.csv"))
    ids = sorted(os.path.basename(f)[len("instance_"):-len("_packages.csv")] for f in files)
    return ids


def load_split(split: str = "train") -> list[Instance]:
    """Load all instances for a split ('train' or 'test'), with K attached."""
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    meta_name = "metadata_with_K.csv"
    meta_path = os.path.join(split_dir, meta_name)
    meta = pd.read_csv(meta_path).set_index("instance")

    instances = []
    for iid in _instance_ids(split_dir):
        name = f"instance_{iid}"
        pkgs = pd.read_csv(os.path.join(split_dir, f"{name}_packages.csv"))
        ulds = pd.read_csv(os.path.join(split_dir, f"{name}_ulds.csv"))
        if "K" in meta.columns:
            K = int(meta.loc[name, "K"])
        else:
            K = None  # test split has no ground-truth K; caller assigns for evaluation
        instances.append(Instance(instance_id=name, packages=pkgs, ulds=ulds, K=K))
    return instances


def load_instance(split: str, instance_id: str, K: int | None = None) -> Instance:
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    pkgs = pd.read_csv(os.path.join(split_dir, f"{instance_id}_packages.csv"))
    ulds = pd.read_csv(os.path.join(split_dir, f"{instance_id}_ulds.csv"))
    if K is None:
        meta_name = "metadata_with_K.csv"
        meta = pd.read_csv(os.path.join(split_dir, meta_name)).set_index("instance")
        K = int(meta.loc[instance_id, "K"]) if "K" in meta.columns else None
    return Instance(instance_id=instance_id, packages=pkgs, ulds=ulds, K=K)


if __name__ == "__main__":
    train = load_split("train")
    test = load_split("test")
    print(f"train instances: {len(train)}, test instances: {len(test)}")
    print(train[0].instance_id, "K =", train[0].K, "n_packages =", len(train[0].packages), "n_ulds =", len(train[0].ulds))
