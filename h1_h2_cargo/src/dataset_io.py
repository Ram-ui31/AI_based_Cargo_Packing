"""
Dataset I/O for the cargo packing dataset (data-20260608T140057Z-3-001.zip).

Observed layout, confirmed by direct inspection of the unzipped files:

    data/
      toy_example_ulds.csv
      toy_example_packages.csv
      toy_example_K.csv              <- ONLY the toy example stores K standalone
      uld_catalogue.csv              <- master ULD spec catalogue (47k rows), not
                                         per-instance; useful for building custom
                                         fleets, not required to solve instances
      generated_test/
        instance_NNN_ulds.csv
        instance_NNN_packages.csv
        metadata.csv                 <- one row per instance, includes K
      synthetic_train/
        instance_NNN_ulds.csv
        instance_NNN_packages.csv
        metadata.csv                 <- same schema as generated_test/metadata.csv

CSV schemas (verified column-by-column against the actual files):

    *_ulds.csv:
        ULD_ID, Length, Width, Height, Weight_Limit

    *_packages.csv:
        Package_ID, Length, Width, Height, Weight, Type, Delay_Cost
        Type in {"Priority", "Economy"}. Priority rows always have
        Delay_Cost == 0 (confirmed across the full generated_test split).

    metadata.csv (generated_test/synthetic_train):
        instance, seed, n_ulds, n_packages, n_priority, n_economy,
        priority_ratio, priority_vol_frac, priority_wt_frac,
        attempts, b_rejects, a_rejects, K

    toy_example_K.csv:
        K        (single column, single row -- only used for the toy example)

This module never guesses: every loader fails loudly (FileNotFoundError /
KeyError / ValueError) rather than silently defaulting, since a wrong K or a
wrong weight limit would silently produce an invalid evaluation result.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv

from geometry import Package, ULD


# ---------------------------------------------------------------------------
# Single-instance loading
# ---------------------------------------------------------------------------

@dataclass
class ProblemInstance:
    """Everything needed to run the packing engine on one instance."""
    instance_id: str
    ulds: List[ULD]
    packages: List[Package]
    k_penalty: float


def load_ulds_csv(path: Path) -> List[ULD]:
    ulds = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"ULD_ID", "Length", "Width", "Height", "Weight_Limit"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            ulds.append(ULD(
                id=row["ULD_ID"],
                length=float(row["Length"]),
                width=float(row["Width"]),
                height=float(row["Height"]),
                weight_limit=float(row["Weight_Limit"]),
            ))
    if not ulds:
        raise ValueError(f"{path}: no ULD rows found")
    return ulds


def load_packages_csv(path: Path) -> List[Package]:
    packages = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Package_ID", "Length", "Width", "Height", "Weight",
                     "Type", "Delay_Cost"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            type_str = row["Type"].strip()
            if type_str not in ("Priority", "Economy"):
                raise ValueError(
                    f"{path}: unexpected Type '{type_str}' for {row['Package_ID']} "
                    "(expected 'Priority' or 'Economy')"
                )
            packages.append(Package(
                id=row["Package_ID"],
                length=float(row["Length"]),
                width=float(row["Width"]),
                height=float(row["Height"]),
                weight=float(row["Weight"]),
                is_priority=(type_str == "Priority"),
                delay_cost=float(row["Delay_Cost"]),
            ))
    if not packages:
        raise ValueError(f"{path}: no package rows found")
    return packages


def load_metadata_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """Returns {instance_name: {column: value}} for every row in a split's
    metadata.csv. Values stay as strings here; callers cast what they need
    (e.g. int(row['K']))."""
    by_instance = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "instance" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing 'instance' column")
        for row in reader:
            by_instance[row["instance"]] = row
    return by_instance


def load_toy_example(data_root: Path) -> ProblemInstance:
    """Loads the standalone toy example, which stores K in its own file
    rather than a metadata.csv row."""
    ulds = load_ulds_csv(data_root / "toy_example_ulds.csv")
    packages = load_packages_csv(data_root / "toy_example_packages.csv")
    with open(data_root / "toy_example_K.csv", newline="") as f:
        reader = csv.DictReader(f)
        k_row = next(reader)
        k_penalty = float(k_row["K"])
    return ProblemInstance(
        instance_id="toy_example",
        ulds=ulds,
        packages=packages,
        k_penalty=k_penalty,
    )


def load_split_instance(data_root: Path, split: str, instance_name: str,
                         metadata_cache: Optional[Dict[str, Dict[str, str]]] = None
                         ) -> ProblemInstance:
    """
    Loads one instance from a split directory ('generated_test' or
    'synthetic_train'). K is read from that split's metadata.csv.

    Pass a pre-loaded metadata_cache (from load_metadata_csv) when loading
    many instances from the same split, to avoid re-reading metadata.csv
    once per instance.
    """
    split_dir = data_root / split
    ulds = load_ulds_csv(split_dir / f"{instance_name}_ulds.csv")
    packages = load_packages_csv(split_dir / f"{instance_name}_packages.csv")

    if metadata_cache is None:
        metadata_cache = load_metadata_csv(split_dir / "metadata.csv")

    if instance_name not in metadata_cache:
        raise KeyError(
            f"{instance_name} not found in {split_dir / 'metadata.csv'}"
        )
    k_penalty = float(metadata_cache[instance_name]["K"])

    return ProblemInstance(
        instance_id=instance_name,
        ulds=ulds,
        packages=packages,
        k_penalty=k_penalty,
    )


# ---------------------------------------------------------------------------
# Whole-split iteration
# ---------------------------------------------------------------------------

def list_instance_names(data_root: Path, split: str) -> List[str]:
    """Discovers every instance in a split by scanning for *_packages.csv
    files, so this stays correct even if instance numbering has gaps
    (confirmed: generated_test goes 000..049 but synthetic_train's numbering
    should never be assumed contiguous either)."""
    split_dir = data_root / split
    names = []
    for p in sorted(split_dir.glob("*_packages.csv")):
        names.append(p.name[: -len("_packages.csv")])
    if not names:
        raise ValueError(f"No *_packages.csv files found in {split_dir}")
    return names


def iter_split(data_root: Path, split: str):
    """Yields ProblemInstance objects for every instance in a split, reading
    metadata.csv once up front for efficiency."""
    split_dir = data_root / split
    metadata_cache = load_metadata_csv(split_dir / "metadata.csv")
    for instance_name in list_instance_names(data_root, split):
        yield load_split_instance(data_root, split, instance_name, metadata_cache)


# ---------------------------------------------------------------------------
# ULD catalogue (optional, for building custom fleets)
# ---------------------------------------------------------------------------

def load_uld_catalogue(data_root: Path) -> List[ULD]:
    """
    Loads the master ULD catalogue (uld_catalogue.csv). This is a reference
    table of available container specs, NOT a per-instance fleet -- it's
    useful if you want to construct your own test instances or look up what
    container sizes exist, but solving an instance only needs that
    instance's own *_ulds.csv file.
    """
    path = data_root / "uld_catalogue.csv"
    ulds = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"catalogue_id", "Length", "Width", "Height", "Weight_Limit"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            ulds.append(ULD(
                id=f"CAT-{row['catalogue_id']}",
                length=float(row["Length"]),
                width=float(row["Width"]),
                height=float(row["Height"]),
                weight_limit=float(row["Weight_Limit"]),
            ))
    return ulds
