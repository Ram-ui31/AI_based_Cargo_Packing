"""Runs py3dbp (a mature, much better-performing 3D packer per direct
comparison) on a (ULD, packages) set, and returns its placement decisions in
placement order -- used as an imitation-learning teacher for our own
placement policy, the same way a greedy heuristic was used to bootstrap
Phase B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeacherStep:
    package_id: str
    dx: int
    dy: int
    dz: int
    x: int
    y: int
    z: int


def _make_bin(uld_row):
    from py3dbp import Bin
    return Bin("ULD", int(uld_row["Length"]), int(uld_row["Height"]),
               int(uld_row["Width"]), float(uld_row["Weight_Limit"]))


def _extract_steps(bin_) -> tuple[list[TeacherStep], list[str]]:
    steps = []
    for item in bin_.items:
        x0, z0, y0 = int(item.position[0]), int(item.position[1]), int(item.position[2])
        # item.width/.height/.depth are the *original* (pre-rotation) constructor
        # values, not the actual placed extents -- must use get_dimension(),
        # which returns [w, h, d] already permuted for item.rotation_type.
        w, h, d = item.get_dimension()
        steps.append(TeacherStep(
            package_id=item.name,
            dx=int(round(w)), dz=int(round(h)), dy=int(round(d)),
            x=x0, y=y0, z=z0,
        ))
    unfit = [item.name for item in bin_.unfitted_items]
    return steps, unfit


def run_py3dbp(uld_row, packages_df) -> tuple[list[TeacherStep], list[str]]:
    """Returns (ordered placement steps, unfit package ids), largest-volume-first,
    with NO priority preference -- see run_py3dbp_priority_first for that.

    py3dbp's constructor is Bin/Item(name, width, height, depth, weight) where
    `height` is its real gravity/vertical axis, and position/get_dimension()
    both report values in (width, height, depth) order. Our own coordinate
    system uses (x=Length, y=Width, z=Height) with z as the vertical axis --
    so we must map our Height to py3dbp's `height` param (not positionally
    to its 3rd arg), and remap the output the same way, or py3dbp silently
    packs against the wrong physical vertical limit.
    """
    from py3dbp import Packer, Item

    packer = Packer()
    packer.add_bin(_make_bin(uld_row))
    for _, row in packages_df.iterrows():
        packer.add_item(Item(str(row["Package_ID"]), int(row["Length"]), int(row["Height"]),
                              int(row["Width"]), float(row["Weight"])))

    packer.pack(bigger_first=True, distribute_items=False, number_of_decimals=0)
    return _extract_steps(packer.bins[0])


def run_py3dbp_priority_first(uld_row, packages_df) -> tuple[list[TeacherStep], list[str]]:
    """Same packer, but priority packages are placed in their own pass before
    any economy package is even considered -- matching the hard "priority
    always ships" constraint our own placement env enforces. `pack(bigger_first=True)`
    globally re-sorts ALL items by volume regardless of add order, so simply
    adding priority items first to one Packer call does NOT achieve this (found
    the hard way) -- priority and economy must go through separate pack_to_bin
    passes into the same Bin object."""
    from py3dbp import Packer, Item

    bin_ = _make_bin(uld_row)
    bin_.format_numbers(0)  # pack() normally does this; we bypass pack() so must do it ourselves
    packer = Packer()  # stateless helper here; pack_to_bin only touches (bin, item)

    for is_priority in (True, False):
        sub = packages_df[(packages_df["Type"] == "Priority") == is_priority]
        items = [
            Item(str(row["Package_ID"]), int(row["Length"]), int(row["Height"]),
                 int(row["Width"]), float(row["Weight"]))
            for _, row in sub.iterrows()
        ]
        for item in items:
            item.format_numbers(0)
        items.sort(key=lambda it: it.get_volume(), reverse=True)
        for item in items:
            packer.pack_to_bin(bin_, item)

    return _extract_steps(bin_)
