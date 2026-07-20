"""Analyze QBP polymer-restraint E2E predictions, including bench-rgi MolProbity."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys

import gemmi
import numpy as np

BENCH = pathlib.Path("/home/hori/works/misc/impl_rgi/bench-rgi")
sys.path.insert(0, str(BENCH))
from common.quality import MOLPROBITY_FIELDS, molprobity_of  # noqa: E402


def _atom(residue, name):
    atom = residue.find_atom(name, "*")
    if atom is None:
        return None
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def _dihedral(a, b, c, d):
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _protein_residues(path):
    structure = gemmi.read_structure(str(path))
    for chain in structure[0]:
        if chain.name == "A":
            return [residue for residue in chain if _atom(residue, "CA") is not None]
    raise ValueError(f"chain A not found in {path}")


def _kabsch_rmsd(moving, target):
    moving = moving - moving.mean(axis=0)
    target = target - target.mean(axis=0)
    u, _s, vt = np.linalg.svd(moving.T @ target)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return float(np.sqrt(np.mean(np.sum((moving @ rotation - target) ** 2, axis=1))))


def local_metrics(path, reference):
    residues = _protein_residues(path)
    ref_residues = _protein_residues(reference)
    bond_errors = []
    omega_errors = []
    for index, (previous, current) in enumerate(zip(residues, residues[1:])):
        ca0 = _atom(previous, "CA")
        c = _atom(previous, "C")
        n = _atom(current, "N")
        ca1 = _atom(current, "CA")
        if c is None or n is None:
            continue
        bond_errors.append(abs(float(np.linalg.norm(c - n)) - 1.329))
        if ca0 is not None and ca1 is not None:
            omega = _dihedral(ca0, c, n, ca1)
            ref_previous = ref_residues[index]
            ref_current = ref_residues[index + 1]
            ref_omega = _dihedral(
                _atom(ref_previous, "CA"),
                _atom(ref_previous, "C"),
                _atom(ref_current, "N"),
                _atom(ref_current, "CA"),
            )
            omega_errors.append(abs((omega - ref_omega + 180.0) % 360.0 - 180.0))

    lo, hi = 89, 180
    moving_ca = np.array([_atom(r, "CA") for r in residues[lo:hi]])
    target_ca = np.array([_atom(r, "CA") for r in ref_residues[lo:hi]])
    return {
        "region_ca_rmsd": _kabsch_rmsd(moving_ca, target_ca),
        "max_peptide_bond_error": max(bond_errors),
        "peptide_bond_outliers": sum(error > 0.10 for error in bond_errors),
        "max_omega_error": max(omega_errors),
        "omega_outliers": sum(error > 30.0 for error in omega_errors),
    }


def prediction_path(base, tag):
    matches = sorted((base / tag).glob("boltz_results_*/predictions/*/*_model_0.cif"))
    if len(matches) != 1:
        raise ValueError(f"expected one prediction for {tag}, got {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        default=pathlib.Path("/home/hori/works/misc/impl_rgi/bench_m_ref_boltz.cif"),
    )
    args = parser.parse_args()
    tags = (
        "rmsd_only_target0_seed0",
        "polymer_target0_seed0",
        "polymer_target0_seed1",
        "polymer_target05_seed0",
    )
    rows = []
    for tag in tags:
        path = prediction_path(args.base, tag)
        row = {"tag": tag, "path": str(path), **local_metrics(path, args.reference)}
        if os.environ.get("PHENIX_ENV"):
            workdir = args.base / "molprobity" / tag
            row.update(molprobity_of(path, workdir=workdir))
        else:
            row.update({field: float("nan") for field in MOLPROBITY_FIELDS})
        rows.append(row)

    output = args.base / "analysis.csv"
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(
            row["tag"],
            f"CA_RMSD={row['region_ca_rmsd']:.3f}",
            f"bond_max={row['max_peptide_bond_error']:.3f}",
            f"bond_out={row['peptide_bond_outliers']}",
            f"omega_max={row['max_omega_error']:.1f}",
            f"omega_out={row['omega_outliers']}",
            f"MolProbity={row['molprobity_score']}",
        )


if __name__ == "__main__":
    main()
