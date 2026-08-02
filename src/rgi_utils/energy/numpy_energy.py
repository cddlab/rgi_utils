"""NumPy restraint energies backed by the shared geometry kernels."""

from __future__ import annotations

from functools import partial

import numpy as np

from rgi_utils import _geometry as G
from rgi_utils._array_ops import get_ops
from rgi_utils.energy import _kernels as K
from rgi_utils.energy import _runtime
from rgi_utils.energy._terms import pack_spec

_OPS = get_ops("numpy")

bond_energy = partial(K.bond_energy, _OPS)
angle_energy = partial(K.angle_energy, _OPS)
chiral_energy = partial(K.chiral_energy, _OPS)
cistrans_energy = partial(K.cistrans_energy, _OPS)
vdw_energy = partial(K.vdw_energy, _OPS)
distance_energy = partial(K.distance_energy, _OPS)
group_angle_energy = partial(K.group_angle_energy, _OPS)
group_dihedral_energy = partial(K.group_dihedral_energy, _OPS)
group_improper_energy = group_dihedral_energy
rmsd_energy = partial(K.rmsd_energy, _OPS)
plane_energy = partial(K.plane_energy, _OPS)
group_plane_energy = partial(K.group_plane_energy, _OPS)

_group_centroid = partial(K._group_centroid, _OPS)
_move_centroid = partial(K._move_centroid, _OPS)
_dihedral_angle = partial(G.dihedral_points, _OPS)
_kabsch_R = partial(G.kabsch_rotation, _OPS)
_plane_rms = partial(G.plane_rms, _OPS)


def _group_delta(value, harmonic_deviation, target1, target2, geom_type):
    return G.restraint_delta(
        _OPS,
        value,
        target1,
        target2,
        geom_type,
        harmonic_deviation=harmonic_deviation,
    )


def _plane_normal(covariance):
    _values, vectors = _OPS.eigh(covariance)
    return vectors[..., :, 0]


_LEAF_FNS = {
    "bond_energy": bond_energy,
    "angle_energy": angle_energy,
    "chiral_energy": chiral_energy,
    "plane_energy": plane_energy,
    "cistrans_energy": cistrans_energy,
    "vdw_energy": vdw_energy,
    "distance_energy": distance_energy,
    "rmsd_energy": rmsd_energy,
    "group_angle_energy": group_angle_energy,
    "group_dihedral_energy": group_dihedral_energy,
    "group_improper_energy": group_improper_energy,
    "group_plane_energy": group_plane_energy,
}


def total_energy(positions, prepared, sigma=None, step=None):
    return _runtime.total_energy(
        _OPS, _LEAF_FNS, positions, prepared, sigma=sigma, step=step
    )


def energy_breakdown(positions, prepared, sigma=None, step=None):
    return _runtime.energy_breakdown(
        _OPS, _LEAF_FNS, positions, prepared, sigma=sigma, step=step
    )


def prepare_spec(spec):
    """Convert a backend-agnostic ``RestraintSpec`` into NumPy arrays."""
    return pack_spec(
        spec,
        lambda value: np.asarray(value, dtype=np.int64),
        lambda value: np.asarray(value, dtype=np.float64),
    )
