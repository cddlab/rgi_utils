"""Atom identity must agree across selectors and reference-file conventions."""

import numpy as np
import pytest

from rgi_utils.atom_context import AtomRecord
from rgi_utils.pdb_ref import PdbAtom, read_cif_atoms
from rgi_utils.rmsd_restr_data import pair_target_to_ref
from rgi_utils.selection import AtomSelector


@pytest.mark.parametrize(
    "name,kind", [("C1*", "rna"), (" C1' ", "dna"), (" ca ", "protein")]
)
def test_backbone_names_use_the_same_normalization_as_name_selector(name, kind):
    atom = {"name": name, "mol_type": kind}
    assert AtomSelector("backbone").matches(atom)
    assert not AtomSelector("sidechain").matches(atom)


@pytest.mark.parametrize("dtype", [int, np.int64])
def test_large_range_and_discrete_selections_preserve_membership(dtype):
    # The range must stay compact even when its endpoints exceed any real structure.
    interval = AtomSelector("index 0 to 1000000000000")
    discrete = AtomSelector("index 1 1000000000000")
    for value in [0, 1, 5000, 1000000000000, 1000000000001]:
        assert interval.matches({"index": dtype(value)}) == (value <= 1000000000000)
        assert discrete.matches({"index": dtype(value)}) == (
            value in {1, 1000000000000}
        )


@pytest.mark.parametrize(
    "target_name,ref_name",
    [
        ("C1'", "C1*"),
        ("H2''", 'H2"'),
        (" CA ", "ca"),
    ],
)
def test_rmsd_identity_uses_normalized_atom_names(target_name, ref_name):
    target = [AtomRecord("A", 1, 7, name=target_name)]
    reference = [PdbAtom("A", 1, 0, ref_name, "C", 1.0, 2.0, 3.0)]
    sites, positions = pair_target_to_ref(
        target, reference, None, None, "fit", ref_path="reference.cif"
    )
    assert list(sites) == [7]
    np.testing.assert_array_equal(positions, [[1.0, 2.0, 3.0]])


def test_rmsd_rejects_duplicate_normalized_reference_names():
    target = [AtomRecord("A", 1, 0, name="C1'")]
    reference = [
        PdbAtom("A", 1, i, name, "C", 1.0, 2.0, 3.0)
        for i, name in enumerate(["C1'", "C1*"])
    ]
    with pytest.raises(ValueError, match="duplicate reference atom"):
        pair_target_to_ref(
            target, reference, None, None, "fit", ref_path="reference.cif"
        )


def _write_cif(path, missing=None, auth=False):
    fields = {
        "label_asym_id": "A",
        "label_seq_id": "8",
        "label_atom_id": "CA",
        "label_comp_id": "ALA",
        "type_symbol": "C",
        "Cartn_x": "1",
        "Cartn_y": "2",
        "Cartn_z": "3",
    }
    if auth:
        fields.update(
            auth_asym_id="?", auth_seq_id=".", auth_atom_id="?", auth_comp_id="."
        )
    if missing:
        del fields[missing]
    path.write_text(
        "data_reference\nloop_\n"
        + "".join(f"_atom_site.{key}\n" for key in fields)
        + " ".join(fields.values())
        + "\n"
    )


@pytest.mark.parametrize("auth", [False, True])
def test_cif_optional_columns_and_rowwise_label_fallback(tmp_path, auth):
    path = tmp_path / "reference.cif"
    _write_cif(path, auth=auth)
    atoms = read_cif_atoms(str(path))
    assert len(atoms) == 1
    atom = atoms[0]
    assert (atom.chain, atom.resid, atom.name, atom.res_name) == ("A", 1, "CA", "ALA")
    assert (atom.x, atom.y, atom.z) == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("field", ["label_seq_id", "label_atom_id", "label_comp_id"])
def test_cif_missing_required_columns_raise_clear_error(tmp_path, field):
    path = tmp_path / "reference.cif"
    _write_cif(path, missing=field)
    with pytest.raises(ValueError, match="missing required.*" + field):
        read_cif_atoms(str(path))
