import pytest

from rgi_utils.selection import AtomSelector


def mol(chain="A", resid=1, index=0, name=None, mol_type=None, resname=None):
    d = {"chain": chain, "resid": resid, "index": index}
    if name is not None:
        d["name"] = name
    if mol_type is not None:
        d["mol_type"] = mol_type
    if resname is not None:
        d["resname"] = resname
    return d


class TestChainSelection:
    def test_single_chain(self):
        sel = AtomSelector("chain A")
        assert sel.eval(mol("A")) is True
        assert sel.eval(mol("B")) is False

    def test_multiple_chains(self):
        sel = AtomSelector("chain A B")
        assert sel.eval(mol("A")) is True
        assert sel.eval(mol("B")) is True
        assert sel.eval(mol("C")) is False


class TestResidSelection:
    def test_single_resid(self):
        sel = AtomSelector("resid 5")
        assert sel.eval(mol(resid=5)) is True
        assert sel.eval(mol(resid=6)) is False

    def test_resid_range(self):
        sel = AtomSelector("resid 1 to 5")
        for i in range(1, 6):
            assert sel.eval(mol(resid=i)) is True
        assert sel.eval(mol(resid=0)) is False
        assert sel.eval(mol(resid=6)) is False

    def test_resid_descending_range_raises(self):
        """`resid 5 to 3` (end < start) must raise a clear range-order error. The check
        used to be swallowed by the list-form backtracking (the `except ParseError` that
        handles `resid 5 6 7`), leaving a misleading 'trailing characters: to 3' message."""
        with pytest.raises(ValueError, match="less than start|Range end"):
            AtomSelector("resid 5 to 3")

    def test_multiple_resids(self):
        sel = AtomSelector("resid 1 3 5")
        assert sel.eval(mol(resid=1)) is True
        assert sel.eval(mol(resid=3)) is True
        assert sel.eval(mol(resid=5)) is True
        assert sel.eval(mol(resid=2)) is False


class TestIndexSelection:
    def test_single_index(self):
        sel = AtomSelector("index 10")
        assert sel.eval(mol(index=10)) is True
        assert sel.eval(mol(index=11)) is False


class TestNameSelection:
    # AtomSelector.matches is the public alias of .eval (same semantics).
    def test_single_name(self):
        sel = AtomSelector("name CA")
        assert sel.matches(mol(name="CA")) is True
        assert sel.matches(mol(name="CB")) is False

    def test_multiple_names(self):
        sel = AtomSelector("name N CA C O")
        for n in ("N", "CA", "C", "O"):
            assert sel.matches(mol(name=n)) is True
        assert sel.matches(mol(name="CB")) is False

    def test_name_case_insensitive(self):
        # PyMOL-like: a lowercase query matches an uppercase atom name and vice versa
        assert AtomSelector("name ca").matches(mol(name="CA")) is True
        assert AtomSelector("name CA").matches(mol(name="ca")) is True

    def test_name_none_never_matches(self):
        # an adapter that supplies no atom name must never match a name selector
        sel = AtomSelector("name CA")
        assert sel.matches(mol(name=None)) is False
        assert sel.matches({"chain": "A", "resid": 1, "index": 0}) is False

    def test_name_combined_with_resid(self):
        sel = AtomSelector("name CA and resid 5")
        assert sel.matches(mol(resid=5, name="CA")) is True
        assert sel.matches(mol(resid=6, name="CA")) is False
        assert sel.matches(mol(resid=5, name="CB")) is False

    def test_name_selects_a_prime_atom(self):
        # Nucleic-acid atoms carry a prime, and picking one out by name is what a
        # hydrogen-bond distance restraint needs.
        sel = AtomSelector("name C1'")
        assert sel.matches(mol(name="C1'")) is True
        assert sel.matches(mol(name="C1")) is False
        assert sel.matches(mol(name="N1")) is False

    @pytest.mark.parametrize(
        "query,atom",
        [
            ("C1'", "C1*"),  # selection in PDB v3 spelling, structure in v2
            ("C1*", "C1'"),  # and the other way round
            ("O2'", "O2*"),
            ('H2"', "H2''"),  # double prime, abbreviated vs written out
            ("H2''", 'H2"'),
        ],
    )
    def test_name_folds_the_prime_spellings(self, query, atom):
        # ' (PDB v3/mmCIF), * (PDB v2) and " (double prime) are the same atom under
        # different conventions; which one a file uses is not the user's choice, so a
        # selection must not depend on it.
        assert AtomSelector(f"name {query}").matches(mol(name=atom)) is True

    def test_name_list_and_composition_with_primes(self):
        sel = AtomSelector("name O2' OP1")
        assert sel.matches(mol(name="O2'")) is True
        assert sel.matches(mol(name="OP1")) is True
        assert sel.matches(mol(name="OP2")) is False
        # the name list must still stop at the operator, not swallow it as a name
        sel = AtomSelector("name C1' and resid 5")
        assert sel.matches(mol(resid=5, name="C1'")) is True
        assert sel.matches(mol(resid=6, name="C1'")) is False
        assert sel.matches(mol(resid=5, name="C2'")) is False

    def test_name_still_rejects_a_glued_operator(self):
        # "andresid" must not be swallowed as an atom name, which would silently drop
        # the rest of the selection.
        with pytest.raises(ValueError):
            AtomSelector("name CA andresid 5")


class TestMolTypeSelection:
    # protein/dna/rna match the normalized mol_type. Guards the shared bare-keyword
    # parser (the same path that now also yields backbone/sidechain).
    def test_protein_dna_rna(self):
        assert AtomSelector("protein").matches(mol(mol_type="protein")) is True
        assert AtomSelector("protein").matches(mol(mol_type="dna")) is False
        assert AtomSelector("dna").matches(mol(mol_type="dna")) is True
        assert AtomSelector("rna").matches(mol(mol_type="rna")) is True

    def test_moltype_none_or_ligand_never_matches(self):
        assert AtomSelector("protein").matches(mol()) is False
        assert AtomSelector("protein").matches(mol(mol_type="ligand")) is False

    def test_moltype_combined_with_chain(self):
        sel = AtomSelector("protein and chain A")
        assert sel.matches(mol("A", mol_type="protein")) is True
        assert sel.matches(mol("B", mol_type="protein")) is False

    def test_keyword_is_usable_as_chain_name(self):
        # a reserved keyword is still a valid chain identifier after `chain` (matches
        # how protein/backbone are treated): `chain backbone` selects chain "backbone".
        sel = AtomSelector("chain backbone")
        assert sel.matches(mol("backbone")) is True
        assert sel.matches(mol("A")) is False


class TestBackboneSidechain:
    # backbone/sidechain are POLYMER selectors: name-based but GATED on polymer type,
    # which is mol_type when the adapter sets it (boltz/esm/AF3) else derived from
    # resname (chai/of3/protenix). matches() is the public alias of eval().
    def test_backbone_protein_by_mol_type(self):
        sel = AtomSelector("backbone")
        for n in ("N", "CA", "C", "O", "OXT"):
            assert sel.matches(mol(name=n, mol_type="protein")) is True
        assert sel.matches(mol(name="CB", mol_type="protein")) is False

    def test_sidechain_protein_by_mol_type(self):
        sel = AtomSelector("sidechain")
        assert sel.matches(mol(name="CB", mol_type="protein")) is True
        assert sel.matches(mol(name="CG", mol_type="protein")) is True
        for n in ("N", "CA", "C", "O"):
            assert sel.matches(mol(name=n, mol_type="protein")) is False

    def test_backbone_case_insensitive(self):
        bb = AtomSelector("backbone")
        assert bb.matches(mol(name="ca", mol_type="protein")) is True

    def test_polymer_gate_from_resname_when_mol_type_unset(self):
        # the chai/of3/protenix path: no mol_type, polymer derived from resname
        bb = AtomSelector("backbone")
        sc = AtomSelector("sidechain")
        assert bb.matches(mol(name="CA", resname="ALA")) is True
        assert bb.matches(mol(name="CB", resname="ALA")) is False
        assert sc.matches(mol(name="CB", resname="ALA")) is True
        assert sc.matches(mol(name="CA", resname="ALA")) is False

    def test_ligand_never_matches(self):
        # an organic ligand atom named "C"/"N"/"O" must NOT match (no polymer type:
        # explicit "ligand", an unknown resname, or neither annotation at all)
        bb = AtomSelector("backbone")
        sc = AtomSelector("sidechain")
        assert bb.matches(mol(name="C", mol_type="ligand")) is False
        assert bb.matches(mol(name="N", resname="ATP")) is False  # ATP: not std polymer
        assert bb.matches(mol(name="C")) is False  # neither mol_type nor resname
        assert sc.matches(mol(name="C", mol_type="ligand")) is False

    def test_modified_residue_mse_diverges(self):
        # accepted cross-tool divergence: MSE is polymer only where the framework set
        # mol_type="protein"; from resname alone it derives None -> not polymer
        bb = AtomSelector("backbone")
        assert bb.matches(mol(name="CA", mol_type="protein", resname="MSE")) is True
        assert bb.matches(mol(name="CA", resname="MSE")) is False

    def test_backbone_sidechain_nucleic(self):
        bb = AtomSelector("backbone")
        sc = AtomSelector("sidechain")
        # sugar-phosphate is backbone (incl. primes); a base atom (N1) is sidechain
        assert bb.matches(mol(name="P", mol_type="dna")) is True
        assert bb.matches(mol(name="O5'", mol_type="dna")) is True
        assert bb.matches(mol(name="C1'", resname="DA")) is True  # resname-gated
        assert bb.matches(mol(name="N1", mol_type="dna")) is False
        assert sc.matches(mol(name="N1", mol_type="dna")) is True

    def test_backbone_combined_with_chain(self):
        sel = AtomSelector("backbone and chain A")
        assert sel.matches(mol("A", name="CA", mol_type="protein")) is True
        assert sel.matches(mol("B", name="CA", mol_type="protein")) is False

    def test_not_backbone_differs_from_sidechain_on_nonpolymer(self):
        # `not backbone` also matches non-polymer atoms (Not just negates), whereas
        # `sidechain` is polymer-gated -> they differ on a ligand atom. Documents the
        # semantics so neither is mistaken for the other.
        nb = AtomSelector("not backbone")
        sc = AtomSelector("sidechain")
        assert nb.matches(mol(name="C", mol_type="ligand")) is True
        assert sc.matches(mol(name="C", mol_type="ligand")) is False


class TestBooleanOperations:
    def test_and(self):
        sel = AtomSelector("chain A and resid 1")
        assert sel.eval(mol("A", resid=1)) is True
        assert sel.eval(mol("A", resid=2)) is False
        assert sel.eval(mol("B", resid=1)) is False

    def test_or(self):
        sel = AtomSelector("chain A or chain B")
        assert sel.eval(mol("A")) is True
        assert sel.eval(mol("B")) is True
        assert sel.eval(mol("C")) is False

    def test_not(self):
        sel = AtomSelector("not chain A")
        assert sel.eval(mol("A")) is False
        assert sel.eval(mol("B")) is True

    def test_parentheses(self):
        sel = AtomSelector("(not chain A) and resid 1 to 11")
        assert sel.eval(mol("A", resid=5)) is False
        assert sel.eval(mol("B", resid=5)) is True
        assert sel.eval(mol("B", resid=12)) is False

    def test_complex(self):
        sel = AtomSelector("chain A and resid 1 to 10 or chain B and resid 1 to 5")
        assert sel.eval(mol("A", resid=5)) is True
        assert sel.eval(mol("A", resid=11)) is False
        assert sel.eval(mol("B", resid=3)) is True
        assert sel.eval(mol("B", resid=6)) is False


class TestParseErrors:
    def test_invalid_syntax(self):
        with pytest.raises(ValueError):
            AtomSelector("invalid_keyword 1")

    def test_empty_string(self):
        with pytest.raises((ValueError, Exception)):
            AtomSelector("")
