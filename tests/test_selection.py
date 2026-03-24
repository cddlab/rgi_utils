import pytest

from rgi_utils.selection import AtomSelector


def mol(chain="A", resid=1, index=0):
    return {"chain": chain, "resid": resid, "index": index}


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
