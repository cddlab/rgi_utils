"""Framework-free rgi_utils adapter for AlphaFold 3.

The in-tool shim (``alphafold3_restr`` ``model/restraints/adapter.py``) holds all
alphafold3 coupling — it reads the fold_input + featurised batch and resolves each
ligand's RDKit mol — then constructs ``AF3RestraintAdapter`` from plain data, exactly
like the torch tools' ``rgi_utils/<tool>/adapter.py``.
"""

from rgi_utils.alphafold3.adapter import AF3RestraintAdapter

__all__ = ["AF3RestraintAdapter"]
