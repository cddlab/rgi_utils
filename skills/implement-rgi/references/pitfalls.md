# Pitfalls (the ones that actually bit during real integrations)

## 1. Use a per-structure instance, not a singleton
A singleton `CombinedRestraints` shared across a batch leaks the last structure's
config into all of them: parse-time `set_config` and predict-time `setup` are
separated in time, so by predict time the singleton holds whatever parsed last.
The symptom: two structures with different targets both end up using one target.
**Fix**: build a fresh `CombinedRestraints()` per structure and pass that
structure's config via `setup(config=...)`. For a tool that parses all inputs up
front (e.g. boltz), carry the config on the per-structure record (add a
`restraints_config` field to its Record dataclass) so predict time reads the
right one. This also makes batch + retry correct with no extra flags.

## 2. Leaving atoms — a CCD ligand may model fewer atoms than the mol
A CCD ligand can have atoms flagged `pdbx_leaving_atom_flag` (glucose **O1**,
amino-acid **OXT**) that the tool's tokenization drops, so the RDKit mol has more
atoms than the structure. Building restraints from the full mol then mismatches
indices / constrains wrong atoms. **Fix**: in `iter_ligand_confs`, find which mol
atoms are actually present (match by atom name against the structure) and
**subset the mol** to those (preserving bonds, chiral tags, conformer). The
restraint count drops accordingly — correct, not a bug. When comparing tools for
parity, pick a leaving-atom-free ligand (ATP, NAD, caffeine) so counts match.

## 3. The flat-index convention is tool-specific
`AtomRecord.index` and `LigandConf.global_indices` must be the atom's row in the
*exact* coordinate tensor you hand to `minimize` (after any reshape). boltz:
padded atom index. protenix: atom_array index. AF3:
`token*max_atoms_per_token + within`. Get it wrong and the optimizer moves the
wrong atoms (silently). Sanity check: `get_elements()[index]` should be the
element you expect for a known atom.

## 4. `resid` is the per-chain 1-based ordinal
Yield `resid` as the residue/token ordinal **within the chain** (resets at each
chain) — not a cumulative global index, not the raw author residue number. This
is what makes one selection string mean the same atom in every tool. An earlier
bug had boltz/AF3 using a global token index and protenix a per-chain res_id, so
the same `chain B and resid 5` selected different atoms across tools and could
crash on multi-chain inputs.

## 5. Multi-ligand: disjoint global indices keep restraints separate
Each `LigandConf` supplies its own `global_indices`; the featurizer unions all
referenced atoms into `active_sites` and remaps to local indices. Two ligands
have disjoint global indices (different atoms), so their restraints never
collide — no per-ligand special-casing needed. One value of conformer
`start_sigma` covers all ligands.

## 6. Verify conformer extraction equivalence before trusting parity
The featurizer derives bonds from `mol.GetBonds()`, angles from
`GetSubstructMatches("*~*~*")`, and chiral volumes from CW/CCW-tagged atoms over
`combinations(neighbors, 3)`. If the tool had a bespoke extractor, diff its
output against the featurizer on the *same* mol (counts and values) before
deleting it — e.g. a glucose mol gives bond=12, angle=17, chiral=5; intramolecular
VdW with default scale/dmax gives 34 pairs.

## 7. Don't add flags — route the one config dict
The entire config (distance + conformer + start_sigma + gpu/method/max_iter) is a
single dict parsed by `rgi_utils.config`. The tool only surfaces that dict from
its input (YAML/JSON) and passes it to `setup(config=...)`. Per-feature CLI flags
duplicate parsing and drift the tool out of parity — the dict is the single
source of truth. (start_sigma is per-distance + one for all conformer terms,
already handled in the dict.)

## 8. batch and retry come (almost) for free
With per-structure instances, **batch** (many inputs in one run) just works.
**Retry** (rerun a failed batch, skip the finished structures) is the tool's
*standard* output-exists check — reuse it, don't invent a new mechanism. If the
tool's default is to make a fresh timestamped dir when output exists, change the
default to "skip completed jobs" instead of adding a `--skip_existing` flag, and
let the existing force/override flag handle recompute. The decisive test that the
fix works: run two structures with *different* configs in one batch and confirm
each reaches its own target (e.g. COM 25 Å and 45 Å, not both the same).

## 9. Don't re-port features that are already shared
If a tool needs a feature another tool already has (e.g. intramolecular VdW), add
it to `rgi_utils` (gated/opt-in so other tools are unaffected) rather than in the
tool. That keeps one implementation and brings the feature to every tool at once.
Example: AF3's intramolecular VdW became `featurizer._build_intramolecular_vdw`,
opt-in via `vdw: {mode: intramolecular}`, leaving boltz/protenix's dynamic VdW
untouched.
