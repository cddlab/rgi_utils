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

## 10. The structure's atom coords may be ZEROED — take conformer geometry from the reference
Some tools zero their atom-array coordinates at inference (a placeholder the
diffusion fills in). openfold-3 does exactly this (`atom_array.coord[:] = 0.0`,
"for consistency"). If `iter_ligand_confs` reads those coords, every conformer
restraint TARGET (bond length, bond angle, chiral volume) is built from a
degenerate origin cloud → bonds target 0, angles target 90°, and the minimizer
**collapses the ligand**. **Fix**: take conformer geometry from the tool's
*reference conformer* (a separate per-atom feature — openfold `ref_pos`, chai
`atom_ref_pos` — built from the RDKit reference mol). Per-mol random
rotation/translation is fine: rigid transforms preserve bond/angle/chiral values.
Check at build time that `LigandConf.conf_coords` are non-zero and spread out.

## 11. No intra-ligand bonds in the structure context — perceive them, then re-enable implicit-H
A tool may expose the reference conformer coords but NOT the ligand's intra-molecular
bonds. chai's `atom_covalent_bond_indices` holds only inter-residue/glycan links, so
a normal ligand arrives with zero bonds → a bond-less mol → bond/angle/chiral all
empty (a silent no-op). **Fix**: build the mol from the conformer and perceive
connectivity — `build_ligand_mol(elements, ref_coords, [], perceive_bonds=True)`
(RDKit `DetermineConnectivity`). **GOTCHA inside the fix**: `DetermineConnectivity`
leaves every atom `noImplicit=True` with 0 implicit H, so heavy-atom stereocentres
look 3-coordinate and `AssignStereochemistryFrom3D` assigns ZERO chiral tags — the
chiral restraints then silently vanish while bond/angle (pure geometry) look fine.
`build_ligand_mol` handles this by running `SetNoImplicit(False)` +
`UpdatePropertyCache(strict=False)` before stereo perception. Do NOT reach for
`DetermineBonds`/`DetermineBondOrders` instead — they raise on charged ligands
(e.g. ATP triphosphate).

## 12. Identify ligand atoms by entity type, not biotite `hetero`
biotite sets `hetero=True` for ANY non-standard CCD residue — including a
non-canonical residue inserted into a protein/NA chain — so keying ligand detection
on `hetero` misclassifies modified polymer residues as ligands. **Fix**: use the
tool's entity/molecule-type signal instead (openfold `molecule_type_id == LIGAND`,
chai `token_entity_type == LIGAND`). Fall back to `hetero` only if no such
annotation exists.

## 13. A `finalize` energy of 0.0 can mean "no restraints", not "satisfied"
The decisive proof that a restraint type is working is the **setup spec count**, not
the finalize energy. A term that built **zero** restraints reports energy `0.00000`
— indistinguishable at a glance from a perfectly-satisfied restraint. Always read
the setup log (`built spec: bonds=.. angles=.. chirals=.. distances=.. vdw=..`) and
confirm the count is non-zero for what you requested. The chai chiral no-op (pitfall
11) hid for a full GPU run precisely because its `chiral=0.00000` was read as
"satisfied" when it was actually "0 chiral restraints built".

## 14. The tool may import an undeclared dep, or resolve a CPU build
Integration code that adds `import yaml` (chai) needs PyYAML, which the tool may not
declare — install it into the tool's env. And a bare `torch>=X` dependency resolves
to the **CPU** wheel from PyPI by default; install from the CUDA index
(`--index-url https://download.pytorch.org/whl/cu12x`) or RGI's GPU path is silently
unavailable (`torch.cuda.is_available() == False`). These bite only at real-GPU run
time, not at import — see pitfall 13's lesson about not trusting a clean-looking run.

## 15. A bare `resid`/`index` range matches across ALL chains — chain-qualify it
`resid 5 to 84` (no `chain` qualifier) matches that per-chain ordinal in EVERY chain,
including a ligand chain whose atoms each carry an ordinal (1..N). A protein distance
group written as a bare resid range silently sweeps in the ligand's atoms, shifting the
group's COM. This is standard selection-DSL behaviour (like PyMOL `resi`), NOT a bug —
but **qualify protein groups with `chain A and (...)`**. It is invisible in a
distance-only or conformer-only run; it only surfaces when a distance restraint and a
ligand coexist. The giveaway is an `n_active` / group size larger than expected — e.g.
a 1628-atom protein group becoming 1659 when a 31-atom ligand leaks in. When writing a
combined (distance + conformer) example, run it and check the group sizes, not just that
it completes.
