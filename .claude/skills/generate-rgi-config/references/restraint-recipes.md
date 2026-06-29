# Restraint recipes — goal → restraint type, penalty, example block

This is the interview's hardest step: turn a plain-language goal into the right restraint
type, penalty shape, and a concrete config block. Each recipe below is a *starting point* —
copy it, swap the selections/targets, and confirm against `doc/config.md` for any key you
change. Blocks are shown in YAML; for JSON tools (protenix/AF3/openfold) it is the same
structure in JSON.

Recurring decisions, in plain terms:

- **Pin vs bound.** Does the user want a quantity to be *exactly* a value (`harmonic`), or
  just kept *within a range / above a floor / below a ceiling* (`flat-bottomed` family)?
- **Always on vs late.** By default a restraint is active at every denoising step (omit
  `start_sigma`). Gate it late (`start_sigma: 1.0`) only if the user wants the model to
  fold freely first and be nudged near the end (e.g. conformer terms often read better
  applied late, once the pocket exists).
- **Degrees, not radians.** `angle` / `dihedral` targets are in degrees in the config.

---

## 1. Distance — "keep these two parts ~X Å apart"

The most common restraint. Pulls the **centroid distance** between two atom groups to a
target. It is CG-minimised (with a reduced-mass `_move_centroid` rescale so each group
translates rigidly), reaching the target at convergence.

> "Hold the N-terminal domain and the C-terminal domain about 25 Å apart."

```yaml
distance_restraints_config:
  - atom_selection1: "chain A and (resid 5 to 84)"     # group 1 (its centroid)
    atom_selection2: "chain A and (resid 90 to 180)"   # group 2 (its centroid)
    harmonic:
      target_distance: 25.0                            # Å
```

Variants (swap the penalty block):

| user says | block |
|---|---|
| "exactly 25 Å" | `harmonic: {target_distance: 25.0}` |
| "no closer than 25 Å" (floor) | `flat-bottomed1: {target_distance1: 25.0}` |
| "no farther than 25 Å" (ceiling) | `flat-bottomed2: {target_distance2: 25.0}` |
| "between 20 and 25 Å" (band) | `flat-bottomed: {target_distance1: 20.0, target_distance2: 25.0}` |

`move` (optional): `both` (default, both groups move toward each other) / `1` / `2` (move
only that group, **pin** the other — e.g. move a ligand toward a fixed pocket).

---

## 2. Angle / dihedral — "set the bend / twist between regions"

The angle of **3 group centroids** (vertex = group 2) or the dihedral of **4 group
centroids** (axis = groups 2–3). Targets in **degrees**.

> "Make the three-lobe arrangement open to 90°." / "Put these four regions in a trans
> (180°) arrangement."

```yaml
angle_restraints_config:
  - atom_selection1: "chain A and (resid 1 to 80)"
    atom_selection2: "chain A and (resid 81 to 140)"   # vertex
    atom_selection3: "chain A and (resid 141 to 224)"
    harmonic:
      target_angle: 90.0          # degrees

dihedral_restraints_config:
  - atom_selection1: "chain A and (resid 1 to 60)"
    atom_selection2: "chain A and (resid 61 to 120)"   # axis start
    atom_selection3: "chain A and (resid 121 to 180)"  # axis end
    atom_selection4: "chain A and (resid 181 to 224)"
    harmonic:
      target_dihedral: 180.0      # degrees
```

Same four penalty shapes as distance (`target_angle1/2`, `target_dihedral1/2` for the
flat-bottomed family). `weight: 1.0` drives any group size (the engine moves a whole group
rigidly). `move` picks which groups are free; default frees the arms/ends and pins the
vertex/axis. Pick groups whose centroids are **not collinear** (a collinear arrangement has
an ill-defined gradient — see config.md).

---

## 3. Conformer — "keep the ligand chemically sensible"

Holds a **ligand** near its ideal RDKit geometry while the pocket forms: bond lengths,
bond angles, chirality (`chiral`), cis/trans of double bonds (`cistrans`), sp2 planarity
(`improper`, opt-in), and clash avoidance (`vdw`). It is a single dict (not a list).

> "Don't let the bound ATP distort into a weird shape."

```yaml
conformer_restraints_config:
  start_sigma: 1            # often applied late, once the pocket exists (optional)
  bond:     {weight: 1.0}
  angle:    {weight: 1.0}
  chiral:   {weight: 1.0}
  cistrans: {weight: 1.0}
  vdw:      {weight: 1.0}   # mode defaults to "both"
  # improper: {weight: 1.0}  # OFF by default; add it to enforce sp2 double-bond planarity
```

Each sub-block is **off unless present** (a listed term defaults to `weight: 1.0`). Include
only the terms the user wants; `bond` + `angle` + `chiral` is a sensible default set.

> ⚠ **A conformer block does nothing without the per-ligand opt-in flag** (placement
> differs per tool — see `tools.md`). This is the single most common silent no-op. Always
> write the opt-in alongside the block.

Notes worth telling the user: `cistrans` / `improper` only fire on ligands that actually
have acyclic non-aromatic double bonds (ATP/NAD/caffeine have none → those counts are 0,
which is correct, not a bug). A glutamine/sugar CCD drops a leaving atom, changing counts.

---

## 4. RMSD — "pull this region onto a reference structure"

Drives the Kabsch-superposed RMSD of a group toward a reference **PDB** (which must already
exist — usually a vanilla prediction saved to PDB). Use it to bias the fold toward a known
conformation / homolog.

> "Keep the backbone close to this reference PDB." / "Pin this loop onto the reference but
> let the model relax it at the very end."

```yaml
rmsd_restraints_config:
  - ref_pdb: "ref.pdb"          # OR ref_cif: "ref.cif" (mmCIF) — mutually exclusive
    atom_selection_target: "chain A and backbone"   # which atoms to fit+measure (both sides)
    atom_selection_ref:    "chain A and backbone"
    harmonic:
      target_rmsd: 0.0          # 0 = match the reference; flat-bottomed2 {target_rmsd2: X} = within X Å
    # stop_sigma: 1.0           # release late so the model re-idealises a strained terminus
```

Key points: the reference is `ref_pdb` (PDB) **or** `ref_cif` (mmCIF), mutually exclusive —
use `ref_cif` to point straight at a `.cif` prediction. `target_rmsd: 0` matches the ref;
`flat-bottomed2: {target_rmsd2: 2.0}` keeps it within 2 Å. `pairing` defaults to `align` (a
homolog ref maps on by sequence). Fit and
measured atoms can be chosen separately (`atom_selection_*_fit` / `_calc`); restrict the
fit to `backbone` / `name CA` so a substituted side chain is not pinned. There is **no bare
`atom_selection` key** — it raises; use the suffixed/target/ref keys. Full surface in
config.md.

---

## 5. Custom — "something the five built-ins don't cover"

Write the restraint as a **math formula** over named atom-group centroids — no Python, no
tool change. Use it for symmetry, equidistance, radius-of-gyration, or any algebraic combo.

> "Keep these two inter-domain distances equal." / "Make this domain more compact."

```yaml
custom_restraints_config:
  - name: symmetric
    energy: "(distance(A, B) - distance(C, D))**2"
    selections:
      A: "chain A and resid 10"
      B: "chain B and resid 10"
      C: "chain A and resid 90"
      D: "chain B and resid 90"
  - name: compact
    energy: "harmonic(rg(dom), 12.0)"        # pull radius of gyration toward 12 Å
    selections: {dom: "chain A and resid 1 to 80"}
```

Vocabulary (geometry on centroids; **angles here are in radians**, unlike the built-in
configs): `centroid` `distance` `angle` `dihedral` `rg` `norm` `dot`; penalties `harmonic`
`flat_bottomed{,1,2}`; math `sqrt exp log abs sin cos clip sum minimum maximum where`. No
`if`, no imports — it is parsed safely. The formula must reduce to a scalar. Full list +
semantics: config.md "custom_restraints_config".

---

## Combining restraints

All blocks compose — a single config can carry a distance restraint, a conformer block, and
an RMSD restraint at once (see the repo's `bench_in_<tool>_distconf*` and `doc/<tool>.md`
"Full config" examples, which exercise every block together). Add `verbose: true` so the
setup log reports the per-type counts and the user can confirm each block built what they
expect.
