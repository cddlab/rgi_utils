# Atom-selection DSL — choosing the atoms a restraint acts on

Every restraint names its atom groups with a small boolean language. A group's **centroid**
(mean position) is what distance/angle/dihedral restraints actually use, so a "group" is
usually a residue range, not a single atom.

## The tokens

| token | matches | example |
|---|---|---|
| `chain <ids>` | atoms in those chains | `chain A` · `chain A B` |
| `resid <n…>` / `resid A to B` | residues by **per-chain 1-based ordinal** | `resid 5` · `resid 1 to 84` · `resid 1 3 7` |
| `index <n…>` | atoms by flat row in the coordinate tensor | `index 42` |
| `name <names>` | atom name, case-insensitive | `name CA` · `name N CA C O` |
| `protein` / `dna` / `rna` | atoms of that polymer type | `protein` |
| `backbone` / `sidechain` | polymer backbone / sidechain heavy atoms | `backbone and chain A` |
| `and` `or` `not` `( )` | boolean composition (precedence: `not` > `and` > `or`) | `chain A and (resid 5 to 84 or resid 186 to 224)` |

## The one rule that causes most silent failures

> **`resid` is the per-chain 1-based ordinal.** It restarts at 1 on every chain, and a
> ligand chain gets its own ordinal. It is NOT the author/PDB residue number, and NOT a
> global index.

Consequences you must design around:

- **Always qualify a protein group with its chain.** `resid 90 to 180` alone will also
  match residues 90–180 of *every other chain*, including the ligand. Write
  `chain A and (resid 90 to 180)`.
- The first residue of chain A is `resid 1` even if the PDB calls it residue 27.
- A single-atom ligand in chain B is `chain B and resid 1` (or just `chain B`).

When the user gives you author residue numbers from a PDB, you usually need to convert to
the 1-based ordinal (subtract the offset of the first modelled residue). If you can't be
sure of the offset, say so and suggest they confirm via the run-time spec count.

## Ready-made patterns

| goal | selection |
|---|---|
| a whole protein chain | `chain A` |
| one contiguous domain | `chain A and (resid 5 to 84)` |
| two stretches as one group | `chain A and ((resid 5 to 84) or (resid 186 to 224))` |
| the ligand (its own chain) | `chain B` |
| backbone only (for an RMSD fit) | `chain A and backbone` |
| just the Cα atoms | `chain A and name CA` |
| everything except the ligand | `protein` |

## Validation reaches only the syntax

The bundled validator parses every selection string (so `chain A and (resid 5 to 84`
without the closing paren, or a misspelled keyword, fails early). But **a syntactically
valid selection that matches zero atoms passes validation** — the parser has no structure to
resolve against at config time. The wrong-range / forgotten-chain mistakes above produce
valid-but-empty selections.

The only way to confirm a selection picked the atoms you meant is at run time: set
`verbose: true` and read the setup log line `built spec: ... distances=N angles=N ...` — the
count must be non-zero, and for a distance restraint the log also reports the two group
sizes (e.g. `938 / 690`), which you can sanity-check against the residue ranges. This is why
SKILL.md insists on `verbose: true` and the spec-count check.

Full grammar and the polymer-type semantics of `backbone`/`sidechain`: `doc/config.md`,
"Atom-selection DSL".
