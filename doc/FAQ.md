# Frequently asked questions

[Documentation index](README.md)

## Why does an atom-to-atom distance restraint break the structure when modeling an enzyme reaction?

An atom-level restraint can be interpreted as noise by the structure-prediction model during
reverse diffusion. Without conformer restraints, local geometry is not guaranteed to remain
valid.

If you do not want to use conformer restraints, define restraints between appropriate domains or
other sufficiently large atom groups. Domain-level guidance allows the model to treat the
restraint as a coherent structural change rather than local noise.

Possible remedies are:

- Move domains or other suitable atom groups instead of individual atoms.
- If necessary, use `move` to choose which `atom_selection` groups are allowed to move; the
  other selected groups are pinned for that restraint.
- Combine the distance restraint with conformer restraints.
- Set `stop_sigma` so that the restraint is released during the final denoising steps.

Note that releasing a restraint with `stop_sigma` may cause the final structure to no longer
satisfy the restraint target.

## Why does an RMSD restraint break the predicted structure?

Consider combining the RMSD restraint with conformer restraints or setting `stop_sigma` to release
the RMSD restraint during the final denoising steps.

## Why does applying a distance restraint only near the end (`start_sigma: 1`) break the protein structure?

During reverse diffusion, larger sigma values allow guidance toward larger-scale structural
changes. As sigma decreases, the scale of changes that can be induced also decreases; around
`sigma = 1`, guidance is approximately limited to atom-scale changes.

If the restraint requires a domain motion or another large rearrangement, applying it only at low
sigma can force the model to satisfy it through local distortion instead. Choose the restraint's
activation timing according to the scale of the desired structural change and any conformer
correction that is required. See [Sigma gating](config.md#sigma-gating-start_sigma--stop_sigma)
for configuration details.

## Why did enabling RGI slow down inference?

RGI minimizes the restraint energy at every diffusion step where restraints are active, so it is
generally more computationally expensive than standard guidance. The cost increases as the number
of restraints grows, both from evaluating the additional restraints and from resolving conflicts
among them. Conformer and RMSD restraints are especially prone to longer runtimes because they
often involve many restrained terms or atoms.

Consider reducing the number of restraints, using `start_sigma` and `stop_sigma` to limit the
diffusion steps on which they are active, and ensuring that GPU execution and JIT compilation are
enabled appropriately for your integration.

## Where can I ask questions or report bugs about RGI?

Please use [GitHub Issues](https://github.com/cddlab/rgi_utils/issues).
