"""Shared energy aggregation and sigma/step gating runtime."""

from __future__ import annotations

from rgi_utils.energy._terms import BREAKDOWN_KEYS, term_energies


def gates(ops, prepared, positions, sigma, step=None):
    """Build the conformer gate and the per-entry gate callable."""
    sigma_value = None if sigma is None else ops.scalar_like(sigma, positions)
    step_value = None if step is None else ops.scalar_like(step, positions)
    conformer = ops.scalar_like(1.0, positions)
    if sigma_value is not None:
        condition = (sigma_value <= prepared.get("conf_start_sigma", 1e30)) & (
            sigma_value >= prepared.get("conf_stop_sigma", -1.0)
        )
        conformer = conformer * ops.astype_like(condition, conformer)
    if step_value is not None:
        condition = (step_value >= prepared.get("conf_start_step", float("-inf"))) & (
            step_value <= prepared.get("conf_stop_step", float("inf"))
        )
        conformer = conformer * ops.astype_like(condition, conformer)

    def per_entry(start_sigma, stop_sigma, start_step, stop_step, mask):
        result = mask
        if sigma_value is not None:
            result = result * ops.astype_like(sigma_value <= start_sigma, mask)
            if stop_sigma is not None:
                result = result * ops.astype_like(sigma_value >= stop_sigma, mask)
        if step_value is not None:
            if start_step is not None:
                result = result * ops.astype_like(step_value >= start_step, mask)
            if stop_step is not None:
                result = result * ops.astype_like(step_value <= stop_step, mask)
        return result

    return conformer, per_entry


def total_energy(ops, leaf_fns, positions, prepared, sigma=None, step=None):
    """Sum every active registered restraint term."""
    conformer, per_entry = gates(ops, prepared, positions, sigma, step)
    total = ops.scalar_like(0.0, positions)
    for value in term_energies(
        leaf_fns, prepared, positions, conformer, per_entry
    ).values():
        total = total + value
    return total


def energy_breakdown(ops, leaf_fns, positions, prepared, sigma=None, step=None):
    """Return every registered restraint term as a Python float."""
    conformer, per_entry = gates(ops, prepared, positions, sigma, step)
    output = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
    for key, value in term_energies(
        leaf_fns, prepared, positions, conformer, per_entry
    ).items():
        output[key] = float(value)
    return output
