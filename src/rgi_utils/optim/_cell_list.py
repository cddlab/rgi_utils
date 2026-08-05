"""Framework-independent constants for dynamic VdW cell lists."""

from __future__ import annotations

# Process hash buckets in bounded-width chunks. This is a traversal width, not a
# capacity: every bucket is consumed, so dense or hash-colliding cells lose no atoms.
CELL_CHUNK_SIZE = 32

# Three odd spatial-hash multipliers. Hash collisions are harmless because both backend
# implementations verify the full integer cell coordinate after the hash lookup.
CELL_HASH_PRIMES = (73856093, 19349663, 83492791)

# With cell width equal to dmax, every point within dmax lies in the same cell or one of
# the 26 immediately adjacent cells.
CELL_OFFSETS = tuple(
    (x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
)
