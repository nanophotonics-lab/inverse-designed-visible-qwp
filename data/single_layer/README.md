# Single-layer rectangular nanopillar baseline

The single-layer reference is defined analytically rather than by a voxelized
MaterialGrid file.

- Unit-cell period: 200 nm × 200 nm
- Rectangular TiO2 nanopillar width: wx = 60 nm
- Rectangular TiO2 nanopillar width: wy = 140 nm
- Pillar height: 600 nm
- Substrate: 500-nm-thick SiO2

The nanopillar is centered in the unit cell and aligned with the x and y axes.
The CSV file is the authoritative geometry definition. No voxelized `.npy`
file is provided because that would introduce grid-dependent approximation
into an exactly parameterized geometry.
