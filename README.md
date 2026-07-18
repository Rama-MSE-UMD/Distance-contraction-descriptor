# Distance Contraction Descriptor

**Author:** Ramanuja Srinivasan Saravanan (University of Maryland)  
**Supervisor:** Prof. Yifei Mo  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rama-MSE-UMD/Distance-contraction-descriptor/blob/main/Distance_contraction.ipynb)

---

## Overview

This repository estimates distance-contraction descriptor values and predicts ion-migration pathways. It can be used to generate pre-NEB pathway geometries, visualize migration pathways, and pre-screen candidate solid-state electrolytes.

---

## How It Works

When a mobile ion (Na⁺, Li⁺, K⁺, Mg²⁺) hops between two lattice sites, it squeezes past neighbouring ions. At the tightest point — the transition state — those neighbours are closer than their equilibrium bond lengths. The **distance contraction** descriptor quantifies that squeeze: a large contraction indicates a high energy barrier; a small contraction indicates an open, low-barrier pathway.

---

## Features

- Enumerates all candidate ion-migration paths within a user-defined hop-distance cutoff
- Reduces paths to symmetry-unique hops using space-group operations
- Generates transverse-optimised NEB pathway geometries (pre-NEB structures)
- Identifies the transition-state image via maximum distance contraction
- Maps descriptor values back to all symmetry-equivalent paths for full percolation network reconstruction
- Determines percolation network dimensionality (1-D / 2-D / 3-D)
- Exports pathway CIF files for visualization (e.g. VESTA)

---

## Required Input Files

Place these files in the same directory as the notebook before running:

| File | Description |
|---|---|
| `<ICSD>.cif` or `<ICSD>_ord_rama.cif` | Crystal structure of the material |
| `periodic_table.json` | Shannon ionic radii and periodic table data |
| `Na_bond_stddev.csv` | Bond-distance statistics for Na conductors |
| `Li_bond_stddev.csv` | Bond-distance statistics for Li conductors |
| `K_bond_stddev.csv` | Bond-distance statistics for K conductors |
| `Mg_bond_stddev.csv` | Bond-distance statistics for Mg conductors |

---

## Usage

### Option 1 — Google Colab (recommended)

Click the badge at the top of this page. All dependencies are installed automatically inside the notebook.

### Option 2 — Local Jupyter

Install dependencies:

```bash
pip install pymatgen==2025.6.14 ase natsort
```

Then open `Distance_contraction.ipynb` in Jupyter and run all cells.

---

## Dependencies

- Python ≥ 3.9
- [pymatgen](https://pymatgen.org/) == 2025.6.14
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- numpy, scipy, pandas, matplotlib
- natsort

---

## Output Files

| File | Contents |
|---|---|
| `<ICSD>_data.csv` | Per-path distance-contraction metrics |
| `<ICSD>_deviation.csv` | Per-structure dimensionality + contraction summary |
| `<ICSD>_uniquepair.csv` | All pairs with symmetry-mapped descriptor values |
| `<ICSD>_uniquesymm.csv` | Symmetry-unique paths only |
| `neb_<N>_linear.cif` | Linear (unoptimised) NEB path geometry |
| `neb_<N>_predicted_*.cif` | Transverse-optimised pathway geometries |
| `image_0<N>.cif` | Individual NEB image structures |
| `*Dim_struct_path.cif` | Mobile-ion network at each dimensionality threshold |

---

## License

MIT License — free to use, modify, and distribute with attribution.
