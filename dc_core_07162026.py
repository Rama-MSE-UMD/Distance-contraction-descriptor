"""
dc_core.py
==========
Core library for the Distance-Contraction Ion-Migration Pathway Screener.

All data loading, geometry helper functions, and the main `run_pipeline`
live here. The companion notebook only holds the intro, the interactive
widgets, and the results display — everything else is imported from this
module.

Usage in the notebook
----------------------
    import dc_core as dc
    dc.download_base_files()
    dc.load_reference_data()
    df_out, low_energy_df, df1, dim_struct, dim_path, dim_stddev, st_name, structure = \\
        dc.run_pipeline(element_symbol="Na", element_oxi=1, st_name="230141",
                         max_path_length=7, nimages=5, create_folder="Yes",
                         compute_extra=False)
"""

import os
import json
import math
import warnings
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from pymatgen.io.cif import CifParser
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.core.periodic_table import Element, Species
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# Data download / loading
# ════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://raw.githubusercontent.com/Rama-MSE-UMD/Distance-contraction-descriptor/main/"

DATA_FILES = [
    "periodic_table.json",
    "Na_bond_stddev.csv",
    "Li_bond_stddev.csv",
    "K_bond_stddev.csv",
    "Mg_bond_stddev.csv",
    "Al_bond_stddev.csv",
    "Ca_bond_stddev.csv",
    # "Zn_bond_stddev.csv",
]
CIF_FILES = [
    "39248_ord_rama.cif",
    "1529809.cif",
    "29962_ord_rama.cif",
    "230141.cif",
    "18640.cif",
    "942733.cif",
]

Periodic_table_data = None
most_common_oxidation_states = None
bond_mean_stddev_na = None
bond_mean_stddev_li = None
bond_mean_stddev_k = None
bond_mean_stddev_mg = None
bond_mean_stddev_ca = None
bond_mean_stddev_al = None


def download_base_files():
    """periodic_table.json, the per-element bond_stddev CSVs, and all CIF
    files. Run once at the start of each session."""
    os.chdir("/content")
    for f in DATA_FILES:
        if os.path.exists(f"/content/{f}"):
            print(f"⏭ Already exists: {f}")
        else:
            subprocess.run(["wget", "-q", "-P", "/content", BASE_URL + f])
            print(f"✓ Downloaded: {f}")
    for f in CIF_FILES:
        if os.path.exists(f"/content/{f}"):
            print(f"⏭ Already exists: {f}")
        else:
            subprocess.run(["wget", "-q", "-P", "/content", BASE_URL + "cif_files/" + f])
            print(f"✓ Downloaded: {f}")
    print("\nDone.")


def load_reference_data():
    """Load periodic_table.json, the bond_stddev CSVs, and the common
    oxidation-state table into module-level globals. Call once after
    download_base_files()."""
    global Periodic_table_data, most_common_oxidation_states
    global bond_mean_stddev_na, bond_mean_stddev_li, bond_mean_stddev_k
    global bond_mean_stddev_mg, bond_mean_stddev_ca, bond_mean_stddev_al

    bond_mean_stddev_na = pd.read_csv("Na_bond_stddev.csv")
    bond_mean_stddev_li = pd.read_csv("Li_bond_stddev.csv")
    bond_mean_stddev_k  = pd.read_csv("K_bond_stddev.csv")
    bond_mean_stddev_mg = pd.read_csv("Mg_bond_stddev.csv")
    bond_mean_stddev_ca = pd.read_csv("Ca_bond_stddev.csv")
    bond_mean_stddev_al = pd.read_csv("Al_bond_stddev.csv")

    with open("periodic_table.json") as f:
        raw = json.load(f)
    Periodic_table_data = pd.DataFrame(raw).transpose()

    most_common_oxidation_states = {
        el.symbol: (el.common_oxidation_states[0] if el.common_oxidation_states else None)
        for el in Element
    }
    print("Reference databases loaded.")


# ════════════════════════════════════════════════════════════════════════════
# Lookup tables
# ════════════════════════════════════════════════════════════════════════════

OXI_STATE = {"Na": 1, "Li": 1, "K": 1, "Mg": 2, "Ca": 2, "Al": 3}

ION_PARAMS = {
    # dist_cutoff kept at 0.65 for all elements (notebook behaviour, intentional)
    "Na": {"cutoff_prefactor": 1.5, "ionic_radius": 1.00, "dist_cutoff": 0.65},
    "Li": {"cutoff_prefactor": 2.0, "ionic_radius": 0.60, "dist_cutoff": 0.65},
    "K":  {"cutoff_prefactor": 1.5, "ionic_radius": 1.38, "dist_cutoff": 0.65},
    "Mg": {"cutoff_prefactor": 2.0, "ionic_radius": 0.72, "dist_cutoff": 0.65},
    "Ca": {"cutoff_prefactor": 2.0, "ionic_radius": 1.00, "dist_cutoff": 0.65},
    "Al": {"cutoff_prefactor": 2.0, "ionic_radius": 0.54, "dist_cutoff": 0.65},
    "Zn": {"cutoff_prefactor": 2.0, "ionic_radius": 0.6,  "dist_cutoff": 0.65},
}


# ════════════════════════════════════════════════════════════════════════════
# 3.1  Structure utilities
# ════════════════════════════════════════════════════════════════════════════

def frac_to_cart(frac_coords, lattice):
    """Convert fractional coordinates to Cartesian coordinates (Å).

    Multiplies the fractional coordinate vector [x, y, z] by the 3×3 lattice
    matrix, giving the real-space position in Ångströms.
    """
    return np.dot(frac_coords, lattice.matrix)


def shift_atoms(structure, migrating_element):
    """Fold mobile-ion sites sitting at the unit-cell boundary back inside.

    Periodic structures occasionally store atoms at fractional coordinate
    ≈ 1.0 (numerically equivalent to 0.0 by periodicity).  This function
    maps those sites to ≈ 0.0 to avoid artefacts during linear path
    interpolation.
    """
    shifted = structure.copy()
    for site in shifted:
        if site.species.elements[0].symbol == migrating_element:
            if any(c > 0.99 for c in site.frac_coords):
                site.frac_coords = [
                    abs(c - 1.0) if c > 0.99 else c
                    for c in site.frac_coords
                ]
    return shifted


def replicate_structure(structure, replication):
    """Build a supercell by tiling the unit cell along each lattice direction.

    Parameters
    ----------
    structure   : pymatgen Structure
    replication : tuple of int, e.g. (3, 3, 3)

    Returns a new Structure whose lattice vectors are scaled by *replication*.
    Used internally by `check_connectivity` to evaluate 3-D percolation.
    """
    lattice   = structure.lattice
    positions = structure.cart_coords
    species   = structure.species

    new_positions, new_species = [], []
    for i in range(replication[0]):
        for j in range(replication[1]):
            for k in range(replication[2]):
                shift = (i * lattice.matrix[0]
                         + j * lattice.matrix[1]
                         + k * lattice.matrix[2])
                for pos, sp in zip(positions, species):
                    new_positions.append(pos + shift)
                    new_species.append(sp)

    new_lattice = Lattice(np.dot(np.diag(replication), lattice.matrix))
    return Structure(new_lattice, new_species, new_positions,
                     coords_are_cartesian=True)


def fallback_radius(element: str, default: float = 1.0) -> float:
    """Return the smallest Shannon ionic radius available for *element*.

    Used when the exact coordination-number entry is missing from the Shannon
    table (e.g. rare oxidation state or unusual CN).  Returns *default* (1.0 Å)
    if no entry at all is found.
    """
    data = Periodic_table_data["Shannon radii"].get(element, {})
    if not data:
        return default
    values = [
        v2["ionic_radius"]
        for v1 in data.values()
        for v2 in v1.values()
        if "ionic_radius" in v2
    ]
    return min(values) if values else default


# ════════════════════════════════════════════════════════════════════════════
# 3.2  Symmetry reduction of migration paths
# ════════════════════════════════════════════════════════════════════════════

def symmetry_paths(unique_pairs_list, struct, tolerance=0.5):
    """Reduce all candidate site pairs to symmetry-unique migration paths.

    Two paths are symmetry-equivalent if one can be mapped onto the other by
    a space-group symmetry operation (rotation, reflection, screw axis, or
    glide plane).  The descriptor needs to be computed only for the unique
    paths; the results are then propagated back to all equivalent paths via
    the returned *mapping* array.

    Plain language: if two hops look identical after rotating the crystal,
    they will have the same energy barrier — so we compute it once and reuse
    it.  The mapping tells us which unique path each original hop corresponds
    to, enabling reconstruction of the full network for percolation analysis.

    Parameters
    ----------
    unique_pairs_list : list of [PeriodicSite, PeriodicSite]
        All candidate start–end site pairs (before symmetry reduction).
    struct            : pymatgen Structure
        Parent structure used to obtain space-group symmetry operations.
    tolerance         : float
        Cartesian distance tolerance (Å) for comparing transformed pairs.
        Default 0.5 Å — do not change without updating post-processing scripts.

    Returns
    -------
    unique_symm_paths : list of [PeriodicSite, PeriodicSite]
        Symmetry-inequivalent paths (subset of unique_pairs_list).
    mapping           : list of int
        mapping[i] gives the index into unique_symm_paths that pair i maps to.
        Used to propagate descriptor values back to all original pairs.
    """
    try:
        sga      = SpacegroupAnalyzer(struct, symprec=0.01)
        symm_ops = sga.get_symmetry_operations()
    except Exception:
        sga      = SpacegroupAnalyzer(struct, symprec=0.001)
        symm_ops = sga.get_symmetry_operations()

    tolerance         = round(tolerance, 1)
    unique_symm_paths = []
    mapping           = [-1] * len(unique_pairs_list)

    for i, pair in enumerate(unique_pairs_list):
        is_unique = True
        for op in symm_ops:
            t_frac = [op.operate(pair[0].frac_coords),
                      op.operate(pair[1].frac_coords)]
            t_cart = [frac_to_cart(f, struct.lattice) for f in t_frac]
            for k, p in enumerate(unique_symm_paths):
                p_cart = [frac_to_cart(p[0].frac_coords, struct.lattice),
                          frac_to_cart(p[1].frac_coords, struct.lattice)]
                if np.allclose(t_cart, p_cart, atol=tolerance):
                    mapping[i] = k
                    is_unique   = False
                    break
            if not is_unique:
                break
        if is_unique:
            mapping[i] = len(unique_symm_paths)
            unique_symm_paths.append(pair)

    return unique_symm_paths, mapping


# ════════════════════════════════════════════════════════════════════════════
# 3.3  NEB path generation & percolation-network connectivity
# ════════════════════════════════════════════════════════════════════════════

def get_structures(struct, isite, esite, nimages=5, vac_mode=True):
    """Generate linearly interpolated NEB images between two mobile-ion sites.

    The mobile ion is placed at *isite* in the start structure and *esite* in
    the end structure; all framework ions remain fixed.  Linear interpolation
    produces *nimages* intermediate geometries (images).

    *vac_mode=True* (vacancy mechanism, default) removes the mobile ion from
    all other equivalent sites before interpolating, so only one ion hops
    at a time.

    Parameters
    ----------
    struct  : pymatgen Structure  (full host lattice)
    isite   : PeriodicSite        (start position of mobile ion)
    esite   : PeriodicSite        (end   position of mobile ion)
    nimages : int                 (intermediate images; total structures = nimages + 2)
    vac_mode: bool

    Returns
    -------
    list of Structure  (length = nimages + 2, including both end-points)
    """
    migrating_sites, other_sites = [], []
    for site in struct.sites:
        if site.specie != isite.specie:
            other_sites.append(site)
        elif vac_mode and isite.distance(site) > 1e-8 and esite.distance(site) > 1e-8:
            migrating_sites.append(site)

    start = Structure.from_sites([isite] + migrating_sites + other_sites)
    end   = Structure.from_sites([esite] + migrating_sites + other_sites)
    # pbc=False: interpolate along the direct Cartesian vector between sites,
    # not through the periodic boundary (which would choose the wrong image).
    return start.interpolate(end, nimages=nimages + 1, pbc=False)


def check_connectivity(structure, cutoff):
    """Determine the dimensionality of the mobile-ion percolation network.

    Builds a 3×3×3 supercell and constructs a graph where any two sites
    within *cutoff* Å are connected.  For each Cartesian axis the function
    tests whether the graph is connected from one face of the supercell to
    the opposite face.  The count of connected axes gives the network
    dimensionality.

    Plain language: checks whether the mobile-ion sites form a continuous
    pathway spanning the crystal — 3-D connectivity is the target for a
    good solid-state electrolyte.

    Returns
    -------
    int  (0 = isolated pockets, 1 = 1-D channels, 2 = 2-D layers, 3 = 3-D network)
    """
    rep       = replicate_structure(structure, (3, 3, 3))
    positions = rep.cart_coords
    lattice   = rep.lattice.matrix

    kd_tree   = KDTree(positions)
    neighbors = kd_tree.query_ball_point(positions, r=cutoff)

    def is_connected(start, end_set, visited):
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in end_set:
                return True
            stack.extend(neighbors[cur])
        return False

    connected_axes = 0
    for axis in range(3):
        start_set = set(np.where(rep.cart_coords[:, axis]
                                 < lattice[axis, axis] / 3)[0])
        end_set   = set(np.where(rep.cart_coords[:, axis]
                                 > 2 * lattice[axis, axis] / 3)[0])
        if any(is_connected(s, end_set, set()) for s in start_set):
            connected_axes += 1

    return connected_axes


# ════════════════════════════════════════════════════════════════════════════
# 3.4  Bond-distance statistics along a migration path
# ════════════════════════════════════════════════════════════════════════════

def path_bond_distances_cutoff(structure1, structure2, coords1, coords2,
                               element_symbol, bond_std, na_cat_dist,
                               min_distances_by_element,
                               cutoff=4.0):
    """Collect nearest-neighbour bond distances at both ends of a hop path.

    Returns
    -------
    dict  { (mobile_element, neighbour_element) : {min, max, avg}_distance }
    """
    neighbor_data = defaultdict(list)
    for coords, st in [(coords1, structure1), (coords2, structure2)]:
        st = st.copy()
        st.remove_species([element_symbol])
        for nbr in st.get_sites_in_sphere(coords, cutoff):
            neighbor_data[(element_symbol, nbr[0].specie.symbol)].append(round(nbr[1], 3))

    distance_stats = {}
    for pair, distances in neighbor_data.items():
        min_dist = min(distances)
        filtered = [d for d in distances if d and d <= min_dist + 0.7] 
        distance_stats[pair] = {
            "min_distance": min_dist,
            "max_distance": max(filtered),
            "avg_distance": float(np.mean(filtered)),
        }
    return distance_stats


def is_neighbor_in_sites(neigh, site_list, tol=1e-3):
    """Check whether *neigh* matches any site in *site_list*.

    Matches by species symbol and Cartesian distance < *tol* Å.
    Used to exclude the mobile ion's own endpoint sites from the
    neighbour lists when building the TS environment.
    """
    return any(
        neigh.specie.symbol == s.specie.symbol and neigh.distance(s) < tol
        for s in site_list
    )


# ════════════════════════════════════════════════════════════════════════════
# 3.5  Transverse displacement optimisation
# ════════════════════════════════════════════════════════════════════════════
#
# The linearly interpolated NEB path places the mobile ion on the straight
# line between its start and end sites.  In reality the ion deviates sideways
# to avoid neighbours.  The three functions below find and apply that
# transverse displacement to produce a more realistic pathway geometry.
#
# Step A — find_average_away_vector
#   Sweeps *n_samples* candidate directions in the plane perpendicular to
#   the migration vector.  For each direction the ion is displaced by *step*
#   Å and the worst-case distance contraction (ref_distance − actual_distance)
#   across all neighbours is recorded.  The direction that minimises this
#   worst-case contraction is the local "away vector" for that NEB image.
#   The average across all images gives a single representative direction.
#
# Step B — get_common_transverse_direction
#   Projects the away vector onto the plane perpendicular to the migration
#   vector (Gram–Schmidt orthogonalisation) so the correction has no
#   component along the hop direction.  Returns None if the away vector is
#   nearly collinear with the migration direction (dot product > 0.95).
#
# Step C — optimize_ion_position
#   Brute-force scan from bound[0] to bound[1] in *step* Å increments along
#   the transverse direction.  For each trial position the actual neighbours
#   are recomputed and the worst-case contraction is evaluated.  The
#   displacement that minimises worst-case contraction is selected.

def find_average_away_vector(structures, ion_index, interp_vector,
                              path_distances_avg, element_symbol, anion_symbol,
                              r_cut=5.0, step=1.0, n_samples=360,
                              include_anion=False):
    """Find the average transverse 'away' direction across all NEB images.

    Parameters
    ----------
    structures        : list of pymatgen Structure  (NEB images)
    ion_index         : int   (index of mobile ion in each structure)
    interp_vector     : ndarray (3,)  (migration direction, need not be unit)
    path_distances_avg: dict  { element_symbol : avg bond distance (Å) }
    element_symbol    : str   (mobile ion, e.g. 'Na')
    anion_symbol      : str   (primary anion, e.g. 'O')
    r_cut             : float (neighbour search radius, Å)
    step              : float (probe displacement magnitude, Å)
    n_samples         : int   (angular grid points in [0, 2π))
    include_anion     : bool  (True → include anion in away-vector search;
                               False → use only non-mobile cations)

    Returns
    -------
    total_away_vector   : ndarray (3,)  normalised average away direction
    ref_structure_index : int  (image index with the most neighbours)
    """
    interp_unit = interp_vector / np.linalg.norm(interp_vector)
    # Build an orthonormal basis {u, v} in the plane ⊥ interp_unit
    ref = np.array([1., 0., 0.]) if abs(interp_unit[0]) < 0.9           else np.array([0., 1., 0.])
    u = np.cross(interp_unit, ref);  u /= np.linalg.norm(u)
    v = np.cross(interp_unit, u);    v /= np.linalg.norm(v)

    total_away_vector   = np.zeros(3)
    num_contributions   = 0
    ref_structure_index = 0

    excluded = ({element_symbol} if not include_anion   ## mobile-ions are excluded when predicting pathway
                else {element_symbol, anion_symbol})

    for idx, struct in enumerate(structures):
        ion_site = struct.sites[ion_index]
        ion_pos  = ion_site.coords

        nbrs          = struct.get_neighbors(ion_site, r_cut)
        neighbor_info = [
            (n[0].coords, n[0].specie.symbol) for n in nbrs
            if n.specie.symbol not in excluded
               and n.specie.symbol in path_distances_avg
        ]
        if not neighbor_info:
            continue

        best_vec, best_min_dev = None, np.inf
        for theta in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
            cand      = np.cos(theta) * u + np.sin(theta) * v
            displaced = ion_pos + step * cand
            deviations = [
                path_distances_avg[sym] - np.linalg.norm(coords - displaced)
                for coords, sym in neighbor_info
            ]
            worst = np.max(deviations)
            if worst < best_min_dev:
                best_min_dev = worst
                best_vec     = cand

        if best_vec is not None:
            total_away_vector += best_vec
            num_contributions  += 1

        ref_nbrs = structures[ref_structure_index].get_neighbors(ion_site, r_cut)
        if len(nbrs) > len(ref_nbrs):
            ref_structure_index = idx

    if num_contributions == 0 or np.linalg.norm(total_away_vector) < 1e-6:
        raise ValueError("No valid away vectors found across structures.")
    total_away_vector /= np.linalg.norm(total_away_vector)
    return total_away_vector, ref_structure_index


def get_common_transverse_direction(structures, ion_index, interp_vector,
                                    path_distances_avg, element_symbol,
                                    anion_symbol, include_anion=False):
    """Return the transverse unit vector perpendicular to *interp_vector*.

    Calls `find_average_away_vector` and then removes any component along
    the migration direction (Gram–Schmidt).  Returns None if the away vector
    is nearly collinear with the migration direction (|dot| > 0.95), meaning
    no meaningful transverse correction exists for this path.
    """
    away, _ = find_average_away_vector(
        structures, ion_index, interp_vector,
        path_distances_avg, element_symbol, anion_symbol,
        include_anion=include_anion
    )
    dot = np.dot(away, interp_vector)
    if abs(dot) > 0.95:
        return None   # collinear — skip transverse correction
    transverse = away - dot * interp_vector
    return transverse / np.linalg.norm(transverse)


def optimize_ion_position(struct, ion_index, transverse_vec,
                           path_distances_avg, element_symbol, anion_symbol,
                           bound, step=0.05, r_cut=5.0,
                           include_anion=False, plot=False):
    """Brute-force scan along *transverse_vec* to minimise steric contraction.

    For each trial displacement *x* in [bound[0], bound[1]]:
    1. Move the mobile ion to  ion_pos + x · transverse_vec.
    2. Recompute actual neighbours at that position.
    3. Evaluate max(ref_distance − actual_distance) — the worst contraction.
    The displacement minimising worst-case contraction is selected and the
    corresponding fractional coordinates are returned.

    Parameters
    ----------
    struct           : pymatgen Structure  (single NEB image)
    ion_index        : int
    transverse_vec   : ndarray (3,)  (unit transverse direction)
    path_distances_avg: dict  { element : avg bond distance }
    element_symbol   : str
    anion_symbol     : str
    bound            : tuple (float, float)  scan range in Å
    step             : float  scan step size (Å)
    r_cut            : float  neighbour search radius (Å)
    include_anion    : bool
    plot             : bool   if True, plot the clearance vs. displacement curve

    Returns
    -------
    optimized_frac_coords : ndarray (3,)
    """
    ion_site = struct.sites[ion_index]
    ion_cart = ion_site.coords
    t_hat    = transverse_vec / np.linalg.norm(transverse_vec)
    excluded = ({element_symbol} if not include_anion
                else {element_symbol, anion_symbol})

    def worst_contraction(x):
        new_cart = ion_cart + x * t_hat
        new_frac = struct.lattice.get_fractional_coords(new_cart)
        new_site = PeriodicSite(ion_site.species, new_frac, struct.lattice)
        nbrs     = [n for n in struct.get_neighbors(new_site, r_cut)
                    if n.specie.symbol not in excluded
                    and n.specie.symbol in path_distances_avg]
        if not nbrs:
            return 0.0
        nbr_coords = np.array([n[0].coords for n in nbrs])
        dists      = np.linalg.norm(nbr_coords - new_cart, axis=1)
        devs       = np.array([
            path_distances_avg[n[0].specie.symbol] - d
            for d, n in zip(dists, nbrs)
        ])
        return float(np.max(devs))

    xs     = np.arange(bound[0], bound[1] + step, step)
    scores = [worst_contraction(x) for x in xs]
    best_x = xs[np.argmin(scores)]

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(xs, scores, marker="o")
        plt.axvline(best_x, color="red", linestyle="--",
                    label=f"Best x = {best_x:.2f} Å")
        plt.xlabel("Transverse displacement (Å)")
        plt.ylabel("Worst-case distance contraction (Å)")
        plt.title("Transverse position optimisation")
        plt.legend(); plt.tight_layout(); plt.show()

    opt_cart = ion_cart + best_x * t_hat
    return struct.lattice.get_fractional_coords(opt_cart)


# ════════════════════════════════════════════════════════════════════════════
# Main analysis pipeline
# ════════════════════════════════════════════════════════════════════════════

_BOND_STDDEV = None


def _bond_stddev_table():
    global _BOND_STDDEV
    if _BOND_STDDEV is None:
        _BOND_STDDEV = {
            "Na": bond_mean_stddev_na,  "Li": bond_mean_stddev_li,
            "K":  bond_mean_stddev_k,   "Mg": bond_mean_stddev_mg,
            "Ca": bond_mean_stddev_ca,  "Al": bond_mean_stddev_al,
            # "Zn":  bond_mean_stddev_zn,
        }
    return _BOND_STDDEV


def run_pipeline(element_symbol, element_oxi, st_name,
                 max_path_length, nimages, create_folder, compute_extra=False):
    """Execute the full distance-contraction screening pipeline.

    Parameters
    ----------
    element_symbol : str   mobile ion (e.g. 'Na')
    element_oxi    : int   formal oxidation state
    st_name        : str   ICSD code (numeric string)
    max_path_length: float maximum hop distance (Å)
    nimages        : int   number of NEB images (excluding end-points)
    create_folder  : str   'Yes' or 'No'
    compute_extra  : bool  If True, also compute the 'struct' and 'stddev'
                           reference categories in addition to 'path'.
    """
    params   = ION_PARAMS[element_symbol]
    bond_std = _bond_stddev_table().get(element_symbol, bond_mean_stddev_na)
    distance_cutoff = params["cutoff_prefactor"] * params["ionic_radius"]

    # Sort key: mobile ion first, then by electronegativity
    def custom_sort_key(site):
        return (0 if element_symbol in str(site.specie) else 1,
                site.specie.X)

    # ── Load structure ────────────────────────────────────────────────────
    structure = None
    for fname in [f"{st_name}_ord_rama.cif", f"{st_name}.cif"]:
        try:
            parser    = CifParser(fname)
            structure = parser.get_structures(primitive=False)[0]
            print(f"Loaded: {fname}")
            break
        except Exception:
            continue
    if structure is None:
        raise FileNotFoundError(
            f"No CIF found for ICSD {st_name}. "
            f"Expected '{st_name}_ord_rama.cif' or '{st_name}.cif'."
        )

    if create_folder == "Yes":
        folder = f"{st_name}_neb"
        os.makedirs(folder, exist_ok=True)
        os.chdir(folder)

    # ── Build minimum supercell (target: ≥ 8 Å per axis) ─────────────────
    con_range = [round(i, 3) for i in np.arange(8.0, 100, 0.001)]
    def _sc_reps(length):
        return 1 if length in con_range else math.ceil(10 / math.ceil(length))

    struct = structure.copy()
    struct.make_supercell([
        _sc_reps(struct.lattice.a),
        _sc_reps(struct.lattice.b),
        _sc_reps(struct.lattice.c),
    ])

    # Framework reference (mobile ion removed, no oxidation states)
    baseref = struct.copy()
    baseref.remove_oxidation_states()
    baseref.remove_species([element_symbol])
    print("Framework composition:", baseref.composition)

    # ── Shift boundary atoms, sort, and keep reference copies ─────────────
    try:
        _ = struct.sites[0].specie.oxi_state
        struct.remove_oxidation_states()
    except Exception:
        pass
    struct     = shift_atoms(struct, element_symbol)
    struct     = struct.sort(key=custom_sort_key)
    struct_ref = struct.copy()

    # ── Enumerate all candidate hop pairs ─────────────────────────────────
    na_sites = [s for s in struct_ref
                if element_symbol in s.species_string]
    initial_final_pairs = []
    for na_site in na_sites:
        for nbr in struct_ref.get_sites_in_sphere(
                na_site.coords, max_path_length,
                include_index=True, include_image=True):
            if element_symbol in nbr.species_string and nbr != na_site:
                initial_final_pairs.append([na_site, nbr])

    unique_pairs      = {frozenset(sorted(p)) for p in initial_final_pairs}
    unique_pairs_list = [list(p) for p in unique_pairs]
    print(f"Total candidate pairs: {len(unique_pairs_list)}")

    # Pre-generate linear path site lists for all pairs (used in connectivity)
    initial_final_path_sites = []
    for p in unique_pairs_list:
        s0 = struct_ref.copy()
        tmp = get_structures(struct=s0, isite=p[0], esite=p[1], nimages=5)
        initial_final_path_sites.append([t[0] for t in tmp])

    # ── Symmetry reduction ────────────────────────────────────────────────
    # The mapping array records which unique path each original pair belongs to.
    # After computing the descriptor for each unique path, values are propagated
    # back to all equivalent pairs so the full percolation network can be built.
    TOLERANCE = 0.5   # Å — do not change without updating post-processing
    unique_symm_paths, mapping = symmetry_paths(
        unique_pairs_list, struct_ref, TOLERANCE)
    print(f"Symmetry-unique paths: {len(unique_symm_paths)} "
          f"(reduced from {len(unique_pairs_list)})")

    # Write full linear NEB connectivity CIF (all pairs)
    sites_interp = []
    for p in unique_pairs_list:
        s0   = struct_ref.copy()
        strs = get_structures(struct=s0, isite=p[0], esite=p[1], nimages=5)
        sites_interp += [strs[0][0], strs[-1][0]]
        sites_interp += [PeriodicSite("H", s[0].frac_coords, s.lattice)
                         for s in strs[1:-1]]
    sites_interp.extend(strs[0].sites[1:])
    if create_folder == "Yes":
        Structure.from_sites(sites_interp).to(
            filename=f"{st_name}_linear_neb_paths.cif")

    # ── Oxidation-state assignment ─────────────────────────────────────────
    struct_oxi = structure.copy()
    struct_oxi = struct_oxi.add_oxidation_state_by_guess()
    if not len({s.oxi_state for s in struct_oxi.species}) > 1:
        struct_oxi.add_oxidation_state_by_element(most_common_oxidation_states)
    oxidation_dict = {
        site.specie.element.symbol: site.specie.oxi_state
        for site in struct_oxi
    }
    print("Oxidation states:", oxidation_dict)

    struct_orig = struct_ref.copy()
    struct_orig.add_oxidation_state_by_element(oxidation_dict)

    # Select primary anion (priority: O > S > Cl > Br > I > F)
    anions = {el: ox for el, ox in oxidation_dict.items() if ox < 0}
    anion_symbol = next(
        (el for el in ["O", "S", "Cl", "Br", "I", "F"] if el in anions),
        next(iter(anions), None)
    )
    anion_oxi = round(anions[anion_symbol]) if anion_symbol else None
    print(f"Primary anion: {anion_symbol} ({anion_oxi:+d})")

    # ── CrystalNN coordination-number table ───────────────────────────────
    crystal_nn = CrystalNN()   # match .py: bulk min/max/avg distances use CrystalNN() defaults (cation_anion=False)
    CNN_wide   = CrystalNN(cation_anion=True, search_cutoff=12,
                            distance_cutoffs=(0, 7.0))

    sites_str, sites_cn = [], []
    for i, site in enumerate(struct_orig.sites):
        sites_str.append(site.species_string)
        try:
            sites_cn.append(CNN_wide.get_cn(struct_orig, i))
        except ValueError:
            sites_cn.append(float("nan"))

    sites_cn_df   = pd.DataFrame({"sites": sites_str, "sites_cn": sites_cn})
    sites_cn_uniq = (
        sites_cn_df.groupby(list(sites_cn_df.columns))
        .apply(lambda x: list(x.index))
        .reset_index(name="indices")
    )
    sites_cn_uniq["Oxi_state"] = [int(Species(s).oxi_state)
                                   for s in sites_cn_uniq["sites"]]
    sites_cn_uniq["Element"]   = [Species(s).element.symbol
                                   for s in sites_cn_uniq["sites"]]

    _ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",
              7:"VII",8:"VIII",9:"IX",10:"X",11:"XI",12:"XII"}

    def _safe_ionic_radius(row):
        try:
            roman = _ROMAN.get(int(row["sites_cn"]), "VI")
            return list(
                Periodic_table_data["Shannon radii"]
                [row["Element"]][str(row["Oxi_state"])][roman].values()
            )[0]["ionic_radius"]
        except Exception:
            return float("nan")

    sites_cn_uniq["ionic_radius"] = sites_cn_uniq.apply(
        _safe_ionic_radius, axis=1)
    sites_cn_uniq["ionic_radius"] = (
        sites_cn_uniq.groupby("Element")["ionic_radius"].transform(
            lambda x: x.fillna(
                x.dropna().iloc[0] if not x.dropna().empty
                else fallback_radius(x.name)
            )
        )
    )
    weighted_ionic_radius = (
        sites_cn_uniq
        .assign(weight=sites_cn_uniq["indices"].apply(len))
        .groupby("Element")
        .apply(lambda g: np.average(g["ionic_radius"], weights=g["weight"]))
    )

    # ── Per-element bond-distance statistics in the bulk structure ────────
    cation_groups = defaultdict(list)
    for site in struct_orig.sites:
        if site.specie.oxi_state > 0:
            cation_groups[site.specie.symbol].append(site)

    max_distances_by_element = {}
    min_distances_by_element = {}
    avg_distances_by_element = {}
    for element, sites in cation_groups.items():
        mn, mx, av = [], [], []
        for site in sites:
            idx  = struct_orig.sites.index(site)
            info = crystal_nn.get_nn_info(struct_orig, idx)
            ds   = [site.distance(nn["site"]) for nn in info]
            if ds:
                mn.append(min(ds)); mx.append(max(ds))
                av.append(sum(ds) / len(ds))
        max_distances_by_element[element] = max(mx) if mx else 4.0
        min_distances_by_element[element] = min(mn) if mn else 1.5
        avg_distances_by_element[element] = min(av) if av else 2.5

    # Minimum mobile-ion – cation distances (used as neighbour cutoffs)
    struct_nooxi = struct_orig.copy()
    struct_nooxi.remove_oxidation_states()
    dm = pd.DataFrame(
        struct_nooxi.distance_matrix,
        columns=[s.species_string for s in struct_nooxi.sites],
        index  =[s.species_string for s in struct_nooxi.sites],
    )
    na_cat_dist = {}
    for el in cation_groups:
        if el == element_symbol:
            val = dm.loc[element_symbol, el]
            na_cat_dist[el] = (
                val.apply(lambda x: x.nsmallest(2).iloc[-1]).min()
                if isinstance(val, pd.DataFrame) else sorted(val)[1]
            )
        else:
            sl = dm[element_symbol].loc[el]
            na_cat_dist[el] = (sl.min().min()
                               if isinstance(sl, pd.DataFrame) else sl.min())
    neigh_cutoff = max(na_cat_dist.values())
    print("Mobile-ion → cation min distances:", na_cat_dist)

    # ════════════════════════════════════════════════════════════════════════
    # Per-path analysis loop (symmetry-unique paths only)
    # ════════════════════════════════════════════════════════════════════════
    neb_number, oxi_dict_list = [], []
    max_reldist_struct, max_reldist_path, max_reldist_stddev = [], [], []
    cat_reldist_struct, cat_reldist_path, cat_reldist_stddev = [], [], []
    ani_reldist_struct, ani_reldist_path, ani_reldist_stddev = [], [], []
    energy_cat_struct,  energy_cat_path,  energy_cat_stddev  = [], [], []

    def _new_dict():
        return {el: [] for el in list(na_cat_dist) + [anion_symbol]}

    optimized_path_sites = []
    for p_index, p in enumerate(unique_symm_paths):
        print(f"\n{'─'*60}")
        print(f"Path {p_index + 1} / {len(unique_symm_paths)}")

        struct_orig = struct_ref.copy()
        structures  = get_structures(struct=struct_orig,
                                     isite=p[0], esite=p[1],
                                     nimages=nimages)
        neb_structures = structures.copy()

        # Linear (unoptimised) path CIF
        lin_sites = [s[0] for s in structures]
        if create_folder == "Yes":
            lin_full = lin_sites.copy()
            lin_full.extend(structures[0].sites[1:])
            Structure.from_sites(lin_full).to(
                filename=f"neb_{p_index+1}_linear.cif")

        # Reference bond distances along this path
        path_dist_raw = path_bond_distances_cutoff(
            structures[0], structures[-1],
            structures[0][0].coords, structures[-1][0].coords,
            element_symbol, bond_std, na_cat_dist, min_distances_by_element)
        path_distances_min = {pr[1]: st["min_distance"]
                              for pr, st in path_dist_raw.items()}
        path_distances_avg = {pr[1]: st["avg_distance"]
                              for pr, st in path_dist_raw.items()}
        print(f"  Path avg distances: {path_distances_avg}")

        struct_orig = struct_ref.copy()
        struct_orig.add_oxidation_state_by_element(oxidation_dict)

        # Unwrap fractional coordinates (minimum-image convention)
        fc_arr = np.array([s.sites[0].frac_coords for s in structures])
        fc_unw = [fc_arr[0]]
        for fc in fc_arr[1:]:
            d = fc - fc_unw[-1]; d -= np.round(d)
            fc_unw.append(fc_unw[-1] + d)
        cart_arr   = np.array(fc_unw) @ structures[0].lattice.matrix
        interp_vec = cart_arr[-1] - cart_arr[0]
        interp_vec /= np.linalg.norm(interp_vec)

        # ── Four transverse configurations ────────────────────────────────
        # (cation-only vs. all neighbours) × (positive vs. negative direction)
        results = {}
        for mode in ["cation", "all"]:
            include_anion = (mode == "all")
            try:
                t_vec = get_common_transverse_direction(
                    structures, 0, interp_vec,
                    path_distances_avg, element_symbol, anion_symbol,
                    include_anion=include_anion)
            except ValueError as e:
                print(f"  [{mode}] Skipping: {e}"); t_vec = None

            for direction in ["pos", "neg"]:
                if t_vec is None:
                    new_structs = structures[:]
                else:
                    max_bound = 2.5 if direction == "pos" else -2.5
                    n         = len(structures)
                    disps     = (np.linspace(0, max_bound, (n+1)//2).tolist()
                                 + np.linspace(max_bound, 0, n//2+1).tolist()[1:])
                    new_structs = []
                    for si, st in enumerate(structures):
                        bound = (0, disps[si]) if direction == "pos"                                 else (disps[si], 0)
                        st = st.copy()
                        opt = optimize_ion_position(
                            st, 0, t_vec,
                            path_distances_avg, element_symbol, anion_symbol,
                            bound=bound, include_anion=include_anion)
                        st.replace(0, element_symbol, opt,
                                   properties=st.sites[0].properties)
                        new_structs.append(st)

                mod_sites = [s[0] for s in new_structs]
                key = f"{mode}_{direction}"

                if create_folder == "Yes":
                    full = mod_sites.copy()
                    full.extend(new_structs[0].sites[1:])
                    Structure.from_sites(full).to(
                        filename=f"neb_{p_index+1}_predicted_{mode}_{direction}.cif")

                # Find TS image: scan all images for max contraction
                max_across_images = _new_dict()
                for ts_i, mod_site in enumerate(mod_sites):
                    na_neigh = defaultdict(list)
                    for sp in struct_orig.composition.as_dict():
                        na_neigh[sp] = []
                    for nbr in struct_orig.get_sites_in_sphere(
                            mod_site.coords, neigh_cutoff):
                        dist   = mod_site.distance(nbr)
                        sp_sym = nbr.specie.element.symbol
                        na_neigh[nbr.species_string].append(dist)

                    for speci, dists in na_neigh.items():
                        sp1   = Species(speci)
                        el    = sp1.element.symbol
                        if sp1.oxi_state > 0:
                            try:
                                ref_d = path_distances_avg[el]
                            except KeyError:
                                ref_d = na_cat_dist[el]
                        else:
                            try:
                                ref_d = path_distances_avg[el]
                            except KeyError:
                                ref_d = min_distances_by_element[element_symbol]
                        if el not in max_across_images:
                            max_across_images[el] = []
                        if dists:
                            max_across_images[el].append(max(ref_d - d for d in dists))
                        else:
                            max_across_images[el].append(0)

                max_across_images.pop(element_symbol, None)
                max_val, ts_idx = -np.inf, None
                for sp, deviations in max_across_images.items():
                    for ti, v in enumerate(deviations):
                        if v > max_val:
                            max_val = v; ts_idx = ti

                results[key] = {
                    "Max-contraction value": max_val,
                    "Max-contraction index": ts_idx,
                    "working_sites":         mod_sites,
                }

        # ── Select best configuration ─────────────────────────────────────
        best_key  = min(results, key=lambda k: results[k]["Max-contraction value"])
        best      = results[best_key]
        ts_idx    = best["Max-contraction index"]
        mod_sites = best["working_sites"]
        print(f"  Best config: {best_key}  |  "
              f"TS contraction = {best['Max-contraction value']:.3f} Å")

        # ── Full TS analysis under three reference schemes ─────────────────
        def _analyse_ts(reference_type, max_rd, sum_rd, avg_rd):
            """Compute distance contraction at the TS under *reference_type*.

            reference_type options:
                'struct'  – reference = minimum cation–anion distance in bulk
                'path'    – reference = average bond distance along the hop path
                'stddev'  – reference = statistical mean from bond_stddev tables
            """
            na_neigh = defaultdict(list)
            for sp in struct_orig.composition.as_dict():
                na_neigh[sp] = []
            for nbr in struct_orig.get_sites_in_sphere(
                    mod_sites[ts_idx].coords,
                    max(na_cat_dist.values()) + 0.5):
                dist   = mod_sites[ts_idx].distance(nbr)
                sp_sym = nbr.specie.element.symbol
                na_neigh[nbr.species_string].append(dist)

            for speci, dists in na_neigh.items():
                sp1 = Species(speci)
                el  = sp1.element.symbol

                if reference_type == "struct":
                    ref_d = (na_cat_dist[el] if sp1.oxi_state > 0
                             else min_distances_by_element[element_symbol])
                elif reference_type == "path":
                    ref_d = (path_distances_avg.get(el,
                             na_cat_dist.get(el, 2.5))
                             if sp1.oxi_state > 0
                             else path_distances_avg.get(
                                 el, min_distances_by_element[element_symbol]))
                else:  # stddev (global reference)
                    match = bond_std.loc[bond_std["Element"] == el, "Mean_Distance"]
                    ref_d = match.iloc[0] if not match.empty else na_cat_dist[el]

                contractions = [ref_d - d for d in dists]
                abs_c        = [abs(c) for c in contractions]

                for dd, vals, agg in [
                    (max_rd, contractions, max),
                    (sum_rd, abs_c,        sum),
                    (avg_rd, abs_c,        lambda x: sum(x)/len(x)),
                ]:
                    if el not in dd: dd[el] = []
                    dd[el].append(agg(vals) if vals else None)

        mr_s, sr_s, ar_s = _new_dict(), _new_dict(), _new_dict()
        mr_p, sr_p, ar_p = _new_dict(), _new_dict(), _new_dict()
        mr_d, sr_d, ar_d = _new_dict(), _new_dict(), _new_dict()
        _analyse_ts("path",   mr_p, sr_p, ar_p)
        if compute_extra:
            _analyse_ts("struct", mr_s, sr_s, ar_s)
            _analyse_ts("stddev", mr_d, sr_d, ar_d)

        def _last_nonmobile(d):
            vals = [v[-1] for k, v in d.items()
                    if k != element_symbol and v and v[-1] is not None]
            return max(vals) if vals else 0.0

        def _last_cation(d):
            vals = [v[-1] for k, v in d.items()
                    if k in cation_groups and k != element_symbol
                    and v and v[-1] is not None]
            return max(vals) if vals else 0.0

        def _last_anion(d):
            vals = [v[-1] for k, v in d.items()
                    if k == anion_symbol and v and v[-1] is not None]
            return max(vals) if vals else 0.0

        dist_cut = params["dist_cutoff"]
        _schemes = [
            (mr_p, cat_reldist_path,   ani_reldist_path,
             energy_cat_path,   max_reldist_path,   "path"),
        ]
        if compute_extra:
            _schemes += [
                (mr_s, cat_reldist_struct, ani_reldist_struct,
                 energy_cat_struct, max_reldist_struct, "struct"),
                (mr_d, cat_reldist_stddev,  ani_reldist_stddev,
                 energy_cat_stddev, max_reldist_stddev, "stddev"),
            ]
        for mr, cat_l, ani_l, cat_cat, rd_list, ref_label in _schemes:
            v = _last_nonmobile(mr)
            rd_list.append(v)
            cat_l.append(_last_cation(mr))
            ani_l.append(_last_anion(mr))
            cat_cat.append("High" if v > dist_cut else "Low")
            print(f"  [{ref_label}]  max contraction = {v:.3f} Å  "
                  f"→ {cat_cat[-1]} barrier")

        # ── Write individual NEB image CIFs ───────────────────────────────
        if create_folder == "Yes":
            img_dir = f"neb_{p_index + 1}"
            os.makedirs(img_dir, exist_ok=True)
            os.chdir(img_dir)
            merged_sites = []
            for img_i, neb_st in enumerate(neb_structures):
                neb_st.replace(
                    0, element_symbol,
                    best["working_sites"][img_i].frac_coords,
                    properties=struct_ref.sites[0].properties,
                    coords_are_cartesian=False)
                neb_st.to(f"image_0{img_i}.cif", "cif")
                merged_sites.append(neb_st.sites[0])
            merged_sites.extend(neb_structures[0].sites[1:])
            Structure.from_sites(merged_sites).to(f"{img_dir}_merged.cif", "cif")
            os.chdir("../")

        neb_number.append(str(p_index + 1))
        oxi_dict_list.append(oxidation_dict)
        optimized_path_sites.append(best["working_sites"])
        print(50 * "─")

    # ════════════════════════════════════════════════════════════════════════
    # Assemble results DataFrame
    # ════════════════════════════════════════════════════════════════════════
    ref_len   = len(neb_number)
    data_dict = {
        "NEB":                         neb_number,
        "Oxi":                         oxi_dict_list,
        "Category_struct":             energy_cat_struct,
        "Max_contraction_struct":      max_reldist_struct,
        "Category_path":               energy_cat_path,
        "Max_contraction_path":        max_reldist_path,
        "Category_stddev":             energy_cat_stddev,
        "Max_contraction_stddev":      max_reldist_stddev,
        "Max_contraction_struct-cation": cat_reldist_struct,
        "Max_contraction_struct-anion":  ani_reldist_struct,
        "Max_contraction_path-cation":   cat_reldist_path,
        "Max_contraction_path-anion":    ani_reldist_path,
        "Max_contraction_stddev-cation": cat_reldist_stddev,
        "Max_contraction_stddev-anion":  ani_reldist_stddev,
    }

    print("\nColumn-length audit:")
    for k, v in data_dict.items():
        status = "✓" if len(v) == ref_len else f"✗ ({len(v)} ≠ {ref_len})"
        print(f"  {k:45s} {status}")

    filtered = {k: v for k, v in data_dict.items() if len(v) == ref_len}
    df1      = pd.DataFrame(filtered)

    # Propagate unique-path values back to all pairs via symmetry mapping
    mapping_proper = [str(i + 1) for i in mapping]
    low_energy_df  = (
        pd.DataFrame(mapping_proper, columns=["NEB"])
        .merge(df1, on="NEB", how="left")
    )
    low_energy_df["NEB1"] = low_energy_df.index + 1
    low_energy_df.to_csv(f"{st_name}_data.csv", index=False)

    # ── Percolation-network dimensionality analysis ────────────────────────
    # Paths are sorted by contraction (lowest first = most open).
    # They are added one by one to the mobile-ion site set.
    # After each addition, connectivity is checked.
    # The contraction value at which 3-D connectivity is first achieved
    # is the key screening threshold.
    def _dim_analysis(sort_col, contraction_col, label):
        sorted_df  = low_energy_df.sort_values(by=[sort_col]).reset_index(drop=True)
        sites_acc  = []
        rel_dists  = {}
        for neb_idx in sorted_df["NEB1"]:
            existing = {tuple(np.round(s.coords, 6)) for s in sites_acc}
            for site in initial_final_path_sites[int(neb_idx) - 1]:
                key = tuple(np.round(site.coords, 6))
                if key not in existing:
                    sites_acc.append(site); existing.add(key)
            rel_struct = Structure.from_sites(sites_acc)
            dim        = check_connectivity(rel_struct, distance_cutoff)
            dev        = sorted_df.loc[
                sorted_df["NEB1"] == neb_idx, contraction_col].values[0]
            if not label in ["struct","stddev"]:
                print(f"  [{label}] path {neb_idx:>2d}  dim = {dim}  "
                  f"contraction = {dev:.3f} Å")
            if dim not in rel_dists:
                rel_dists[dim] = dev
                if dim == 1 and create_folder == "Yes":
                    Structure.from_sites(
                        baseref.sites + rel_struct.sites
                    ).to(f"1Dim_struct_{label}.cif", "cif")
                elif dim == 2 and create_folder == "Yes":
                    Structure.from_sites(
                        baseref.sites + rel_struct.sites
                    ).to(f"2Dim_struct_{label}.cif", "cif")
                elif dim == 3:
                    if create_folder == "Yes":
                        Structure.from_sites(
                            baseref.sites + rel_struct.sites
                        ).to(f"3Dim_struct_{label}.cif", "cif")
                    break
        return rel_dists

    print("\n── Dimensionality analysis ──")
    dim_path = _dim_analysis("Max_contraction_path", "Max_contraction_path", "path")
    if compute_extra:
        dim_struct = _dim_analysis("Max_contraction_struct",
                                   "Max_contraction_struct", "struct")
        dim_stddev = _dim_analysis("Max_contraction_stddev",
                                   "Max_contraction_stddev", "stddev")
    else:
        dim_struct, dim_stddev = {}, {}

    df_out = pd.DataFrame({
        "Struct":                     [st_name],
        "Distance contraction-struct": [dim_struct],
        "Distance contraction-path":   [dim_path],
        "Distance contraction-stddev": [dim_stddev],
        "Composition":                 [structure.composition],
    })

    if create_folder == "Yes":
        df_out.to_csv(f"{st_name}_dev.csv", index=False)
        low_energy_df.to_csv(f"{st_name}_uniquepair.csv", index=False)
        df1.to_csv(f"{st_name}_uniquesymm.csv", index=False)
        os.chdir("..")
    else:
        df_out.to_csv(f"{st_name}_deviation.csv", index=False)

    print("\nDone.")
    return df_out, low_energy_df, df1, dim_struct, dim_path, dim_stddev, st_name, structure


# ════════════════════════════════════════════════════════════════════════════
# Post-processing helper
# ════════════════════════════════════════════════════════════════════════════

def find_unique_neb_paths(df, value_col="Max_contraction_path", tolerance=0.02):
    """Group symmetry-unique NEB paths (neb_1, neb_2, ...) into clusters whose
    *value_col* values agree within *tolerance* Å, and pick one representative
    folder per cluster.

    Parameters
    ----------
    df         : DataFrame with an "NEB" column (folder index) and *value_col*
                 — typically df1, the symmetry-unique-path table, since its
                 NEB numbering matches the neb_N folder names directly.
    value_col  : column holding the contraction value to compare on.
    tolerance  : Å, max allowed distance from a cluster's anchor value for a
                 path to be considered "the same path".

    Returns
    -------
    unique_folders : list of str   representative folder names, e.g. ["neb_1", "neb_4", ...]
    cluster_map    : dict          {NEB (str) : representative NEB (str)}
    """
    sdf = df[["NEB", value_col]].dropna().copy()
    sdf[value_col] = sdf[value_col].astype(float)
    sdf = sdf.sort_values(value_col).reset_index(drop=True)

    cluster_map    = {}
    unique_folders = []
    cluster_anchor_val = None
    cluster_anchor_neb = None

    for _, row in sdf.iterrows():
        neb, val = str(row["NEB"]), row[value_col]
        if cluster_anchor_val is None or abs(val - cluster_anchor_val) > tolerance:
            cluster_anchor_val = val
            cluster_anchor_neb = neb
            unique_folders.append(f"neb_{neb}")
        cluster_map[neb] = cluster_anchor_neb

    return unique_folders, cluster_map
