"""
Generalized Phonon DOS Calculator (Phonopy API — no qpoints.yaml needed)
=========================================================================
Supports any material. Reads force constants directly from Phonopy input files.

Input modes (mutually exclusive):
    --params   phonopy_params.yaml or phonopy_params.yaml.xz   [recommended]
    --forces   FORCE_SETS file  +  --poscar POSCAR  (+ optional --born BORN)

Q-point sampling modes (mutually exclusive):
    --mesh  NX NY NZ        Standard uniform Monkhorst-Pack mesh
    --adaptive              Uniform coarse mesh + dense logarithmic shell near Γ
                            (fixes flat fractional DOS at low frequencies)

Core options:
    --sigma       Gaussian smearing width [THz]          (default: 0.1)
    --nfreq       Frequency grid points                  (default: 1000)
    --fmin        Min frequency [THz]                    (default: auto)
    --fmax        Max frequency [THz]                    (default: auto)
    --smearing    gaussian | heaviside                   (default: gaussian)

Fractional acoustic DOS:
    --fdos              Enable fractional DOS computation
    --acoustic TS TF L  1-based branch indices           (default: 1 2 3)
    --fmin-fdos         Lower frequency bound [THz]      (default: 0.0)
    --fmax-fdos         Upper frequency bound [THz]      (default: 1.5)
    --output-fdos       Output file                      (default: FDOS_acoustic.txt)

Selective branch plotting:
    --branches N [N ...]   1-based branch numbers to plot individually

Adaptive mesh tuning:
    --coarse-mesh  NX NY NZ   Coarse uniform background mesh  (default: 11 11 11)
    --gamma-radius            Radius of dense Γ shell in r.l.u (default: 0.15)
    --gamma-points            Number of extra q-points in shell (default: 2000)
    --gamma-shells            Number of logarithmic shells      (default: 30)

Output:
    --output      Full DOS output file                   (default: DOS_output.txt)
    --plot        Show plots
    --pdos        Show all partial DOS on DOS plot
    --save-qpts   Save the adaptive q-point set to a file for inspection

Examples:
    # From phonopy_params.yaml.xz, uniform 21x21x21 mesh
    python phonon_dos.py --params phonopy_params.yaml.xz --mesh 21 21 21 --plot

    # Adaptive mesh for flat fractional DOS, branches 1=TS 2=TF 3=L
    python phonon_dos.py --params phonopy_params.yaml.xz --adaptive \\
        --fdos --acoustic 1 2 3 --fmax-fdos 1.5 --plot

    # From FORCE_SETS, adaptive, plot branches 1 and 10
    python phonon_dos.py --forces FORCE_SETS --poscar POSCAR --born BORN \\
        --adaptive --branches 1 10 --fdos --plot

    # Fine-tune the adaptive shell
    python phonon_dos.py --params phonopy_params.yaml.xz --adaptive \\
        --coarse-mesh 15 15 15 --gamma-radius 0.12 --gamma-points 3000 \\
        --fdos --fmax-fdos 1.5 --plot
"""

import argparse
import csv
import math
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# ── Phonopy ───────────────────────────────────────────────────────────────────
try:
    import phonopy
    from phonopy import Phonopy
    from phonopy.interface.calculator import read_crystal_structure
    from phonopy.file_IO import parse_FORCE_SETS
except ImportError as e:
    sys.exit(
        f"ERROR: failed to import phonopy component: {e}\n"
        f"Your phonopy version may be incompatible.\n"
        f"Check with: python -c \"import phonopy; print(phonopy.__version__)\"\n"
        f"Recommended: phonopy >= 2.20   (you have {getattr(phonopy,'__version__','unknown')})"
    )


# =============================================================================
#  Crystal system detection
# =============================================================================

def detect_crystal_system(lattice):
    """
    Detect crystal system from a 3x3 lattice matrix (rows = lattice vectors).

    Returns one of: 'cubic', 'tetragonal', 'hexagonal', 'orthorhombic',
                    'trigonal', 'monoclinic', 'triclinic'
    """
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])

    # Angles between lattice vectors (degrees)
    def angle(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return math.degrees(math.acos(np.clip(cos, -1, 1)))

    alpha = angle(lattice[1], lattice[2])
    beta  = angle(lattice[0], lattice[2])
    gamma = angle(lattice[0], lattice[1])

    tol_len = 0.01   # Angstrom tolerance for lengths
    tol_ang = 0.5    # degree tolerance for angles

    def eq(x, y, tol): return abs(x - y) < tol
    def right(ang):     return eq(ang, 90.0, tol_ang)
    def hex_ang(ang):   return eq(ang, 120.0, tol_ang)

    ab = eq(a, b, tol_len)
    ac = eq(a, c, tol_len)
    bc = eq(b, c, tol_len)

    all_right  = right(alpha) and right(beta) and right(gamma)
    all_equal  = ab and ac

    if all_equal and all_right:
        return "cubic"
    if ab and not ac and all_right:
        return "tetragonal"
    if ab and right(alpha) and right(beta) and hex_ang(gamma):
        return "hexagonal"
    if ab and eq(alpha, beta, tol_ang) and eq(beta, gamma, tol_ang) and not right(alpha):
        return "trigonal"
    if all_right:
        return "orthorhombic"
    if (right(alpha) and right(gamma) and not right(beta)) or \
       (right(alpha) and right(beta) and not right(gamma)):
        return "monoclinic"
    return "triclinic"


# =============================================================================
#  Phonopy loader
# =============================================================================

def load_from_params(params_file):
    """Load Phonopy object from phonopy_params.yaml[.xz]."""
    print(f"Loading force constants from: {params_file}")
    ph = phonopy.load(params_file, is_nac=False, log_level=0)
    return ph


def load_from_force_sets(force_sets_file, poscar_file, born_file=None):
    """
    Load Phonopy object from FORCE_SETS + POSCAR [+ BORN].
    Compatible with phonopy >= 2.20 (tested on 2.38).
    """
    print(f"Loading structure from : {poscar_file}")
    print(f"Loading force sets from: {force_sets_file}")

    unitcell, _ = read_crystal_structure(poscar_file, interface_mode="vasp")

    # Infer supercell scale from FORCE_SETS header (line 2 = natoms in supercell)
    with open(force_sets_file) as f:
        lines = f.readlines()
    natom_super = int(lines[1].strip())
    natom_unit  = len(unitcell.masses)
    scale       = round((natom_super / natom_unit) ** (1 / 3))
    supercell_matrix = np.diag([scale, scale, scale])

    ph = Phonopy(unitcell, supercell_matrix, log_level=0)
    force_sets = parse_FORCE_SETS(filename=force_sets_file)
    ph.dataset = force_sets
    ph.produce_force_constants()

    if born_file:
        # phonopy 2.20+: parse_BORN_file removed; use read_born_file_and_set_nac_params
        print(f"Loading NAC parameters from: {born_file}")
        ph.read_born_file_and_set_nac_params(born_file)

    return ph


# =============================================================================
#  Q-point mesh generators
# =============================================================================

def uniform_mesh_qpoints(mesh):
    """
    Generate a uniform Monkhorst-Pack q-point grid.
    Returns q-points in fractional reciprocal coordinates, shape (N, 3).
    """
    nx, ny, nz = mesh
    gi = (np.arange(nx) + 0.5) / nx - 0.5
    gj = (np.arange(ny) + 0.5) / ny - 0.5
    gk = (np.arange(nz) + 0.5) / nz - 0.5
    ii, jj, kk = np.meshgrid(gi, gj, gk, indexing="ij")
    qpts = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    print(f"  Uniform mesh {nx}x{ny}x{nz} → {len(qpts)} q-points")
    return qpts


def adaptive_mesh_qpoints(coarse_mesh, gamma_radius, gamma_points, gamma_shells,
                           crystal_system="cubic", reciprocal_lattice=None):
    """
    Build an adaptive q-point set:
        1. Coarse uniform background grid (Monkhorst-Pack)
        2. Dense logarithmically-spaced spherical shells around Γ

    The shell density compensates for the acoustic DOS ∝ ω² ∝ |q|² deficit
    in a uniform mesh, yielding a flatter fractional DOS at low frequencies.

    Parameters
    ----------
    coarse_mesh       : list[int] — [NX, NY, NZ] background grid
    gamma_radius      : float — shell radius in fractional r.l.u.
    gamma_points      : int   — total extra q-points in the shell
    gamma_shells      : int   — number of logarithmic radial shells
    crystal_system    : str   — used to scale anisotropic shells
    reciprocal_lattice: np.ndarray (3,3) or None

    Returns
    -------
    qpts : np.ndarray (N_total, 3) — fractional coordinates
    """
    # 1. Coarse background grid
    qpts_coarse = uniform_mesh_qpoints(coarse_mesh)

    # 2. Anisotropy scaling based on crystal system
    #    For non-cubic cells, scale the Γ-shell to be ellipsoidal
    if reciprocal_lattice is not None:
        rec_lengths = np.linalg.norm(reciprocal_lattice, axis=1)
        scale = rec_lengths / rec_lengths.mean()
    else:
        scale = np.ones(3)

    # 3. Logarithmic radial shells: r from very small to gamma_radius
    r_min = gamma_radius / (gamma_shells * 10)  # start close to Γ but not at it
    r_max = gamma_radius
    radii = np.logspace(np.log10(r_min), np.log10(r_max), gamma_shells)

    # Points per shell proportional to shell surface area (4πr²)
    weights  = radii ** 2
    pts_per_shell = np.round(weights / weights.sum() * gamma_points).astype(int)
    pts_per_shell = np.maximum(pts_per_shell, 4)  # at least 4 per shell

    gamma_qpts = []
    rng = np.random.default_rng(seed=42)  # reproducible

    for r, npts in zip(radii, pts_per_shell):
        # Uniform random points on a sphere (Marsaglia method)
        vecs = rng.standard_normal((npts, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        # Scale to ellipsoid for anisotropic cells, then to radius r
        vecs = vecs * (r / scale)
        gamma_qpts.append(vecs)

    gamma_qpts = np.vstack(gamma_qpts)

    # 4. Remove coarse-mesh points that fall inside the Γ-sphere
    #    (they will be replaced by the finer shell points)
    r_coarse = np.linalg.norm(qpts_coarse / scale, axis=1)
    qpts_outside = qpts_coarse[r_coarse > gamma_radius]

    # 5. Combine
    qpts_all = np.vstack([qpts_outside, gamma_qpts])

    n_coarse = len(qpts_outside)
    n_gamma  = len(gamma_qpts)
    print(f"  Adaptive mesh: {n_coarse} coarse + {n_gamma} Γ-shell = {len(qpts_all)} q-points")
    print(f"  Crystal system: {crystal_system}  |  Γ-shell radius: {gamma_radius} r.l.u.")
    return qpts_all


# =============================================================================
#  Frequency extraction
# =============================================================================

def get_frequencies(ph, qpoints):
    """
    Compute phonon frequencies for all q-points using Phonopy API.

    Returns
    -------
    freqs : np.ndarray (N_qpoints, N_branches) — frequencies [THz]
    """
    print(f"  Computing frequencies for {len(qpoints)} q-points ...")
    ph.run_qpoints(qpoints, with_eigenvectors=False)
    freqs = ph.get_qpoints_dict()["frequencies"]   # shape (NQ, N_branches)
    print(f"  Done. Branches: {freqs.shape[1]}")
    return freqs


# =============================================================================
#  Smearing
# =============================================================================

def gaussian_smearing(freq_grid, phonon_freqs, sigma):
    diff    = freq_grid[:, None] - phonon_freqs[None, :]
    weights = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    return weights.sum(axis=1)


def heaviside_smearing(freq_grid, phonon_freqs, delta):
    diff = np.abs(freq_grid[:, None] - phonon_freqs[None, :])
    return (diff <= delta / 2).sum(axis=1).astype(float)


# =============================================================================
#  DOS computation
# =============================================================================

def compute_dos(freqs, freq_grid, sigma, smearing="gaussian"):
    """
    Vectorized DOS for all branches.

    Returns
    -------
    dos_total   : (N_freq,)
    dos_partial : (N_branches, N_freq)
    """
    n_branches  = freqs.shape[1]
    dos_partial = np.zeros((n_branches, len(freq_grid)))
    smear_fn    = gaussian_smearing if smearing == "gaussian" else heaviside_smearing

    for b in range(n_branches):
        dos_partial[b] = smear_fn(freq_grid, freqs[:, b], sigma)

    return dos_partial.sum(axis=0), dos_partial


# =============================================================================
#  Fractional acoustic DOS
# =============================================================================

def compute_fractional_dos(dos_partial, freq_grid, acoustic_branches,
                            fmin_fdos=0.0, fmax_fdos=1.5):
    """
    F_i(ω) = DOS_i(ω) / (DOS_TS + DOS_TF + DOS_L)
    Computed only in [fmin_fdos, fmax_fdos].
    """
    labels      = ["TS", "TF", "L"]
    mask        = (freq_grid >= fmin_fdos) & (freq_grid <= fmax_fdos)
    freq_window = freq_grid[mask]

    dos_acoustic = {
        label: dos_partial[b1 - 1][mask]
        for label, b1 in zip(labels, acoustic_branches)
    }
    total = sum(dos_acoustic.values())

    with np.errstate(invalid="ignore", divide="ignore"):
        fdos = {
            label: np.where(total > 0, dos_acoustic[label] / total, 0.0)
            for label in labels
        }

    return freq_window, fdos, dos_acoustic


# =============================================================================
#  Linear fit of fractional DOS around 1 THz
# =============================================================================

def linear_fit_fdos(freq_window, fdos,
                    fit_fmin=0.5, fit_fmax=1.2, eval_freq=1.0):
    """
    Fit each fractional DOS branch to F(omega) = a*omega + b
    over [fit_fmin, fit_fmax] THz, then evaluate at eval_freq.

    Parameters
    ----------
    freq_window : np.ndarray (N,)
    fdos        : dict {label: np.ndarray (N,)}
    fit_fmin    : float  lower bound of fit window [THz]
    fit_fmax    : float  upper bound of fit window [THz]
    eval_freq   : float  frequency at which to evaluate [THz]

    Returns
    -------
    fit_results : dict {label: dict} with keys:
        slope, intercept, value (at eval_freq), r2, fit_freq, fit_curve
    """
    mask = (freq_window >= fit_fmin) & (freq_window <= fit_fmax)
    if mask.sum() < 3:
        raise ValueError(
            f"Fewer than 3 points in fit window [{fit_fmin}, {fit_fmax}] THz. "
            f"Widen --fit-fmin / --fit-fmax or increase --nfreq."
        )

    fit_freq    = freq_window[mask]
    fit_results = {}

    for label, values in fdos.items():
        y = values[mask]

        # Linear least-squares: coeffs = [slope, intercept]
        coeffs           = np.polyfit(fit_freq, y, deg=1)
        slope, intercept = coeffs

        # Evaluate at target frequency
        value  = slope * eval_freq + intercept

        # R2 goodness of fit
        y_pred = np.polyval(coeffs, fit_freq)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        fit_results[label] = {
            "slope"     : slope,
            "intercept" : intercept,
            "value"     : value,
            "r2"        : r2,
            "fit_freq"  : fit_freq,
            "fit_curve" : y_pred,
        }

        print(f"  F_{label}:  a = {slope:+.5f}  b = {intercept:.5f}  "
              f"F({eval_freq} THz) = {value:.5f}   R2 = {r2:.5f}")

    total_at_eval = sum(r["value"] for r in fit_results.values())
    print(f"  Sum of fitted values at {eval_freq} THz: {total_at_eval:.5f}  "
          f"(ideal = 1.000)")

    return fit_results


def save_fit_results(path, fit_results, eval_freq, fit_fmin, fit_fmax):
    """Save linear fit parameters and evaluated values to a text file."""
    with open(path, "w") as f:
        f.write("# Linear fit of Fractional Acoustic DOS\n")
        f.write(f"# Fit window  : [{fit_fmin}, {fit_fmax}] THz\n")
        f.write(f"# Evaluated at: {eval_freq} THz\n")
        f.write("# Model: F(omega) = slope * omega + intercept\n")
        f.write("#\n")
        f.write(f"{'Branch':<10} {'Slope':>12} {'Intercept':>12} "
                f"{'F({:.1f}THz)'.format(eval_freq):>14} {'R2':>10}\n")
        f.write("-" * 62 + "\n")
        for label, r in fit_results.items():
            f.write(f"{'F_'+label:<10} {r['slope']:>12.6f} {r['intercept']:>12.6f} "
                    f"{r['value']:>14.6f} {r['r2']:>10.6f}\n")
        total = sum(r["value"] for r in fit_results.values())
        f.write("-" * 62 + "\n")
        f.write(f"{'Sum':<10} {'':>12} {'':>12} {total:>14.6f}\n")
    print(f"Fit results saved -> {path}")


# =============================================================================
#  Plotting
# =============================================================================

def plot_dispersion(freqs, title=""):
    q_idx   = np.arange(len(freqs))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.tab20(np.linspace(0, 1, freqs.shape[1]))
    for b in range(freqs.shape[1]):
        ax.plot(q_idx, freqs[:, b], color=colors[b], linewidth=0.6, alpha=0.7)
    ax.set_xlabel("q-point index")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(f"Phonon frequencies — {freqs.shape[1]} branches  {title}")
    plt.tight_layout()
    plt.savefig("dispersion.png", dpi=150)
    plt.show()


def plot_dos(freq_grid, dos_total, dos_partial, show_partial=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq_grid, dos_total, color="black", linewidth=2, label="Total DOS")
    if show_partial:
        colors = plt.cm.tab20(np.linspace(0, 1, dos_partial.shape[0]))
        for b in range(dos_partial.shape[0]):
            ax.plot(freq_grid, dos_partial[b], color=colors[b],
                    linewidth=0.8, alpha=0.6, label=f"Branch {b+1}")
        if dos_partial.shape[0] <= 12:
            ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("DOS (states/THz)")
    ax.set_title("Phonon Density of States")
    plt.tight_layout()
    plt.savefig("dos.png", dpi=150)
    plt.show()


def plot_selected_branches(freq_grid, dos_partial, branch_list):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(branch_list)))
    for color, b1 in zip(colors, branch_list):
        b0 = b1 - 1
        if b0 < 0 or b0 >= dos_partial.shape[0]:
            print(f"  Warning: branch {b1} out of range — skipped")
            continue
        ax.plot(freq_grid, dos_partial[b0], color=color,
                linewidth=1.5, label=f"Branch {b1}")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("DOS (states/THz)")
    ax.set_title(f"Partial DOS — Selected branches: {branch_list}")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig("dos_selected_branches.png", dpi=150)
    plt.show()


def plot_fractional_dos(freq_window, fdos, fmin_fdos, fmax_fdos,
                        fit_results=None, fit_fmin=None, fit_fmax=None,
                        eval_freq=1.0):
    """
    Plot fractional acoustic DOS. Optionally overlay linear fits and
    mark the evaluated value at eval_freq with a vertical dashed line.
    """
    colors  = {"TS": "royalblue", "TF": "tomato", "L": "seagreen"}
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, values in fdos.items():
        ax.plot(freq_window, values, color=colors[label],
                linewidth=2.0, label=f"$F_{{\\mathrm{{{label}}}}}$")

    # Overlay linear fits if provided
    if fit_results is not None:
        for label, r in fit_results.items():
            ax.plot(r["fit_freq"], r["fit_curve"],
                    color=colors[label], linewidth=1.2,
                    linestyle="--", alpha=0.85)
            # Mark evaluated point
            ax.plot(eval_freq, r["value"], "o",
                    color=colors[label], markersize=8, zorder=5,
                    label=f"$F_{{\\mathrm{{{label}}}}}$({eval_freq} THz) = {r['value']:.3f}")

        # Shade the fit window
        if fit_fmin is not None and fit_fmax is not None:
            ax.axvspan(fit_fmin, fit_fmax, alpha=0.07, color="gray",
                       label=f"Fit window [{fit_fmin}–{fit_fmax} THz]")

        # Vertical line at eval frequency
        ax.axvline(eval_freq, color="black", linewidth=0.8,
                   linestyle=":", alpha=0.7)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlim(fmin_fdos, fmax_fdos)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("Fractional DOS", fontsize=12)
    ax.set_title(
        f"Fractional Acoustic DOS  [{fmin_fdos} – {fmax_fdos} THz]\n"
        r"$F_i = \mathrm{DOS}_i\;/\;(\mathrm{DOS}_\mathrm{TS}"
        r"+\mathrm{DOS}_\mathrm{TF}+\mathrm{DOS}_\mathrm{L})$",
        fontsize=11
    )
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig("fractional_dos.png", dpi=150)
    plt.show()


# =============================================================================
#  Save
# =============================================================================

def save_dos(path, freq_grid, dos_total, dos_partial):
    n_branches = dos_partial.shape[0]
    header = ["Frequency(THz)", "Total_DOS"] + [f"Branch_{b+1}" for b in range(n_branches)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for i, w in enumerate(freq_grid):
            row = [f"{w:.6f}", f"{dos_total[i]:.6f}"] + \
                  [f"{dos_partial[b, i]:.6f}" for b in range(n_branches)]
            writer.writerow(row)
    print(f"DOS saved → {path}")


def save_fractional_dos(path, freq_window, fdos, dos_acoustic):
    labels = list(fdos.keys())
    header = (["Frequency(THz)"]
              + [f"DOS_{l}(raw)" for l in labels]
              + [f"FDOS_{l}" for l in labels])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for i, w in enumerate(freq_window):
            row = ([f"{w:.6f}"]
                   + [f"{dos_acoustic[l][i]:.6f}" for l in labels]
                   + [f"{fdos[l][i]:.6f}" for l in labels])
            writer.writerow(row)
    print(f"Fractional DOS saved → {path}")


def save_qpoints(path, qpoints):
    np.savetxt(path, qpoints, fmt="%.8f",
               header="qx  qy  qz  (fractional reciprocal coordinates)")
    print(f"Q-points saved → {path}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generalized Phonon DOS — Phonopy API, no qpoints.yaml needed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--params",  metavar="FILE",
                     help="phonopy_params.yaml or phonopy_params.yaml.xz")
    inp.add_argument("--forces",  metavar="FILE",
                     help="FORCE_SETS file (also set --poscar)")

    p.add_argument("--poscar", metavar="FILE", default="POSCAR",
                   help="POSCAR file (used with --forces, default: POSCAR)")
    p.add_argument("--born",   metavar="FILE", default=None,
                   help="BORN file for NAC correction (optional)")

    # ── Q-point sampling ──────────────────────────────────────────────────────
    qmesh = p.add_mutually_exclusive_group(required=True)
    qmesh.add_argument("--mesh", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
                        help="Uniform Monkhorst-Pack mesh  e.g. --mesh 21 21 21")
    qmesh.add_argument("--adaptive", action="store_true",
                        help="Adaptive mesh: uniform + dense Γ-shell (best for FDOS)")

    # ── Adaptive tuning ───────────────────────────────────────────────────────
    p.add_argument("--coarse-mesh",    type=int, nargs=3, default=[11, 11, 11],
                   metavar=("NX", "NY", "NZ"),
                   help="Background mesh for adaptive mode  (default: 11 11 11)")
    p.add_argument("--gamma-radius",   type=float, default=0.15,
                   help="Γ-shell radius in r.l.u.           (default: 0.15)")
    p.add_argument("--gamma-points",   type=int,   default=2000,
                   help="Extra q-points inside Γ-shell      (default: 2000)")
    p.add_argument("--gamma-shells",   type=int,   default=30,
                   help="Number of logarithmic shells       (default: 30)")

    # ── DOS core ──────────────────────────────────────────────────────────────
    p.add_argument("--sigma",    type=float, default=0.1,
                   help="Gaussian smearing width [THz]      (default: 0.1)")
    p.add_argument("--nfreq",    type=int,   default=1000,
                   help="Frequency grid points               (default: 1000)")
    p.add_argument("--fmin",     type=float, default=None,
                   help="Min frequency [THz]                 (default: auto)")
    p.add_argument("--fmax",     type=float, default=None,
                   help="Max frequency [THz]                 (default: auto)")
    p.add_argument("--smearing", default="gaussian",
                   help="gaussian | heaviside                (default: gaussian)")

    # ── Fractional DOS ────────────────────────────────────────────────────────
    p.add_argument("--fdos", action="store_true",
                   help="Compute fractional acoustic DOS")
    p.add_argument("--acoustic", type=int, nargs=3, default=[1, 2, 3],
                   metavar=("TS", "TF", "L"),
                   help="1-based branch indices for TS TF L  (default: 1 2 3)")
    p.add_argument("--fmin-fdos", type=float, default=0.0, dest="fmin_fdos",
                   help="FDOS lower bound [THz]              (default: 0.0)")
    p.add_argument("--fmax-fdos", type=float, default=1.5, dest="fmax_fdos",
                   help="FDOS upper bound [THz]              (default: 1.5)")
    p.add_argument("--output-fdos", default="FDOS_acoustic.txt",
                   help="Output file for fractional DOS")

    # Linear fit of fractional DOS
    p.add_argument("--fit", action="store_true",
                   help="Linear fit of FDOS in [--fit-fmin, --fit-fmax], "
                        "evaluated at --eval-freq  (requires --fdos)")
    p.add_argument("--fit-fmin",   type=float, default=0.5, dest="fit_fmin",
                   help="Lower bound of linear fit window [THz]  (default: 0.5)")
    p.add_argument("--fit-fmax",   type=float, default=1.2, dest="fit_fmax",
                   help="Upper bound of linear fit window [THz]  (default: 1.2)")
    p.add_argument("--eval-freq",  type=float, default=1.0, dest="eval_freq",
                   help="Frequency at which to evaluate the fit [THz] (default: 1.0)")
    p.add_argument("--output-fit", default="FDOS_fit.txt",
                   help="Output file for fit results               (default: FDOS_fit.txt)")

    # ── Branches ──────────────────────────────────────────────────────────────
    p.add_argument("--branches", type=int, nargs="+", default=None, metavar="N",
                   help="1-based branch numbers to plot  e.g. --branches 1 3 10")

    # ── Output / plot ─────────────────────────────────────────────────────────
    p.add_argument("--output",     default="DOS_output.txt",
                   help="Full DOS output file               (default: DOS_output.txt)")
    p.add_argument("--plot",       action="store_true", help="Show plots")
    p.add_argument("--pdos",       action="store_true", help="Show all partial DOS")
    p.add_argument("--save-qpts",  default=None, metavar="FILE",
                   help="Save q-points to file for inspection")

    return p.parse_args()


# =============================================================================
#  Main
# =============================================================================

def main():
    args = parse_args()

    # ── 1. Load Phonopy object ─────────────────────────────────────────────
    if args.params:
        ph = load_from_params(args.params)
    else:
        ph = load_from_force_sets(args.forces, args.poscar, args.born)

    # ── 2. Detect crystal system ───────────────────────────────────────────
    lattice        = ph.primitive.cell          # (3,3) real-space lattice [Angstrom]
    crystal_system = detect_crystal_system(lattice)
    # Reciprocal lattice vectors (rows)
    rec_lattice    = np.linalg.inv(lattice).T * 2 * math.pi
    print(f"\n  Crystal system detected: {crystal_system.upper()}")
    print(f"  Lattice parameters:")
    for i, (vec, label) in enumerate(zip(lattice, ["a", "b", "c"])):
        print(f"    {label} = {np.linalg.norm(vec):.4f} Å")

    # ── 3. Build q-point set ───────────────────────────────────────────────
    print("\nBuilding q-point mesh...")
    if args.mesh:
        qpoints = uniform_mesh_qpoints(args.mesh)
    else:
        qpoints = adaptive_mesh_qpoints(
            coarse_mesh    = args.coarse_mesh,
            gamma_radius   = args.gamma_radius,
            gamma_points   = args.gamma_points,
            gamma_shells   = args.gamma_shells,
            crystal_system = crystal_system,
            reciprocal_lattice = rec_lattice,
        )

    if args.save_qpts:
        save_qpoints(args.save_qpts, qpoints)

    # ── 4. Compute frequencies ─────────────────────────────────────────────
    print("\nRunning Phonopy frequency calculation...")
    freqs       = get_frequencies(ph, qpoints)
    n_qpoints   = freqs.shape[0]
    n_branches  = freqs.shape[1]

    # ── 5. Frequency grid ──────────────────────────────────────────────────
    fmin      = args.fmin if args.fmin is not None else max(0.0, freqs.min() - args.sigma)
    fmax      = args.fmax if args.fmax is not None else freqs.max() + args.sigma
    freq_grid = np.linspace(fmin, fmax, args.nfreq)

    print(f"\n  Frequency range : [{fmin:.3f}, {fmax:.3f}] THz  ({args.nfreq} pts)")
    print(f"  Smearing        : {args.smearing},  sigma = {args.sigma} THz")
    print(f"  Branches        : {n_branches}")

    # ── 6. Compute DOS ────────────────────────────────────────────────────
    print("\nComputing DOS...")
    dos_total, dos_partial = compute_dos(freqs, freq_grid, args.sigma, args.smearing)

    # ── 7. Summary ────────────────────────────────────────────────────────
    ac = args.acoustic
    print(f"\nAcoustic branches — TS={ac[0]}  TF={ac[1]}  L={ac[2]}")
    for label, b1 in zip(["TS", "TF", "L"], ac):
        print(f"  Max freq branch {b1} ({label}): {freqs[:, b1-1].max():.4f} THz")
    print(f"  Max freq (all branches)       : {freqs.max():.4f} THz")

    # ── 8. Save full DOS ──────────────────────────────────────────────────
    save_dos(args.output, freq_grid, dos_total, dos_partial)

    # ── 9. Fractional acoustic DOS ────────────────────────────────────────
    if args.fdos:
        print(f"\nComputing fractional acoustic DOS "
              f"[{args.fmin_fdos} – {args.fmax_fdos} THz]...")

        for b1 in args.acoustic:
            if b1 < 1 or b1 > n_branches:
                sys.exit(f"ERROR: --acoustic branch {b1} out of range (1–{n_branches})")

        freq_window, fdos, dos_acoustic = compute_fractional_dos(
            dos_partial, freq_grid,
            acoustic_branches = args.acoustic,
            fmin_fdos         = args.fmin_fdos,
            fmax_fdos         = args.fmax_fdos,
        )
        save_fractional_dos(args.output_fdos, freq_window, fdos, dos_acoustic)

        # ── Linear fit of fractional DOS ──────────────────────────────────
        fit_results = None
        if args.fit:
            print(f"\nLinear fit of fractional DOS "
                  f"[{args.fit_fmin} – {args.fit_fmax} THz], "
                  f"evaluated at {args.eval_freq} THz:")
            fit_results = linear_fit_fdos(
                freq_window, fdos,
                fit_fmin  = args.fit_fmin,
                fit_fmax  = args.fit_fmax,
                eval_freq = args.eval_freq,
            )
            save_fit_results(
                args.output_fit, fit_results,
                eval_freq = args.eval_freq,
                fit_fmin  = args.fit_fmin,
                fit_fmax  = args.fit_fmax,
            )

        if args.plot:
            plot_fractional_dos(
                freq_window, fdos, args.fmin_fdos, args.fmax_fdos,
                fit_results = fit_results,
                fit_fmin    = args.fit_fmin  if args.fit else None,
                fit_fmax    = args.fit_fmax  if args.fit else None,
                eval_freq   = args.eval_freq,
            )

    # ── 10. Plots ─────────────────────────────────────────────────────────
    if args.plot:
        mesh_label = ("adaptive" if args.adaptive
                      else f"mesh {'x'.join(str(m) for m in args.mesh)}")
        plot_dispersion(freqs, title=f"({mesh_label})")
        plot_dos(freq_grid, dos_total, dos_partial, show_partial=args.pdos)

    if args.branches:
        if args.plot:
            print(f"\nPlotting selected branches: {args.branches}")
            plot_selected_branches(freq_grid, dos_partial, args.branches)
        else:
            print("\nNote: --branches requires --plot to display the figure.")


if __name__ == "__main__":
    main()
