"""
Brownian step-length analysis with Rayleigh model (PHY224):
- Build per-frame step sizes Δr from all files in 'Datasets'
- Histogram with sqrt(N) bins (probability density)
- Per-bin uncertainty: σ_density = sqrt(n_i) / (N * Δr_bin)
- Unbinned MLE for D and weighted curve_fit to binned density
- Combined overlay plot: Rayleigh (MLE) and Rayleigh (bin-fit)
- Residual plots with χ² and reduced χ²
- k estimates with full propagated uncertainty (D, η, bead r, T)
Measured on Nikkon x40 Objective lens Microscope
Source: PHY224 Thermal Motion Lab, University of Toronto Physics Department
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# ---------------- Parameters (with fallbacks to safe defaults) ----------------
try:
    import Parameter as pm
except Exception:
    class _PM: pass
    pm = _PM()

# Frame/scale
FrameRate_fps   = getattr(pm, "FrameRate_fps", 2.0)            # images per second
DeltaT_s        = getattr(pm, "DeltaT_s", 1.0/FrameRate_fps)   # seconds between frames
PixelSize_um    = getattr(pm, "PixelSize_um", 0.1204)          # µm/pixel (nominal)

# Physical params (values + uncertainties)
BeadDiameter_um     = getattr(pm, "BeadDiameter_um", 1.9)      # µm
BeadDiameterUnc_um  = getattr(pm, "BeadDiameterUnc_um", 0.1)   # µm  (fallback)
Viscosity_cP        = getattr(pm, "Viscosity_cP", 1.00)        # cP
ViscosityUnc_cP     = getattr(pm, "ViscosityUnc_cP", 0.05)     # cP  (~5% default)
Temperature_K       = getattr(pm, "Temperature_K", 296.5)      # K
TemperatureUnc_K    = getattr(pm, "TemperatureUnc_K", 0.5)     # K

# ---------------- Data ingest: build step sizes Δr (pixels) ----------------
def file_read_xy(filename):
    """Robust TSV/CSV reader expecting 'X' and 'Y' columns (pixels)."""
    try:
        df = pd.read_csv(filename, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(filename, sep=None, engine="python")  # auto-detect
    df.columns = [c.replace("(pixels)", "").strip() for c in df.columns]
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("Missing X/Y columns")
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Not enough points")
    return x, y

def step_size_pixels(x, y):
    """Per-frame step sizes between consecutive frames (pixels)."""
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx*dx + dy*dy)

def collect_all_steps(root_dir="Dataset", max_um_cutoff=None):
    """Collect Δr (pixels) from all files; optional cutoff in µm."""
    steps_pix = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith((".txt", ".tsv", ".csv")):
                continue
            fpath = os.path.join(root, fname)
            try:
                x, y = file_read_xy(fpath)
                dr_pix = step_size_pixels(x, y)
                if max_um_cutoff is not None:
                    mask = (dr_pix * PixelSize_um) <= max_um_cutoff
                    dr_pix = dr_pix[mask]
                if dr_pix.size:
                    steps_pix.append(dr_pix)
            except Exception as e:
                print(f"Skip {fpath}: {e}")
    if len(steps_pix) == 0:
        raise RuntimeError("No valid data files found in 'Datasets'.")
    return np.concatenate(steps_pix)

# Collect steps; keep prior 5.7 µm cutoff if desired
delta_r_pix = collect_all_steps("Dataset", max_um_cutoff=5.7)
if delta_r_pix.size == 0:
    raise RuntimeError("No step sizes after filtering.")
delta_r_um = delta_r_pix * PixelSize_um

# Optional gentle outlier mask (retain central bulk)
if delta_r_um.size >= 10:
    q99 = np.percentile(delta_r_um, 99)
    mask_bulk = delta_r_um <= (q99 * 1.5)
    delta_r_um_used = delta_r_um[mask_bulk]
else:
    delta_r_um_used = delta_r_um.copy()

# ---------------- Rayleigh PDF and MLE for D ----------------
def rayleigh_pdf(r, D_um2_per_s, dt):
    # p(r) = r / (2 D dt) * exp(-r^2 / (4 D dt)), r >= 0
    return (r / (2.0 * D_um2_per_s * dt)) * np.exp(-(r**2) / (4.0 * D_um2_per_s * dt))

# Unbinned MLE for D:
# (2 D Δt)_est = (1/(2N)) * sum(r_i^2)  =>  D̂ = mean(r^2)/(4 Δt)
N_used = len(delta_r_um_used)
Mean_r2_um2 = np.mean(delta_r_um_used**2)
D_hat_um2_per_s = Mean_r2_um2 / (4.0 * DeltaT_s)
D_SE_um2_per_s = D_hat_um2_per_s / np.sqrt(max(N_used, 1))  # SE ≈ D/√N

print("---- MLE (no bins) ----")
print(f"N used for MLE: {N_used:d}")
print(f"Δt = {DeltaT_s:.6g} s  |  PixelSize = {PixelSize_um:.5g} µm/pixel")
print(f"D̂ = {D_hat_um2_per_s:.5g} ± {D_SE_um2_per_s:.2g}  µm²/s")

# ---------------- Histogram with sqrt(N) bins (as probability density) ----------------
N_all = len(delta_r_um)
nbins = max(5, int(np.sqrt(N_all)))
counts, edges = np.histogram(delta_r_um, bins=nbins, range=(0, np.max(delta_r_um)))
bin_widths = np.diff(edges)
centers = 0.5 * (edges[:-1] + edges[1:])
N_total = counts.sum()

# Probability density and its uncertainty (Poisson → density)
density = counts / (N_total * bin_widths)
sigma_density = np.sqrt(np.maximum(counts, 1.0)) / (N_total * bin_widths)  # guard zeros

print(f"sqrt(N) bins: {nbins}, total steps plotted: {N_total}")

# ---------------- Fit D to the binned density (weighted) ----------------
mask_fit = np.isfinite(density) & np.isfinite(sigma_density) & (sigma_density > 0)
x_fit = centers[mask_fit]
y_fit = density[mask_fit]
yerr_fit = sigma_density[mask_fit]

p0 = [max(D_hat_um2_per_s, 1e-9)]  # start at MLE
bounds = (1e-12, np.inf)

popt_bins, pcov_bins = curve_fit(
    lambda r, D: rayleigh_pdf(r, D, DeltaT_s),
    x_fit, y_fit, p0=p0, sigma=yerr_fit, absolute_sigma=True, bounds=bounds
)
D_fit_um2_per_s = float(popt_bins[0])
sD_fit_um2_per_s = float(np.sqrt(pcov_bins[0, 0])) if pcov_bins.size else np.nan

print("---- Fit to binned density (curve_fit) ----")
print(f"D_fit = {D_fit_um2_per_s:.5g} ± {sD_fit_um2_per_s:.2g}  µm²/s")

# ---------------- Combined overlay plot (with Poisson error bars) ----------------
rgrid = np.linspace(0, np.max(delta_r_um), 500)
pdf_mle_grid = rayleigh_pdf(rgrid, D_hat_um2_per_s, DeltaT_s)
pdf_fit_grid = rayleigh_pdf(rgrid, D_fit_um2_per_s, DeltaT_s)

fig, ax = plt.subplots(figsize=(9, 5))

# Bars = empirical probability density
ax.bar(
    centers, density, width=bin_widths,
    edgecolor='k', alpha=0.70, align='center',
    label="Data (Δr) density", facecolor = 'none'
)

# ⟂ Poisson (density) uncertainty: σ = √n / (N * Δr_bin)
# Plot as vertical error bars at bin centers
ax.errorbar(
    centers, density, yerr=sigma_density,
    fmt='none', ecolor='black', elinewidth=1.1, capsize=2,
    label="Poisson 1σ"
)

# Rayleigh overlays
ax.plot(
    rgrid, pdf_mle_grid, linewidth=2.0, color='blue',
    label=f"Rayleigh (MLE)  D={D_hat_um2_per_s:.3g} µm²/s", ls='--'
)
ax.plot(
    rgrid, pdf_fit_grid, linewidth=2.0, color='green',
    label=f"Rayleigh (bin fit)  D={D_fit_um2_per_s:.3g}±{sD_fit_um2_per_s:.1g} µm²/s"
)

ax.set_xlabel("Step size Δr (µm)")
ax.set_ylabel("Probability density")
ax.set_title("Distribution of Step Lengths (µm) with Rayleigh Overlays")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("Pt2_overlay_mle_binf.png", dpi=500)
plt.show()


# ---------------- Residual plots (your template) ----------------
def residual_plot(x, data_y, model_y, yerr, title, filename, label="Residual (density)"):
    rdata = data_y - model_y

    fig, ax = plt.subplots(figsize=(10,5))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.errorbar(x, rdata, yerr=yerr, fmt='.', capsize=2, ecolor='blue', elinewidth=1, label=label)
    ax.set(title=title, xlabel=r"Δr bin center (µm)", ylabel=r"Density residual")
    plt.legend(fontsize=12, loc='upper right')

    # Chi-square and reduced chi-square (parameters: 1 => D only)
    chi2 = np.sum((rdata**2) / (yerr**2))
    Npts = np.count_nonzero(np.isfinite(rdata) & np.isfinite(yerr))
    p = 1
    dof = max(Npts - p, 1)
    reduced_chi2 = chi2 / dof
    print(f"{title}: Chi Square: {chi2:.4g} | Reduced Chi Square: {reduced_chi2:.4g}  (N={Npts}, p={p})")

    plt.tight_layout()
    ax.grid(alpha=0.25)
    plt.savefig(filename, dpi=500)
    plt.show()

# Residuals vs MLE
residual_plot(
    x=centers[mask_fit],
    data_y=density[mask_fit],
    model_y=rayleigh_pdf(centers[mask_fit], D_hat_um2_per_s, DeltaT_s),
    yerr=sigma_density[mask_fit],
    title=r"Residuals: Rayleigh (MLE) vs binned density",
    filename="Pt2_residuals_mle.png",
    label="Residual (density)"
)

# Residuals vs bin-fit
residual_plot(
    x=centers[mask_fit],
    data_y=density[mask_fit],
    model_y=rayleigh_pdf(centers[mask_fit], D_fit_um2_per_s, DeltaT_s),
    yerr=sigma_density[mask_fit],
    title=r"Residuals: Rayleigh (curve_fit on bins) vs binned density",
    filename="Pt2_residuals_binf.png",
    label="Residual (density)"
)

# ---------------- k estimates with propagated uncertainties ----------------
# Units:
#   η in Pa·s (1 cP = 1e-3 Pa·s)
#   r in m (diameter in µm -> radius = d/2 * 1e-6)
eta_Pa_s   = Viscosity_cP * 1e-3
seta_Pa_s  = ViscosityUnc_cP * 1e-3
r_m        = (BeadDiameter_um * 1e-6) / 2.0
sr_m       = (BeadDiameterUnc_um * 1e-6) / 2.0
T          = Temperature_K
sT         = TemperatureUnc_K

# Stokes drag γ and its uncertainty
gamma = 6.0 * np.pi * eta_Pa_s * r_m
# relative uncertainty of γ: sqrt((sη/η)^2 + (sr/r)^2)
rel_gamma = np.sqrt(
    (seta_Pa_s / max(eta_Pa_s, 1e-30))**2 +
    (sr_m / max(r_m, 1e-30))**2
)
sgamma = abs(gamma) * rel_gamma

# D uncertainties in SI
D_hat_m2_per_s = D_hat_um2_per_s * 1e-12
sD_hat_m2_per_s = D_SE_um2_per_s * 1e-12

D_fit_m2_per_s = D_fit_um2_per_s * 1e-12
sD_fit_m2_per_s = sD_fit_um2_per_s * 1e-12

# k estimates
k_hat = (D_hat_m2_per_s * gamma) / T
k_fit = (D_fit_m2_per_s * gamma) / T

# relative T uncertainty
rel_T = sT / max(T, 1e-30)

# propagated uncertainties for k (MLE and bin-fit)
rel_D_hat = sD_hat_m2_per_s / max(abs(D_hat_m2_per_s), 1e-30)
rel_D_fit = sD_fit_m2_per_s / max(abs(D_fit_m2_per_s), 1e-30)

sk_hat = abs(k_hat) * np.sqrt(rel_D_hat**2 + rel_gamma**2 + rel_T**2)
sk_fit = abs(k_fit) * np.sqrt(rel_D_fit**2 + rel_gamma**2 + rel_T**2)

print("\n---- k estimates with uncertainties ----")
print(f"η = {eta_Pa_s:.3e} ± {seta_Pa_s:.1e} Pa·s  "
      f"| r = {r_m:.3e} ± {sr_m:.1e} m  "
      f"| T = {T:.2f} ± {sT:.2f} K  "
      f"| γ = {gamma:.3e} ± {sgamma:.1e} N·s/m")

print(f"k (from MLE D̂):    {k_hat:.5g} ± {sk_hat:.2g} J/K")
print(f"k (from bin-fit D): {k_fit:.5g} ± {sk_fit:.2g} J/K")

# Optional: show relative contributions (percent) to k uncertainty
def percent_contrib(rel_D, rel_g, rel_T):
    # fraction of total variance
    total = rel_D**2 + rel_g**2 + rel_T**2
    if total <= 0:
        return (0.0, 0.0, 0.0)
    return (100*rel_D**2/total, 100*rel_g**2/total, 100*rel_T**2/total)

cD_hat, cG_hat, cT_hat = percent_contrib(rel_D_hat, rel_gamma, rel_T)
cD_fit, cG_fit, cT_fit = percent_contrib(rel_D_fit, rel_gamma, rel_T)

print("\nUncertainty budget for k (percent of variance):")
print(f"  MLE:   D: {cD_hat:5.1f}%  |  γ(η,r): {cG_hat:5.1f}%  |  T: {cT_hat:5.1f}%")
print(f"  BinFit: D: {cD_fit:5.1f}% |  γ(η,r): {cG_fit:5.1f}% |  T: {cT_fit:5.1f}%")
