'''
Brownian mean squared distance analysis:
- Build r^2 from all datasets in 'Dataset'
- curve_fit to mean squared distance as a function of Time
- Residual plots with chi-square
Measured on Nikkon x40 Objective lens Microscope
Source: PHY224 Thermal Motion Lab, University of Toronto Physics Department
'''

# Import Modules
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import os

# Custom Param Module
import Parameter as pm

# Get Critical Attributues
PixelSize_um = getattr(pm, "PixelSize_um", 0.1204)  # µm/pixel (nominal)
PixelSizeUnc = getattr(pm, "PixelSizeUnc", 0.003)        # µm/pixel (manual)
LocalizationUnc_um = getattr(pm, "LocalizationUnc_um", 0.00)  # µm per axis (tracking/focus)
TimeUnc = getattr(pm, "TimeUnc", 0.03)                   # s per frame (manual)

# -- Load run --
# Usable Datasets: 1, 3, 5, 7(maybe), 8, 9, 10, 11, 15 pt2, 16
DatasetName = "Dataset" 

#  Per-point uncertainty for r^2 (µm^2) 
# Model: Δx_um = s*Δx_pix;  σ(Δx_um)^2 = (Δx_pix*σ_s)^2 + σ_loc^2  (same for y)
sigma_s = PixelSizeUnc            # µm/pixel
sigma_loc = LocalizationUnc_um    # µm

dxT = []
dyT = []


# -- Uncertainty Calculator --
def rsqr_and_sigma_per_run(xdata, ydata, um_per_pix, pix_unc, loc_unc):
    """
      - r2[i] = (Δx_um^2 + Δy_um^2) from frame 0 to frame i (i>=1)
      - r2_unc[i] = propagated Type B uncertainty for that r^2 point
    """
    # Displacements in pixels relative to frame 0
    dx = xdata[1:] - xdata[0]
    dy = ydata[1:] - ydata[0]

    # Convert to micrometers
    dx_um = um_per_pix * dx
    dy_um = um_per_pix * dy

    # Per-axis displacement uncertainty in µm:
    # scale term + (two localizations) term
    dx_unc = np.sqrt((dx * pix_unc)**2 + 2.0 * (loc_unc**2))
    dy_unc = np.sqrt((dy * pix_unc)**2 + 2.0 * (loc_unc**2))

    # r^2 and its propagated uncertainty
    r2 = dx_um**2 + dy_um**2
    r2_unc = np.sqrt((2.0 * dx_um * dx_unc)**2 +
                           (2.0 * dy_um * dy_unc)**2)

    return r2, r2_unc



# -- calculate mean r^2 across runs & combined mean uncertainty (Type A + Type B)
def stack_runs_and_mean(all_r2_list, all_sigma_r2_list):
    M = len(all_r2_list)
    L = min(len(r) for r in all_r2_list)  # trim to same length

    R = np.vstack([r[:L] for r in all_r2_list])
    S = np.vstack([s[:L] for s in all_sigma_r2_list])

    mean_r2 = np.mean(R, axis=0)
    # sample std across runs (ddof=1)
    sample_std = np.std(R, axis=0, ddof=1)
    typeA_sem = sample_std / np.sqrt(M)

    # Average Type B into the mean: var(mean) = (1/M^2) * sum_k var_k
    typeB_on_mean = np.sqrt(np.sum(S**2, axis=0)) / M

    # Calulate Type A & Type B Uncertainty
    sigma_mean = np.sqrt(typeA_sem**2 + typeB_on_mean**2)
    return mean_r2, sigma_mean, typeA_sem, typeB_on_mean


# -- Find r^2 for every file --
def compute_all_runs(directory_path, um_per_pix, pix_unc, loc_unc):
    r2_runs = []
    run_unc = []
    for root, _, files in os.walk(directory_path):
        for fname in files:
            if not fname.lower().endswith((".txt", ".tsv", ".csv")):  # guard
                continue
            fpath = os.path.join(root, fname)
            try:
                df = pd.read_csv(fpath, sep="\t", skiprows=(0,))
                cols = [c.strip() for c in df.columns]
                if "X" not in cols or "Y" not in cols:
                    df = pd.read_csv(fpath, sep="\t")
                df.columns = [c.strip(" (pixels)").strip() for c in df.columns]
                xdata = df["X"].to_numpy()
                ydata = df["Y"].to_numpy()
                #dxT.append(xdata)
                #dyT.append(ydata)
                if len(xdata) < 2 or len(ydata) < 2:
                    continue
            except Exception as e:
                print("Skip file (read error):", fpath, e)
                continue
            r2, s_r2 = rsqr_and_sigma_per_run(
                xdata, ydata, um_per_pix, pix_unc, loc_unc
            )
            r2_runs.append(r2)
            run_unc.append(s_r2)
    return r2_runs, run_unc






s = PixelSize_um
s_unc = PixelSizeUnc
loc_unc = LocalizationUnc_um

r2_runs, run_unc = compute_all_runs(DatasetName, s, s_unc, loc_unc)
'''
X = np.mean(np.vstack([d[:118] for d in dxT]), axis=0)
Y = np.mean(np.vstack([d[:118] for d in dyT]), axis=0)

totX = []
totY = []

for i in range(1, len(X)):
    totX.append((X[i] - X[0])*PixelSize_um)
    totY.append((Y[i] - Y[0])*PixelSize_um)'''

#print(X)

if len(r2_runs) == 0:
    raise RuntimeError(f"No valid runs found in {DatasetName}.")

Rsqr, sigma_Rsqr, sem_typeA, typeB_mean = stack_runs_and_mean(r2_runs, run_unc)

# Build the time axis from the *trimmed* common length
FPS = pm.FPS
t = np.arange(1, len(Rsqr) + 1) / FPS  # since r^2 starts at frame 1 relative to 0

# Debugging Lines
#print(dx_um)
#print(len(sigma_rsqr))

# -- Linear fit (curve_fit) --
def fittingModel(x, a, b):
    return a*x + b

# Use y-errors; curve_fit ignores x-errors (common practice here)
popt, pcov = curve_fit(fittingModel, xdata=t, ydata=Rsqr,sigma=sigma_Rsqr, absolute_sigma=True)
a, b = popt
sa, sb = np.sqrt(np.diag(pcov))

# -- Calculate: D and k --
# MSD in 2D: <r^2> = 4 D t + b  ⇒  D = a/4
# Convert µm^2/s → m^2/s via 1e-12
D = (a/4.0) * 1e-12  # m^2/s
sD = (sa/4.0) * 1e-12

# Import Stokes Drag
gamma = pm.StokesDrag  # kg/s
# Gamma Uncertainty
sgamma = getattr(pm, "StokesDragUnc", 0.0)

# Boltzmann: k = D γ / T
T = pm.roomTemp  # K
k_est = D * gamma / T

# error propagation for k (independent D, γ, T):
rel_D = sD/abs(D) 
rel_g = sgamma/abs(gamma)
sT = getattr(pm, "roomTempUnc", 0.5)# K (manual default)
rel_T = sT/abs(T)
sk = abs(k_est) * np.sqrt(rel_D**2 + rel_g**2 + rel_T**2)

print(f"k = {k_est:.3e} ± {sk:.3e} J/K")

# Find Fitted Values 
yfitted = fittingModel(t, a, b)

R = np.sqrt(Rsqr)

# -- Plot --
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel(r"t (s)")
ax.set_ylabel(r"$\langle r^2\rangle$ ($\mu$m$^2$)")
ax.set_title(r"Brownian Motion: Mean-squared displacement ($\mu$m$^2$) vs time (s)")

# Data with y-error bars (curve_fit used these sigmas)
ax.errorbar(t, Rsqr, yerr=sigma_Rsqr, fmt=".", ms=5, capsize=2, elinewidth=0.7,
            ecolor='skyblue', label="Experimental data ± σ")
#ax.errorbar(totX, totY)
# Best-fit line
ax.plot(t, yfitted, label=fr"Fit: $({a:.3g}\pm{sa:.1g})t + ({b:.3g}\pm{sb:.1g})$ $\mu$m$^2$")

# Annotation for D and k
ax.legend()
ax.grid(alpha=0.25)
plt.savefig("Pt1.png", dpi=500)
plt.tight_layout()
plt.show()

# -- Console summary --
print("\nFit: <r^2> = a t + b")
print(f"a = {a:.6g} ± {sa:.2g}  µm^2/s")
print(f"b = {b:.6g} ± {sb:.2g}  µm^2")
print(f"D = {D:.3e} ± {sD:.3e}  m^2/s  (D = a/4 with µm^2→m^2)")
print(f"γ (Stokes drag) = {gamma:.3e} kg/s")
print(f"k = {k_est:.3e} ± {sk:.3e} J/K   (accepted 1.380e-23 J/K)")


# Find & Plot residual data

# Calculate Residual Array
rdata = Rsqr - yfitted

fig, ax = plt.subplots(figsize=(10,5))
# Set x-axis to 0 y-value
ax.axhline(0, color="black", linestyle="--",linewidth=1)

# Plot values with error bars & graph format
ax.errorbar(t, rdata, label="Residual (T)", yerr=sigma_Rsqr, fmt='.', capsize=2, ecolor='blue', elinewidth=1)
ax.set(title=r"Residual of Mean-squared displacement ($\mu$m$^2$) & Fitted Linear Model", 
          xlabel=r"t (s)", ylabel=r"$\langle r^2\rangle$ ($\mu$m$^2$)")
plt.legend(fontsize=12, loc='upper right')


# Find chi squared & reduced chi squared value
chi2 = np.sum((rdata ** 2) / sigma_Rsqr ** 2)
N = len(Rsqr)
p = len(popt)
reduced_chi2 = chi2 / (N - p)

# -- Console Summary --
print(f"Chi Square: {chi2} | Reduced Chi Square: {reduced_chi2}")

plt.tight_layout()
ax.grid(alpha=0.25)
plt.savefig("Pt1R.png", dpi=500)
plt.show()
