# This code find the fitting model of the experimental data set of 
# Coils Magnetic Field Bc (T) as Function of Inverse Electron Path Radius (m)
# Data were collected using KeySight 34461A digital multimeter during the Electron Mass to Charge Ratio Experiment. 
# Source: PHY224 Electron Mass to Charge Ratio Lab, University of Toronto.

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
LocalizationUnc_um = getattr(pm, "LocalizationUnc_um", 0.10)  # µm per axis (tracking/focus)
TimeUnc = getattr(pm, "TimeUnc", 0.03)                   # s per frame (manual)

# -- Load run --
# Usable Datasets: 1, 3, 5, 7(maybe), 8, 9, 10, 11, 15 pt2, 16

delta_r = []

# -- Traverse Dataset Folder & Process Data --
def Traverse_files(directory_path):
    # Traverses files in the provided director path, calculates 
    # delta_r for each step and append it to delta_r array
    for root, _, files in os.walk(directory_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            #print(filepath)
            xdata, ydata = FileRead(filepath)
            if (len(xdata) != len(ydata)):
                print("File Corupted")
                continue
            # -- Traverse Data to compute for r (in PIXELS here)
            for i in range(len(xdata) - 1):
                dr = CalculateStepSize(xdata[i], xdata[i+1], ydata[i], ydata[i+1])
                if (dr < 1e-9):  # null check
                    print("null value")
                # keep only reasonable steps (still in pixels at this point)
                if dr*PixelSize_um <= 5.7: # If the cut of is at 5.7, percent error is smallest
                    delta_r.append(dr)

# Reading File
def FileRead(filename):
    try:
        data = pd.read_csv(filename, sep='\t', skiprows=(0,))
        cols = [c.strip() for c in data.columns]
        if "X" not in cols or "Y" not in cols:
            # try without skipping
            data = pd.read_csv(filename, sep='\t', skiprows=(0,))
    except Exception:
        data = pd.read_csv(filename, sep='\t', skiprows=(0,))

    data.columns = [c.strip(" (pixels)") for c in data.columns]
    #print(data.columns) <- Debugging Line
    xdata = data["X"].to_numpy()
    ydata = data["Y"].to_numpy()
    return xdata, ydata

# -- Calculate Step Size --
def CalculateStepSize(xi, xf, yi, yf):
    # returns step size in PIXELS
    delta_x = xf - xi
    delta_y = yf - yi
    return np.sqrt(delta_x**2 + delta_y**2)

Traverse_files("Datasets")

# print(delta_r) # <- Debugging Line

# ---- Build Histogram of Step Sizes ----

# Make sure delta_r is a NumPy array (PIXELS -> µm conversion below)
delta_r = np.asarray(delta_r, dtype=float)

if delta_r.size == 0:  # FIX: guard empty
    raise ValueError("No step sizes found. Check your dataset path/format.")

# Convert to µm (recommended for axis units)
delta_r_um = delta_r * PixelSize_um

# Freedman–Diaconis bin width for most appropriate bins
def FreedmanDiaconisWidth(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return None
    q75, q25 = np.percentile(a, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return None
    h = 2 * iqr * (n ** (-1/3))
    return h

# Compute bin width; fallback to Scott's rule if needed
h = FreedmanDiaconisWidth(delta_r_um)
if (h is None) or (h <= 0):
    sr = np.std(delta_r_um, ddof=1) if len(delta_r_um) > 1 else 0.0
    h = 3.5 * sr * (len(delta_r_um) ** (-1/3)) if sr > 0 else 1.0  # safe fallback

# Build bin edges from 0 (Rayleigh support) to max
rmax = np.max(delta_r_um) if delta_r_um.size else 1.0
if h <= 0 or not np.isfinite(h):  # FIX: guard pathological h
    h = max(rmax, 1.0) / 50.0
bins = np.arange(0, rmax + h, h)

print(f"Histogram uses {len(bins)-1} bins; bin width ≈ {h:.4g} µm")

plt.figure()
plt.hist(delta_r_um, bins=bins, density=True, edgecolor='k', alpha=0.75)
plt.xlabel("Step size Δr (µm)")
plt.ylabel("Probability density")
plt.title("Distribution of Step Lengths")
plt.tight_layout()
plt.show()

# ---- MLE estimate of D (no histogram/bins) ----
# Assumptions (can be overridden in Parameter.py)
FrameRate_fps   = getattr(pm, "FrameRate_fps", 2.0)        # images per second (manual default: 2 fps)
DeltaT_s        = getattr(pm, "DeltaT_s", 1.0/FrameRate_fps)  # seconds between frames
PixelSize_um    = getattr(pm, "PixelSize_um", 0.1204)      # µm/pixel (nominal)
BeadDiameter_um = getattr(pm, "BeadDiameter_um", 1.9)      # µm (manual)
Viscosity_cP    = getattr(pm, "Viscosity_cP", 1.00)        # cP (manual ~1.00 at ~20°C)
Temperature_K   = getattr(pm, "Temperature_K", 296.5)      # K (manual)

# Convert step sizes to micrometres (already have delta_r in pixels)
delta_r = np.asarray(delta_r, dtype=float)
delta_r_um = delta_r * PixelSize_um  # NOT double; CalculateStepSize returned pixels

# Optional: basic outlier rejection for tracking glitches (non-destructive)
# Comment out if you want raw MLE
q99 = np.percentile(delta_r_um, 99) if delta_r_um.size > 0 else 0.0
Mask = delta_r_um <= (q99 * 1.5)  # keep the central bulk
delta_r_um_used = delta_r_um[Mask]

# MLE for D from Rayleigh steps:
# (2 D Δt)_est = (1/(2N)) * sum(r_i^2)  =>  D_hat = mean(r^2)/(4 Δt)
N_used = len(delta_r_um_used)
if N_used == 0:
    raise ValueError("No step data available for MLE of D.")

Mean_r2_um2 = np.mean(delta_r_um_used**2)
D_hat_um2_per_s = Mean_r2_um2 / (4.0 * DeltaT_s)        # µm^2/s
D_hat_m2_per_s  = D_hat_um2_per_s * 1e-12                # m^2/s

# Standard error of MLE (Rayleigh): SE(D_hat) = D_hat / sqrt(N)
D_SE_um2_per_s = D_hat_um2_per_s / np.sqrt(N_used)
D_SE_m2_per_s  = D_hat_m2_per_s  / np.sqrt(N_used)

# ---- Compute Boltzmann constant from Stokes–Einstein ----
# k = D * (6 π η r) / T
eta_Pa_s = Viscosity_cP * 1e-3                           # 1 cP = 1e-3 Pa·s
r_m      = (BeadDiameter_um * 1e-6) / 2.0                # bead radius in m
gamma    = 6.0 * np.pi * eta_Pa_s * r_m                  # Stokes drag, N·s/m
k_hat_J_per_K = (D_hat_m2_per_s * gamma) / Temperature_K

# Propagate only counting error from D (dominant for large N):
k_SE_J_per_K  = k_hat_J_per_K / np.sqrt(N_used)

print("---- MLE (no bins) ----")
print(f"N used for MLE: {N_used:d}")
print(f"Δt = {DeltaT_s:.6g} s  |  PixelSize = {PixelSize_um:.5g} µm/pixel")
print(f"D̂ = {D_hat_um2_per_s:.5g} ± {D_SE_um2_per_s:.2g}  µm²/s")
print(f"D̂ = {D_hat_m2_per_s:.5g} ± {D_SE_m2_per_s:.2g}  m²/s")
print(f"k̂ = {k_hat_J_per_K:.5g} ± {k_SE_J_per_K:.1g}  J/K")
print("Accepted k ≈ 1.380649e-23 J/K")
print(f"Percent diff (|k̂-k|/k*100): {abs(k_hat_J_per_K-1.380649e-23)/1.380649e-23*100:.2f}%")


# ---- Overlay Rayleigh PDF on the histogram (using MLE D̂ and Δt) ----

# Use the same data array you histogrammed
rgrid = np.linspace(0, np.max(delta_r_um) if delta_r_um.size else 1.0, 400)

def rayleigh_pdf(r, D, dt):
    # p(r) = r/(2 D dt) * exp(-r^2/(4 D dt))
    return (r / (2.0 * D * dt)) * np.exp(-(r**2) / (4.0 * D * dt))

pdf_mle = rayleigh_pdf(rgrid, D_hat_um2_per_s, DeltaT_s)

plt.figure()
# Replot the histogram so the overlay is on the same axes
plt.hist(delta_r_um, bins=bins, density=True, edgecolor='k', alpha=0.75, label="Data (Δr)")
plt.plot(rgrid, pdf_mle, linewidth=2.0, label=f"Rayleigh (MLE)  D̂={D_hat_um2_per_s:.3g} µm²/s")
plt.xlabel("Step size Δr (µm)")
plt.ylabel("# of Steps")
plt.title("Distribution of Step Lengths (µm) with Rayleigh Fit (MLE)")
plt.legend()
plt.tight_layout()
plt.savefig("Pt2.png", dpi=500)
plt.show()
