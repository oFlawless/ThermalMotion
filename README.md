# Thermal Motion Analysis — PHY224

This repository contains full analysis of Brownian motion bead tracking, including MSD fits, unbinned MLE diffusion estimation, binned Rayleigh fits, and propagated uncertainties for the Boltzmann constant.

# 1. Mean-Squared Displacement (MSD) Analysis

Fit model:
⟨r²⟩ = a t + b

a = 1.23071 ± 0.024 µm²/s
b = –0.495975 ± 0.034 µm²
D = 3.077e–13 ± 5.898e–15 m²/s (D = a/4 with µm² → m²)
γ (Stokes drag) = 1.671e–08 kg/s
k = 1.734e–23 ± 1.315e–24 J/K
Accepted k = 1.380e–23 J/K

Goodness of fit:
Chi Square = 117.18
Reduced Chi Square = 1.0016

# 2. Maximum Likelihood Estimate (Unbinned)

N (steps) = 2614
Δt = 0.5 s
Pixel scale = 0.1204 µm/pixel

MLE diffusion coefficient:
D̂ = 0.11508 ± 0.0023 µm²/s

Histogram:
Number of bins = sqrt(N) = 51
Total steps plotted = 2618

# 3. Fit to Binned Rayleigh Distribution

Rayleigh curve_fit diffusion estimate:
D_fit = 0.095981 ± 0.0019 µm²/s

Goodness of fit:

Rayleigh (MLE curve vs binned density): Chi² = 222.5, Reduced χ² = 4.45

Rayleigh (curve_fit on bins): Chi² = 119.8, Reduced χ² = 2.397

# 4. Boltzmann Constant Estimates (With Uncertainties)

Parameter values used:
η = 1.000e–03 ± 5.0e–05 Pa·s
r = 9.500e–07 ± 5.0e–08 m
T = 296.50 ± 0.50 K
γ = 1.791e–08 ± 1.3e–09 N·s/m

k estimates:

From MLE D̂: 6.95e–24 ± 5.2e–25 J/K

From binned-fit D: 5.7968e–24 ± 4.4e–25 J/K

# 5. Uncertainty Budget for Boltzmann Constant

From MLE D̂:

D: 6.8%

γ (from η,r): 93.2%

T: 0.1%

From binned-fit D:

D: 6.6%

γ (from η,r): 93.4%

T: 0.1%
