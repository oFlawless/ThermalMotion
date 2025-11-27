# ThermalMotion

---- MSD ----
Fit: <r^2> = a t + b
a = 1.23071 ± 0.024  µm^2/s
b = -0.495975 ± 0.034  µm^2
D = 3.077e-13 ± 5.898e-15  m^2/s  (D = a/4 with µm^2→m^2)
γ (Stokes drag) = 1.671e-08 kg/s
k = 1.734e-23 ± 1.315e-24 J/K   (accepted 1.380e-23 J/K)
Chi Square: 117.18375652161508 | Reduced Chi Square: 1.0015705685608127

---- MLE (no bins) ----
N used for MLE: 2614
Δt = 0.5 s  |  PixelSize = 0.1204 µm/pixel
D̂ = 0.11508 ± 0.0023  µm²/s
sqrt(N) bins: 51, total steps plotted: 2618
---- Fit to binned density (curve_fit) ----
D_fit = 0.095981 ± 0.0019  µm²/s
Residuals: Rayleigh (MLE) vs binned density: Chi Square: 222.5 | Reduced Chi Square: 4.45  (N=51, p=1)
Residuals: Rayleigh (curve_fit on bins) vs binned density: Chi Square: 119.8 | Reduced Chi Square: 2.397  (N=51, p=1)

---- k estimates with uncertainties ----
η = 1.000e-03 ± 5.0e-05 Pa·s  | r = 9.500e-07 ± 5.0e-08 m  | T = 296.50 ± 0.50 K  | γ = 1.791e-08 ± 1.3e-09 N·s/m
k (from MLE D̂):    6.95e-24 ± 5.2e-25 J/K
k (from bin-fit D): 5.7968e-24 ± 4.4e-25 J/K

Uncertainty budget for k (percent of variance):
  MLE:   D:   6.8%  |  γ(η,r):  93.2%  |  T:   0.1%
  BinFit: D:   6.6% |  γ(η,r):  93.4% |  T:   0.1%