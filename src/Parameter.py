# -- Paramter.py, dedicated constants module extrapolated & calculated 
# Source: PHY224 Thermal Motion Lab, University of Toronto Physics Department

# Import Module
import math

# Camera FPS
FPS = 2.0  # frames per second

# Pixel scale (µm/pixel)
PixelSize = 0.1204
PixelSizeUnc = 0.003

# Bead (µm)
BeadDiameter = 1.9
BeadDiameterUnc = 0.1

# Temperature (K)
roomTemp = 296.5
roomTempUnc = 0.5

# ----- Viscosity η(T) -----
# Baseline at 20°C: 1.00 ± 0.05 cP  =>  (1.00e-3 ± 0.05e-3) Pa·s
ETA20 = 1.00e-3
ETA20_unc = 0.05e-3

# Lab Manual Viscosity Temp depdence: ~2% decrease per °C above 20°C
TEMP_SLOPE = -0.02  # per °C
deltaC = (roomTemp - 273.15) - 20.0
viscosity = ETA20 * (1.0 + TEMP_SLOPE * deltaC)

# Propagate η uncertainty from baseline and T:
# η = η20 * (1 + s ΔT)  =>  ∂η/∂η20 = (1 + sΔT),  ∂η/∂T = η20 * s
deta_deta20 = (1.0 + TEMP_SLOPE * deltaC)
deta_dT = ETA20 * TEMP_SLOPE
viscosityUnc = math.sqrt((deta_deta20*ETA20_unc)**2 + (deta_dT*roomTempUnc)**2)

# Stokes drag γ = 6πηr
#  - Bead Radius -
radius_m = (BeadDiameter/2.0) * 1e-6 
radiusUnc_m = (BeadDiameterUnc/2.0) * 1e-6

StokesDrag = 6 * math.pi * viscosity * radius_m

# Relative uncertainty from η and r (independent) to find Stokes Drag Unc:
rel_eta = viscosityUnc / viscosity
rel_r = radiusUnc_m / radius_m
StokesDragUnc = StokesDrag * math.sqrt(rel_eta**2 + rel_r**2)
