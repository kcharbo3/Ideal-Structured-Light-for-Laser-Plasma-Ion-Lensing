from inspect import getmembers
from configs import base_config
import numpy as np

# Import all constants from base_config
for k, v in getmembers(base_config):
    if k.isupper():
        globals()[k] = v

SIM_DIRECTORY = "./"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "mem_files/"

WAVELENGTH = 1.
SPATIAL_GAUSSIAN_ORDER = 6
TEMPORAL_GAUSSIAN_ORDER = 6
ALPHA_PARABOLA = 0.
PEAK_A0 = 80
SPOT_SIZE = 10.
PULSE_FWHM = 40.

# ------------ Simulation Grid Parameters ---------------
# In microns
Y_HEIGHT = 28.16
DY_SIM = 1 / 25.
Z_HEIGHT = 28.16
DZ_SIM = 1 / 25.
T_LENGTH = 300.
DT_SIM = (1 / 50.) * (0.95 / np.sqrt(3.))  # To satisfy CFL condition

# Not needed for the interpolator, just for sims
X_LENGTH = 48.
DX_SIM = 1 / 50.


FOIL_THICKNESS = np.array([0.25])
