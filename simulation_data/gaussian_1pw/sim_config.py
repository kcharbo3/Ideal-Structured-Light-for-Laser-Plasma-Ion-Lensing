from inspect import getmembers
from configs import base_config
from fourier_prop.laser_input import constants
import numpy as np

# Import all constants from base_config
for k, v in getmembers(base_config):
    if k.isupper():
        globals()[k] = v

SIM_DIRECTORY = "./"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "mem_files/"

WAVELENGTH = 0.800
REF_FREQ = (2*np.pi*constants.C_SPEED) / (WAVELENGTH * 1.e-6)
OMEGA0 = REF_FREQ * 1e-15  # rad / PHz

SPATIAL_GAUSSIAN_ORDER = 1
TEMPORAL_GAUSSIAN_ORDER = 1
PEAK_A0 = 16.9
SPOT_SIZE = 7.
PULSE_FWHM = 30.

LASER_TIME_START = 120.

# In microns
Y_HEIGHT = 28.16
DY_SIM = 1 / 20.
Z_HEIGHT = 28.16
DZ_SIM = 1 / 20.
T_LENGTH = 1000.
DT_SIM = (1/40.) * (0.95 / np.sqrt(3.))  # To satisfy CFL condition

# Not needed for the interpolator, just for sims
X_LENGTH = 48.
DX_SIM = 1 / 40.

N0 = np.array([30])
FOIL_THICKNESS = np.array([0.6])
CENTERY = np.array([Y_HEIGHT / 2.])
CENTERZ = np.array([Z_HEIGHT / 2.])

