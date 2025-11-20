from inspect import getmembers
from configs import base_config
from fourier_prop.laser_input import constants
import numpy as np

# Import all constants from base_config
for k, v in getmembers(base_config):
    if k.isupper():
        globals()[k] = v

# Simulation Parameters
# INPUT PLANE
Y_INPUT_RANGE = 10. * 20e4
Z_INPUT_RANGE = Y_INPUT_RANGE
N_Y_INPUT = 2 ** 10
N_Z_INPUT = 2 ** 10
Y_VALS_INPUT = np.linspace(-Y_INPUT_RANGE, Y_INPUT_RANGE, N_Y_INPUT)
Z_VALS_INPUT = np.linspace(-Z_INPUT_RANGE, Z_INPUT_RANGE, N_Z_INPUT)

# OUTPUT PLANE
Y_OUTPUT_RANGE = 30  # -50 to 50 um
Z_OUTPUT_RANGE = Y_OUTPUT_RANGE
# Recommended +1 to have center bin at 0, especially for radially polarized beams
N_Y_OUTPUT = (2 ** 9) + 1
N_Z_OUTPUT = (2 ** 9) + 1
Y_VALS_OUTPUT = np.linspace(-Y_OUTPUT_RANGE, Y_OUTPUT_RANGE, N_Y_OUTPUT)
Z_VALS_OUTPUT = np.linspace(-Z_OUTPUT_RANGE, Z_OUTPUT_RANGE, N_Z_OUTPUT)

# TIME DIMENSION
T_RANGE = 1000
N_T = 2 ** 14

TIMES = np.linspace(-T_RANGE, T_RANGE, N_T)
TIMES -= 0.000001
DT = TIMES[1] - TIMES[0]
OMEGAS = np.fft.fftshift(np.fft.fftfreq(len(TIMES), DT / (2*np.pi)))
OMEGAS -= 0.0000001

SIM_DIRECTORY = "./"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "mem_files/"

# Laser Parameters
WAVELENGTH = 0.800
REF_FREQ = (2*np.pi*constants.C_SPEED) / (WAVELENGTH * 1.e-6)
OMEGA0 = REF_FREQ * 1e-15  # rad / PHz

POLARIZATION = constants.CIRCULAR_L

SPATIAL_SHAPE = constants.ECHELON
SPATIAL_GAUSSIAN_ORDER = 1
TEMPORAL_SHAPE = constants.GAUSSIAN_T
TEMPORAL_GAUSSIAN_ORDER = 1
PHASE_OFFSET = 0.

SPOT_SIZE = 0.6
PULSE_FWHM = 30.

WAIST_IN = 20e4
DELTAX = 0 * WAIST_IN

OUTPUT_DISTANCE_FROM_FOCUS = -38.5

NORMALIZE_TO_A0 = False
TOTAL_ENERGY = 212340651.929383

# Advanced Parameters
USE_GRATING_EQ = False
GRATING_SEPARATION = [-102e4]
GRATING_ANGLE_OF_INCIDENCE = [np.deg2rad(22.8)]
ECHELON_DELAY = 70

# Simulation Grid Parameters
LASER_TIME_START = 120.  # fs
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

# Other Simulation Parameters
N0 = np.array([30])
FOIL_THICKNESS = np.array([0.6])
CENTERY = np.array([Y_HEIGHT / 2.])
CENTERZ = np.array([Z_HEIGHT / 2.])


