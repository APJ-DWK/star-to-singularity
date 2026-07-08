"""
Thermodynamics and Radiation Physics.

Blackbody radiation, temperature-to-color mapping, luminosity
calculations, and Wien's displacement law.

Temperature → RGB conversion uses Tanner Helland's algorithm
which approximates the Planck blackbody spectrum mapping to sRGB
for temperatures from 1000 K to 40000 K.

Milestone: 3 (Physics Engine)
Status: Completed (Color Temperature Mapping)
"""

import numpy as np


def temperature_to_rgb(temp_k):
    """
    Convert a blackbody temperature in Kelvin to an RGB color vector.

    Based on Tanner Helland's algorithm (2013) approximating blackbody
    radiation in the sRGB color space.

    Args:
        temp_k: Temperature in Kelvin (valid range: 1000 K to 40000 K).

    Returns:
        rgb: A numpy array [r, g, b] with values in the range [0.0, 1.0].
    """
    # Clamp temperature to algorithm limits
    temp = np.clip(temp_k, 1000.0, 40000.0)
    t = temp / 100.0

    # Calculate Red
    if t <= 66.0:
        r = 1.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592) / 255.0

    # Calculate Green
    if t <= 66.0:
        g = (99.4708025861 * np.log(t) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492) / 255.0

    # Calculate Blue
    if t >= 66.0:
        b = 1.0
    else:
        if t <= 19.0:
            b = 0.0
        else:
            b = (138.5177312231 * np.log(t - 10.0) - 305.0447927307) / 255.0

    # Clamp results to [0.0, 1.0]
    rgb = np.clip(np.array([r, g, b]), 0.0, 1.0)
    return rgb
