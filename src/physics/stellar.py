"""
Stellar Structure and Main-Sequence Evolution.

Models the physical properties of a massive O/B-type star on the main sequence,
including mass-luminosity, mass-radius, Stefan-Boltzmann temperature calculations,
and main sequence lifetime.

Reference: Carroll & Ostlie, "An Introduction to Modern Astrophysics" (2017)

Milestone: 4 (Stellar Evolution)
Status: Completed
"""

import numpy as np


def compute_stellar_properties(mass_m_sun):
    """
    Compute the physical properties of a massive main sequence star.

    Args:
        mass_m_sun: Mass in solar units (M_sun), expected range: 8.0 to 50.0.

    Returns:
        properties: A dictionary containing:
            - radius (R_sun)
            - luminosity (L_sun)
            - temperature (Kelvin)
            - lifetime (years)
    """
    m = float(mass_m_sun)

    # 1. Mass-Luminosity Relation (L ∝ M^3.5 for intermediate, flatter for massive stars)
    # Calibrated for massive stars where radiation pressure is significant.
    if m >= 20.0:
        # Flatter slope due to electron scattering opacity and radiation pressure dominance
        luminosity = 1.2 * (m ** 3.2)
    else:
        # Standard intermediate-to-high mass slope
        luminosity = 1.0 * (m ** 3.5)

    # 2. Mass-Radius Relation (R ∝ M^0.6 for massive stars on the main sequence)
    # Massive stars have radiative envelopes and convective cores.
    radius = 1.0 * (m ** 0.6)

    # 3. Stefan-Boltzmann Temperature Relation: T = T_sun * (L_ratio / R_ratio^2)^(1/4)
    # Solar effective temperature is 5778 Kelvin
    t_sun = 5778.0
    temperature = t_sun * ((luminosity / (radius ** 2)) ** 0.25)

    # 4. Main Sequence Lifetime: t_ms ≈ 1e10 * (M / L) years
    # Solar main sequence lifetime is ~10 billion years.
    lifetime = 1e10 * (m / luminosity)

    return {
        "radius": radius,
        "luminosity": luminosity,
        "temperature": temperature,
        "lifetime": lifetime
    }
