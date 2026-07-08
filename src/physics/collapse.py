"""
Core Collapse Mechanics.

Models the collapse of a massive star's iron core when nuclear fusion ceases:
- Chandrasekhar limit (1.4 M_sun) for electron degeneracy pressure.
- Tolman-Oppenheimer-Volkoff (TOV) limit (~2.1-2.5 M_sun) for neutron degeneracy pressure.
- Core density and temperature escalation.
- Free-fall collapse timescale calculations.

Reference:
- Shapiro & Teukolsky, "Black Holes, White Dwarfs, and Neutron Stars" (1983)
- Carroll & Ostlie, "An Introduction to Modern Astrophysics" (2017)

Milestone: 5 (Core Collapse)
Status: Completed
"""

import numpy as np
from src.physics.constants import G, M_SUN, R_SUN


def get_remnant_mass(initial_mass_m_sun):
    """
    Compute the mass of the collapsed core remnant based on initial stellar mass.
    
    A fraction of the initial mass is lost in the supernova envelope ejection,
    leaving behind a compact core.
    
    Args:
        initial_mass_m_sun: Initial mass of the star in solar masses.
        
    Returns:
        remnant_mass_m_sun: Remnant core mass in solar masses.
    """
    # Simple astrophysically-inspired relation:
    # A 20 M_sun star leaves a ~3-5 M_sun black hole core remnant.
    # An 8 M_sun star leaves a ~1.4 M_sun neutron star remnant.
    if initial_mass_m_sun < 15.0:
        return 1.4 + 0.1 * (initial_mass_m_sun - 8.0)
    else:
        return 2.1 + 0.15 * (initial_mass_m_sun - 15.0)


def calculate_collapse_physics(progress, initial_mass_m_sun):
    """
    Calculate the physical state of the stellar core during collapse.
    
    Args:
        progress: Normalized progress of the collapse phase (0.0 to 1.0).
        initial_mass_m_sun: Initial mass of the star in solar masses.
        
    Returns:
        dict containing:
            - core_density: In g/cm^3
            - core_temperature: In Kelvin
            - free_fall_time: In seconds
            - degeneracy_pressure_type: Description string of dominant pressure
    """
    remnant_m = get_remnant_mass(initial_mass_m_sun)
    
    # 1. Core Density (g/cm^3)
    # Starts around iron core density (~1e6 g/cm^3) and increases exponentially
    # past the Chandrasekhar density to nuclear density (~1e14 g/cm^3).
    start_density = 1e6
    end_density = 2e14 if remnant_m < 2.5 else 1e16
    core_density = start_density * np.exp(progress * np.log(end_density / start_density))
    
    # 2. Core Temperature (Kelvin)
    # Starts around silicon burning temperature (~3e9 K) and goes up to ~1e11 K
    start_temp = 3.0e9
    end_temp = 5.0e11
    core_temperature = start_temp * np.exp(progress * np.log(end_temp / start_temp))
    
    # 3. Free-fall collapse timescale (t_ff = sqrt(3*pi / (32 * G * rho)))
    # Note: conversion of core_density from g/cm^3 to kg/m^3 (multiply by 1000)
    rho_si = core_density * 1000.0
    free_fall_time = np.sqrt((3.0 * np.pi) / (32.0 * G * rho_si))
    
    # 4. Identify degeneracy regime
    # Chandrasekhar Limit is 1.4 M_sun (electron degeneracy pressure)
    # TOV Limit is ~2.1-2.5 M_sun (neutron degeneracy pressure)
    if remnant_m <= 1.4:
        regime = "Electron Degeneracy"
    elif remnant_m <= 2.2:
        regime = "Neutron Degeneracy"
    else:
        regime = "Gravitational Collapse"
        
    return {
        "core_density": core_density,
        "core_temperature": core_temperature,
        "free_fall_time": free_fall_time,
        "regime": regime,
        "remnant_mass": remnant_m
    }
