"""
Fundamental Physical Constants.

All values are from NIST CODATA 2018 and IAU 2015 Resolution B3.
Each constant includes its SI unit and source reference.

These constants are the ground truth for all physics computations
in the simulation. No module should hardcode physical constants
— always import from here.

Milestone: 2 (Physics Constants)
Status: Completed
"""

# Speed of light in vacuum (exact, SI definition)
# Unit: m / s
C = 299792458.0

# Gravitational constant (CODATA 2018)
# Unit: m^3 / (kg * s^2)
G = 6.67430e-11

# Reduced Planck constant (CODATA 2018)
# Unit: J * s
HBAR = 1.054571817e-34

# Boltzmann constant (CODATA 2018)
# Unit: J / K
KB = 1.380649e-23

# Stefan-Boltzmann constant (CODATA 2018)
# Unit: W / (m^2 * K^4)
SIGMA = 5.670374419e-8

# ── Astronomical Constants (IAU 2015 Resolution B3) ─────────────────

# Nominal solar mass (M☉)
# Unit: kg
M_SUN = 1.98847e30

# Nominal solar radius (R☉)
# Unit: m
R_SUN = 6.957e8

# Nominal solar luminosity (L☉)
# Unit: W
L_SUN = 3.828e26

# Year in seconds (Julian year)
# Unit: s
YEAR_SEC = 365.25 * 24.0 * 3600.0
