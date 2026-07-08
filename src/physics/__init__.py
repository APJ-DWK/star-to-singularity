"""
Physics subpackage.

Contains all physical models, constants, and computations.
The physics layer is the single source of truth for all physical
quantities. No other module may invent or override physical values.

Modules:
    constants       — Fundamental physical constants (NIST/IAU 2015 values)
    stellar         — Stellar structure and main-sequence evolution
    gravity         — Schwarzschild metric, geodesics, gravitational quantities
    collapse        — Core collapse mechanics (Chandrasekhar, TOV, free-fall)
    thermodynamics  — Blackbody radiation, temperature-to-color mapping
    accretion       — Accretion disk physics (Novikov-Thorne thin disk model)
    lensing         — Gravitational lensing and photon sphere
"""
