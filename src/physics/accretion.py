"""
Accretion Disk Physics.

Models a geometrically thin, optically thick accretion disk around
a Schwarzschild black hole using the Novikov-Thorne model.

Key quantities:
- Inner edge at ISCO (r = 6GM/c² for Schwarzschild)
- Temperature profile: T(r) ∝ r^(-3/4) with inner boundary correction
- Radiative efficiency: η ≈ 1 - √(1 - 2/(3 r_ISCO/r_s))
- Orbital velocities (Keplerian, with GR corrections near ISCO)
- Doppler beaming for approaching/receding disk material

Reference: Novikov, I.D. & Thorne, K.S. (1973). "Astrophysics of
           Black Holes." In: Black Holes (Les Houches).

Milestone: 8 (Accretion Disk)
Status: STUB — will be implemented in Milestone 8.
"""
