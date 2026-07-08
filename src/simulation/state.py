"""
Simulation State.

Centralized, single source of truth for all simulation data.
All subsystems read from this state. Only the physics engine
and user input handlers write to it.

Milestone: 3 (Physics Engine)
Status: Completed (Basic Framework)
"""

import config


class SimulationState:
    """
    Holds the complete state of the simulation.

    No computation is performed here; this is a pure data container
    to ensure decoupling between physics, rendering, and UI.
    """

    def __init__(self):
        # ── Time and Controls ────────────────────────────────────────
        self.time = 0.0          # Accumulated simulation time (seconds)
        self.dt = 0.0            # Timestep for the current frame (seconds)
        self.time_scale = config.DEFAULT_TIME_SCALE  # Time acceleration factor
        self.paused = False      # Pause state

        # ── Phase Control ────────────────────────────────────────────
        self.current_phase = "stellar_birth"
        self.phase_progress = 0.0  # Normalized progress within the phase (0.0 to 1.0)
        self.core_density = 0.0     # g/cm^3
        self.core_temperature = 0.0 # Kelvin
        self.supernova_flash = 0.0  # Brightness multiplier for explosion flash
        self.remnant_mass = 0.0     # M_sun
        self.supernova_trigger = False # Spawns particles in renderer when True

        # ── Stellar Parameters ───────────────────────────────────────
        self.stellar_mass = config.DEFAULT_STELLAR_MASS  # Initial mass in M☉
        self.stellar_radius = 1.0       # Solar radii (computed)
        self.stellar_luminosity = 1.0   # Solar luminosities (computed)
        self.stellar_temperature = 5778.0 # Kelvin (computed)
        self.stellar_hydrogen = 1.0     # Remaining core hydrogen fraction (1.0 to 0.0)
        self.stellar_age = 0.0          # Current star age in years

        # ── Black Hole Parameters ────────────────────────────────────
        self.black_hole_mass = 0.0      # In kg
        self.schwarzschild_radius = 0.0 # In meters (r_s = 2GM/c^2)
        self.photon_sphere_radius = 0.0 # In meters (r_ph = 1.5 r_s)
        self.isco_radius = 0.0          # In meters (r_ISCO = 3.0 r_s)

        # ── Accretion Disk Parameters ────────────────────────────────
        self.accretion_rate = 0.0       # In kg/s
        self.accretion_inner = 0.0      # In meters
        self.accretion_outer = 0.0      # In meters
        self.accretion_max_temp = 0.0   # In Kelvin

        # ── Camera State ─────────────────────────────────────────────
        self.camera_zoom = 1.0
        self.camera_center = [0.0, 0.0]

    def reset(self):
        """Reset the simulation state to initial conditions."""
        self.time = 0.0
        self.dt = 0.0
        self.time_scale = config.DEFAULT_TIME_SCALE
        self.paused = False
        self.current_phase = "stellar_birth"
        self.phase_progress = 0.0
        self.core_density = 0.0
        self.core_temperature = 0.0
        self.supernova_flash = 0.0
        self.remnant_mass = 0.0
        self.supernova_trigger = False

        self.stellar_mass = config.DEFAULT_STELLAR_MASS
        self.stellar_radius = 1.0
        self.stellar_luminosity = 1.0
        self.stellar_temperature = 5778.0
        self.stellar_hydrogen = 1.0
        self.stellar_age = 0.0

        self.black_hole_mass = 0.0
        self.schwarzschild_radius = 0.0
        self.photon_sphere_radius = 0.0
        self.isco_radius = 0.0

        self.accretion_rate = 0.0
        self.accretion_inner = 0.0
        self.accretion_outer = 0.0
        self.accretion_max_temp = 0.0

        self.camera_zoom = 1.0
        self.camera_center = [0.0, 0.0]
