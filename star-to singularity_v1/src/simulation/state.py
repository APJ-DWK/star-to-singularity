from enum import Enum, auto
import config
from src.physics.constants import R_SUN, G, C, M_SUN  # Fixed: Import from constants directly

class SimulationPhase(Enum):
    STELLAR_LIFE = auto()       # Main sequence / Supergiant phase
    CORE_COLLAPSE = auto()      # Supernova explosion & collapse event
    BLACK_HOLE_STABLE = auto()  # Active accretion disk & lensing
    HAWKING_EVAPORATION = auto()# Final decay phase

class SimulationState:
    """
    Centralized, thread-safe simulation state parameters.
    Tracks state values accessible by both the physics engine and UI.
    """
    def __init__(self):
        # Current stage state
        self.phase = SimulationPhase.STELLAR_LIFE
        self.is_paused = False
        self.time_scale = config.DEFAULT_TIME_SCALE
        self.elapsed_time = 0.0  # Normalized simulation time
        
        # Stellar / Compact Object Properties
        self.initial_mass_msun = config.DEFAULT_STELLAR_MASS
        self.current_mass_msun = config.DEFAULT_STELLAR_MASS
        self.radius_m = R_SUN * 5.0  # Fixed: Using the imported physical constant directly
        
        # Derived Dynamic Metrics
        self.schwarzschild_radius_m = 0.0
        self.core_temperature_k = 1.57e7 # Baseline core temp
        
        # Recalculate static relational metrics on boot
        self.update_derived_values()

    def update_derived_values(self):
        """Calculates derived metrics like Schwarzschild Radius: $R_s = \frac{2GM}{c^2}$"""
        mass_kg = self.current_mass_msun * M_SUN
        self.schwarzschild_radius_m = (2.0 * G * mass_kg) / (C ** 2)