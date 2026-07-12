"""
Simulation Engine.

Coordinates physics updates each frame. Calls the appropriate
physics modules based on the current simulation phase and updates
the simulation state.

Milestone: 3 (Physics Engine)
Status: Completed (Basic Coordinator)
"""

import numpy as np
from src.physics.collapse import calculate_collapse_physics
from src.simulation.phases import PhaseManager


class SimulationEngine:
    """
    Main driver of the physical simulation state.

    Responsible for advancing the simulation clock and invoking
    physics computations depending on the active phase.
    """

    def __init__(self, state):
        self.state = state
        self.phase_manager = PhaseManager(state)

    def update(self, real_dt):
        """
        Advance the physics simulation.

        Args:
            real_dt: The actual wall-clock elapsed time since the last frame.
        """
        # If simulation is paused, dt = 0.0, but we still update the HUD/camera
        if self.state.paused:
            self.state.dt = 0.0
            return

        # Calculate simulation-scale dt
        sim_dt = real_dt * self.state.time_scale
        self.state.dt = sim_dt
        self.state.time += sim_dt

        # ── Call Physics Updates Based on Current Phase ──────────────────────
        # (Physics modules are stubbed for now and will be populated in Milestones 4-10)
        if self.state.current_phase == "stellar_birth":
            self._update_stellar_birth(sim_dt)
        elif self.state.current_phase == "stellar_death":
            self._update_stellar_death(sim_dt)
        elif self.state.current_phase == "supernova":
            self._update_supernova(sim_dt)
        elif self.state.current_phase == "black_hole":
            self._update_black_hole(sim_dt)

        # Check for automatic phase transition conditions
        self.phase_manager.check_automatic_transitions()

    def _update_stellar_birth(self, dt):
        """Physics update for the stellar birth and evolution phase."""
        from src.physics.stellar import compute_stellar_properties

        # 1. Compute physical properties based on current mass
        props = compute_stellar_properties(self.state.stellar_mass)

        self.state.stellar_radius = props["radius"]
        self.state.stellar_luminosity = props["luminosity"]
        self.state.stellar_temperature = props["temperature"]

        # 2. Advance age and deplete hydrogen core fuel
        lifetime = props["lifetime"]
        
        # We design the main sequence to take 25 seconds of visual simulation time at time_scale=1.0
        target_ms_seconds = 25.0
        age_increment_years = (dt / target_ms_seconds) * lifetime
        
        self.state.stellar_age = min(self.state.stellar_age + age_increment_years, lifetime)
        self.state.stellar_hydrogen = max(0.0, 1.0 - (self.state.stellar_age / lifetime))

    def _update_stellar_death(self, dt):
        """Physics update for red supergiant expansion and core collapse phase."""
        import numpy as np
        from src.physics.stellar import compute_stellar_properties
        from src.physics.collapse import calculate_collapse_physics

        # Let's make the red supergiant + collapse phase take 12 seconds in visual simulation time
        target_death_seconds = 8.0
        self.state.phase_progress = min(1.0, self.state.phase_progress + dt / target_death_seconds)

        props = compute_stellar_properties(self.state.stellar_mass)
        r_main = props["radius"]
        t_main = props["temperature"]

        # We divide the phase into:
        # - progress [0.0, 0.7]: Red Supergiant Expansion
        # - progress [0.7, 1.0]: Core Instability & Core Collapse
        if self.state.phase_progress < 0.7:
            t_exp = self.state.phase_progress / 0.7
            
            # Radius expands up to 30x the main sequence radius (Red Supergiant stage)
            # Smoothly expand using a cubic ease-out shape
            scale_factor = 1.0 + (3.0 - 1.0) * (t_exp ** 1.5)
            self.state.stellar_radius = r_main * scale_factor
            
            # Temperature drops to ~3200 K (Red Supergiant temperature)
            self.state.stellar_temperature = t_main + (3200.0 - t_main) * (1.0 - (1.0 - t_exp) ** 2.0)
            
            # Luminosity = R^2 * (T/5778)^4
            r_ratio = self.state.stellar_radius
            t_ratio = self.state.stellar_temperature / 5778.0
            self.state.stellar_luminosity = (r_ratio ** 2.0) * (t_ratio ** 4.0)
            
            # Age continues past main sequence into supergiant lifetime (~10% extension)
            self.state.stellar_age = props["lifetime"] + (0.1 * props["lifetime"]) * self.state.phase_progress
            
            # Core parameters are pre-collapse
            self.state.core_density = 1.5e5
            self.state.core_temperature = 1.2e8
        else:
            t_collapse = (self.state.phase_progress - 0.7) / 0.3
            
            # Rapid core collapse & outer shell instability (pulsations)
            pulsation = 1.0 + 0.001 * np.sin(self.state.time * 6.0)
            r_supergiant = r_main * 3.0
            self.state.stellar_radius = r_supergiant * pulsation
            print("Red Supergiant Radius =", self.state.stellar_radius)
            self.state.stellar_temperature = 3200.0
            
            r_ratio = self.state.stellar_radius
            t_ratio = self.state.stellar_temperature / 5778.0
            self.state.stellar_luminosity = (r_ratio ** 2.0) * (t_ratio ** 4.0)
            
            # Compute physical core collapse metrics
            phys = calculate_collapse_physics(t_collapse, self.state.stellar_mass)
            self.state.core_density = phys["core_density"]
            self.state.core_temperature = phys["core_temperature"]
            self.state.remnant_mass = phys["remnant_mass"]

    def _update_supernova(self, dt):
        """Physics update for supernova explosion phase with core collapse and fallback accretion."""
        import numpy as np
        from src.physics.stellar import compute_stellar_properties
        from src.physics.collapse import calculate_collapse_physics, get_remnant_mass
        
        # Supernova phase takes 8 seconds in visual simulation time
        target_supernova_seconds = 8.0
        self.state.phase_progress = min(1.0, self.state.phase_progress + dt / target_supernova_seconds)
        progress = self.state.phase_progress

        props = compute_stellar_properties(self.state.stellar_mass)
        r_main = props["radius"]
        r_supergiant = r_main * 3.0
        remnant_m = get_remnant_mass(self.state.stellar_mass)
        self.state.remnant_mass = remnant_m

        # 1. Supernova flash intensity (dual-peak: first at explosion, second at event horizon formation)
        flash1 = 4.5 * np.exp(-7.5 * progress)
        flash2 = 0.0
        if progress >= 0.60:
            flash2 = 2.5 * np.exp(-80.0 * ((progress - 0.78) ** 2.0))
        self.state.supernova_flash = flash1 + flash2
        
        # 2. Envelope expansion (gas shell expands rapidly outwards)
        if progress < 0.25:
            expand = progress / 0.25
            self.state.stellar_radius = (
                r_supergiant * (1.0 + 0.30 * expand)
            )
        else:
            t = (progress - 0.25) / 0.75
            self.state.stellar_radius = (
                r_supergiant * (1.30 + 0.45 * t)
            )
        print("Supernova Radius =", self.state.stellar_radius)
        
        # ----------------------------------------------------------
        # 3. Remnant Core Collapse
        # ----------------------------------------------------------

        if progress < 0.30:

            # Core hidden behind dense envelope
            self.state.core_radius = 0.0
            self.state.core_density = 1e6
            self.state.core_temperature = 3e9

        else:

            # Collapse progress
            t = (progress - 0.30) / 0.70
            t = min(max(t, 0.0), 1.0)

            # Slow → Fast collapse
            collapse = t * t * (3.0 - 2.0 * t)

            r_start = 0.15 * r_main
            r_end = 0.04

            self.state.core_radius = (
                r_start * (1.0 - collapse)
                + r_end * collapse
            )

            phys = calculate_collapse_physics(
                t,
                self.state.stellar_mass
            )

            self.state.core_density = phys["core_density"]
            self.state.core_temperature = phys["core_temperature"]

        # 4. Cooling of expanding ejecta envelope
        self.state.stellar_temperature = 100000.0 * np.exp(-3.0 * progress) + 3000.0
        
        # 5. Supernova luminosity (decays over time)
        self.state.stellar_luminosity = 2.0e9 * np.exp(-3.5 * progress)

    def _update_black_hole(self, dt):
        """Physics update for event horizon, accretion, and lensing phase."""
        import numpy as np
        from src.physics.collapse import get_remnant_mass
        from src.physics.constants import M_SUN, G, C
        
        remnant_m = get_remnant_mass(self.state.stellar_mass)
        self.state.remnant_mass = remnant_m
        
        # Calculate black hole parameters
        self.state.black_hole_mass = remnant_m * M_SUN
        
        # Schwarzschild radius: r_s = 2GM/c^2
        self.state.schwarzschild_radius = 2.0 * G * self.state.black_hole_mass / (C ** 2.0)
        self.state.photon_sphere_radius = 1.5 * self.state.schwarzschild_radius
        self.state.isco_radius = 3.0 * self.state.schwarzschild_radius
        
        # Accretion rate simulation (accretion grows to a steady state)
        self.state.accretion_rate = 1.2e18 * (1.0 - np.exp(-0.2 * self.state.time))
        self.state.accretion_inner = self.state.isco_radius
        self.state.accretion_outer = 12.0 * self.state.schwarzschild_radius
        self.state.accretion_max_temp = 1.5e7
