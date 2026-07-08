"""
Simulation Engine.

Coordinates physics updates each frame. Calls the appropriate
physics modules based on the current simulation phase and updates
the simulation state.

Milestone: 3 (Physics Engine)
Status: Completed (Basic Coordinator)
"""

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

        # If hydrogen fuel is depleted, freeze simulation time expansion
        if self.state.stellar_hydrogen <= 0.0:
            self.state.paused = True
            print("  [Simulation] Hydrogen exhausted! Stellar core collapse imminent.")

    def _update_stellar_death(self, dt):
        """Physics update for core collapse phase."""
        # Stub: to be populated in Milestone 5.
        pass

    def _update_supernova(self, dt):
        """Physics update for supernova explosion phase."""
        # Stub: to be populated in Milestone 6.
        pass

    def _update_black_hole(self, dt):
        """Physics update for event horizon, accretion, and lensing phase."""
        # Stub: to be populated in Milestones 7-10.
        pass
