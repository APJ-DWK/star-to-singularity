"""
Educational Text Overlays.

Renders educational annotations, phase descriptions, and
contextual explanations on the simulation canvas.

Milestone: 12 (Educational Overlays)
Status: Completed (Basic stub panel)
"""


class EducationalOverlays:
    """
    Manages text annotations overlaying the simulation.

    Displays academic context and notes on physical approximations.
    """

    def __init__(self, state):
        self.state = state

    def render(self, gui):
        """
        Draw the educational annotation container.

        Args:
            gui: The Taichi Window GUI instance.
        """
        # ── 1. General Notes Panel ───────────────────────────────────────────
        gui.begin("Educational Context", 0.02, 0.70, 0.25, 0.28)

        gui.text("Physics Notes:")

        if self.state.current_phase == "stellar_birth":
            gui.text("Massive Star Evolution:")
            gui.text("A star with mass M > 8 M_sun will end")
            gui.text("its life in a supernova collapse.")
            gui.text("It is held in hydrostatic equilibrium")
            gui.text("by fusion radiation pressure.")

        elif self.state.current_phase == "stellar_death":
            gui.text("Gravity Dominance:")
            gui.text("As nuclear fuel exhausts, radiation")
            gui.text("pressure drops. Gravity collapses the")
            gui.text("core past the TOV limit.")

        elif self.state.current_phase == "supernova":
            gui.text("Supernova Shockwave:")
            gui.text("Infalling layers bounce off the dense")
            gui.text("degenerate neutron core, causing a")
            gui.text("cataclysmic explosion.")

        elif self.state.current_phase == "black_hole":
            gui.text("General Relativity Regime:")
            gui.text("Light rays are bent around the black")
            gui.text("hole. The photon sphere lies at")
            gui.text("r = 1.5 r_s. The event horizon is")
            gui.text("the boundary at r = r_s.")

        gui.end()

        # ── 2. Active Structure Labels Panel ──────────────────────────────────
        gui.begin("Active Structures Guide", 0.32, 0.80, 0.36, 0.18)
        
        phase = self.state.current_phase
        progress = self.state.phase_progress
        
        gui.text("Active Structures:")
        gui.text("-" * 38)
        
        if phase == "stellar_birth":
            gui.text("  [Stellar Core] - Hydrogen Fusion Zone")
            gui.text("  [Atmosphere] - Hydrostatic Radiative Envelope")
            gui.text("  [Solar Prominences] - Ballistic Magnetic Loops")
        elif phase == "stellar_death":
            gui.text("  [Red Giant Envelope] - Expanded gas shell (R ~ 100 R_main)")
            gui.text("  [Stellar Core] - Silicon Burning Phase")
            gui.text("  [Core Collapse] - Gravity exceeds Degeneracy Pressure")
        elif phase == "supernova":
            if progress < 0.3:
                gui.text("  [Shockwave Front] - Expanding Ejecta shockwave layer")
                gui.text("  [Core Bounce] - Blinding neutrino pressure flash")
            else:
                gui.text("  [Shockwave Front] - Expanding outer gas shell")
                gui.text("  [Collapsing Remnant Core] - Shrinking core remnant")
                gui.text("  [Infalling Matter] - Gravitational fallback accretion")
        elif phase == "black_hole":
            gui.text("  [Event Horizon] - Schwarzschild boundary (r = r_s)")
            gui.text("  [Photon Sphere] - Unstable circular photon orbit (r = 1.5 r_s)")
            gui.text("  [Accretion Disk] - Keplerian sheared hot plasma")
            gui.text("  [Lensing Ring] - Gravitational warping of space-time")
            
        gui.end()
