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
        # Panel position: bottom-left side, below controls
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
