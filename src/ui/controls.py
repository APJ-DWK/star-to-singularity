"""
Interactive Controls.

Parameter sliders, buttons, and keyboard shortcuts for controlling
the simulation. Uses Taichi GUI's built-in widget system.

Milestone: 11 (Interactive Controls)
Status: Completed (Basic controls panel)
"""

import config


class InteractiveControls:
    """
    Renders and processes user inputs and GUI controls.

    Exposes sliders and buttons to modify simulation state parameters.
    """

    def __init__(self, state):
        self.state = state

    def render(self, gui):
        """
        Draw the control panel widget container.

        Args:
            gui: The Taichi Window GUI instance.
        """
        # Panel position: top-left side, width: 280px, height: almost full screen
        gui.begin("Controls & Configuration", 0.02, 0.02, 0.25, 0.96)

        # ── Time Controls ────────────────────────────────────────────
        gui.text("Simulation Playback")
        if self.state.paused:
            if gui.button("Resume (Space)"):
                self.state.paused = False
        else:
            if gui.button("Pause (Space)"):
                self.state.paused = True

        self.state.time_scale = gui.slider_float(
            "Time Scale", self.state.time_scale, 0.01, 100.0
        )

        gui.text("-" * 28)

        # ── Parameter Controls ───────────────────────────────────────
        gui.text("Physical Parameters")

        # Mass control (can only change mass in the initial evolution phase)
        is_birth = self.state.current_phase == "stellar_birth"
        if is_birth:
            self.state.stellar_mass = gui.slider_float(
                "Stellar Mass (M_sun)",
                self.state.stellar_mass,
                config.MIN_STELLAR_MASS,
                config.MAX_STELLAR_MASS,
            )
        else:
            gui.text(f"Stellar Mass fixed at: {self.state.stellar_mass:.2f} M_sun")

        gui.text("-" * 28)

        # ── Phase Selection ──────────────────────────────────────────
        gui.text("Life Cycle Phases")
        if gui.button("Prev Phase"):
            # Handled via app controller or state machine
            pass
        if gui.button("Next Phase"):
            # Handled via app controller or state machine
            pass

        gui.text("-" * 28)

        if gui.button("Reset Simulation (R)"):
            self.state.reset()


        gui.end()
