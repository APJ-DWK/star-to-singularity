"""
Heads-Up Display (HUD).

Renders real-time physical quantities (mass, radius, temperature,
luminosity, etc.) with proper SI units and scientific notation.

Milestone: 11 (Interactive Controls)
Status: Completed
"""


class HUD:
    """
    Renders a physical data readout panel on the screen.

    Displays active state variables formatted for educational utility.
    """

    def __init__(self, state):
        self.state = state

    def render(self, gui):
        """
        Draw the HUD data readout panel.

        Args:
            gui: The Taichi Window GUI instance.
        """
        # Panel position: top-right side, width: 280px, height: variable
        gui.begin("Physical Readouts", 0.73, 0.02, 0.25, 0.45)

        gui.text("Simulation Status:")
        gui.text(f"  Phase: {self.state.current_phase.replace('_', ' ').title()}")
        gui.text(f"  Time Elapsed: {self.state.time:.2f} s")
        gui.text("-" * 28)

        gui.text("Active Object State:")

        if self.state.current_phase in ["stellar_birth", "stellar_death"]:
            gui.text(f"  Mass: {self.state.stellar_mass:.2f} M_sun")
            gui.text(f"  Radius: {self.state.stellar_radius:.2f} R_sun")
            gui.text(f"  Luminosity: {self.state.stellar_luminosity:.2e} L_sun")
            gui.text(f"  Temp: {self.state.stellar_temperature:.0f} K")
            gui.text(f"  Age: {self.state.stellar_age / 1e6:.3f} Myr")
            
            if self.state.current_phase == "stellar_birth":
                gui.text(f"  Hydrogen: {self.state.stellar_hydrogen * 100.0:.2f}%")
            else:
                gui.text("-" * 28)
                gui.text("Core Instability:")
                gui.text(f"  Core Density: {self.state.core_density:.2e} g/cm3")
                gui.text(f"  Core Temp: {self.state.core_temperature:.2e} K")
                gui.text(f"  Regime: {'Grav. Collapse' if self.state.remnant_mass > 2.2 else 'Neutron Degen.' if self.state.remnant_mass > 1.4 else 'Electron Degen.'}")

        elif self.state.current_phase == "supernova":
            gui.text("Supernova Shockwave:")
            gui.text(f"  Remnant Core: {self.state.remnant_mass:.2f} M_sun")
            gui.text(f"  Envelope Temp: {self.state.stellar_temperature:.0f} K")
            gui.text(f"  Luminosity: {self.state.stellar_luminosity:.2e} L_sun")
            gui.text(f"  Ejecta Size: {self.state.stellar_radius:.1f} R_sun")
            gui.text(f"  Flash Power: {self.state.supernova_flash * 100.0:.1f}%")

        elif self.state.current_phase == "black_hole":
            gui.text(f"  Remnant Mass: {self.state.remnant_mass:.2f} M_sun")
            gui.text(f"  BH Mass: {self.state.black_hole_mass:.2e} kg")
            gui.text(f"  Sch. Radius (r_s): {self.state.schwarzschild_radius / 1e3:.3f} km")
            gui.text(f"  Photon Sphere: {self.state.photon_sphere_radius / 1e3:.3f} km")
            gui.text(f"  ISCO Radius: {self.state.isco_radius / 1e3:.3f} km")

            if self.state.accretion_rate > 0.0:
                gui.text(f"  Accretion Rate: {self.state.accretion_rate:.2e} kg/s")
                gui.text(f"  Disk Temp (max): {self.state.accretion_max_temp:.0f} K")

        gui.end()
