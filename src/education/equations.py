"""
Key Equation Definitions.

Stores the key astrophysics equations used in the simulation,
with their physical explanations and the approximations made.

Milestone: 12 (Educational Overlays)
Status: Completed
"""


class EquationLibrary:
    """Contains descriptions of scientific formulas for display in the UI."""

    EQUATIONS = {
        "schwarzschild_radius": {
            "symbol": "r_s = 2GM / c^2",
            "name": "Schwarzschild Radius",
            "description": "Calculates the radius of the event horizon for a non-rotating black hole.",
            "approximation": "None. This is an exact solution of Einstein's field equations for spherical symmetry."
        },
        "tov_limit": {
            "symbol": "M_TOV ~ 2.1 - 2.5 M_sun",
            "name": "Tolman-Oppenheimer-Volkoff Limit",
            "description": "The upper bound to the mass of cold, non-rotating neutron degenerate matter.",
            "approximation": "Approximate range based on uncertainties in nuclear equation of state at high density."
        },
        "free_fall_time": {
            "symbol": "t_ff = sqrt(3 * pi / (32 * G * rho))",
            "name": "Free-Fall Time",
            "description": "The characteristic time that a gas cloud or stellar core takes to collapse under gravity if pressure forces are negligible.",
            "approximation": "Assumes uniform density, spherical symmetry, and zero internal pressure opposition."
        }
    }

    @classmethod
    def get_equation(cls, eq_id):
        """Retrieve equation data by its ID."""
        return cls.EQUATIONS.get(eq_id, None)
