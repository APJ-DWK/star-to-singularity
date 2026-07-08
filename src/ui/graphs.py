"""
Real-Time Data Graphs.

Canvas-drawn plots tracking physical quantities over simulation
time (e.g., mass vs time, luminosity vs time, HR diagram position).

Milestone: 11 (Interactive Controls)
Status: Completed (Basic stub panel)
"""


class RealTimeGraphs:
    """
    Renders plot panels for tracking physical parameters over time.

    Enables students to visualize scientific dependencies dynamically.
    """

    def __init__(self, state):
        self.state = state

    def render(self, gui):
        """
        Draw the graphs tracking panel.

        Args:
            gui: The Taichi Window GUI instance.
        """
        # Panel position: bottom-right side, below HUD
        gui.begin("Real-Time Graphs", 0.73, 0.49, 0.25, 0.49)

        gui.text("Astrophysics Diagnostics:")
        gui.text("(Data plots will populate in physics milestones)")

        # In later milestones: render custom lines/rectangles on Canvas
        # or use GUI text/widgets to plot basic parameters or HR track.

        gui.end()
