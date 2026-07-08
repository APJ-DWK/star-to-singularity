import taichi as ti
import config
from src.simulation.state import SimulationState, SimulationPhase

class Application:
    """
    Main system orchestrator. Initializes Taichi runtime engine,
    manages UI drawing context, and tracks execution loops.
    """
    def __init__(self):
        # Initialize Taichi environment using configured backend
        arch_type = ti.gpu if config.TAICHI_ARCH == "gpu" else ti.cpu
        ti.init(arch=arch_type, log_level=ti.INFO)
        
        # Instantiate window environment context
        self.window = ti.ui.Window(
            name=config.WINDOW_TITLE,
            res=(config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            fps_limit=config.TARGET_FPS
        )
        
        # Capture context access handles
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        
        # Instantiate systemic states
        self.state = SimulationState()

    def _render_ui(self):
        """Draws layout matching constraints dynamically scaled to window bounds."""
        # Get current window dimensions from configuration
        import config
        padding = 20
        panel_width = 320
        panel_height = 680
        
        # 1. Left Menu: Controls & Configuration
        with self.gui.sub_window("Controls & Configuration", padding, padding, panel_width, panel_height):
            self.gui.text(f"Simulation Status: {'PAUSED' if self.state.is_paused else 'RUNNING'}")
            if self.gui.button("Toggle Pause"):
                self.state.is_paused = not self.state.is_paused
                
            self.gui.text(f"Current Phase: {self.state.phase.name}")
            self.gui.text("--------------------------------")
            
            # Parametric control inputs
            self.gui.text("Stellar Configuration:")
            self.state.initial_mass_msun = self.gui.slider_float(
                "Initial Mass (M_sun)", 
                self.state.initial_mass_msun, 
                config.MIN_STELLAR_MASS, 
                config.MAX_STELLAR_MASS
            )

        # 2. Right Menu: Physical Readouts (Dynamically anchored to the right side)
        right_panel_x = config.WINDOW_WIDTH - panel_width - padding
        with self.gui.sub_window("Physical Readouts", right_panel_x, padding, panel_width, panel_height // 2):
            self.gui.text("System Metrics:")
            self.gui.text(f"Current Mass: {self.state.current_mass_msun:.2f} M☉")
            self.gui.text(f"Est. R_s Radius: {self.state.schwarzschild_radius_m / 1000.0:.2f} km")

        # Dynamic System Metric Readouts Panel
        with self.gui.sub_window("Physical Readouts", 1542, 55, 320, 324):
            self.gui.text(f"Current Mass: {self.state.current_mass_msun:.2f} M☉")
            self.gui.text(f"Est. R_s Radius: {self.state.schwarzschild_radius_m / 1000.0:.2f} km")

    def run(self):
        """Primary execution loop lifecycle driver."""
        while self.window.running:
            # 1. Coordinate and update physics engine frames
            if not self.state.is_paused:
                # Physics step tracking logic will hook here
                self.state.update_derived_values()
                
            # 2. Render background visuals onto the GPU graphics canvas
            self.canvas.set_background_color((0.03, 0.03, 0.05)) # Deep space template blue-black
            
            # 3. Process immediate mode GUI frames
            self._render_ui()
            
            # 4. Swap frame buffers
            self.window.show()
