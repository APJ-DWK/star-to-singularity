"""
Main Application Controller for Star-to-Singularity.

Initializes Taichi, creates the simulation window, and runs the
main loop. Coordinates the physics engine, renderer, UI, and
education systems.
"""

import time
import numpy as np
import taichi as ti

import config
from src.renderer.renderer import Renderer
from src.simulation.engine import SimulationEngine
from src.simulation.state import SimulationState
from src.ui.controls import InteractiveControls
from src.ui.graphs import RealTimeGraphs
from src.ui.hud import HUD
from src.ui.overlays import EducationalOverlays


class Application:
    """
    Top-level application controller.

    Responsibilities:
        - Initialize Taichi runtime
        - Create the GUI window
        - Instantiate and wire together all subsystems
        - Run the main simulation loop
        - Handle user input (keyboard shortcuts and mouse camera orbit)
    """

    def __init__(self):
        self._init_taichi()
        self.state = SimulationState()
        self.engine = SimulationEngine(self.state)
        self.renderer = Renderer(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        # UI Subsystems
        self.controls = InteractiveControls(self.state, self.engine.phase_manager)
        self.hud = HUD(self.state)
        self.graphs = RealTimeGraphs(self.state)
        self.overlays = EducationalOverlays(self.state)

        # Application state
        self.window = None
        self.running = False
        self.last_time = 0.0

        # Input state
        self.last_mouse_pos = np.array([0.5, 0.5], dtype=float)
        self.is_dragging = False

        # Options
        self.ambient_rotation_enabled = False

    def _init_taichi(self):
        """Initialize Taichi with GPU architecture fallback to CPU."""
        arch = ti.cpu
        if config.TAICHI_ARCH == "gpu":
            arch = ti.gpu

        try:
            ti.init(arch=arch)
            print(f"  [Taichi] Initialized on architecture: {config.TAICHI_ARCH.upper()}")
        except Exception as e:
            ti.init(arch=ti.cpu)
            print(f"  [Taichi] GPU unavailable. Fallen back to CPU. Details: {e}")

    def run(self):
        """
        Main application frame loop.

        Runs the physics updates, projects stars on the GPU, renders the frame,
        composites UI overlays, and manages user input at 60 FPS (vsync).
        """
        self.window = ti.ui.Window(
            name=config.WINDOW_TITLE,
            res=(config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            vsync=True,
        )
        canvas = self.window.get_canvas()
        gui = self.window.get_gui()
        self.running = True

        print("  [Application] Window initialized.")
        print("  [Application] Loop controls: SPACE to play/pause, R to reset, ESC to quit.")
        print("  [Application] Camera controls: Left-click and DRAG mouse to rotate view.")

        self.last_time = time.perf_counter()

        while self.running:
            # 1. Update timings
            current_time = time.perf_counter()
            real_dt = current_time - self.last_time
            self.last_time = current_time

            # Compute and show FPS in window title
            fps = 1.0 / max(real_dt, 1e-5)
            self.window.title = f"{config.WINDOW_TITLE} | FPS: {fps:.1f}"

            # 2. Process Input events
            self._handle_input()

            # 3. Update Physics
            # Cap real_dt to avoid large time jumps during lag spikes
            capped_dt = min(real_dt, 0.1)
            self.engine.update(capped_dt)

            # 4. Camera Ambient Rotation (slow drift if not user orbiting)
            if self.ambient_rotation_enabled and not self.is_dragging and not self.state.paused:
                self.renderer.camera.yaw += 0.04 * capped_dt

            # 5. Render Scene (GPU parallel projection and canvas composite)
            # Clear canvas to dark space color
            canvas.set_background_color((0.01, 0.01, 0.02))
            self.renderer.render(self.state, canvas)

            # 6. Render UI Panels
            self.controls.render(gui)
            self.hud.render(gui)
            self.graphs.render(gui)
            self.overlays.render(gui)

            # Present to screen
            self.window.show()

        print("  [Application] Window closed cleanly. Exiting.")

    def _handle_input(self):
        """Process keyboard shortcuts and mouse rotation drag events."""
        # ── Window Exit ──────────────────────────────────────────────
        if not self.window.running:
            self.running = False
            return

        # ── Keyboard Controls ────────────────────────────────────────
        if self.window.get_event(ti.ui.PRESS):
            key = self.window.event.key
            if key == ti.ui.ESCAPE:
                self.running = False
            elif key == ' ':
                # Toggle Pause state
                self.state.paused = not self.state.paused
                print(f"  [Input] Playback toggled. Paused = {self.state.paused}")
            elif key in ['r', 'R']:
                # Reset simulation state
                self.state.reset()
                self.renderer.camera.yaw = 0.0
                self.renderer.camera.pitch = 0.0
                print("  [Input] Simulation state reset.")

        # ── Mouse Drag Orbit Controls ────────────────────────────────
        mouse_pos = np.array(self.window.get_cursor_pos(), dtype=float)

        # Taichi check for left-click
        if self.window.is_pressed(ti.ui.LMB):
            if not self.is_dragging:
                # Drag started this frame
                self.is_dragging = True
                self.last_mouse_pos = mouse_pos
            else:
                # Drag continuing: calculate drag delta
                delta = mouse_pos - self.last_mouse_pos
                self.last_mouse_pos = mouse_pos

                # Adjust camera view angles based on screen drag
                # Sensitivity scaling: yaw (horizontal) and pitch (vertical)
                sensitivity = 4.0
                self.renderer.camera.yaw -= delta[0] * sensitivity
                self.renderer.camera.pitch = np.clip(
                    self.renderer.camera.pitch - delta[1] * sensitivity,
                    -np.pi / 2.2,  # Limit look-up angle to avoid gimbal lock/flip
                    np.pi / 2.2   # Limit look-down angle
                )
        else:
            self.is_dragging = False
