"""
Application-level configuration for Star-to-Singularity.

This module contains display, rendering, and simulation configuration
parameters. Physical constants belong in src/physics/constants.py.
"""


# ── Window Configuration ─────────────────────────────────────────────
WINDOW_TITLE = "Star to Singularity — Black Hole Simulation"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# ── Taichi Configuration ─────────────────────────────────────────────
# Taichi backend: 'gpu' tries CUDA/Vulkan/Metal, falls back to CPU
TAICHI_ARCH = "gpu"

# ── Simulation Defaults ──────────────────────────────────────────────
# Default initial stellar mass in solar masses (M_sun)
# Range restricted to 8–50 M_sun (masses that produce black holes)
DEFAULT_STELLAR_MASS = 20.0
MIN_STELLAR_MASS = 8.0
MAX_STELLAR_MASS = 50.0


# Default simulation time scale (1.0 = real-time ratio, higher = faster)
DEFAULT_TIME_SCALE = 1.0

# ── Rendering Configuration ──────────────────────────────────────────
# Target frames per second
TARGET_FPS = 60

# Maximum number of particles in the particle system
MAX_PARTICLES = 50000

# Background star count for the starfield
BACKGROUND_STAR_COUNT = 2000
