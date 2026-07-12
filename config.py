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

# ── Phase Duration Configuration (seconds of wall-clock time) ────────
PHASE_DURATION_STELLAR_BIRTH = 25.0
PHASE_DURATION_STELLAR_DEATH = 8.0
PHASE_DURATION_SUPERNOVA = 10.0
PHASE_DURATION_SN_TO_BH = 8.0

# ── Black Hole Renderer ─────────────────────────────────────────────
# Event horizon visual radius in NDC (normalized device coords)
BH_EVENT_HORIZON_RADIUS = 0.04
# Photon ring width as fraction of event horizon radius
BH_PHOTON_RING_WIDTH = 0.03
# Accretion disk inner edge as multiple of event horizon radius
BH_DISK_INNER_MULT = 1.5
# Accretion disk outer edge as multiple of event horizon radius
BH_DISK_OUTER_MULT = 7.0
# Noise octaves for turbulent plasma (lower = faster on Intel UHD)
BH_NOISE_OCTAVES = 3
# Maximum Doppler velocity factor
BH_MAX_DOPPLER_VELOCITY = 0.62

# ── Camera Configuration ─────────────────────────────────────────────
CAMERA_LERP_SPEED = 2.0
CAMERA_DRAG_SENSITIVITY = 4.0
CAMERA_AMBIENT_ROTATION_SPEED = 0.04
CAMERA_DAMPING = 0.92
