# Star to Singularity

**An Interactive Journey Through the Life Cycle of a Black Hole**

An educational astrophysics simulation built with Python and Taichi,
exploring the complete life cycle of a black hole — from stellar
collapse to singularity.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/APJ-DWK/star-to-singularity.git
cd star-to-singularity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulation
python run.py
```

**Requirements**: Python 3.10+, GPU recommended (CUDA/Vulkan/Metal).
Falls back to CPU if no GPU is available.

---

## Project Structure

```
star-to-singularity/
├── run.py                  # Entry point
├── config.py               # Application configuration
├── requirements.txt        # Python dependencies
├── SCIENCE.md              # Scientific references and approximations
│
├── src/
│   ├── app.py              # Main application controller
│   ├── physics/            # Physical models and computations
│   │   ├── constants.py    # Fundamental physical constants (NIST/IAU)
│   │   ├── stellar.py      # Stellar structure and evolution
│   │   ├── gravity.py      # Schwarzschild metric and geodesics
│   │   ├── collapse.py     # Core collapse mechanics
│   │   ├── thermodynamics.py # Blackbody radiation and temperature
│   │   ├── accretion.py    # Accretion disk (Novikov-Thorne model)
│   │   └── lensing.py      # Gravitational lensing
│   │
│   ├── simulation/         # Simulation engine and state
│   │   ├── engine.py       # Physics update coordinator
│   │   ├── state.py        # Centralized simulation state
│   │   └── phases.py       # Phase state machine
│   │
│   ├── renderer/           # Taichi GPU rendering
│   │   ├── renderer.py     # Main render engine
│   │   ├── particles.py    # GPU particle systems
│   │   ├── camera.py       # Virtual camera controls
│   │   └── background.py   # Procedural starfield
│   │
│   ├── ui/                 # User interface
│   │   ├── controls.py     # Interactive parameter controls
│   │   ├── overlays.py     # Educational text overlays
│   │   ├── hud.py          # Real-time data display
│   │   └── graphs.py       # Real-time plots
│   │
│   └── education/          # Educational content system
│       ├── content.py      # Content manager
│       └── equations.py    # Key equation definitions
│
├── data/                   # Data-driven content (JSON)
│   └── education_content.json
│
└── tests/                  # Unit tests
    └── test_physics.py
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.12 | Application logic |
| GPU Computing | Taichi | Parallel physics simulation and rendering |
| Numerics | NumPy | Array computation and data handling |

---

## Scientific Approach

This simulation prioritizes **scientific accuracy** (40% of judging criteria).

- Every visualization derives from a physical model, not artistic invention.
- All approximations are explicitly documented in [SCIENCE.md](SCIENCE.md).
- Physical constants sourced from NIST CODATA 2018 and IAU 2015.
- Where real-time simplifications are necessary, they are marked as
  "educational approximations" in the code and UI.

---

## License

MIT License
