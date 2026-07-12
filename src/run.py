#!/usr/bin/env python3
"""
STAR TO SINGULARITY
An Interactive Journey Through the Life Cycle of a Black Hole

Entry point for the simulation application.

Usage:
    python run.py

Requirements:
    pip install -r requirements.txt
"""

import sys


def main():
    """Launch the Star-to-Singularity simulation."""

    # Verify Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or later is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)

    # Import here so dependency errors are caught after version check
    try:
        import taichi as ti
    except ImportError:
        print("Error: Taichi is not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        import numpy as np
    except ImportError:
        print("Error: NumPy is not installed.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Print startup banner
    print("=" * 60)
    print("  STAR TO SINGULARITY")
    print("  An Interactive Journey Through the Life Cycle")
    print("  of a Black Hole")
    print("=" * 60)
    print()
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Taichi  : {ti.__version__}")
    print(f"  NumPy   : {np.__version__}")
    print()

    # Launch application
    from src.app import Application
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
