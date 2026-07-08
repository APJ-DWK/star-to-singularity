"""
Educational Content Manager.

Loads educational text, equation descriptions, and glossary
entries from JSON data files. Provides content keyed by
simulation phase and parameter name.

Milestone: 12 (Educational Overlays)
Status: Completed
"""

import json
import os


class EducationManager:
    """
    Manages loading and querying of curriculum text data.

    Decouples educational copy from code modules.
    """

    def __init__(self, data_path="data/education_content.json"):
        self.data_path = data_path
        self.content = {}
        self.load_content()

    def load_content(self):
        """Load JSON data or initialize with fallback content."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.content = json.load(f)
                    print(f"  [Education] Loaded content from {self.data_path}")
                    return
            except Exception as e:
                print(f"  [Education] Error loading JSON: {e}")

        # Fallback dictionary if file not found or corrupted
        self.content = {
            "phases": {
                "stellar_birth": {
                    "title": "Stellar Evolution",
                    "text": "Stars form from nebulae and burn hydrogen via nuclear fusion. Outward radiation pressure balances inward gravity."
                },
                "stellar_death": {
                    "title": "Core Collapse",
                    "text": "When nuclear fuel is exhausted, the star's core collapses under gravity as radiation pressure drops to zero."
                },
                "supernova": {
                    "title": "Supernova Explosion",
                    "text": "The outer layers of the star are blown away in a dramatic shockwave, leaving behind a compact core remnant."
                },
                "black_hole": {
                    "title": "Black Hole Formation",
                    "text": "If the core remnant exceeds the TOV limit, it collapses completely into a singularity, forming a Schwarzschild black hole."
                }
            }
        }

    def get_phase_info(self, phase_name):
        """Retrieve the educational title and text for a given phase."""
        phases_data = self.content.get("phases", {})
        phase_info = phases_data.get(phase_name, {})
        title = phase_info.get("title", phase_name.replace("_", " ").title())
        text = phase_info.get("text", "No detailed information available for this phase.")
        return title, text
