"""
Simulation Phase Definitions and Transitions.

Defines the phase state machine:
    stellar_birth → stellar_death → supernova → black_hole

Each phase has entry/exit logic and determines which physics
modules are active and which rendering mode is used.

Milestone: 3 (Physics Engine)
Status: Completed (Basic State Machine)
"""


class Phase:
    """Constants for simulation phases."""
    STELLAR_BIRTH = "stellar_birth"
    STELLAR_DEATH = "stellar_death"
    SUPERNOVA = "supernova"
    BLACK_HOLE = "black_hole"

    ALL_PHASES = [STELLAR_BIRTH, STELLAR_DEATH, SUPERNOVA, BLACK_HOLE]

    PHASE_NAMES = {
        STELLAR_BIRTH: "Stellar Birth & Evolution",
        STELLAR_DEATH: "Core Collapse",
        SUPERNOVA: "Supernova Explosion",
        BLACK_HOLE: "Black Hole & Accretion",
    }


class PhaseManager:
    """Coordinates phase transitions and lifecycle callbacks."""

    def __init__(self, state):
        self.state = state
        # Order of phases in the simulation
        self.phase_order = Phase.ALL_PHASES

    def get_current_name(self):
        """Get the human-readable name of the current phase."""
        return Phase.PHASE_NAMES.get(self.state.current_phase, "Unknown Phase")

    def set_phase(self, phase_name):
        """Set the active simulation phase and trigger lifecycle updates."""
        if phase_name in self.phase_order:
            print(f"[Phase] Transitioning from {self.state.current_phase} to {phase_name}")
            self.state.current_phase = phase_name
            self.state.phase_progress = 0.0
            return True
        return False

    def next_phase(self):
        """Advance to the next phase in the sequence."""
        try:
            curr_idx = self.phase_order.index(self.state.current_phase)
            if curr_idx < len(self.phase_order) - 1:
                return self.set_phase(self.phase_order[curr_idx + 1])
        except ValueError:
            pass
        return False

    def prev_phase(self):
        """Revert to the previous phase in the sequence."""
        try:
            curr_idx = self.phase_order.index(self.state.current_phase)
            if curr_idx > 0:
                return self.set_phase(self.phase_order[curr_idx - 1])
        except ValueError:
            pass
        return False

    def check_automatic_transitions(self):
        """
        Check state parameters to see if phase transitions should occur automatically.

        For example:
        - Once hydrogen is fully depleted in stellar_birth, transition to stellar_death.
        - Once core collapse completes in stellar_death, trigger supernova.
        - Once supernova explosion finishes, form black_hole.
        """
        # This will be fully implemented in physics milestone integrations.
        # For now, it remains a stub called by the simulation engine.
        pass
