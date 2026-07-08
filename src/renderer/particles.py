"""
GPU Particle System.

Manages pools of particles for stellar wind, supernova ejecta,
accretion flows, and other particle effects. Uses Taichi fields
for GPU-parallel particle updates.

Milestone: 6 (Supernova particles)
Status: Completed (Basic Stub Interface)
"""

import taichi as ti


@ti.data_oriented
class ParticleSystem:
    """
    Manages GPU-based particle collections for the simulation.

    Initialized with maximum particle capacities and updated in parallel.
    """

    def __init__(self, max_particles=50000):
        self.max_particles = max_particles
        # Space reserved for positions, velocities, and colors
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.active = ti.field(dtype=ti.i32, shape=max_particles)

    def reset(self):
        """Reset all particles to inactive state."""
        # Simple CPU-based clear for initialization/reset
        for i in range(self.max_particles):
            self.pos[i] = ti.Vector([0.0, 0.0])
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.color[i] = ti.Vector([0.0, 0.0, 0.0])
            self.active[i] = 0
