"""
GPU Particle System.

Manages collections of particles on the GPU using Taichi parallel kernels.
Simulates supernova ejecta shells, shockwaves, expansion velocities,
and color/brightness decay.

Milestone: 6 (Supernova particles)
Status: Completed
"""

import taichi as ti


@ti.data_oriented
class ParticleSystem:
    """
    GPU-accelerated particle system for simulating cosmic gas ejecta and flows.
    """

    def __init__(self, max_particles=50000):
        self.max_particles = max_particles

        # ── GPU Fields ───────────────────────────────────────────────
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        
        # Remaining life of each particle in seconds
        self.life = ti.field(dtype=ti.f32, shape=max_particles)
        self.max_life = ti.field(dtype=ti.f32, shape=max_particles)
        self.active = ti.field(dtype=ti.i32, shape=max_particles)

    @ti.kernel
    def reset_kernel(self):
        """GPU-side clear of all particle states."""
        for i in range(self.max_particles):
            self.pos[i] = ti.Vector([-10.0, -10.0])
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.color[i] = ti.Vector([0.0, 0.0, 0.0])
            self.life[i] = 0.0
            self.max_life[i] = 1.0
            self.active[i] = 0

    def reset(self):
        """CPU interface to clear particles."""
        self.reset_kernel()

    @ti.kernel
    def spawn_supernova_ejecta(
        self,
        center_x: ti.f32,
        center_y: ti.f32,
        speed_min: ti.f32,
        speed_max: ti.f32,
        base_color: ti.template(),
        aspect_ratio: ti.f32
    ):
        """GPU kernel to spawn a radial shell of supernova shockwave particles."""
        for i in range(self.max_particles):
            # Distribute directions evenly/randomly
            angle = (i / self.num_active_guess_factor()) * 2.0 * 3.14159265
            speed = speed_min + (speed_max - speed_min) * (ti.sin(i * 987.65) * 0.5 + 0.5)
            
            # Spherical expansion coordinates corrected for screen aspect ratio
            dx = ti.cos(angle)
            dy = ti.sin(angle) * aspect_ratio
            
            self.pos[i] = ti.Vector([center_x, center_y])
            self.vel[i] = ti.Vector([dx * speed, dy * speed])
            
            # High-energy ejecta colors (mix white-cyan in the core, orange-red at outer edge)
            color_rand = ti.sin(i * 123.45) * 0.5 + 0.5
            self.color[i] = base_color * (0.8 + 0.2 * color_rand)
            if color_rand > 0.6:
                # Add hot white/yellow shock front particles
                self.color[i] = ti.Vector([1.0, 0.9, 0.8])
                
            life_sec = 2.0 + 3.0 * (ti.sin(i * 456.78) * 0.5 + 0.5)
            self.life[i] = life_sec
            self.max_life[i] = life_sec
            self.active[i] = 1

    @ti.func
    def num_active_guess_factor(self):
        """Helper to get scaling factor for spawning distributions."""
        return float(self.max_particles)

    @ti.kernel
    def update_particles_kernel(self, dt: ti.f32):
        """Update particle positions and fade color intensities based on remaining lifetime."""
        for i in range(self.max_particles):
            if self.active[i] == 1:
                # Move particle
                self.pos[i] += self.vel[i] * dt
                
                # Apply slow drag / expansion cooling slowing down
                self.vel[i] *= ti.exp(-0.4 * dt)
                
                # Age decay
                self.life[i] -= dt
                if self.life[i] <= 0.0:
                    self.active[i] = 0
                    self.pos[i] = ti.Vector([-10.0, -10.0])
                    self.color[i] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    # Calculate remaining life ratio
                    ratio = self.life[i] / self.max_life[i]
                    
                    # Shift color from hot bright yellow/white to cool dark red as gas cools
                    if ratio < 0.6:
                        # Fade to orange/red
                        self.color[i].y *= ti.exp(-0.8 * dt) # Dim green (creates red shift)
                        self.color[i].z *= ti.exp(-1.5 * dt) # Dim blue (creates red shift)
                    
                    # Overall intensity fade out
                    self.color[i] *= ratio

    def update(self, dt):
        """Advance the particle simulation step."""
        self.update_particles_kernel(dt)

    def render(self, canvas):
        """Draw particles onto the canvas."""
        # Use small radius for debris particles
        radius = 0.003
        canvas.circles(self.pos, radius=radius, per_vertex_color=self.color)
