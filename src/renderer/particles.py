"""
GPU Particle System.

Manages collections of particles on the GPU using Taichi parallel kernels.
Simulates supernova ejecta shells, shockwaves, expansion velocities,
ballistic stellar prominence loops, infalling accretion spirals,
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
            self.color[i] = base_color * (0.9 + 0.3 * color_rand)
            if color_rand > 0.5:
                # Add hot white/yellow shock front particles
                self.color[i] = ti.Vector([1.0, 0.9, 0.8])
                
            life_sec = 2.0 + 3.0 * (ti.sin(i * 456.78) * 0.5 + 0.5)
            self.life[i] = life_sec
            self.max_life[i] = life_sec
            self.active[i] = 1

    @ti.kernel
    def spawn_stellar_prominence(
        self,
        center_x: ti.f32,
        center_y: ti.f32,
        radius: ti.f32,
        aspect_ratio: ti.f32,
        time: ti.f32,
        count: ti.i32
    ):
        """GPU kernel to spawn solar prominences (H-alpha pinkish-red loops) along the stellar limb."""
        for i in range(self.max_particles):
            # Only spawn in a small subset of fields that are inactive
            if self.active[i] == 0 and i < count:
                # Distribute angles around the limb
                angle = time * 0.2 + (i * 153.25) % 6.28318
                
                # Position on the limb
                self.pos[i] = ti.Vector([
                    center_x + ti.cos(angle) * radius,
                    center_y + ti.sin(angle) * radius * aspect_ratio
                ])
                
                # Eject particles slightly outward and tangentially (magnetic looping)
                radial_dir = ti.Vector([ti.cos(angle), ti.sin(angle) * aspect_ratio])
                tangent_dir = ti.Vector([-ti.sin(angle), ti.cos(angle) * aspect_ratio])
                
                # Ballistic velocity
                v_out = 0.08 + 0.04 * ti.sin(i * 45.67)
                v_tang = 0.04 * ti.cos(i * 78.91)
                self.vel[i] = radial_dir * v_out + tangent_dir * v_tang
                
                # Prominences emit primarily in Hydrogen-alpha (deep pinkish red)
                self.color[i] = ti.Vector([1.0, 0.15, 0.25])
                
                life_sec = 0.6 + 0.5 * (ti.sin(i * 99.9) * 0.5 + 0.5)
                self.life[i] = life_sec
                self.max_life[i] = life_sec
                self.active[i] = 1

    @ti.kernel
    def spawn_infall_gas(
        self,
        center_x: ti.f32,
        center_y: ti.f32,
        radius: ti.f32,
        aspect_ratio: ti.f32,
        count: ti.i32
    ):
        """GPU kernel to spawn infalling accretion plasma particles at the outer edge."""
        for i in range(self.max_particles):
            if self.active[i] == 0 and i < count:
                angle = (i * 245.71) % 6.28318
                self.pos[i] = ti.Vector([
                    center_x + ti.cos(angle) * radius,
                    center_y + ti.sin(angle) * radius * aspect_ratio
                ])
                
                # Velocity: Keplerian swirl + slight inward spiral
                radial_dir = ti.Vector([ti.cos(angle), ti.sin(angle) * aspect_ratio])
                tangent_dir = ti.Vector([-ti.sin(angle), ti.cos(angle) * aspect_ratio])
                
                self.vel[i] = tangent_dir * 0.12 - radial_dir * 0.05
                self.color[i] = ti.Vector([0.9, 0.45, 0.1]) # Hot accretion color
                
                life_sec = 3.0
                self.life[i] = life_sec
                self.max_life[i] = life_sec
                self.active[i] = 1

    @ti.func
    def num_active_guess_factor(self):
        """Helper to get scaling factor for spawning distributions."""
        return float(self.max_particles)

    @ti.kernel
    def update_particles_kernel(
        self,
        dt: ti.f32,
        center_x: ti.f32,
        center_y: ti.f32,
        gravity_accel: ti.f32,
        spin_accel: ti.f32
    ):
        """Update particle positions, apply gravitational pull, swirling force, and fade color intensities."""
        for i in range(self.max_particles):
            if self.active[i] == 1:
                # 1. Move particle
                self.pos[i] += self.vel[i] * dt
                
                # 2. Physics: Gravitational pull towards the center
                dx = self.pos[i][0] - center_x
                dy = self.pos[i][1] - center_y
                r = ti.sqrt(dx*dx + dy*dy)
                
                if r > 1e-3:
                    # Acceleration ∝ -G * M / r^2
                    # Radial direction vector
                    rx = dx / r
                    ry = dy / r
                    
                    self.vel[i][0] -= (gravity_accel * dt / (r * r)) * rx
                    self.vel[i][1] -= (gravity_accel * dt / (r * r)) * ry
                    
                    # Rotational / Swirling force (conservation of angular momentum / Keplerian shear)
                    tx = -ry
                    ty = rx
                    self.vel[i][0] += (spin_accel * dt / (r * r)) * tx
                    self.vel[i][1] += (spin_accel * dt / (r * r)) * ty
                
                # Apply gas drag / friction
                self.vel[i] *= ti.exp(-0.25 * dt)
                
                # 3. Age decay
                self.life[i] -= dt
                if self.life[i] <= 0.0 or (r < 0.015 and gravity_accel > 0.0):
                    # Particle falls into event horizon or dies
                    self.active[i] = 0
                    self.pos[i] = ti.Vector([-10.0, -10.0])
                    self.color[i] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    ratio = self.life[i] / self.max_life[i]
                    
                    # Thermal cooling shift
                    if ratio < 0.6:
                        # Fade to red-orange
                        self.color[i].y *= ti.exp(-0.8 * dt)
                        self.color[i].z *= ti.exp(-1.5 * dt)
                    
                    self.color[i] *= ratio

    def update(self, dt, center_x=0.5, center_y=0.5, gravity=0.0, spin=0.0):
        """Advance the particle simulation step, applying central forces."""
        self.update_particles_kernel(dt, center_x, center_y, gravity, spin)

    def render(self, canvas):
        """Draw particles onto the canvas."""
        radius = 0.003
        canvas.circles(self.pos, radius=radius, per_vertex_color=self.color)
