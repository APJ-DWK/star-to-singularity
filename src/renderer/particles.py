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

        # Accretion orbital plane coordinates & velocity (for Type 1)
        self.pos_orbit = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.vel_orbit = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        # Particle type: 0 = Ejecta/Prominences (screen-space), 1 = Accretion/Infall (orbit-plane lensed)
        self.type = ti.field(dtype=ti.i32, shape=max_particles)

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
            self.pos_orbit[i] = ti.Vector([0.0, 0.0])
            self.vel_orbit[i] = ti.Vector([0.0, 0.0])
            self.type[i] = 0

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
            angle = (i / self.num_active_guess_factor()) * 2.0 * 3.14159265
            speed = speed_min + (speed_max - speed_min) * (ti.sin(i * 987.65) * 0.5 + 0.5)
            
            dx = ti.cos(angle)
            dy = ti.sin(angle) * aspect_ratio
            
            self.pos[i] = ti.Vector([center_x, center_y])
            self.vel[i] = ti.Vector([dx * speed, dy * speed])
            
            color_rand = ti.sin(i * 123.45) * 0.5 + 0.5
            self.color[i] = base_color * (0.9 + 0.3 * color_rand)
            if color_rand > 0.5:
                self.color[i] = ti.Vector([1.0, 0.9, 0.8])
                
            life_sec = 3.5 + 4.5 * (ti.sin(i * 456.78) * 0.5 + 0.5)
            self.life[i] = life_sec
            self.max_life[i] = life_sec
            self.active[i] = 1
            self.type[i] = 0

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
            if self.active[i] == 0 and i < count:
                angle = time * 0.2 + (i * 153.25) % 6.28318
                
                self.pos[i] = ti.Vector([
                    center_x + ti.cos(angle) * radius,
                    center_y + ti.sin(angle) * radius * aspect_ratio
                ])
                
                radial_dir = ti.Vector([ti.cos(angle), ti.sin(angle) * aspect_ratio])
                tangent_dir = ti.Vector([-ti.sin(angle), ti.cos(angle) * aspect_ratio])
                
                v_out = 0.08 + 0.04 * ti.sin(i * 45.67)
                v_tang = 0.04 * ti.cos(i * 78.91)
                self.vel[i] = radial_dir * v_out + tangent_dir * v_tang
                
                self.color[i] = ti.Vector([1.0, 0.15, 0.25])
                
                life_sec = 0.6 + 0.5 * (ti.sin(i * 99.9) * 0.5 + 0.5)
                self.life[i] = life_sec
                self.max_life[i] = life_sec
                self.active[i] = 1
                self.type[i] = 0

    @ti.kernel
    def spawn_infall_gas(
        self,
        center_x: ti.f32,
        center_y: ti.f32,
        radius: ti.f32,
        aspect_ratio: ti.f32,
        count: ti.i32
    ):
        """GPU kernel to spawn infalling accretion plasma particles in the accretion disk plane."""
        for i in range(self.max_particles):
            if self.active[i] == 0 and i < count:
                angle = (i * 245.71) % 6.28318
                
                r_orbit = radius * (0.8 + 0.4 * (ti.sin(i * 111.11) * 0.5 + 0.5))
                self.pos_orbit[i] = ti.Vector([
                    ti.cos(angle) * r_orbit,
                    ti.sin(angle) * r_orbit
                ])
                
                radial_dir = ti.Vector([ti.cos(angle), ti.sin(angle)])
                tangent_dir = ti.Vector([-ti.sin(angle), ti.cos(angle)])
                
                self.vel_orbit[i] = tangent_dir * 0.18 - radial_dir * 0.04
                
                self.pos[i] = ti.Vector([-10.0, -10.0])
                self.vel[i] = ti.Vector([0.0, 0.0])
                
                color_rand = ti.sin(i * 876.54) * 0.5 + 0.5
                self.color[i] = ti.Vector([1.0, 0.45 + 0.3 * color_rand, 0.05])
                
                life_sec = 4.0 + 3.0 * (ti.sin(i * 321.0) * 0.5 + 0.5)
                self.life[i] = life_sec
                self.max_life[i] = life_sec
                self.active[i] = 1
                self.type[i] = 1

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
        spin_accel: ti.f32,
        yaw: ti.f32,
        pitch: ti.f32,
        zoom: ti.f32,
        aspect_ratio: ti.f32,
        phase: ti.i32,
        progress: ti.f32,
        remnant_mass: ti.f32
    ):
        """Update particle positions, apply gravitational pull, swirling force, and fade color intensities."""
        for i in range(self.max_particles):
            if self.active[i] == 1:
                self.life[i] -= dt
                if self.life[i] <= 0.0:
                    self.active[i] = 0
                    self.pos[i] = ti.Vector([-10.0, -10.0])
                    self.color[i] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    if self.type[i] == 1:
                        # ── 1. Update accretion particles in orbital plane ──
                        px = self.pos_orbit[i][0]
                        py = self.pos_orbit[i][1]
                        r = ti.sqrt(px*px + py*py) + 1e-5
                        
                        if r < 0.045:
                            self.active[i] = 0
                            self.pos[i] = ti.Vector([-10.0, -10.0])
                            self.color[i] = ti.Vector([0.0, 0.0, 0.0])
                        else:
                            dx_dir = px / r
                            dy_dir = py / r
                            tx_dir = -dy_dir
                            ty_dir = dx_dir
                            
                            self.vel_orbit[i][0] -= (gravity_accel * dt / (r * r)) * dx_dir
                            self.vel_orbit[i][1] -= (gravity_accel * dt / (r * r)) * dy_dir
                            self.vel_orbit[i][0] += (spin_accel * dt / (r * r)) * tx_dir
                            self.vel_orbit[i][1] += (spin_accel * dt / (r * r)) * ty_dir
                            
                            self.vel_orbit[i] *= ti.exp(-0.20 * dt)
                            self.pos_orbit[i] += self.vel_orbit[i] * dt
                            
                            # ── 2. Project to screen with rotation, tilt, and lensing ──
                            cy = ti.cos(yaw)
                            sy = ti.sin(yaw)
                            tilt = ti.max(0.15, ti.abs(ti.sin(pitch)))
                            
                            ux = px * cy + py * tilt * sy
                            uy = -px * sy + py * tilt * cy
                            
                            ndc_x = ux * zoom
                            ndc_y = uy * zoom * aspect_ratio
                            
                            r_screen = ti.sqrt(ndc_x*ndc_x + ndc_y*ndc_y / (aspect_ratio*aspect_ratio)) + 1e-5
                            
                            lensing_factor = 0.0
                            if phase == 3:
                                lensing_factor = 1.0
                            elif phase == 2 and progress >= 0.3:
                                lensing_factor = (progress - 0.3) / 0.7
                                
                            is_occluded = False
                            if lensing_factor > 0.0:
                                theta_E = 0.12 * (remnant_mass / 3.0) * lensing_factor
                                scale = 0.5 * (1.0 + ti.sqrt(1.0 + 4.0 * theta_E * theta_E / (r_screen * r_screen)))
                                if r_screen < theta_E * 2.5:
                                    scale *= 1.0 + 0.6 * (1.0 - r_screen / (theta_E * 2.5))
                                    
                                lensed_ndc_x = ndc_x * scale
                                lensed_ndc_y = ndc_y * scale
                                
                                r_eh_screen = 0.065 * lensing_factor
                                if ti.sqrt(lensed_ndc_x*lensed_ndc_x + lensed_ndc_y*lensed_ndc_y / (aspect_ratio*aspect_ratio)) <= r_eh_screen:
                                    is_occluded = True
                                    
                                self.pos[i] = ti.Vector([center_x + lensed_ndc_x, center_y + lensed_ndc_y])
                            else:
                                self.pos[i] = ti.Vector([center_x + ndc_x, center_y + ndc_y])
                                
                            if is_occluded:
                                self.active[i] = 0
                                self.pos[i] = ti.Vector([-10.0, -10.0])
                                
                    else:
                        # ── 3. Update Type 0 particles in flat screen space ──
                        self.pos[i] += self.vel[i] * dt
                        
                        dx = self.pos[i][0] - center_x
                        dy = self.pos[i][1] - center_y
                        r = ti.sqrt(dx*dx + dy*dy)
                        if r > 1e-3:
                            rx = dx / r
                            ry = dy / r
                            
                            prom_gravity = 0.015
                            self.vel[i][0] -= (prom_gravity * dt / (r * r)) * rx
                            self.vel[i][1] -= (prom_gravity * dt / (r * r)) * ry
                            
                        self.vel[i] *= ti.exp(-0.15 * dt)
                        
                    # ── 4. Apply age color/brightness decay ──
                    ratio = self.life[i] / self.max_life[i]
                    if ratio < 0.6:
                        self.color[i].y *= ti.exp(-0.8 * dt)
                        self.color[i].z *= ti.exp(-1.5 * dt)
                        
                    self.color[i] *= ratio

    def update(self, dt, center_x, center_y, gravity, spin, camera, phase_int, remnant_mass, progress):
        """Advance the particle simulation step, applying central forces and projecting to screen."""
        self.update_particles_kernel(
            dt, center_x, center_y, gravity, spin,
            camera.yaw, camera.pitch, camera.zoom, camera.aspect_ratio,
            phase_int, progress, remnant_mass
        )

    def render(self, canvas):
        """Draw particles onto the canvas."""
        radius = 0.003
        canvas.circles(self.pos, radius=radius, per_vertex_color=self.color)
