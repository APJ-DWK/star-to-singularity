"""
Procedural Starfield Background.

Generates a static starfield on a 3D celestial sphere with physically-motivated
brightness distribution (apparent magnitudes) and color variation based on
stellar spectral types. Stars are projected to 2D screen coordinates and their
intensity modulated on the GPU in a parallel Taichi kernel.

Milestone: 4 (Stellar Evolution — provides visual context)
Status: Completed
"""

import numpy as np
import taichi as ti

import config
from src.physics.thermodynamics import temperature_to_rgb


@ti.data_oriented
class BackgroundStarfield:
    """
    Manages and renders a 3D procedural starfield.

    Uses Taichi fields and parallel kernels to project stars to screen space
    and apply camera zoom, rotation, pan, and aspect ratio.
    """

    def __init__(self, num_stars=config.BACKGROUND_STAR_COUNT):
        self.num_stars = num_stars

        # ── Taichi Fields ────────────────────────────────────────────
        # 3D unit vectors representing direction to the stars
        self.star_dirs = ti.Vector.field(3, dtype=ti.f32, shape=num_stars)
        # sRGB base colors of the stars
        self.star_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_stars)
        # Apparent brightness multipliers [0.0, 1.0]
        self.star_brightness = ti.field(dtype=ti.f32, shape=num_stars)

        # Output fields (computed each frame on GPU)
        self.screen_pos = ti.Vector.field(2, dtype=ti.f32, shape=num_stars)
        self.render_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_stars)

        self._generate_starfield_data()

    def _generate_starfield_data(self):
        """Generate stellar positions, colors, and brightnesses on the CPU."""
        dirs = np.zeros((self.num_stars, 3), dtype=np.float32)
        colors = np.zeros((self.num_stars, 3), dtype=np.float32)
        brightness = np.zeros(self.num_stars, dtype=np.float32)

        # Standard spectral distribution of stars in the Milky Way
        # O: 0.2%, B: 0.8%, A: 2%, F: 3%, G: 7%, K: 12%, M: 75%
        spectral_probs = [0.002, 0.008, 0.02, 0.03, 0.07, 0.12, 0.75]
        spectral_temps = [
            (30000.0, 40000.0), # O
            (10000.0, 30000.0), # B
            (7500.0, 10000.0),  # A
            (6000.0, 7500.0),   # F
            (5200.0, 6000.0),   # G
            (3700.0, 5200.0),   # K
            (2000.0, 3700.0)    # M
        ]

        np.random.seed(42)  # Seed for reproducible background stars

        for i in range(self.num_stars):
            # 1. Generate uniform direction on sphere
            u = np.random.uniform(0.0, 1.0)
            v = np.random.uniform(0.0, 1.0)

            theta = np.arccos(2.0 * u - 1.0)  # Latitude
            phi = 2.0 * np.pi * v             # Longitude

            dirs[i, 0] = np.sin(theta) * np.cos(phi)
            dirs[i, 1] = np.sin(theta) * np.sin(phi)
            dirs[i, 2] = np.cos(theta)

            # 2. Sample spectral type and temperature
            spec_idx = np.random.choice(7, p=spectral_probs)
            min_t, max_t = spectral_temps[spec_idx]
            temp = np.random.uniform(min_t, max_t)

            # Convert to sRGB color
            colors[i] = temperature_to_rgb(temp)

            # 3. Apparent magnitude simulation (power-law: many dim stars, few bright)
            # Power law mapping: B = 0.05 + 0.95 * (1 - random)^4
            r_val = np.random.uniform(0.0, 1.0)
            brightness[i] = 0.03 + 0.97 * (1.0 - r_val) ** 5.0

        # Load generated data to Taichi fields
        self.star_dirs.from_numpy(dirs)
        self.star_colors.from_numpy(colors)
        self.star_brightness.from_numpy(brightness)

    @ti.kernel
    def _project_stars_kernel(
        self,
        yaw: ti.f32,
        pitch: ti.f32,
        zoom: ti.f32,
        aspect_ratio: ti.f32,
        pan_x: ti.f32,
        pan_y: ti.f32,
        mask_radius: ti.f32,
        phase: ti.i32,
        remnant_mass: ti.f32,
        progress: ti.f32
    ):
        """Project 3D directions of stars into 2D screen positions in parallel, applying gravitational lensing when black hole/collapsing remnant is present."""
        for i in range(self.num_stars):
            x = self.star_dirs[i][0]
            y = self.star_dirs[i][1]
            z = self.star_dirs[i][2]

            # 1. Yaw rotation (around vertical Y axis)
            cy = ti.cos(yaw)
            sy = ti.sin(yaw)
            x_rot1 = cy * x - sy * z
            z_rot1 = sy * x + cy * z

            # 2. Pitch rotation (around horizontal X axis)
            cp = ti.cos(pitch)
            sp = ti.sin(pitch)
            y_rot = cp * y - sp * z_rot1
            z_final = sp * y + cp * z_rot1

            # Determine visibility (only project stars in front of camera plane)
            if z_final > 0.05:
                # Standard perspective projection
                ndc_x = (x_rot1 / z_final) * zoom
                ndc_y = (y_rot / z_final) * zoom * aspect_ratio

                # Convert to screen coordinates [0.0, 1.0]
                u = 0.5 + 0.5 * ndc_x + pan_x
                v = 0.5 + 0.5 * ndc_y + pan_y

                # Apply occlusion or lensing distortion
                is_occluded = False
                
                # Check if we should apply lensing (phase 3 is black hole, phase 2 is supernova)
                # During supernova phase, lensing grows in the final stage (collapse progress >= 0.5)
                is_lensing = False
                lensing_factor = 0.0
                if phase == 3:
                    is_lensing = True
                    lensing_factor = 0.55
                elif phase == 2 and progress >= 0.5:
                    is_lensing = True
                    # Scale from 0.0 to 1.0 as core collapse progresses
                    lensing_factor = (progress - 0.5) / 0.5
                elif phase == 2:
                    is_lensing = True
                    lensing_factor = progress * progress
                    
                if is_lensing:
                    # Gravitational lensing deflection math:
                    # Calculate Einstein radius in NDC coordinates (scales with remnant mass)
                    theta_E = 0.03 * (remnant_mass / 3.0) * lensing_factor
                    
                    du = u - 0.5 - pan_x
                    dv = (v - 0.5 - pan_y) / aspect_ratio
                    dx = du / zoom
                    dy = dv / zoom
                    theta_in = ti.sqrt(dx*dx + dy*dy)

                    # Only lens stars close to the black hole
                    if theta_in > theta_E * 2.0:
                        is_lensing = False

                    # Ignore stars that are far from the black hole.
                    if theta_in > 0.12:
                        is_lensing = False
                    
                    if is_lensing and theta_in > 1e-5:
                        # Solve lens equation: theta_out^2 - theta_in * theta_out - theta_E^2 = 0
                        scale = 0.5 * (1.0 + ti.sqrt(1.0 + 4.0 * theta_E * theta_E / (theta_in * theta_in)))

                        # Stronger bending very close to the event horizon
                        if theta_in < theta_E * 2.5:
                            scale *= 1.0 + 0.6 * (1.0 - theta_in / (theta_E * 2.5))
                        
                        lensed_dx = dx * scale
                        lensed_dy = dy * scale
                        
                        # Re-project to screen coordinates
                        ndc_x_l = lensed_dx * zoom
                        ndc_y_l = lensed_dy * zoom / aspect_ratio
                        u = 0.5 + 0.5 * ndc_x_l + pan_x
                        v = 0.5 + 0.5 * ndc_y_l + pan_y
                        
                        # Event horizon occlusion check (shadow radius is ~2.6 * r_s visually)
                        r_eh = 0.04 * lensing_factor
                        disk_mask = r_eh * 5.6
                        if ti.sqrt(lensed_dx*lensed_dx + lensed_dy*lensed_dy) <= disk_mask:
                            is_occluded = True
                    else:
                        is_occluded = True
                else:
                    # Normal stellar body occlusion mask
                    if mask_radius > 0.0:
                        r_bound = 0.45 * (mask_radius / 6.034)
                        du = u - 0.5 - pan_x
                        dv = (v - 0.5 - pan_y) / aspect_ratio
                        dx = du / zoom
                        dy = dv / zoom
                        if ti.sqrt(dx*dx + dy*dy) <= r_bound:
                            is_occluded = True

                if not is_occluded:
                    self.screen_pos[i] = ti.Vector([u, v])
                    self.render_colors[i] = self.star_colors[i] * self.star_brightness[i]
                else:
                    self.screen_pos[i] = ti.Vector([-10.0, -10.0])
                    self.render_colors[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # Place off-screen and hide
                self.screen_pos[i] = ti.Vector([-10.0, -10.0])
                self.render_colors[i] = ti.Vector([0.0, 0.0, 0.0])

    def update(self, camera, mask_radius, phase, remnant_mass, progress):
        """Update star projections based on the camera view, phase, and lensing parameters."""
        self._project_stars_kernel(
            camera.yaw,
            camera.pitch,
            camera.zoom,
            camera.aspect_ratio,
            camera.pan[0],
            camera.pan[1],
            mask_radius,
            phase,
            remnant_mass,
            progress
        )

    def render(self, canvas):
        """
        Draw the stars onto the canvas.

        Uses sub-pixel anti-aliasing via brightness modulation: all particles are
        drawn with a tiny uniform radius, and their perceived size is governed
        by their relative intensity on screen.
        """
        # Small uniform radius for stars
        star_radius = 0.0018
        canvas.circles(self.screen_pos, radius=star_radius, per_vertex_color=self.render_colors)

