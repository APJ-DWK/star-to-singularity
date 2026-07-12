"""
Taichi Rendering Engine.

Renders the simulation using Taichi GPU kernels. Coordinates the camera,
background starfield, particle systems, and GUI elements.

Milestone: 4 (Stellar Evolution rendering framework)
Status: Completed (Basic Rendering Pipeline)
"""

import taichi as ti

from src.renderer.background import BackgroundStarfield
from src.renderer.camera import Camera
from src.renderer.particles import ParticleSystem
import config


# ── GPU Math & Noise Functions ───────────────────────────────────────

@ti.func
def hash3(p):
    """3D pseudo-random hash generator for noise generation."""
    h = p.dot(ti.Vector([127.1, 311.7, 74.7]))
    val = ti.sin(h) * 43758.5453123
    return val - ti.floor(val)


@ti.func
def noise3(p):
    """3D value noise with smooth interpolation."""
    ip = ti.floor(p)
    fp = p - ip
    
    # Smooth Hermite interpolation (smoothstep)
    u = fp * fp * (3.0 - 2.0 * fp)
    
    # Random values at grid cell corners
    n000 = hash3(ip + ti.Vector([0.0, 0.0, 0.0]))
    n100 = hash3(ip + ti.Vector([1.0, 0.0, 0.0]))
    n010 = hash3(ip + ti.Vector([0.0, 1.0, 0.0]))
    n110 = hash3(ip + ti.Vector([1.0, 1.0, 0.0]))
    n001 = hash3(ip + ti.Vector([0.0, 0.0, 1.0]))
    n101 = hash3(ip + ti.Vector([1.0, 0.0, 1.0]))
    n011 = hash3(ip + ti.Vector([0.0, 1.0, 1.0]))
    n111 = hash3(ip + ti.Vector([1.0, 1.0, 1.0]))
    
    # Trilinear interpolation
    x00 = n000 + (n100 - n000) * u.x
    x10 = n010 + (n110 - n010) * u.x
    x01 = n001 + (n101 - n001) * u.x
    x11 = n011 + (n111 - n011) * u.x
    
    y0 = x00 + (x10 - x00) * u.y
    y1 = x01 + (x11 - x01) * u.y
    
    return y0 + (y1 - y0) * u.z


@ti.func
def fbm(p):
    """Fractional Brownian Motion (multi-scale fractal noise) for granulation."""
    val = 0.0
    amp = 0.5
    freq = 1.0
    for _ in range(3):  # 3 octaves for high detail at low performance cost
        val += amp * noise3(p * freq)
        amp *= 0.5
        freq *= 2.0
    return val


@ti.func
def temp_to_rgb_gpu(temp):
    """GPU-compilable Tanner Helland blackbody color temperature mapping."""
    t = temp / 100.0
    r = 0.0
    g = 0.0
    b = 0.0

    # Red
    if t <= 66.0:
        r = 1.0
    else:
        r = 329.698727446 * ti.pow(t - 60.0, -0.1332047592) / 255.0

    # Green
    if t <= 66.0:
        g = (99.4708025861 * ti.log(t) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ti.pow(t - 60.0, -0.0755148492) / 255.0

    # Blue
    if t >= 66.0:
        b = 1.0
    else:
        if t <= 19.0:
            b = 0.0
        else:
            b = (138.5177312231 * ti.log(t - 10.0) - 305.0447927307) / 255.0

    return ti.Vector([ti.max(0.0, ti.min(1.0, r)),
                      ti.max(0.0, ti.min(1.0, g)),
                      ti.max(0.0, ti.min(1.0, b))])


@ti.data_oriented
class Renderer:
    """
    Main manager for visual presentation.

    Maintains the virtual camera, background starfield, particle systems, and physical star rendering,
    compositing them onto the Taichi window Canvas.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Initialize core components
        self.camera = Camera(width, height)
        self.starfield = BackgroundStarfield()
        self.particles = ParticleSystem(config.MAX_PARTICLES)

        # Render pass toggles
        self.show_background = True

        # Star surface texture buffer (rendered at half resolution to guarantee high FPS on CPU fallback)
        self.star_img_width = 640
        self.star_img_height = 360
        self.star_image = ti.Vector.field(3, dtype=ti.f32, shape=(self.star_img_width, self.star_img_height))

    @ti.kernel
    def _render_scene_kernel(
        self,
        radius: ti.f32,
        temperature: ti.f32,
        time: ti.f32,
        zoom: ti.f32,
        pan_x: ti.f32,
        pan_y: ti.f32,
        aspect_ratio: ti.f32,
        phase: ti.i32,
        progress: ti.f32,
        flash: ti.f32,
        pitch: ti.f32,
        yaw: ti.f32,
        core_radius: ti.f32
    ):
        """Unified GPU pixel shader to render the star surface, plasma turbulence, solar loops, expanding shockwaves, collapsing core, event horizon, and accretion disk."""
        for i, j in self.star_image:
            u = i / self.star_img_width
            v = j / self.star_img_height
            
            du = u - 0.5 - pan_x
            dv = (v - 0.5 - pan_y) / aspect_ratio
            
            dx = du / zoom
            dy = dv / zoom
            dist = ti.sqrt(dx*dx + dy*dy)
            
            color = ti.Vector([0.0, 0.0, 0.0])
            
            if phase == 0 or phase == 1 or phase == 2:
                # ── STAR & SUPERNOVA Pass ────────────────────────────────────
                r_bound = 0.38 * (radius / 6.034)
                
                # Check if inside main stellar/envelope body
                if dist <= r_bound:
                    z = ti.sqrt(r_bound*r_bound - dist*dist)
                    p = ti.Vector([dx, dy, z])
                    
                    angle = time * 0.05
                    ca = ti.cos(angle)
                    sa = ti.sin(angle)
                    p_rot = ti.Vector([ca * p.x - sa * p.z, p.y, sa * p.x + ca * p.z])
                    
                    # Convection/Plasma multi-scale noise
                    freq = 28.0
                    if phase == 1:
                        freq = 12.0
                    elif phase == 2:
                        freq = 8.0
                        
                    granulation = fbm(p_rot * freq + ti.Vector([0.0, 0.0, time * 0.4]))
                    granulation2 = fbm(p_rot * (freq * 2.0) - ti.Vector([time * 0.2, 0.0, 0.0]))
                    
                    mu = z / r_bound
                    u_ld = 0.40
                    limb_factor = 1.0 - u_ld * (1.0 - mu)
                    
                    local_temp = temperature * (1.0 + (granulation - 0.5) * 0.08 + (granulation2 - 0.5) * 0.04)
                    surf_color = temp_to_rgb_gpu(local_temp) * limb_factor
                    
                    if phase == 2:
                        # Envelope dims/disperses as shock front expands
                        fade = ti.max(0.0, 1.0 - progress)
                        color = surf_color * fade
                    else:
                        color = surf_color
                else:
                    # Outside Star: Corona, Solar Flares & Supernova shockwaves
                    dist_ratio = (dist - r_bound) / r_bound
                    
                    # Dynamic corona with streamer rays (fBm noise)
                    corona_noise = fbm(ti.Vector([dx * 18.0, dy * 18.0, time * 0.8]))
                    corona_intensity = ti.exp(-dist_ratio * 12.0) * (1.0 + corona_noise * 0.7)
                    corona_color = temp_to_rgb_gpu(temperature * 1.3)
                    
                    color = corona_color * corona_intensity * 0.45
                    
                    # Solar prominence loop simulation (pinkish H-alpha loops on limb)
                    if phase <= 1:
                        flare_angle = ti.atan2(dy, dx)
                        prominence = ti.sin(flare_angle * 6.0 + time) * ti.cos(flare_angle * 3.0 - time * 0.4) * ti.exp(-dist_ratio * 4.0)
                        if prominence > 0.20:
                            color += ti.Vector([1.0, 0.15, 0.25]) * prominence * 0.3
                            
                    # Supernova expanding shockwave ring layers
                    if phase == 2:
                        fade = ti.max(0.0, 1.0 - progress)
                        color *= fade
                        
                        # Double shockwave layers
                        flare_angle = ti.atan2(dy, dx)
                        
                        # Outer primary shock front
                        ring_dist = dist - (progress * 0.20)
                        ring_int = ti.exp(-ti.abs(ring_dist) * 30.0) * (ti.sin(flare_angle * 12.0) * 0.3 + 0.7)
                        if ring_int > 0.01:
                            color += ti.Vector([1.0, 0.4, 0.1]) * ring_int * 2.0
                            
                        # Inner fallback/ejecta shell
                        ring_dist2 = dist - (progress * 0.1)
                        ring_int2 = ti.exp(-ti.abs(ring_dist2) * 20.0) * (ti.cos(flare_angle * 8.0) * 0.2 + 0.8)
                        if ring_int2 > 0.01:
                            color += ti.Vector([0.9, 0.2, 0.05]) * ring_int2 * 1.8

                # ── Collapsing Remnant Core overlay ──────────────────────────
                if phase == 2 and progress >= 0.3:
                    # Render the shrinking hot protoneutron star core at the center
                    r_core_bound = 0.38 * (core_radius / 6.034)
                    
                    if dist <= r_core_bound:
                        zc = ti.sqrt(r_core_bound*r_core_bound - dist*dist)
                        
                        # Spin increases as core contracts (conservation of angular momentum)
                        angle_c = time * (1.0 + progress * 8.0)
                        ca_c = ti.cos(angle_c)
                        sa_c = ti.sin(angle_c)
                        
                        p_c = ti.Vector([dx, dy, zc])
                        p_c_rot = ti.Vector([ca_c * p_c.x - sa_c * p_c.z, p_c.y, sa_c * p_c.x + ca_c * p_c.z])
                        
                        core_noise = fbm(p_c_rot * 55.0 + ti.Vector([0.0, 0.0, time * 0.5]))
                        core_temp = 40000.0 * (1.0 + progress * 2.5) * (0.85 + core_noise * 0.3)
                        core_color = temp_to_rgb_gpu(core_temp)
                        
                        mu_c = zc / r_core_bound
                        
                        # Smooth core morphing: core fades as it slips inside forming horizon (progress >= 0.8)
                        fade_core = 1.0
                        if progress >= 0.8:
                            fade_core = ti.max(0.0, (0.95 - progress) / 0.15)
                            
                        color = core_color * mu_c * fade_core

                # Core bounce explosion flash overlay
                if phase == 2:
                    color += ti.Vector([1.0, 0.95, 0.9]) * flash
                    
            elif phase == 3:
                # ── BLACK HOLE Pass ──────────────────────────────────────────
                r_eh = 0.04  # Event horizon visual boundary
                
                # Project accretion disk ellipse
                tilt = ti.max(0.15, ti.abs(ti.sin(pitch)))
                
                # Rotate disk coordinates around camera yaw
                cy = ti.cos(yaw)
                sy = ti.sin(yaw)
                rx = dx * cy - dy * sy
                ry = dx * sy + dy * cy
                
                # =====================================================
                # Approximate gravitational lensing warp
                # =====================================================

                r_screen = ti.sqrt(rx * rx + ry * ry) + 1e-5

                warp_strength = 0.020

                warp = warp_strength / (r_screen + r_eh * 0.6)

                warp = ti.min(warp, 0.45)

                wx = rx * (1.0 + warp)
                wy = ry * (1.0 + warp)

                dist_disk = ti.sqrt(wx * wx + (wy / tilt) ** 2.0)
                
                r_in = r_eh * 1.5
                r_out = r_eh * 6.0
                
                if dist <= r_eh:
                    color = ti.Vector([0.0, 0.0, 0.0])  # Singular darkness
                else:
                    # Photon Sphere lensing glow ring right outside EH
                    photon_sphere_radius = r_eh * 1.5

                    # Thin, sharp photon ring
                    ring_width = r_eh * 0.05
                    ps_ring = ti.exp(-((dist - photon_sphere_radius) / ring_width) ** 2)

                    # Bright white photon ring
                    ring_color = ti.Vector([1.0, 0.98, 0.95])

                    ring_angle = ti.atan2(ry, rx)

                    ring_noise = 0.85 + 0.15 * fbm(
                        ti.Vector([
                            ring_angle * 8.0,
                            dist * 180.0,
                            time
                        ])
                    )

                    color += ring_color * ps_ring * 3.4 * ring_noise
                    
                    if dist_disk >= r_in and dist_disk <= r_out:
                        # Keplerian differential shear: inner regions spin faster!
                        shear = time * 2.5 * (r_in / dist_disk)
                        angle = ti.atan2(wy / tilt, wx) - shear
                        
                       # Multi-scale turbulent plasma
                        gas1 = fbm(ti.Vector([dist_disk * 140.0, angle * 4.0, time * 0.25]))
                        gas2 = fbm(ti.Vector([dist_disk * 280.0, angle * 9.0, -time * 0.45]))

                        gas = 0.65 * gas1 + 0.35 * gas2

                        spiral = 0.5 + 0.5 * ti.sin(8.0 * angle - dist_disk * 90.0 + time * 2.5)

                        gas *= (0.8 + 0.4 * spiral)

                        # -------- Physics --------
                        t_disk = 1.8e7 * ti.pow(r_in / dist_disk, 0.75)
                        local_temp = t_disk * (0.70 + gas * 0.65)

                        # -------- Visual rendering --------
                        visual_temp = 4200.0 + 18000.0 * ti.pow(r_in / dist_disk, 0.82)
                        visual_temp *= (0.85 + gas * 0.30)

                        disk_color = temp_to_rgb_gpu(visual_temp)
                        # Warm the accretion disk
                        warm = ti.Vector([1.15, 1.05, 0.75])
                        disk_color *= warm

                        # Clamp to valid range
                        disk_color = ti.Vector([
                            ti.min(disk_color.x, 1.0),
                            ti.min(disk_color.y, 1.0),
                            ti.min(disk_color.z, 1.0)
                        ])
                        # Cubic Doppler Beaming factor: beaming ∝ Doppler^3
                        cp = ti.cos(pitch)
                        velocity = ti.min(0.62, ti.sqrt(r_in / dist_disk) * 0.52)

                        doppler = 1.0 + velocity * (-wx / dist_disk)
                        beaming = ti.pow(doppler, 3.0)

                        # Prevent the far side from disappearing
                        beaming = ti.max(0.35, beaming)
                        
                        # Smooth boundaries fade out
                        edge_fade = ti.sin((dist_disk - r_in) / (r_out - r_in) * 3.14159265)

                        inner_glow = ti.exp(-(dist_disk - r_in) * 16.0)
                        edge_fade *= (0.65 + inner_glow)
                        
                        # Accretion disk color mapping
                        color += disk_color * beaming * edge_fade * 1.2

                        # ----- Secondary lensed image -----
                        lensed = ti.exp(-((ry / tilt) * (ry / tilt)) * 8.0)

                        color += (
                            disk_color
                            * 0.22
                            * lensed
                            * edge_fade
                        )
                    else:
                        # Background bloom from hot inner accretion disk
                        bloom_intensity = ti.exp(-(dist - r_eh) / r_eh * 1.5)
                        color += temp_to_rgb_gpu(55000.0) * bloom_intensity * 0.15
                        
            self.star_image[i, j] = color

    def render(self, state, canvas):
        """
        Compose and render the current frame.

        Args:
            state: The current SimulationState.
            canvas: The Taichi Window Canvas to draw onto.
        """
        # 1. Sync camera from central simulation state
        self.camera.update_from_state(state)

        # 2. Determine background star masking radius
        # For black hole phase, background stars are lensed and occluded on GPU
        mask_radius = 0.0
        if state.current_phase in ["stellar_birth", "stellar_death"]:
            mask_radius = state.stellar_radius

        # 3. Map phase name to integer for unified GPU shader
        phase_int = 0
        if state.current_phase == "stellar_birth":
            phase_int = 0
        elif state.current_phase == "stellar_death":
            phase_int = 1
        elif state.current_phase == "supernova":
            phase_int = 2
        elif state.current_phase == "black_hole":
            phase_int = 3

        # 4. Update background starfield (applying gravitational lensing)
        if self.show_background:
            self.starfield.update(
                self.camera,
                mask_radius,
                phase_int,
                state.remnant_mass,
                state.phase_progress
            )

        # 5. Handle particle spawning and updates
        center_x = 0.5 + self.camera.pan[0]
        center_y = 0.5 + self.camera.pan[1]
        
        if state.current_phase == "stellar_birth":
            # Reset particles on start/reset
            self.particles.reset()
            # Spawn minor solar prominence loops
            self.particles.spawn_stellar_prominence(
                center_x,
                center_y,
                0.45 * (state.stellar_radius / 6.034),
                self.camera.aspect_ratio,
                state.time,
                10
            )
            self.particles.update(state.dt, center_x, center_y, gravity=0.03, spin=0.0)
            
        elif state.current_phase == "stellar_death":
            # Spawn larger RSG prominence loops
            self.particles.spawn_stellar_prominence(
                center_x,
                center_y,
                0.45 * (state.stellar_radius / 6.034),
                self.camera.aspect_ratio,
                state.time,
                18
            )
            self.particles.update(state.dt, center_x, center_y, gravity=0.015, spin=0.0)
            
        elif state.current_phase == "supernova":
            if state.supernova_trigger:
                state.supernova_trigger = False
                base_color = ti.Vector([1.0, 0.82, 0.35]) # hot gas ejecta color
                self.particles.reset()
                self.particles.spawn_supernova_ejecta(
                    center_x,
                    center_y,
                    0.02,  # min speed
                    0.10,  # max speed
                    base_color,
                    self.camera.aspect_ratio
                )
            
            # During fallback core collapse stage (progress >= 0.5)
            if state.phase_progress >= 0.5:
                # Spawn gas collapsing inward
                self.particles.spawn_infall_gas(
                    center_x,
                    center_y,
                    0.45 * (state.stellar_radius / 6.034),
                    self.camera.aspect_ratio,
                    40
                )
                # Strong collapse gravity pull and swirling accretion spin
                self.particles.update(state.dt, center_x, center_y, gravity=0.24, spin=0.14)
            else:
                self.particles.update(state.dt, center_x, center_y, gravity=0.02, spin=0.0)
                
        elif state.current_phase == "black_hole":
            # Spawn infalling gas continuously
            self.particles.spawn_infall_gas(
                center_x,
                center_y,
                0.40,
                self.camera.aspect_ratio,
                30
            )
            # Black hole gravity pull and swirling accretion spin
            self.particles.update(state.dt, center_x, center_y, gravity=0.34, spin=0.20)

        # 6. Execute unified GPU shader scene render pass
        self._render_scene_kernel(
            state.stellar_radius,
            state.stellar_temperature,
            state.time,
            self.camera.zoom,
            self.camera.pan[0],
            self.camera.pan[1],
            self.camera.aspect_ratio,
            phase_int,
            state.phase_progress,
            state.supernova_flash,
            self.camera.pitch,
            self.camera.yaw,
            state.core_radius
        )
        canvas.set_image(self.star_image)

        # 7. Composite background stars
        if self.show_background:
            self.starfield.render(canvas)

        # 8. Composite particles (prominences, supernova ejecta, accretion spirals)
        self.particles.render(canvas)


