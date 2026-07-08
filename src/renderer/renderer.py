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
        yaw: ti.f32
    ):
        """Unified GPU pixel shader to render the star (main sequence / red supergiant), supernova explosion, and event horizon."""
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
                # Map reference radius of M = 20 M_sun (6.034 R_sun) to 0.45 of viewport
                r_bound = 0.45 * (radius / 6.034)
                
                if dist <= r_bound:
                    z = ti.sqrt(r_bound*r_bound - dist*dist)
                    p = ti.Vector([dx, dy, z])
                    
                    angle = time * 0.05
                    ca = ti.cos(angle)
                    sa = ti.sin(angle)
                    p_rot = ti.Vector([ca * p.x - sa * p.z, p.y, sa * p.x + ca * p.z])
                    
                    # Convection noise configuration based on phase
                    freq = 22.0
                    if phase == 1:
                        # Larger convection cells for red supergiant
                        freq = 12.0
                    elif phase == 2:
                        # Huge chaotic filament structures during supernova expansion
                        freq = 6.0
                        
                    granulation = fbm(p_rot * freq + ti.Vector([0.0, 0.0, time * 0.4]))
                    
                    mu = z / r_bound
                    u_ld = 0.40
                    limb_factor = 1.0 - u_ld * (1.0 - mu)
                    
                    local_temp = temperature * (1.0 + (granulation - 0.5) * 0.07)
                    surf_color = temp_to_rgb_gpu(local_temp) * limb_factor
                    
                    if phase == 2:
                        # Supernova envelope gas shell dims/disperses as it expands
                        fade = ti.max(0.0, 1.0 - progress)
                        color = surf_color * fade
                    else:
                        color = surf_color
                else:
                    # Outside Star: Corona & Bloom
                    dist_ratio = (dist - r_bound) / r_bound
                    corona_intensity = ti.exp(-dist_ratio * 14.0)
                    corona_color = temp_to_rgb_gpu(temperature * 1.2)
                    
                    bloom_intensity = ti.exp(-dist_ratio * 1.8)
                    bloom_color = temp_to_rgb_gpu(temperature)
                    
                    color = (corona_color * corona_intensity * 0.45) + (bloom_color * bloom_intensity * 0.18)
                    
                    if phase == 2:
                        # Supernova fade out
                        fade = ti.max(0.0, 1.0 - progress)
                        color *= fade
                
                # Add supernova core bounce flash overlay across the screen
                if phase == 2:
                    color += ti.Vector([1.0, 0.95, 0.9]) * flash
                    
            elif phase == 3:
                # ── BLACK HOLE Pass ──────────────────────────────────────────
                r_eh = 0.04  # Screen-space event horizon visual boundary
                
                # Project the accretion disk based on pitch tilt angle
                tilt = ti.max(0.15, ti.abs(ti.sin(pitch)))
                
                # Rotate disk coordinates around camera yaw to match space orientation
                cy = ti.cos(yaw)
                sy = ti.sin(yaw)
                rx = dx * cy - dy * sy
                ry = dx * sy + dy * cy
                
                dist_disk = ti.sqrt(rx * rx + (ry / tilt) ** 2.0)
                
                r_in = r_eh * 1.5
                r_out = r_eh * 6.0
                
                if dist <= r_eh:
                    color = ti.Vector([0.0, 0.0, 0.0])  # Singular darkness inside event horizon
                elif dist_disk >= r_in and dist_disk <= r_out:
                    # Accretion disk thin gaseous glow
                    # Temperature profile from Novikov-Thorne limit: T ∝ r^-0.75
                    t_disk = 1.2e7 * ti.pow(r_in / dist_disk, 0.75)
                    
                    # Accretion density wave fBm modulation
                    angle = ti.atan2(ry / tilt, rx) - time * 1.2
                    gas = fbm(ti.Vector([dist_disk * 120.0, angle * 4.0, time * 0.3]))
                    local_temp = t_disk * (0.8 + gas * 0.4)
                    
                    disk_color = temp_to_rgb_gpu(local_temp)
                    
                    # Relativistic Doppler Beaming factor
                    # Side moving towards camera (negative rx side) is beamed and shifted
                    cp = ti.cos(pitch)
                    beaming = 1.0 + 0.6 * cp * (-rx / dist_disk)
                    
                    # Smoothly fade boundaries of the disk
                    edge_fade = ti.sin((dist_disk - r_in) / (r_out - r_in) * 3.14159265)
                    
                    color = disk_color * beaming * edge_fade * 0.9
                else:
                    # Background bloom scattering from the accretion disk
                    bloom_intensity = ti.exp(-(dist - r_eh) / r_eh * 1.5)
                    color = temp_to_rgb_gpu(4000.0) * bloom_intensity * 0.08
                    
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
        # For black hole phase, we occlude stars exactly behind the event horizon (0.04 boundary)
        mask_radius = 0.0
        if state.current_phase in ["stellar_birth", "stellar_death"]:
            mask_radius = state.stellar_radius
        elif state.current_phase == "black_hole":
            mask_radius = 0.536  # Calibrated so r_bound matches EH visual boundary

        # 3. Update background starfield
        if self.show_background:
            self.starfield.update(self.camera, mask_radius)

        # 4. Map phase name to integer for unified GPU shader
        phase_int = 0
        if state.current_phase == "stellar_birth":
            phase_int = 0
            # Reset particles when starting over or reset
            self.particles.reset()
        elif state.current_phase == "stellar_death":
            phase_int = 1
        elif state.current_phase == "supernova":
            phase_int = 2
        elif state.current_phase == "black_hole":
            phase_int = 3

        # 5. Handle supernova particle trigger
        if state.supernova_trigger:
            state.supernova_trigger = False
            center_x = 0.5 + self.camera.pan[0]
            center_y = 0.5 + self.camera.pan[1]
            base_color = ti.Vector([1.0, 0.6, 0.3]) # hot gas ejecta color
            self.particles.reset()
            self.particles.spawn_supernova_ejecta(
                center_x,
                center_y,
                0.05,  # min speed
                0.35,  # max speed
                base_color,
                self.camera.aspect_ratio
            )

        # 6. Update particle positions and colors
        self.particles.update(state.dt)

        # 7. Execute unified GPU shader scene render pass
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
            self.camera.yaw
        )
        canvas.set_image(self.star_image)

        # 8. Composite background stars
        if self.show_background:
            self.starfield.render(canvas)

        # 9. Composite supernova particles
        if state.current_phase in ["supernova", "black_hole"]:
            self.particles.render(canvas)

