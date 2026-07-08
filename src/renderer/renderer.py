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

    Maintains the virtual camera, background starfield, and physical star rendering,
    compositing them onto the Taichi window Canvas.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Initialize core components
        self.camera = Camera(width, height)
        self.starfield = BackgroundStarfield()

        # Render pass toggles
        self.show_background = True

        # Star surface texture buffer (rendered at half resolution to guarantee high FPS on CPU fallback)
        self.star_img_width = 640
        self.star_img_height = 360
        self.star_image = ti.Vector.field(3, dtype=ti.f32, shape=(self.star_img_width, self.star_img_height))

    @ti.kernel
    def _render_star_kernel(
        self,
        radius: ti.f32,
        temperature: ti.f32,
        time: ti.f32,
        zoom: ti.f32,
        pan_x: ti.f32,
        pan_y: ti.f32,
        aspect_ratio: ti.f32
    ):
        """GPU pixel shader to render the star surface, limb darkening, convection, corona, and bloom."""
        for i, j in self.star_image:
            # 1. Normalize pixel coordinate to [0, 1]
            u = i / self.star_img_width
            v = j / self.star_img_height
            
            # Displacement from center, correcting for aspect ratio
            du = u - 0.5 - pan_x
            dv = (v - 0.5 - pan_y) / aspect_ratio
            
            # Scale by camera zoom factor to yield Normalized Device Coordinates
            dx = du / zoom
            dy = dv / zoom
            dist = ti.sqrt(dx*dx + dy*dy)
            
            # Reference star of M = 20 M_sun has radius R = 6.034 R_sun.
            # Map this reference radius to 0.45 of the viewport half-height.
            r_bound = 0.45 * (radius / 6.034)
            
            color = ti.Vector([0.0, 0.0, 0.0])
            
            if dist <= r_bound:
                # ── STAR SURFACE Pass ────────────────────────────────────────
                # Reconstruct z-coordinate of sphere
                z = ti.sqrt(r_bound*r_bound - dist*dist)
                
                # 3D surface point on celestial sphere
                p = ti.Vector([dx, dy, z])
                
                # Apply Y-axis axial rotation over time
                angle = time * 0.05
                ca = ti.cos(angle)
                sa = ti.sin(angle)
                p_rot = ti.Vector([ca * p.x - sa * p.z, p.y, sa * p.x + ca * p.z])
                
                # Sample multi-scale noise to simulate surface convection (granulation cells)
                # Offset noise coordinates over time to represent rising/sinking plasma
                granulation = fbm(p_rot * 22.0 + ti.Vector([0.0, 0.0, time * 0.4]))
                
                # Apply Limb Darkening: Standard linear profile I(mu)/I(1) = 1 - u_ld * (1 - mu)
                # Where mu = cos(theta) = z / R
                mu = z / r_bound
                u_ld = 0.40  # Limb darkening coefficient for hot stellar surfaces
                limb_factor = 1.0 - u_ld * (1.0 - mu)
                
                # Granulation modulates the surface temperature locally by ±3.5%
                local_temp = temperature * (1.0 + (granulation - 0.5) * 0.07)
                
                # Color the surface using blackbody approximation and apply limb darkening
                color = temp_to_rgb_gpu(local_temp) * limb_factor
                
            else:
                # ── CORONA & BLOOM Pass ──────────────────────────────────────
                # Distance scale factor starting outside the stellar limb
                dist_ratio = (dist - r_bound) / r_bound
                
                # Corona: faint high-temperature atmosphere (decays rapidly)
                corona_intensity = ti.exp(-dist_ratio * 14.0)
                corona_color = temp_to_rgb_gpu(temperature * 1.2)
                
                # Bloom: optical lens scattering (decays slowly)
                bloom_intensity = ti.exp(-dist_ratio * 1.8)
                bloom_color = temp_to_rgb_gpu(temperature)
                
                color = (corona_color * corona_intensity * 0.45) + (bloom_color * bloom_intensity * 0.18)
                
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
        mask_radius = 0.0
        if state.current_phase in ["stellar_birth", "stellar_death"]:
            mask_radius = state.stellar_radius

        # 3. Update background starfield
        if self.show_background:
            self.starfield.update(self.camera, mask_radius)

        # 4. Render stellar surface to full-screen texture
        if state.current_phase == "stellar_birth":
            self._render_stellar_birth(state, canvas)
        elif state.current_phase == "stellar_death":
            self._render_stellar_death(state, canvas)
        else:
            # Deep space clear for other phases
            canvas.set_background_color((0.01, 0.01, 0.02))

        # 5. Composite background stars ON TOP of background textures/glows
        if self.show_background:
            self.starfield.render(canvas)

    def _render_stellar_birth(self, state, canvas):
        """Render main sequence star surface and solar wind."""
        self._render_star_kernel(
            state.stellar_radius,
            state.stellar_temperature,
            state.time,
            self.camera.zoom,
            self.camera.pan[0],
            self.camera.pan[1],
            self.camera.aspect_ratio
        )
        canvas.set_image(self.star_image)

    def _render_stellar_death(self, state, canvas):
        """Render contracting core and silicon shell burning."""
        self._render_star_kernel(
            state.stellar_radius,
            state.stellar_temperature,
            state.time,
            self.camera.zoom,
            self.camera.pan[0],
            self.camera.pan[1],
            self.camera.aspect_ratio
        )
        canvas.set_image(self.star_image)

    def _render_supernova(self, state, canvas):
        """Render expanding supernova gas shell and central core."""
        pass

    def _render_black_hole(self, state, canvas):
        """Render event horizon, photon sphere, and accretion disk."""
        pass

