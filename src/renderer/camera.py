"""
Virtual Camera.

Manages camera viewing parameters (zoom, orientation, and pan)
and provides projection functions to map 3D directions and 2D physics
coordinates to normalized screen coordinates [0, 1] x [0, 1].

Milestone: 7 (Camera and Interactive Controls)
Status: Completed
"""

import numpy as np


class Camera:
    """
    Simulates a virtual pinhole camera for rendering.

    Provides projection calculations for 3D celestial sphere objects
    and 2D orbital plane coordinates.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height

        # Camera parameters
        self.zoom = 1.0          # Zoom factor (higher = narrower field of view)
        self.yaw = 0.0           # Horizontal rotation angle (radians)
        self.pitch = 0.0         # Vertical rotation angle (radians)
        self.pan = np.array([0.0, 0.0], dtype=float)  # 2D screen pan offset

    def update_from_state(self, state):
        """Sync camera settings with the centralized simulation state using cinematic transitions."""
        # Use state.dt, with a fallback if simulation is paused to keep camera responsive
        dt = state.dt if (state.dt > 0.0 and not state.paused) else 0.016
        
        target_zoom = 1.0
        target_pan = np.array([0.0, 0.0], dtype=float)
        
        phase = state.current_phase
        progress = state.phase_progress
        
        if phase == "stellar_birth":
            target_zoom = 1.0
        elif phase == "stellar_death":
            if progress < 0.7:
                # Expand to red supergiant: zoom out from 1.0 to 0.15
                t_exp = progress / 0.7
                target_zoom = 1.0 + (0.15 - 1.0) * (1.0 - (1.0 - t_exp) ** 2.0)
            else:
                # Stay zoomed out for the start of core collapse
                target_zoom = 0.15
        elif phase == "supernova":
            if progress < 0.35:
                # Supernova explosion: stay zoomed out to frame the expanding ejecta
                target_zoom = 0.15
                # Apply camera shake during the shockwave peak
                shake_t = progress / 0.35
                shake_amp = 0.03 * np.exp(-3.0 * shake_t) * np.sin(state.time * 70.0)
                target_pan[0] += shake_amp
                target_pan[1] += shake_amp * 0.7
            else:
                # Core collapse fallback: zoom in smoothly from 0.15 to 1.8
                t_zoom = (progress - 0.35) / 0.65
                target_zoom = 0.15 + (1.8 - 0.15) * (t_zoom ** 2.0)
        elif phase == "black_hole":
            target_zoom = 1.8
            
        # Smoothly interpolate towards target settings
        lerp_speed = 3.0
        step = min(1.0, dt * lerp_speed)
        self.zoom += (target_zoom - self.zoom) * step
        self.pan += (target_pan - self.pan) * step
        
        # Write back to state so other systems remain in sync
        state.camera_zoom = float(self.zoom)
        state.camera_center[0] = float(self.pan[0])
        state.camera_center[1] = float(self.pan[1])

    def project_3d_direction(self, direction_vec):
        """
        Project a 3D unit direction vector onto the 2D screen.

        Applies camera rotation, zoom, and perspective projection.

        Args:
            direction_vec: A 3D numpy array or vector [x, y, z] representing a direction.

        Returns:
            screen_coords: [u, v] in the range [0, 1] x [0, 1], or None if behind camera.
        """
        # 1. Apply yaw (around Y-axis) and pitch (around X-axis) rotation
        # Rotation matrices:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)

        x, y, z = direction_vec

        # Yaw rotation
        x_rot = cy * x - sy * z
        z_rot = sy * x + cy * z

        # Pitch rotation
        y_rot = cp * y - sp * z_rot
        z_final = sp * y + cp * z_rot

        # Check if the object is behind the camera plane
        if z_final <= 0.0:
            return None

        # 2. Perspective projection scaled by zoom
        # Normalized device coordinates (NDC) range from -1 to 1
        ndc_x = (x_rot / z_final) * self.zoom
        ndc_y = (y_rot / z_final) * self.zoom * self.aspect_ratio

        # 3. Map to screen space [0, 1] x [0, 1] with pan applied
        u = 0.5 + 0.5 * ndc_x + self.pan[0]
        v = 0.5 + 0.5 * ndc_y + self.pan[1]

        return np.array([u, v])

    def project_2d_physics(self, pos_2d):
        """
        Project a 2D physics-space coordinate [x, y] to screen space [0, 1] x [0, 1].

        Used for rendering the accretion disk and stellar surface directly.

        Args:
            pos_2d: 2D vector [x, y] in physics space.

        Returns:
            screen_coords: [u, v] on screen.
        """
        # Apply translation (pan) and zoom
        x = (pos_2d[0] + self.pan[0]) * self.zoom
        y = (pos_2d[1] + self.pan[1]) * self.zoom * self.aspect_ratio

        # Map to [0, 1] screen coordinates
        u = 0.5 + 0.5 * x
        v = 0.5 + 0.5 * y

        return np.array([u, v])
