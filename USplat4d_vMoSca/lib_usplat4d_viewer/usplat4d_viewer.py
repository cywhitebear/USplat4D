from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer

from lib_from_som.playback_panel import add_gui_playback_group
from lib_from_som.render_panel import populate_render_tab

import json
import datetime

import cv2
import torch
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix

class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
        camera_state: Optional[CameraState] = None,
        # camera_state: Optional[CameraState] = CameraState(), # bug
        img_wh: Tuple[int, int] = (1920, 1080),
        look_at_tuple: Optional[Tuple[float, float, float]] = None,
        cams=None,

    ):
        self.camera_state = camera_state
        self.img_wh = img_wh
        self.num_frames = num_frames
        self.work_dir = Path(work_dir)
        self.cams = cams
        self.camera_pose_log = {}
        self.camera_pose_dir = self.work_dir / "camera_poses"
        self.camera_pose_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.generate_camera_pose_filename()

        super().__init__(server, render_fn, mode)

        self.track_client_cam_poses(look_at_tuple=look_at_tuple)

        self.client_id = 0
    
    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._time_folder = server.gui.add_folder("Time")
        with self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        self._render_track_checkbox = server.gui.add_checkbox("Render tracks", False)
        self._render_track_checkbox.on_update(self.rerender)

        ########################################################################
        # ugraph funtion
        # self._show_rendering_checkbox = server.gui.add_checkbox("Show rendering", True)
        # self._show_rendering_checkbox.on_update(self.rerender)

        self._show_fg_checkbox = server.gui.add_checkbox("show fg", True)
        self._show_fg_checkbox.on_update(self.rerender)

        self._show_bg_checkbox = server.gui.add_checkbox("show bg", True)
        self._show_bg_checkbox.on_update(self.rerender)

        self._show_d_model_checkbox = server.gui.add_checkbox("show d_model", False) # just for testing
        self._show_d_model_checkbox.on_update(self.rerender)

        self._show_dqb_directly_checkbox = server.gui.add_checkbox("show DQB directly", False)
        self._show_dqb_directly_checkbox.on_update(self.rerender)

        # self._render_track_checkbox = server.gui.add_checkbox("Render tracks", False) # False
        # self._render_track_checkbox.on_update(self.rerender)

        # self._draw_track_checkbox = server.gui.add_checkbox("Draw tracks", False)
        # self._draw_track_checkbox.on_update(self.rerender)

        self._target_with_edges_checkbox = server.gui.add_checkbox("Target with edges", True)
        self._target_with_edges_checkbox.on_update(self.rerender)

        self._gui_key_pts = server.gui.add_slider(
            "key point id", min=0, max=999, step=1, initial_value=0
        ) # max=self.num_key_pts-1
        self._gui_key_pts.on_update(self.rerender)

        self._draw_lines_checkbox = server.gui.add_checkbox("Draw lines", False)
        self._draw_lines_checkbox.on_update(self.rerender)

        self._render_key_pts_checkbox = server.gui.add_checkbox("Render Key Pts only", False)
        self._render_key_pts_checkbox.on_update(self.rerender)

        self._render_key_pts_contrib_checkbox = server.gui.add_checkbox("Render Key Pts with contrib", False)
        self._render_key_pts_contrib_checkbox.on_update(self.rerender)

        # # render Guassians according to contribs
        # self._gui_ctr_threshold = server.gui.add_slider(
        #     "ctr_threshold", min=0, max=5, step=0.05, initial_value=0
        # )
        # self._gui_ctr_threshold.on_update(self.rerender)

        # self._gui_t_threshold = server.gui.add_slider(
        #     "t_threshold", min=0, max=50, step=1, initial_value=0
        # )
        # self._gui_t_threshold.on_update(self.rerender)

        # self._render_reverse_show_checkbox = server.gui.add_checkbox("Reverse Show", False)
        # self._render_reverse_show_checkbox.on_update(self.rerender)

        ########################################################################
        # Render (add keyframes)
        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )

        self._reset_button = server.gui.add_button("Reset Camera Poses")
        self._reset_button.on_click(lambda _: self.reset_camera_poses())

        self._render_button = server.gui.add_button("Render Image")
        self._render_button.on_click(lambda _: self.render_image())

        ###################################


    def track_client_cam_poses(self,look_at_tuple=None):
        server = self.server

        @server.on_client_connect
        def _(client):
            print(f"Client {client.client_id} connected")
            self.client_id = client.client_id

            # ✅ Initialize camera with first cam pose
            if self.cams is not None:
                # Get first camera's q_wc and t_wc
                first_q_wc = self.cams.q_wc[0].detach().cpu().numpy()
                first_t_wc = self.cams.t_wc[0].detach().cpu().numpy()
                
                # Set camera position and orientation
                client.camera.position = tuple(first_t_wc)
                client.camera.wxyz = tuple(first_q_wc)
                
                # Add frame for visualization
                self.server.scene.add_frame(
                    name="first_cam_frame",
                    position=tuple(first_t_wc),
                    wxyz=tuple(first_q_wc),
                    show_axes=True,
                    axes_length=0.1,
                    axes_radius=0.005,
                    visible=False,
                )

                # ✅ Set custom look-at point
                if look_at_tuple is not None:
                    client.camera.look_at = look_at_tuple #(10.0, 8.0, 3.0)  # <-- Your desired target center
                    self.server.scene.add_frame(
                        name="look_at_axes",
                        position=look_at_tuple,      # Your target center
                        wxyz=tuple(first_q_wc), # (1.0, 0.0, 0.0, 0.0),     # Identity quaternion (no rotation)
                        show_axes=True,                # Show coordinate axes
                        axes_length=0.1,               # Optional: change size
                        axes_radius=0.005,
                        visible=False,
                    )

            if str(client.client_id) not in self.camera_pose_log:
                self.camera_pose_log[str(client.client_id)] = []

            @client.camera.on_update
            def _(_: "viser.CameraHandle") -> None:
                pose_data = {
                    "wxyz": client.camera.wxyz.tolist(),
                    "position": client.camera.position.tolist(),
                    "fov": client.camera.fov,
                    "aspect": client.camera.aspect,
                }

                self.camera_pose_log[str(client.client_id)].append(pose_data) # TODO
                # self.save_camera_poses() # TODO

                # print(f"Client {client.client_id} updated camera pose")
    
    def save_camera_poses(self):
        with open(self.log_file, "w") as f:
            json.dump(self.camera_pose_log, f, indent=4)
            
    def generate_camera_pose_filename(self):
        """Generates a timestamped filename for storing camera poses."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.camera_pose_dir / f"camera_poses_{timestamp}.json"

    def reset_camera_poses(self):
        """Resets all stored camera poses and updates clients."""
        print("🔄 Resetting all camera poses...")
            

        # Clear stored camera poses
        self.camera_pose_log["0"] = self.camera_pose_log["0"][-1:]

        # Generate a new timestamped file for fresh storage
        self.save_camera_poses()  # Save the empty state

        print("✅ All camera poses reset!")
    
    def _add_3d_axes(self):
        """✅ Adds 3D coordinate axes at the world origin with labels."""
        axis_length = 1.5  # Adjustable axis length

        # ✅ Add a world frame to show X, Y, Z axes
        self.axes_frame = self.server.scene.add_frame(
            "world_axes",
            position=(0, 0, 0),  # Axes at world origin
            wxyz=(1, 0, 0, 0),  # Identity quaternion (no rotation)
            show_axes=True,  # Show X (red), Y (green), Z (blue)
            axes_length=axis_length,
        )

        # ✅ Add axis labels
        self.axis_labels = {
            "x_label": self.server.scene.add_label(
                "x_label", position=(axis_length, 0, 0), text="X"
            ),
            "y_label": self.server.scene.add_label(
                "y_label", position=(0, axis_length, 0), text="Y"
            ),
            "z_label": self.server.scene.add_label(
                "z_label", position=(0, 0, axis_length), text="Z"
            ),
        }

    def _toggle_axes(self, visible):
        """✅ Toggles the visibility of the 3D axes and labels."""
        self.axes_frame.visible = visible
        for label in self.axis_labels.values():
            label.visible = visible
    
    def access_camera(self) -> CameraState:
        """✅ Accesses the camera state for the current client."""
        client = self.server.get_clients()[self.client_id]
        camera = client.camera
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=np.concatenate(
                [
                    np.concatenate(
                        [
                            quaternion_to_matrix(torch.Tensor(camera.wxyz)),
                            np.array(camera.position)[:, None],
                        ],
                        1,
                    ),
                    [[0, 0, 0, 1]],
                ],
                0,
            ),
        )

    def render_image(self):
        if self.camera_pose_log is not None:
            try:
                self.camera_state = self.access_camera()
            except Exception as e:
                print(f"❌ Error while setting camera state: {e}")
        
        # Ensure image size exists
        if not hasattr(self, "img_wh") or self.img_wh is None:
            print("❌ No image size set! Using default resolution.")
            self.img_wh = (800, 600)

        # Ensure render function exists
        if not callable(self.render_fn):
            raise ValueError("❌ render_fn is not set properly. It must be a callable function.")
        
        # Call rendering function
        try:
            print("🚀 Calling render function...")
            img = self.render_fn(self.camera_state, self.img_wh)
            print("✅ Render function executed successfully!")
        except Exception as e:
            print(f"❌ Error while calling render_fn: {e}")
            return

        # Ensure valid NumPy array
        if not isinstance(img, np.ndarray):
            print("❌ Rendered image is not a valid NumPy array!")
            return
        
        # Ensure dtype is correct
        if img.dtype != np.uint8:
            print(f"❌ Image has incorrect dtype {img.dtype}. Converting to uint8.")
            img = (img * 255).astype(np.uint8)

        # Save the image
        success = cv2.imwrite("./output.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if success:
            print(f"✅ Image saved to ./output.png")
        else:
            print("❌ Failed to save image! Check the output path.")