import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width_rgb=1280,
                 height_rgb=720,
                 width_depth=1280,
                 height_depth=720,
                 fps=30):

        self.device_id = device_id
        self.width_rgb = width_rgb
        self.height_rgb = height_rgb
        self.width_depth = width_depth
        self.height_depth = height_depth
        self.fps = fps

        self.pipeline = None
        self.align_to_rgb = None
        self.scale = None
        self.rgb_intrinsics = None
        self.depth_intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width_depth, self.height_depth, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width_rgb, self.height_rgb, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        rgb_profile = cfg.get_stream(rs.stream.color)
        depth_profile = cfg.get_stream(rs.stream.depth)
        self.rgb_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        # Alignment processors
        self.align_to_rgb = rs.align(rs.stream.color)

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        # RGB
        cf = frames.get_color_frame()
        rgb = np.asanyarray(cf.get_data())

        # Depth aligned to RGB
        color_aligned_frames = self.align_to_rgb.process(frames)
        aligned_depth_frame = color_aligned_frames.get_depth_frame()

        rgb_aligned_depth = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        rgb_aligned_depth *= self.scale * 1000.0

        depth = np.expand_dims(rgb_aligned_depth, axis=2)

        return{
            'rgb': rgb,
            'aligned_depth': depth,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()
        # plt.savefig('cam435.png')


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    while True:
        cam.plot_image_bundle()
        time.sleep(1)
