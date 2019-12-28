import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 config_json_path=None,
                 width_rgb=1280,
                 height_rgb=720,
                 width_depth=1280,
                 height_depth=720,
                 fps=30):

        self.device_id = device_id
        self.config_json_path = config_json_path

        self.width_rgb = width_rgb
        self.height_rgb = height_rgb
        self.width_depth = width_depth
        self.height_depth = height_depth
        self.fps = fps
        self.pipeline = None
        self.align_to_rgb = None
        self.scale = None

    def connect(self):
        # Select the correct device from all available RealSenses
        ctx = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in ctx.query_devices()}
        device = devices[str(self.device_id)]

        # Load JSON file with RealSense configuration. These files can be generated with the RealSense viewer.
        advanced_device = rs.rs400_advanced_mode(device)

        if self.config_json_path:
            try:
                with open(self.config_json_path, 'r') as f:
                    advanced_device.load_json(f.read())
            except FileNotFoundError:
                raise FileExistsError(
                    "Could not find RealSense configuration file at %s" % self.config_json_path)
            except:  # noqa
                logger.exception('Could not load Realsense JSON configuration file %s', self.config_json_path)
                raise

        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width_depth, self.height_depth, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width_rgb, self.height_rgb, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.infrared, 1, self.width_depth, self.height_depth, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.infrared, 2, self.width_depth, self.height_depth, rs.format.y8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics and extrinsics
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        rgb_profile = cfg.get_stream(rs.stream.color)
        infrared1_profile = cfg.get_stream(rs.stream.infrared, 1)
        infrared2_profile = cfg.get_stream(rs.stream.infrared, 2)
        depth_profile = cfg.get_stream(rs.stream.depth)
        rgb_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        # https://github.com/IntelRealSense/librealsense/issues/867#issuecomment-352733067
        infrared_intrinsics = infrared1_profile.as_video_stream_profile().get_intrinsics()
        extrinsics_infrared1_to_rgb = infrared1_profile.get_extrinsics_to(rgb_profile)
        extrinsics_infrared2_to_rgb = infrared2_profile.get_extrinsics_to(rgb_profile)
        extrinsics_depth_to_rgb = depth_profile.get_extrinsics_to(rgb_profile)

        extrinsics_depth = np.eye(4)
        extrinsics_depth[:3, :3] = np.array(extrinsics_depth_to_rgb.rotation).reshape(3, 3)
        extrinsics_depth[:3, 3] = np.array([1000. * x for x in extrinsics_depth_to_rgb.translation])

        # The rotation in extrinsics_infrared(1/2)_to_rgb is stored as a 9 element list.
        # We save the rotation and translation as a 4x4 homogenous matrix with translation in units of mm.
        extrinsics_ir1 = np.eye(4)
        extrinsics_ir1[:3, :3] = np.array(extrinsics_infrared1_to_rgb.rotation).reshape(3, 3)
        extrinsics_ir1[:3, 3] = np.array([1000. * x for x in extrinsics_infrared1_to_rgb.translation])

        extrinsics_ir2 = np.eye(4)
        extrinsics_ir2[:3, :3] = np.array(extrinsics_infrared2_to_rgb.rotation).reshape(3, 3)
        extrinsics_ir2[:3, 3] = np.array([1000. * x for x in extrinsics_infrared2_to_rgb.translation])

        # Alignment processors
        self.align_to_rgb = rs.align(rs.stream.color)

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)

        color_aligned_frames = self.align_to_rgb.process(frames)
        aligned_depth_frame = color_aligned_frames.get_depth_frame()

        # RGB
        image = np.asanyarray(cf.get_data())

        # Depth aligned to RGB
        rgb_aligned_depth = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        rgb_aligned_depth *= self.scale * 1000.0

        raw_depth = np.asarray(df.get_data(), dtype=np.float32)
        raw_depth *= self.scale * 1000.0

        # Handle unknown values properly https://communities.intel.com/thread/121826
        # rgb_aligned_depth[rgb_aligned_depth <= 0.0001] = np.nan
        # rgb_aligned_depth[rgb_aligned_depth > 65534] = np.nan
        #
        # raw_depth[raw_depth <= 0.0001] = np.nan
        # raw_depth[raw_depth > 65534] = np.nan

        depth = np.expand_dims(rgb_aligned_depth, axis=2)
        unaligned_depth = np.expand_dims(raw_depth, axis=2)

        infrared1 = np.expand_dims(np.asanyarray(ir1.get_data()), axis=2)
        infrared2 = np.expand_dims(np.asanyarray(ir2.get_data()), axis=2)

        return{
            'rgb': image,
            'aligned_depth': depth,
            'unaligned_depth': unaligned_depth,
            'infrared1': infrared1,
            'infrared2': infrared2
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        # yield 'rgb: shape: %s mean: %s' % (rgb.shape, np.mean(rgb))
        depth = images['aligned_depth']
        # yield 'aligned_depth: shape: %s mean: %s' % (depth.shape, np.nanmean(depth))
        infrared1 = images['infrared1']
        infrared2 = images['infrared2']
        # yield 'infrared1: shape: %s mean: %s' % (infrared1.shape, np.mean(infrared1))
        # yield 'infrared2: shape: %s mean: %s' % (infrared2.shape, np.mean(infrared2))

        fig, ax = plt.subplots(2, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[1, 0].imshow(infrared1.squeeze(axis=2), vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[1, 1].imshow(infrared2.squeeze(axis=2), vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')
        ax[1, 0].set_title('infrared1')
        ax[1, 1].set_title('infrared2')

        # plt.show()
        plt.savefig('cam435.png')


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    cam.plot_image_bundle()
