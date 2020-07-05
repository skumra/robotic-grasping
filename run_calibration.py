#!/usr/bin/env python
import numpy as np

from hardware.calibrate_camera import Calibration

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=830112070066,
        calib_grid_step=0.05,
        checkerboard_offset_from_tool=[0.0, 0.0215, 0.0115],
        workspace_limits=np.asarray([[0.55, 0.65], [-0.2, -0.1], [0.0, 0.2]])
    )
    calibration.run()
