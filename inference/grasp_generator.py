import torch
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData


class GraspGenerator:
    def __init__(self, saved_model, camera):
        self.saved_model = saved_model
        self.camera = camera
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

    def load_model(self):
        self.model = torch.load(self.saved_model)
        self.device = torch.device("cuda:0")

    def generate(self):
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

