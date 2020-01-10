from inference.grasp_generator import GraspGenerator


if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='saved_data/jacquard_rgbd_iou_0.94',
        visualize=True
    )
    generator.load_model()
    generator.run()
