from tasks.pnp import PickAndPlace


if __name__ == '__main__':
    pnp = PickAndPlace(
        robot_ip='127.0.0.1',
        robot_port=1000,
        cam_id=830112070066,
        saved_model='saved_data/jacquard_rgbd_iou_0.94',
        hover_distance=0.15,
        place_position=[1, 2, 3]
    )
    pnp.run()
