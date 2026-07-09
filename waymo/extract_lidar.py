from pathlib import Path
import traceback

import numpy as np
import tensorflow as tf
import laspy
from tqdm import tqdm

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def extract_global_las(
    tfrecord_path: str,
    output_las: str,
):
    """
    Extract TOP lidar from a Waymo segment,
    transform all scans into world coordinates,
    merge them into a single point cloud,
    and save as LAS.
    """

    # Count frames for progress bar
    num_frames = sum(
        1 for _ in tf.data.TFRecordDataset(
            str(tfrecord_path),
            compression_type=""
        )
    )

    dataset = tf.data.TFRecordDataset(
        str(tfrecord_path),
        compression_type=""
    )

    all_points_world = []

    for data in tqdm(
        dataset,
        total=num_frames,
        desc=Path(tfrecord_path).stem[:60],
        unit="frame",
        leave=False,
    ):

        frame = dataset_pb2.Frame()
        frame.ParseFromString(data.numpy())

        # Vehicle -> World transform
        pose = np.array(
            frame.pose.transform,
            dtype=np.float64,
        ).reshape(4, 4)

        (
            range_images,
            camera_projections,
            _,
            range_image_top_pose,
        ) = frame_utils.parse_range_image_and_camera_projection(
            frame
        )

        points_return1, _ = (
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
            )
        )

        points_return2, _ = (
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1,
            )
        )

        lidar_ids = [
            calib.name
            for calib in frame.context.laser_calibrations
        ]
        lidar_ids.sort()

        for lidar_id, p1, p2 in zip(
            lidar_ids,
            points_return1,
            points_return2,
        ):

            lidar_name = (
                dataset_pb2.LaserName.Name.Name(
                    lidar_id
                )
            )

            # Only keep TOP lidar
            if lidar_name != "TOP":
                continue

            points_vehicle = np.concatenate(
                [p1, p2],
                axis=0
            )

            # Convert vehicle-frame points to world-frame
            points_h = np.hstack(
                [
                    points_vehicle,
                    np.ones(
                        (points_vehicle.shape[0], 1),
                        dtype=np.float64,
                    ),
                ]
            )

            points_world = (
                pose @ points_h.T
            ).T[:, :3]

            all_points_world.append(points_world)

    if len(all_points_world) == 0:
        raise RuntimeError(
            f"No TOP lidar points found in {tfrecord_path}"
        )

    points = np.concatenate(
        all_points_world,
        axis=0
    )

    # Create LAS
    header = laspy.LasHeader(
        point_format=3,
        version="1.2"
    )

    # 1 mm precision
    header.scales = np.array(
        [0.001, 0.001, 0.001]
    )

    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    las.write(output_las)


if __name__ == "__main__":

    INPUT_DIR = Path("/data_1/claire/waymo")
    OUTPUT_DIR = Path("/data_1/claire/waymo_las")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tfrecord_files = sorted(
        INPUT_DIR.rglob("*.tfrecord")
    )

    print(
        f"Found {len(tfrecord_files)} tfrecord files"
    )

    for tfrecord_path in tqdm(
        tfrecord_files,
        desc="Segments",
        unit="segment",
    ):

        output_las = (
            OUTPUT_DIR /
            f"{tfrecord_path.stem}.las"
        )

        # Skip already processed files
        if output_las.exists():
            continue

        try:

            extract_global_las(
                str(tfrecord_path),
                str(output_las),
            )

        except Exception:

            print(
                f"\nERROR while processing:\n"
                f"{tfrecord_path}"
            )

            traceback.print_exc()

            # Continue with next segment
            continue

    print("Done.")