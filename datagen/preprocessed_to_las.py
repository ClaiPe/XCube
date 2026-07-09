import os
import glob
import torch
import laspy
import numpy as np

INPUT_DIR = "data/preprocessed/320/train"
OUTPUT_DIR = "debug_las/train"


def pkl_to_las(pkl_path, las_path):
    print(f"Converting: {pkl_path}")

    data = torch.load(pkl_path, map_location="cpu")

    grid = data["points"]

    # voxel center coordinates
    xyz = grid.grid_to_world(grid.ijk.float()).jdata.numpy()

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)

    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # semantic labels
    if "semantics" in data:
        las.classification = (
            data["semantics"]
            .numpy()
            .astype(np.uint8)
        )

    os.makedirs(os.path.dirname(las_path), exist_ok=True)
    las.write(las_path)

    print(f"  -> {las_path} ({len(xyz):,} points)")


def main():
    pkl_files = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, "**", "*.pkl"),
            recursive=True,
        )
    )

    print(f"Found {len(pkl_files)} pkl files")

    for pkl_path in pkl_files:

        rel_path = os.path.relpath(pkl_path, INPUT_DIR)

        las_path = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(rel_path)[0] + ".las"
        )

        pkl_to_las(pkl_path, las_path)

    print("Done.")


if __name__ == "__main__":
    main()