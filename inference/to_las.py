import torch
import numpy as np
import laspy
import os
import glob

# Path to your results folder
results_folder = ".results/dales2_2026-06-08_16-55-29/"
output_folder = "./results/dales2_2026-06-08_16-55-29/las_files/"
os.makedirs(output_folder, exist_ok=True)

# Find all pkl files
# pkl_files = sorted(glob.glob(os.path.join(results_folder, "result_dict_*.pkl")))
pkl_files = ["results/dales2_2026-06-08_16-55-29/result_dict_0.pkl"]

for pkl_path in pkl_files:
    # Load the pkl
    result = torch.load(pkl_path, map_location='cpu')
    
    # Get fine point cloud (the high-res one you care about)
    xyz     = result['fine_xyz']       # shape (N, 3)
    normals = result['fine_normal']    # shape (N, 3)
    labels  = result['fine_semantic']  # shape (N,) — integer class per point

    # Create LAS file (version 1.4, point format 0)
    header = laspy.LasHeader(point_format=0, version="1.4")
    las = laspy.LasData(header=header)

    # Write XYZ coordinates
    las.x = xyz[:, 0].astype(np.float64)
    las.y = xyz[:, 1].astype(np.float64)
    las.z = xyz[:, 2].astype(np.float64)

    # Write semantic labels as classification field
    las.classification = labels.astype(np.uint8)

    # Save
    sample_id = os.path.basename(pkl_path).replace("result_dict_", "").replace(".pkl", "")
    out_path = os.path.join(output_folder, f"generated_scene_{sample_id}.las")
    las.write(out_path)
    print(f"Saved {out_path} — {len(xyz)} points")

print("Done!")