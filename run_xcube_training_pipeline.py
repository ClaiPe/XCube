#!/usr/bin/env python3
import os
import subprocess
import sys
from typing import List, Tuple


def run_step(step_index: int, total_steps: int, config_path: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    command: List[str] = ["python", "train.py", config_path]

    print(f"\n[{step_index}/{total_steps}] Running: CUDA_VISIBLE_DEVICES=0 {' '.join(command)}")
    result = subprocess.run(command, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(
            f"\nStep {step_index} failed with exit code {result.returncode}: {config_path}",
            file=sys.stderr,
        )
    else:
        print(f"Step {step_index} completed successfully: {config_path}")

    return result.returncode


def main() -> int:
    configs: List[Tuple[str, str]] = [
        ("train VAE dense", "./configs/dales2/train_vae_dense.yaml"),
        ("train VAE sparse", "./configs/dales2/train_vae_sparse.yaml"),
        ("train diffusion dense", "./configs/dales2/train_diffusion_dense.yaml"),
        ("train diffusion sparse", "./configs/dales2/train_diffusion_sparse.yaml"),
    ]

    print("Starting XCube training pipeline...")

    for idx, (_, config_path) in enumerate(configs, start=1):
        code = run_step(idx, len(configs), config_path)
        if code != 0:
            return code

    print("\nAll training steps completed successfully !")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
