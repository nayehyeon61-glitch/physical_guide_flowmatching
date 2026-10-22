# scripts/run_inference.py
"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms,
with or without modification, are permitted. See the LICENSE file in the project root for
the full license text.
"""
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import AutoregressiveInference, InferenceConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run autoregressive inference (KAI-Atmos)")
    parser.add_argument("--config", type=str, default="./configs/inference.yaml",
                        help="Path to inference YAML config file")
    args = parser.parse_args()

    # Load config, validate, build engine, run — fully automated
    config = InferenceConfig.from_yaml(args.config)
    engine = AutoregressiveInference(config)
    engine.run()


if __name__ == "__main__":
    main()
