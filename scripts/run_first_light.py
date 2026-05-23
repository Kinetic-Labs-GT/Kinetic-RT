import sys
import os
import argparse
import logging

# Add root directory to python path for kinetic_rt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import python.kinetic_rt as kinetic_rt
from python.kinetic_rt.orchestrator import KineticRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the .kin model file")
    args = parser.parse_args()

    logger.info("Initializing AOTEngine and GraphWrapper...")
    engine = kinetic_rt.AOTEngine()
    wrapper = kinetic_rt.GraphWrapper()

    runtime = KineticRuntime(engine, wrapper)
    logger.info(f"Dynamically discovering topology and loading models from {args.model_dir}...")
    runtime.load_model(args.model_dir)

    prompt = "The capital of India is "
    logger.info(f"Generating continuation for: '{prompt}'")

    output = runtime.generate(prompt)
    logger.info(f"Generated text: {output}")

if __name__ == "__main__":
    run_inference()
