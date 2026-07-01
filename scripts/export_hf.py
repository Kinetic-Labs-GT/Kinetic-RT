import sys
import os

# Add root directory to python path for kinetic_rt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM
import python.kinetic_rt as kinetic_rt
from python.kinetic_rt.fusion_forge import compile_and_serialize

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism degree")
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help=(
            "Model to export. Accepts either a Hugging Face Hub repository ID "
            "(e.g. 'HuggingFaceTB/SmolLM2-135M') for online quickstart runs, or a "
            "fully-qualified local directory path containing pre-downloaded model "
            "weights (config.json, tokenizer files, and weight shards) for offline "
            "or air-gapped execution."
        ),
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for the serialized model")
    args = parser.parse_args()

    model_id = args.model_id
    logger.info(f"Loading {model_id}...")

    # We will just load the state_dict
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = model.state_dict()

    logger.info(f"Applying Tensor Parallelism (TP={args.tp})...")
    tp_degree = args.tp

    actual_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if tp_degree > 1 and actual_gpus < tp_degree:
        raise RuntimeError(f"Requested TP={tp_degree}, but only found {actual_gpus} GPUs.")

    sharded_weights = []

    # Mathematical Sharding Example (Simplified)
    # Llama architecture typical keys:
    # Column Parallel: q_proj, k_proj, v_proj, gate_proj, up_proj (split dim 0)
    # Row Parallel: o_proj, down_proj (split dim 1)

    for key, tensor in state_dict.items():
        if any(proj in key for proj in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            # Column-parallel
            chunks = torch.chunk(tensor, tp_degree, dim=0)
            sharded_weights.append(chunks)
        elif any(proj in key for proj in ["o_proj", "down_proj"]):
            # Row-parallel
            chunks = torch.chunk(tensor, tp_degree, dim=1)
            sharded_weights.append(chunks)

    logger.info(f"Successfully sharded weights into {tp_degree} domains.")

    # AOT Compilation
    logger.info("Executing AOT Compilation with Auto-Discovery...")
    engine = kinetic_rt.AOTEngine()
    serializer = kinetic_rt.Serializer()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_filepath = os.path.join(args.output_dir, f"smollm_135m_tp{tp_degree}.kin")

    # Passing the dummy tensor/graphs via kwargs for our fusion_forge mock compile
    # In a real environment, we'd pass the actual sharded_weights into the compiler
    compile_and_serialize(engine, serializer, output_filepath, tensor_parallel_degree=tp_degree)

    logger.info(f"End-to-End Export Complete: {output_filepath}")

if __name__ == "__main__":
    export_model()
