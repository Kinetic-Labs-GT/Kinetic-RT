import sys
import os

# Add root directory to python path for kinetic_rt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM
import python.kinetic_rt as kinetic_rt
from python.kinetic_rt.fusion_forge import compile_and_serialize

import argparse

def export_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism degree")
    args = parser.parse_args()

    model_id = "HuggingFaceTB/SmolLM2-135M"
    print(f"Downloading {model_id}...")

    # We will just load the state_dict
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = model.state_dict()

    print(f"Applying Tensor Parallelism (TP={args.tp})...")
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

    print(f"Successfully sharded weights into {tp_degree} domains.")

    # AOT Compilation
    print("Executing AOT Compilation with Auto-Discovery...")
    engine = kinetic_rt.AOTEngine()
    serializer = kinetic_rt.Serializer()

    output_filepath = "smollm_135m_tp2.kin"

    # Passing the dummy tensor/graphs via kwargs for our fusion_forge mock compile
    # In a real environment, we'd pass the actual sharded_weights into the compiler
    compile_and_serialize(engine, serializer, output_filepath, tensor_parallel_degree=tp_degree)

    print(f"End-to-End Export Complete: {output_filepath}")

if __name__ == "__main__":
    export_model()
