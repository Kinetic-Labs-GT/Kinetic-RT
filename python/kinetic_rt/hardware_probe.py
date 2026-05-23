import subprocess
import os
import torch

def probe_hardware():
    forced_target = os.environ.get("KINETIC_TARGET")
    if forced_target:
        backend = "CUDA" if "sm" in forced_target else "ROCm"
        return "1x Overridden GPU", backend, forced_target

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)

        # Check if we are on ROCm (AMD) via torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # We can use ROCm smi or hip info if we want, but PyTorch doesn't expose
            # an easy target string without torch.cuda.get_device_properties(0).gcnArchName
            # Let's try to get it from properties
            props = torch.cuda.get_device_properties(0)
            target = getattr(props, 'gcnArchName', None)
            if not target:
                raise RuntimeError("Architecture could not be dynamically probed.")
            else:
                # Sometimes gcnArchName has a prefix or suffix, like gfx90a:sramecc+...
                target = target.split(':')[0]
            return f"{num_gpus}x {name}", "ROCm", target
        else:
            # CUDA
            cap = torch.cuda.get_device_capability(0)
            target = f"sm{cap[0]}{cap[1]}"
            return f"{num_gpus}x {name} (Compute {cap[0]}.{cap[1]})", "CUDA", target

    # Headless CI Resilience
    return "CPU Only (Headless)", "CPU", "CPU"

def get_topology_string():
    topology, backend, target = probe_hardware()
    if backend == "CPU":
        return "[Kinetic-RT] Hardware Detected: CPU Only (Headless)"
    return f"[Kinetic-RT Init] Detected Topology: {topology} | Backend: {backend} | Target: {target}"
