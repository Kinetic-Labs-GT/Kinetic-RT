import subprocess
import os
import re
import torch

def probe_hardware():
    forced_arch = os.environ.get("KINETIC_FORCE_ARCH")
    if forced_arch:
        backend = "CUDA" if "sm" in forced_arch else "ROCm"
        return "1x Overridden GPU", backend, forced_arch

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)

        # Check if we are on ROCm (AMD) via torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # We can use ROCm smi or hip info if we want, but PyTorch doesn't expose
            # an easy arch string without torch.cuda.get_device_properties(0).gcnArchName
            # Let's try to get it from properties
            props = torch.cuda.get_device_properties(0)
            arch = getattr(props, 'gcnArchName', None)
            if not arch:
                match = re.search(r"(gfx\d+[a-z0-9]*)", name)
                arch = match.group(1) if match else "gfx90a"
            else:
                # Sometimes gcnArchName has a prefix or suffix, like gfx90a:sramecc+...
                arch = arch.split(':')[0]
            return f"{num_gpus}x {name}", "ROCm", arch
        else:
            # CUDA
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm{cap[0]}{cap[1]}"
            return f"{num_gpus}x {name} (Compute {cap[0]}.{cap[1]})", "CUDA", arch

    # Headless CI Resilience
    return "CPU Only (Headless)", "CPU", "CPU"

def get_topology_string():
    topology, backend, arch = probe_hardware()
    if backend == "CPU":
        return "[Kinetic-RT] Hardware Detected: CPU Only (Headless)"
    return f"[Kinetic-RT Init] Detected Topology: {topology} | Backend: {backend} | Arch: {arch}"
