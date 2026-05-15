import subprocess
import os

def probe_hardware():
    forced_arch = os.environ.get("KINETIC_FORCE_ARCH")
    if forced_arch:
        backend = "CUDA" if "sm" in forced_arch else "ROCm"
        return "1x Overridden GPU", backend, forced_arch

    # If MOCK_HIP is defined or we are in a testing environment without GPUs, mock it
    # We can check for nvidia-smi or rocm-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            num_gpus = len(lines)
            if num_gpus > 0:
                name, cap = lines[0].split(', ')
                arch = f"sm{cap.replace('.', '')}"
                return f"{num_gpus}x {name} (Compute {cap})", "CUDA", arch
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            num_gpus = len([l for l in lines if 'GPU' in l and 'Card series' in l])
            if num_gpus > 0:
                name = "AMD Radeon" # simplified
                return f"{num_gpus}x {name}", "ROCm", os.environ.get('MOCK_ROCM_ARCH', 'gfx1100')
    except FileNotFoundError:
        pass

    # Headless CI Resilience
    return "CPU Only (Headless)", "CPU", "cpu"

def get_topology_string():
    topology, backend, arch = probe_hardware()
    if backend == "CPU":
        return "[Kinetic-RT] Hardware Detected: CPU Only (Headless)"
    return f"[Kinetic-RT Init] Detected Topology: {topology} | Backend: {backend} | Arch: {arch}"
