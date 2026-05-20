import unittest
import sys
from types import ModuleType
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch

if not hasattr(sys.modules.get("torch", {}), "Tensor"):
    sys.modules["torch"].Tensor = type("Tensor", (), {})
from python.kinetic_rt.hardware_probe import get_topology_string, probe_hardware

class TestHardwareProbe(unittest.TestCase):
    def test_get_topology_string(self):
        topology, backend, arch = probe_hardware()
        result = get_topology_string()

        if backend == "CPU":
            self.assertEqual(result, "[Kinetic-RT] Hardware Detected: CPU Only (Headless)")
        else:
            self.assertEqual(result, f"[Kinetic-RT Init] Detected Topology: {topology} | Backend: {backend} | Arch: {arch}")

if __name__ == '__main__':
    unittest.main()
