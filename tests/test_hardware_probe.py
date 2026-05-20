import os
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.cuda = ModuleType("torch.cuda")
    torch.version = ModuleType("torch.version")
    torch.version.hip = None
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = torch.cuda
    sys.modules["torch.version"] = torch.version

if not hasattr(torch, "Tensor"):
    torch.Tensor = type("Tensor", (), {})
if not hasattr(torch, "cuda"):
    torch.cuda = ModuleType("torch.cuda")
if not hasattr(torch, "version"):
    torch.version = ModuleType("torch.version")
if not hasattr(torch.version, "hip"):
    torch.version.hip = None
if not hasattr(torch.cuda, "is_available"):
    torch.cuda.is_available = lambda: False
if not hasattr(torch.cuda, "device_count"):
    torch.cuda.device_count = lambda: 0
if not hasattr(torch.cuda, "get_device_name"):
    torch.cuda.get_device_name = lambda *_: ""
if not hasattr(torch.cuda, "get_device_capability"):
    torch.cuda.get_device_capability = lambda *_: (0, 0)
if not hasattr(torch.cuda, "get_device_properties"):
    torch.cuda.get_device_properties = lambda *_: ModuleType("props")

class TestHardwareProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We patch sys.modules inside tests rather than globally to prevent leaking
        cls.core_mock = ModuleType('python.kinetic_rt._core')
        cls.core_mock.AOTEngine = MagicMock()
        cls.core_mock.GraphWrapper = MagicMock()
        cls.core_mock.Serializer = MagicMock()
        cls.core_mock.Communicator = MagicMock()
        cls.core_mock.HardwareMismatchError = Exception

        cls.sys_modules_patcher = patch.dict(sys.modules, {'python.kinetic_rt._core': cls.core_mock})
        cls.sys_modules_patcher.start()

        # Import the module to test AFTER the mock is in place
        global probe_hardware, get_topology_string
        from python.kinetic_rt.hardware_probe import probe_hardware, get_topology_string

    @classmethod
    def tearDownClass(cls):
        cls.sys_modules_patcher.stop()

    def setUp(self):
        # Clear environment for tests
        if 'KINETIC_FORCE_ARCH' in os.environ:
            del os.environ['KINETIC_FORCE_ARCH']

    @patch.dict(os.environ, {"KINETIC_FORCE_ARCH": "sm80"})
    def test_probe_hardware_forced_cuda(self):
        topology, backend, arch = probe_hardware()
        self.assertEqual(topology, "1x Overridden GPU")
        self.assertEqual(backend, "CUDA")
        self.assertEqual(arch, "sm80")

    @patch.dict(os.environ, {"KINETIC_FORCE_ARCH": "gfx90a"})
    def test_probe_hardware_forced_rocm(self):
        topology, backend, arch = probe_hardware()
        self.assertEqual(topology, "1x Overridden GPU")
        self.assertEqual(backend, "ROCm")
        self.assertEqual(arch, "gfx90a")

    @patch('torch.cuda.is_available', return_value=False)
    def test_probe_hardware_cpu_fallback(self, mock_is_available):
        topology, backend, arch = probe_hardware()
        self.assertEqual(topology, "CPU Only (Headless)")
        self.assertEqual(backend, "CPU")
        self.assertEqual(arch, "CPU")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_capability', return_value=(7, 0))
    def test_probe_hardware_cuda(self, mock_get_device_capability, mock_get_device_name, mock_device_count, mock_is_available):
        device_id = 0
        mock_get_device_name.side_effect = lambda idx: f"GPU-{idx}"
        # ensure we're not on ROCm (hasattr torch.version.hip is False or None)
        with patch('torch.version') as mock_version:
            mock_version.hip = None
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, f"2x {torch.cuda.get_device_name(device_id)} (Compute 7.0)")
            self.assertEqual(backend, "CUDA")
            self.assertEqual(arch, "sm70")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_with_props(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        device_id = 0
        mock_get_device_name.side_effect = lambda idx: f"GPU-{idx}"
        mock_props = MagicMock()
        mock_props.gcnArchName = 'gfx1100:sramecc+'
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, f"1x {torch.cuda.get_device_name(device_id)}")
            self.assertEqual(backend, "ROCm")
            self.assertEqual(arch, "gfx1100")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_fallback_name(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        device_id = 0
        mock_get_device_name.side_effect = lambda idx: "GPU gfx906"
        mock_props = MagicMock()
        del mock_props.gcnArchName
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, f"1x {torch.cuda.get_device_name(device_id)}")
            self.assertEqual(backend, "ROCm")
            self.assertEqual(arch, "gfx906")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_fallback_default(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        device_id = 0
        mock_get_device_name.side_effect = lambda idx: f"GPU-{idx}"
        mock_props = MagicMock()
        del mock_props.gcnArchName
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, f"1x {torch.cuda.get_device_name(device_id)}")
            self.assertEqual(backend, "ROCm")
            self.assertEqual(arch, "gfx90a")

    def test_get_topology_string(self):
        with patch('python.kinetic_rt.hardware_probe.probe_hardware', return_value=("1x Overridden GPU", "CUDA", "sm80")):
            result = get_topology_string()
            self.assertEqual(result, "[Kinetic-RT Init] Detected Topology: 1x Overridden GPU | Backend: CUDA | Arch: sm80")

    def test_get_topology_string_cpu(self):
        with patch('python.kinetic_rt.hardware_probe.probe_hardware', return_value=("CPU Only (Headless)", "CPU", "CPU")):
            result = get_topology_string()
            self.assertEqual(result, "[Kinetic-RT] Hardware Detected: CPU Only (Headless)")


if __name__ == '__main__':
    unittest.main()
