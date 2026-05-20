import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from types import ModuleType

# Mock the core module properly
core_mock = ModuleType('python.kinetic_rt._core')
core_mock.AOTEngine = MagicMock()
core_mock.GraphWrapper = MagicMock()
core_mock.Serializer = MagicMock()
core_mock.Communicator = MagicMock()
core_mock.HardwareMismatchError = Exception
sys.modules['python.kinetic_rt._core'] = core_mock

import torch
from python.kinetic_rt.hardware_probe import probe_hardware, get_topology_string

class TestHardwareProbe(unittest.TestCase):
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
    @patch('torch.cuda.get_device_name', return_value='Tesla V100-SXM2-16GB')
    @patch('torch.cuda.get_device_capability', return_value=(7, 0))
    def test_probe_hardware_cuda(self, mock_get_device_capability, mock_get_device_name, mock_device_count, mock_is_available):
        # ensure we're not on ROCm (hasattr torch.version.hip is False or None)
        with patch('torch.version') as mock_version:
            mock_version.hip = None
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, "2x Tesla V100-SXM2-16GB (Compute 7.0)")
            self.assertEqual(backend, "CUDA")
            self.assertEqual(arch, "sm70")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='AMD Radeon RX 7900 XTX')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_with_props(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        mock_props = MagicMock()
        mock_props.gcnArchName = 'gfx1100:sramecc+'
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, "1x AMD Radeon RX 7900 XTX")
            self.assertEqual(backend, "ROCm")
            self.assertEqual(arch, "gfx1100")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='AMD Radeon RX gfx906')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_fallback_name(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        mock_props = MagicMock()
        del mock_props.gcnArchName
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, "1x AMD Radeon RX gfx906")
            self.assertEqual(backend, "ROCm")
            self.assertEqual(arch, "gfx906")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_name', return_value='Unknown AMD GPU')
    @patch('torch.cuda.get_device_properties')
    def test_probe_hardware_rocm_fallback_default(self, mock_get_device_properties, mock_get_device_name, mock_device_count, mock_is_available):
        mock_props = MagicMock()
        del mock_props.gcnArchName
        mock_get_device_properties.return_value = mock_props

        with patch('torch.version') as mock_version:
            mock_version.hip = '5.7.0'
            topology, backend, arch = probe_hardware()
            self.assertEqual(topology, "1x Unknown AMD GPU")
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
