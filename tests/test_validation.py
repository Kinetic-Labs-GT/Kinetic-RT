import os
import unittest
import sys
from types import ModuleType

# Mock dependencies to allow importing fusion_forge without installing triton/torch
if "triton" not in sys.modules:
    triton = ModuleType("triton")
    triton.jit = lambda x: x
    sys.modules["triton"] = triton
    triton.language = ModuleType("triton.language")
    triton.language.constexpr = int
    sys.modules["triton.language"] = triton.language

if "torch" not in sys.modules:
    torch = ModuleType("torch")
    sys.modules["torch"] = torch

from python.kinetic_rt.fusion_forge import validate_compilation, TritonCompilationError

class TestValidateCompilation(unittest.TestCase):
    def test_valid_hsaco(self):
        # Create a valid mock HSACO
        compiled_hsaco = bytearray(64)
        compiled_hsaco[0:4] = b"\x7fELF"
        compiled_hsaco[4] = 2 # 64-bit class
        compiled_hsaco[18:20] = b"\xE0\x00" # EM_AMDGPU
        compiled_hsaco = bytes(compiled_hsaco)

        # Should not raise any error
        validate_compilation(compiled_hsaco, "ROCm")

    def test_empty_binary(self):
        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(b"", "ROCm")
        self.assertEqual(str(cm.exception), "Triton compilation yielded an empty binary.")

    def test_missing_elf_magic(self):
        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(b"NOT_AN_ELF_BINARY", "ROCm")
        self.assertEqual(str(cm.exception), "Triton binary lacks the standard ELF magic header.")

    def test_too_short_binary(self):
        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(b"\x7fELF", "ROCm") # Only 4 bytes
        self.assertEqual(str(cm.exception), "Triton binary is too short to be a valid ELF.")

    def test_incorrect_class_identifier(self):
        compiled_hsaco = bytearray(64)
        compiled_hsaco[0:4] = b"\x7fELF"
        compiled_hsaco[4] = 1 # 32-bit instead of 64-bit
        compiled_hsaco[18:20] = b"\xE0\x00"

        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(bytes(compiled_hsaco), "ROCm")
        self.assertEqual(str(cm.exception), "Triton binary is not a 64-bit ELF.")

    def test_incorrect_architecture_identifier(self):
        compiled_hsaco = bytearray(64)
        compiled_hsaco[0:4] = b"\x7fELF"
        compiled_hsaco[4] = 2
        compiled_hsaco[18:20] = b"\x3E\x00" # x86-64 instead of AMDGPU

        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(bytes(compiled_hsaco), "ROCm")
        self.assertEqual(str(cm.exception), "Triton binary architecture is not AMDGPU.")

    def test_partial_architecture_identifier(self):
        # Case where only the first byte matches 0xE0 but it's not EM_AMDGPU (0x00E0)
        compiled_hsaco = bytearray(64)
        compiled_hsaco[0:4] = b"\x7fELF"
        compiled_hsaco[4] = 2
        compiled_hsaco[18:20] = b"\xE0\xFF"

        with self.assertRaises(TritonCompilationError) as cm:
            validate_compilation(bytes(compiled_hsaco), "ROCm")
        self.assertEqual(str(cm.exception), "Triton binary architecture is not AMDGPU.")

if __name__ == "__main__":
    unittest.main()


import python.kinetic_rt as kinetic_rt

class TestHardwareMismatchError(unittest.TestCase):
    def test_hardware_mismatch_exception_propagation(self):
        engine = kinetic_rt.AOTEngine()
        serializer = kinetic_rt.Serializer()

        # Test file mismatch
        filepath_mismatch = "test_mismatch.kin"
        try:
            serializer.save_kin_file(filepath_mismatch, "gfx942", "ROCm_gfx942", 12345, [], [])
            # AOTEngine expects "gfx1100" by default
            engine.load_model(filepath_mismatch)
            self.fail("Should have raised HardwareMismatchError")
        except Exception as e:
            # We must catch kinetic_rt.HardwareMismatchError
            self.assertTrue(isinstance(e, kinetic_rt.HardwareMismatchError))
            self.assertIn("Hardware mismatch", str(e))
        finally:
            if os.path.exists(filepath_mismatch):
                os.remove(filepath_mismatch)
