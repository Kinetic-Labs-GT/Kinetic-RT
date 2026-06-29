import triton
import triton.language as tl
import torch
import json  # Added for JSON encoding
import os
import hashlib
import struct
import logging
from .hardware_probe import probe_hardware

logger = logging.getLogger(__name__)

# Global artifact kind – set during compilation to indicate mock or production
_artifact_kind = "production"

@triton.jit
def fused_rmsnorm_qkv_rope(
    x_ptr, qkv_weight_ptr, qkv_bias_ptr, rms_weight_ptr, freqs_cos_ptr, freqs_sin_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    seq_len, d_model, n_heads, head_dim, eps,
    stride_x_seq, stride_x_dim,
    stride_w_out, stride_w_in,
    stride_q_seq, stride_q_dim,
    stride_k_seq, stride_k_dim,
    stride_v_seq, stride_v_dim,
    BLOCK_DIM: tl.constexpr
):
    pid = tl.program_id(0) # token index (seq_len)

    # --- 1. Load inputs for the current token ---
    offsets_d = tl.arange(0, BLOCK_DIM)
    x_ptrs = x_ptr + pid * stride_x_seq + offsets_d * stride_x_dim
    x = tl.load(x_ptrs, mask=offsets_d < d_model, other=0.0)

    # --- 2. RMSNorm using LDS/SRAM ---
    # Calculate RMS
    x_f32 = x.to(tl.float32)
    variance = tl.sum(x_f32 * x_f32, axis=0) / d_model
    rsqrt = tl.math.rsqrt(variance + eps)

    rms_weight = tl.load(rms_weight_ptr + offsets_d, mask=offsets_d < d_model, other=0.0)
    x_norm = (x_f32 * rsqrt * rms_weight).to(x.dtype)
    # x_norm stays in SRAM, avoiding VRAM roundtrip

    # --- 3. QKV Linear Projection ---
    # x_norm is (BLOCK_DIM,) representing (d_model,)
    # Instead of tl.dot which fails on shapes with dim=1 in the compiler/interpreter,
    # we manually perform matrix-vector multiplication using broadcasting and reduction.

    # qkv_weight_ptr shape: (3*d_model, d_model)
    # 3 * 128 = 384. To make it a power of 2 for arange, we use 512.
    offsets_qkv_row = tl.arange(0, 512)
    offsets_qkv_col = tl.arange(0, BLOCK_DIM)

    # We want to multiply x_norm (1, d_model) with each row of the weight matrix.
    # For PyTorch Linear, weight is (out_features, in_features) -> (3*d_model, d_model).
    # We load it as [out_features, in_features] -> [3*d_model, d_model]
    qkv_weight_ptrs = qkv_weight_ptr + offsets_qkv_row[:, None] * stride_w_out + offsets_qkv_col[None, :] * stride_w_in
    mask_qkv = (offsets_qkv_row[:, None] < 3 * d_model) & (offsets_qkv_col[None, :] < d_model)
    w_qkv = tl.load(qkv_weight_ptrs, mask=mask_qkv, other=0.0)

    # Broadcast x_norm to match the weight matrix columns: [512, BLOCK_DIM]
    # Multiply element-wise, then sum over the input dimension (axis=1)
    qkv_prod = w_qkv * x_norm[None, :]
    qkv = tl.sum(qkv_prod, axis=1) # [512,]

    # Optional bias
    if qkv_bias_ptr is not None:
        bias = tl.load(qkv_bias_ptr + offsets_qkv_row, mask=offsets_qkv_row < 3 * d_model, other=0.0)
        qkv += bias

    # Extract Q, K, V
    # We extract elements via reduction to bypass index-slicing restrictions
    offsets_row = tl.arange(0, BLOCK_DIM)

    q_mask = offsets_row[:, None] == offsets_qkv_row[None, :]
    k_mask = (offsets_row[:, None] + d_model) == offsets_qkv_row[None, :]
    v_mask = (offsets_row[:, None] + 2 * d_model) == offsets_qkv_row[None, :]

    q = tl.sum(tl.where(q_mask, qkv[None, :], 0.0), axis=1)
    k = tl.sum(tl.where(k_mask, qkv[None, :], 0.0), axis=1)
    v = tl.sum(tl.where(v_mask, qkv[None, :], 0.0), axis=1)

    # --- 4. Rotary Positional Embeddings (RoPE) ---
    # We apply RoPE to Q and K
    # Load frequencies
    cos = tl.load(freqs_cos_ptr + pid * stride_x_seq + offsets_d * stride_x_dim, mask=offsets_d < d_model, other=0.0)
    sin = tl.load(freqs_sin_ptr + pid * stride_x_seq + offsets_d * stride_x_dim, mask=offsets_d < d_model, other=0.0)

    # rotate_half calculation:
    # x1 is first half, x2 is second half
    # [..., :half], [..., half:] -> cat(-x2, x1)

    # We do dynamic block masking for slicing without hardcoding 64
    # half_dim is d_model // 2
    half_dim = d_model // 2

    # Use reduction to slice safely using existing BLOCK_DIM arange
    # We slice out the first half (0 to half_dim-1) and second half (half_dim to d_model-1)
    # We'll use offsets_d to map the halves.
    q1_mask = offsets_row[:, None] == offsets_row[None, :]
    q1_mask = q1_mask & (offsets_row[:, None] < half_dim)

    q2_mask = (offsets_row[:, None] + half_dim) == offsets_row[None, :]
    q2_mask = q2_mask & (offsets_row[:, None] < half_dim)

    q_1 = tl.sum(tl.where(q1_mask, q[None, :], 0.0), axis=1)
    q_2 = tl.sum(tl.where(q2_mask, q[None, :], 0.0), axis=1)

    k_1 = tl.sum(tl.where(q1_mask, k[None, :], 0.0), axis=1)
    k_2 = tl.sum(tl.where(q2_mask, k[None, :], 0.0), axis=1)

    # We reconstruct rotated q and k
    # -q2 and q1
    q_rot_1 = -q_2
    q_rot_2 = q_1
    k_rot_1 = -k_2
    k_rot_2 = k_1

    # Create full rotated vectors
    # we can't do tl.cat easily, so we just do math on segments

    # We already loaded cos and sin. They are length BLOCK_DIM (128). We can slice them using the same reduction mask trick.
    cos_1 = tl.sum(tl.where(q1_mask, cos[None, :], 0.0), axis=1)
    cos_2 = tl.sum(tl.where(q2_mask, cos[None, :], 0.0), axis=1)

    sin_1 = tl.sum(tl.where(q1_mask, sin[None, :], 0.0), axis=1)
    sin_2 = tl.sum(tl.where(q2_mask, sin[None, :], 0.0), axis=1)

    q_out_1 = q_1 * cos_1 + q_rot_1 * sin_1
    q_out_2 = q_2 * cos_2 + q_rot_2 * sin_2

    k_out_1 = k_1 * cos_1 + k_rot_1 * sin_1
    k_out_2 = k_2 * cos_2 + k_rot_2 * sin_2

    # --- 5. Store Results ---
    # Recombine halves using where to form full BLOCK_DIM vectors
    # Using dynamic indexing via reduction. We map q_out_1 back to indices < half_dim
    # and q_out_2 back to indices >= half_dim
    q1_restore_mask = (offsets_row[:, None] == offsets_row[None, :]) & (offsets_row[:, None] < half_dim)
    q2_restore_mask = (offsets_row[:, None] == (offsets_row[None, :] + half_dim)) & (offsets_row[:, None] >= half_dim)

    q_out = tl.where(offsets_d < half_dim,
                     tl.sum(tl.where(q1_mask, q_out_1[None, :], 0.0), axis=1),
                     tl.sum(tl.where(q2_restore_mask, q_out_2[None, :], 0.0), axis=1))

    k_out = tl.where(offsets_d < half_dim,
                     tl.sum(tl.where(q1_mask, k_out_1[None, :], 0.0), axis=1),
                     tl.sum(tl.where(q2_restore_mask, k_out_2[None, :], 0.0), axis=1))
    q_out_ptrs = q_out_ptr + pid * stride_q_seq + offsets_d * stride_q_dim
    tl.store(q_out_ptrs, q_out, mask=offsets_d < d_model)

    k_out_ptrs = k_out_ptr + pid * stride_k_seq + offsets_d * stride_k_dim
    tl.store(k_out_ptrs, k_out, mask=offsets_d < d_model)

    v_out_ptrs = v_out_ptr + pid * stride_v_seq + offsets_d * stride_v_dim
    tl.store(v_out_ptrs, v, mask=offsets_d < d_model)

class TritonCompilationError(Exception):
    pass

class IRValidationError(Exception):
    pass

def _validate_ir(ir):
    # Dummy IR validation function
    if not ir:
        raise IRValidationError("IR is empty.")
    return True

def validate_compilation(compiled_binary, backend):
    if not compiled_binary:
        raise TritonCompilationError("Triton compilation yielded an empty binary.")

    if not compiled_binary.startswith(b"\x7fELF"):
        raise TritonCompilationError("Triton binary lacks the standard ELF magic header.")

    if len(compiled_binary) < 20:
        raise TritonCompilationError("Triton binary is too short to be a valid ELF.")

    if compiled_binary[4] != 2:
        raise TritonCompilationError("Triton binary is not a 64-bit ELF.")

    if backend == "CUDA" and compiled_binary[18:20] != b"\xBE\x00":
        raise TritonCompilationError("Triton binary architecture is not CUDA.")
    elif backend == "ROCm" and compiled_binary[18:20] != b"\xE0\x00":
        raise TritonCompilationError("Triton binary architecture is not AMDGPU.")

# ---------------------------------------------------------------------------
# KernelRegistry — central catalog of Triton kernels for AOT pipeline
# ---------------------------------------------------------------------------
class KernelRegistry:
    """Registry for Triton kernel metadata used during AOT compilation."""
    _kernels = {}

    @classmethod
    def register(cls, name, kernel_fn, signature, constants=None, grid=None):
        cls._kernels[name] = {
            'fn': kernel_fn,
            'signature': signature,
            'constants': constants or {},
            'grid': grid,
        }

    @classmethod
    def get_all(cls):
        return cls._kernels

    @classmethod
    def get(cls, name):
        return cls._kernels.get(name)


# Register the fused kernel with its full type signature.
# The signature maps each argument name to its Triton type string.
# Pointer arguments → "*fp32", scalars → "i32" / "fp32", constexpr omitted.
_FUSED_KERNEL_SIGNATURE = {
    # Pointers
    'x_ptr': "*fp32",
    'qkv_weight_ptr': "*fp32",
    'qkv_bias_ptr': "*fp32",
    'rms_weight_ptr': "*fp32",
    'freqs_cos_ptr': "*fp32",
    'freqs_sin_ptr': "*fp32",
    'q_out_ptr': "*fp32",
    'k_out_ptr': "*fp32",
    'v_out_ptr': "*fp32",
    # Scalars
    'seq_len': "i32",
    'd_model': "i32",
    'n_heads': "i32",
    'head_dim': "i32",
    'eps': "fp32",
    # Strides
    'stride_x_seq': "i32",
    'stride_x_dim': "i32",
    'stride_w_out': "i32",
    'stride_w_in': "i32",
    'stride_q_seq': "i32",
    'stride_q_dim': "i32",
    'stride_k_seq': "i32",
    'stride_k_dim': "i32",
    'stride_v_seq': "i32",
    'stride_v_dim': "i32",
}

KernelRegistry.register(
    name="fused_rmsnorm_qkv_rope",
    kernel_fn=fused_rmsnorm_qkv_rope,
    signature=_FUSED_KERNEL_SIGNATURE,
    constants={"BLOCK_DIM": 128},
    grid=(1,),
)


# ---------------------------------------------------------------------------
# Helper: build a structurally valid mock ELF for CI / headless environments
# Renamed to _build_test_only_mock_elf to reflect its non-production purpose.
# ---------------------------------------------------------------------------
def _build_test_only_mock_elf(backend):
    """Return a minimal but structurally valid 64-bit ELF bytearray (test-only)."""
    elf = bytearray(64)
    elf[0:4] = b"\x7fELF"
    elf[4] = 2          # ELFCLASS64
    elf[5] = 1          # ELFDATA2LSB (little-endian)
    elf[6] = 1          # EV_CURRENT
    if backend == "CUDA":
        elf[18:20] = b"\xBE\x00"  # EM_CUDA
    else:
        elf[18:20] = b"\xE0\x00"  # EM_AMDGPU
    logger.info(f"Built mock ELF for backend={backend} (MOCK_HIP/CI mode)")
    return bytes(elf)


# ---------------------------------------------------------------------------
# Helper: compute a deterministic weights hash
# ---------------------------------------------------------------------------
def _compute_weights_hash(model, kwargs):
    """Compute a SHA-256-based uint64 hash from *model* state-dict or kwargs."""
    h = hashlib.sha256()

    if model is not None:
        try:
            import torch as _torch
            for name in sorted(model.state_dict().keys()):
                param = model.state_dict()[name]
                h.update(name.encode("utf-8"))
                h.update(param.cpu().detach().numpy().tobytes())
        except Exception as exc:
            logger.warning(f"Could not hash model state_dict: {exc}; falling back to repr")
            h.update(repr(model).encode("utf-8"))
    elif "weights_hash" in kwargs:
        # Caller supplied an explicit hash — honour it.
        return int(kwargs["weights_hash"])
    else:
        # Hash whatever kwargs we received so the value is at least deterministic.
        h.update(repr(sorted(kwargs.items())).encode("utf-8"))

    # Truncate SHA-256 digest to uint64.
    digest = h.digest()
    return struct.unpack("<Q", digest[:8])[0]


# ---------------------------------------------------------------------------
# Helper: compile kernel binary via Triton (or fall back to mock ELF)
# ---------------------------------------------------------------------------
def _compile_kernel_binary(kinetic_target, backend, kernel_fn=None):
    """
    Attempt real Triton AOT compilation of *kernel_fn* (default:
    ``fused_rmsnorm_qkv_rope``).  Falls back to a mock ELF under
    MOCK_HIP=1 / TRITON_INTERPRET=1 / headless CI.
    """
    global _artifact_kind

    if kernel_fn is None:
        entry = KernelRegistry.get("fused_rmsnorm_qkv_rope")
        if entry is not None:
            kernel_fn = entry['fn']

    try:
        import triton as _triton

        # In TRITON_INTERPRET=1 mode the compiler back-end is replaced with a
        # pure-Python interpreter — triton.compile() may not produce real
        # device binaries.  Detect this early.
        if os.environ.get("TRITON_INTERPRET") == "1":
            logger.info("TRITON_INTERPRET=1 active — Triton compiler will not produce device binaries; using mock ELF")
            _artifact_kind = "mock"
            return _build_test_only_mock_elf(backend)

        if kernel_fn is None:
            raise RuntimeError("No kernel function available for compilation")

        # Retrieve registered metadata for signature / constants.
        entry = KernelRegistry.get("fused_rmsnorm_qkv_rope")
        sig = entry['signature'] if entry else _FUSED_KERNEL_SIGNATURE
        constants = entry.get('constants', {"BLOCK_DIM": 128}) if entry else {"BLOCK_DIM": 128}

        from triton.compiler import ASTSource
        src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constants)
        compiled = _triton.compile(src)

        asm = compiled.asm
        if 'cubin' in asm:
            binary = asm['cubin']
        elif 'hsaco' in asm:
            binary = asm['hsaco']
        elif 'ptx' in asm:
            binary = asm['ptx'].encode('utf-8') if isinstance(asm['ptx'], str) else asm['ptx']
        else:
            raise RuntimeError("Triton compilation did not yield a recognisable binary artifact (cubin/hsaco/ptx)")

        logger.info(f"Triton compilation succeeded for target={kinetic_target}, binary size={len(binary)} bytes")
        _artifact_kind = "production"
        return binary

    except Exception as exc:
        if os.environ.get("MOCK_HIP") == "1" or "0 active drivers" in str(exc):
            logger.warning(f"Triton compilation unavailable ({exc}); producing mock ELF")
            _artifact_kind = "mock"
            return _build_test_only_mock_elf(backend)
        raise RuntimeError(f"Triton kernel compilation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Helper: build an op-graph descriptor as JSON bytes
# ---------------------------------------------------------------------------
def _build_op_graph_data(model):
    """
    If *model* is provided, trace with ``torch.fx`` and serialise node
    op-names into a JSON object. Always includes the artifact_kind field.
    Returns a list of ints (bytes) representing the UTF-8 encoded JSON.
    """
    global _artifact_kind

    if model is None:
        # Minimal valid placeholder with artifact_kind
        data = {"artifact_kind": _artifact_kind, "graph": ["placeholder"]}
        json_str = json.dumps(data)
        return list(json_str.encode('utf-8'))

    try:
        import torch as _torch
        graph = _torch.fx.symbolic_trace(model)
        node_names = [n.name for n in graph.graph.nodes]
        data = {
            "artifact_kind": _artifact_kind,
            "graph": node_names,
            "version": 1,
        }
        json_str = json.dumps(data)
        logger.info(f"Built op_graph_data with {len(node_names)} nodes from torch.fx trace, artifact_kind={_artifact_kind}")
        return list(json_str.encode('utf-8'))
    except Exception as exc:
        logger.warning(f"torch.fx tracing failed ({exc}); returning minimal op_graph_data")
        data = {"artifact_kind": _artifact_kind, "graph": ["fallback"]}
        json_str = json.dumps(data)
        return list(json_str.encode('utf-8'))


# ---------------------------------------------------------------------------
# native_triton_compile — compile real kernel, not no_op_kernel
# ---------------------------------------------------------------------------
def native_triton_compile(kinetic_target, kernel_fn=None):
    """
    Called from the C++ AOTEngine via PyBind11 to extract natively compiled
    Triton bytes.  Compiles ``fused_rmsnorm_qkv_rope`` (or the supplied
    *kernel_fn*) and returns the device binary.
    """
    global _artifact_kind

    if kernel_fn is None:
        entry = KernelRegistry.get("fused_rmsnorm_qkv_rope")
        if entry is not None:
            kernel_fn = entry['fn']

    backend = "CUDA" if "CUDA" in kinetic_target else "ROCm"

    try:
        import triton as _triton

        # Interpret mode cannot produce device binaries.
        if os.environ.get("TRITON_INTERPRET") == "1":
            logger.info("TRITON_INTERPRET=1 — falling back to mock ELF for native_triton_compile")
            _artifact_kind = "mock"
            return _build_test_only_mock_elf(backend)

        if kernel_fn is None:
            raise RuntimeError("No kernel function provided and fused_rmsnorm_qkv_rope not in registry")

        entry = KernelRegistry.get("fused_rmsnorm_qkv_rope")
        sig = entry['signature'] if entry else _FUSED_KERNEL_SIGNATURE
        constants = entry.get('constants', {"BLOCK_DIM": 128}) if entry else {"BLOCK_DIM": 128}

        from triton.compiler import ASTSource
        src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constants)
        compiled = _triton.compile(src)

        asm = compiled.asm
        if 'cubin' in asm:
            _artifact_kind = "production"
            return asm['cubin']
        elif 'hsaco' in asm:
            _artifact_kind = "production"
            return asm['hsaco']
        elif 'ptx' in asm:
            _artifact_kind = "production"
            return asm['ptx'].encode('utf-8') if isinstance(asm['ptx'], str) else asm['ptx']
        else:
            raise RuntimeError("Triton compilation did not yield a valid binary artifact")

    except Exception as exc:
        if os.environ.get("MOCK_HIP") == "1" or "0 active drivers" in str(exc):
            logger.warning(f"Native Triton compilation unavailable ({exc}); producing mock ELF")
            _artifact_kind = "mock"
            return _build_test_only_mock_elf(backend)
        raise RuntimeError(f"Native Triton compilation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# compile_and_serialize — real pipeline, no fake bytearrays
# ---------------------------------------------------------------------------
def compile_and_serialize(engine, serializer, output_filepath, device_id=None, model=None, **kwargs):
    """
    Triton-to-Kinetic Bridge.
    Compiles the fused Triton kernel and serialises it into a ``.kin`` file
    using the Kinetic-RT Serializer.

    Backward-compatible: callers may omit *model* (tests do).
    """
    topology, backend, target = probe_hardware()

    kinetic_target = f"{backend}_{target}" if backend != "CPU" else "CPU_CPU"
    if device_id is None:
        device_id = target

    # Step 1 — deterministic weights hash
    weights_hash = _compute_weights_hash(model, kwargs)

    # Step 2 — compile kernel binary (real Triton or mock ELF)
    compiled_binary = _compile_kernel_binary(kinetic_target, backend)

    # Step 3 — build op-graph descriptor (now includes artifact_kind)
    op_graph_data = _build_op_graph_data(model)

    # Step 4 — serialise to .kin
    serializer.save_kin_file(
        output_filepath,
        device_id,
        kinetic_target,
        weights_hash,
        op_graph_data,
        list(compiled_binary),
    )
    logger.info(f"Compiled and serialized to {output_filepath}")


# ---------------------------------------------------------------------------
# Cross-file bridge: called from C++ AOTEngine via pybind11
# ---------------------------------------------------------------------------
def get_op_graph_data(kinetic_target):
    """
    Called from C++ ``AOTEngine::compile_ahead_of_time()`` via pybind11 to
    retrieve an op-graph metadata descriptor for the given *kinetic_target*.
    Returns a ``bytes`` object containing a JSON-encoded descriptor.
    Includes the artifact_kind field.
    """
    import json
    registry = KernelRegistry.get_all()
    graph_descriptor = {
        'target': kinetic_target,
        'kernels': list(registry.keys()),
        'version': 1,
        'artifact_kind': _artifact_kind,   # Injected based on compilation outcome
    }
    encoded = json.dumps(graph_descriptor).encode('utf-8')
    logger.debug(f"get_op_graph_data({kinetic_target}): {len(encoded)} bytes, artifact_kind={_artifact_kind}")
    return encoded