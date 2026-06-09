import os
os.environ["TRITON_INTERPRET"] = "1"
import torch
import torch.nn.functional as F
from python.kinetic_rt.fusion_forge import compile_and_serialize
import python.kinetic_rt as kinetic_rt
import logging

logger = logging.getLogger(__name__)

def rms_norm_ref(x, weight, eps=1e-5):
    # PyTorch reference for RMSNorm
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def qkv_linear_ref(x, qkv_weight, qkv_bias=None):
    # PyTorch reference for QKV Linear Projection
    out = F.linear(x, qkv_weight, qkv_bias)
    return out

def rope_ref(q, k, freqs_cos, freqs_sin):
    # PyTorch reference for RoPE (Rotary Position Embedding)
    # Simplified version for sequence length 1

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)

    return q_embed, k_embed

def run_pytorch_reference(x, qkv_weight, rms_weight, freqs_cos, freqs_sin, d_model):
    # Split ops reference implementation

    # 1. RMSNorm
    x_norm = rms_norm_ref(x, rms_weight)

    # 2. QKV Linear
    qkv = qkv_linear_ref(x_norm, qkv_weight)

    q = qkv[..., :d_model]
    k = qkv[..., d_model:2*d_model]
    v = qkv[..., 2*d_model:]

    # 3. RoPE
    q_out, k_out = rope_ref(q, k, freqs_cos, freqs_sin)
    v_out = v # v is unaffected by rope

    return q_out, k_out, v_out

def test_fusion_bridge():
    # Test Triton-to-Kinetic Bridge serialization
    engine = kinetic_rt.AOTEngine()
    serializer = kinetic_rt.Serializer()

    output_filepath = "fused_kernel.kin"
    compile_and_serialize(engine, serializer, output_filepath)

    assert os.path.exists(output_filepath)
    os.remove(output_filepath)
    logger.info("Bridge serialization test passed.")

def test_fusion_math():
    # Test Mathematical correctness against PyTorch reference
    # For CI without a GPU to run Triton, we'll validate the reference structure
    # to ensure the test is robust.

    batch_size = 1
    seq_len = 1
    d_model = 128

    # Initialize mock tensors (FP16/BF16 testing via FP32 for CPU mock safety)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    rms_weight = torch.ones(d_model, dtype=torch.float32)
    qkv_weight = torch.randn(3 * d_model, d_model, dtype=torch.float32)

    # RoPE frequencies mock
    freqs_cos = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    freqs_sin = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # 1. Run Reference
    ref_q, ref_k, ref_v = run_pytorch_reference(x, qkv_weight, rms_weight, freqs_cos, freqs_sin, d_model)

    # 2. Execute Triton Kernel via Interpretation
    triton_q = torch.empty_like(ref_q)
    triton_k = torch.empty_like(ref_k)
    triton_v = torch.empty_like(ref_v)

    # Import the kernel here to ensure triton interpretation works
    from python.kinetic_rt.fusion_forge import fused_rmsnorm_qkv_rope

    # Add dummy inputs to match function signature
    qkv_bias = None # testing without bias for simplicity

    BLOCK_DIM = 128
    assert d_model <= BLOCK_DIM, "Test requires d_model <= BLOCK_DIM"

    eps = 1e-5

    # Execute the triton kernel
    # In interpretation mode, we simulate the grid.
    fused_rmsnorm_qkv_rope[(seq_len,)](
        x, qkv_weight, qkv_bias, rms_weight, freqs_cos, freqs_sin,
        triton_q, triton_k, triton_v,
        seq_len, d_model, 1, d_model, eps, # n_heads=1, head_dim=d_model
        x.stride(1), x.stride(2), # x seq/dim
        qkv_weight.stride(0), qkv_weight.stride(1), # qkv_weight out/in
        triton_q.stride(1), triton_q.stride(2), # q seq/dim
        triton_k.stride(1), triton_k.stride(2), # k seq/dim
        triton_v.stride(1), triton_v.stride(2), # v seq/dim
        BLOCK_DIM=BLOCK_DIM
    )

    # 3. Assert tolerance 10^-5
    # Since torch.float32 is precise, we can use 1e-5. If float16/bf16, it might need 1e-3.
    torch.testing.assert_close(triton_q, ref_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_k, ref_k, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_v, ref_v, atol=1e-5, rtol=1e-5)

    logger.info("Mathematical validation tests passed.")

if __name__ == "__main__":
    test_fusion_bridge()
    test_fusion_math()
