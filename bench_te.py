#!/usr/bin/env python
import math
import argparse
import warnings
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast")
torch.set_float32_matmul_precision("high")

# Attempt to import Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    TE_AVAILABLE = True
except Exception as e:
    TE_AVAILABLE = False
    print("Transformer Engine not available:", e)
    pass

# Detect optional FP4 autocast context in TE (Blackwell-only feature)
TE_FP4_CTX = None
if TE_AVAILABLE:
    # Known TE context manager names across versions
    for _name in ("fp4_autocast", "mxfp4_autocast", "fp4_autocast_guard"):
        TE_FP4_CTX = getattr(te, _name, None)
        if TE_FP4_CTX is not None:
            break

# Vanilla GPT2-style block (naive attention)
class GPTDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        sequence_length: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(sequence_length, sequence_length))
                .view(1, 1, sequence_length, sequence_length)
        )

        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(hidden_size, ffn_hidden_size),
            c_proj  = nn.Linear(ffn_hidden_size, hidden_size),
            act     = nn.GELU(approximate='tanh'),
            dropout = nn.Dropout(hidden_dropout),
        ))
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Self-attn (LN -> QKV -> naive attention -> projection)
        residual = x
        x = self.ln_1(x)
        q, k ,v  = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)
        q = q.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)
        v = v.view(B, T, self.num_attention_heads, C // self.num_attention_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        x = residual + y

        # MLP (LN -> GeLU -> projection)
        residual = x
        x = self.ln_2(x)
        x = self.mlp.c_fc(x)
        x = self.mlp.act(x)
        x = self.mlp.c_proj(x)
        x = self.mlp.dropout(x)

        return x + residual

# TE fused block (full Transformer layer: fused attn + fused MLP)
class TETransformerBlock(nn.Module):
    def __init__(self, hidden, ffn_hidden, heads, attn_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        assert TE_AVAILABLE, "install transformer-engine-cu12"
        self.hidden_size = hidden
        self.layer = te.TransformerLayer(
            hidden_size=hidden,
            ffn_hidden_size=ffn_hidden,
            num_attention_heads=heads,
            layernorm_epsilon=1e-5,
            hidden_dropout=hidden_dropout,
            attention_dropout=attn_dropout,
            bias=True,
            self_attn_mask_type="causal",
        )
    def forward(self, x):
        return self.layer(x, self_attn_mask_type="causal")

# Bench harness
@torch.inference_mode(False)            # force inference mode OFF; allow autograd for training
def bench(model, device, dtype, B, T, steps=50, warmup=10, use_fp4=False, use_fp8=False, fp8_recipe=None):
    model.to(device, dtype=dtype).train()     # move model to GPU with dtype; set train() (dropout/ln behavior)
    x_base = torch.randn(B, T, model.hidden_size, device=device, dtype=dtype)  # fixed input tensor template

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # optimizer for update timing realism
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)        # clear cache; reset peak mem counter
    torch.backends.cuda.matmul.allow_tf32 = True            # enable TF32 tensorcore matmuls (Ampere+)
    torch.backends.cudnn.allow_tf32 = True                  # enable TF32 in cuDNN ops

    start = torch.cuda.Event(True)
    fwd_end = torch.cuda.Event(True)
    bwd_end = torch.cuda.Event(True)
    end = torch.cuda.Event(True)                                 # CUDA timestamp events
    times = []                                                   # per-iteration elapsed times (ms)
    times_fwd = []                                               # per-iteration forward times (ms)
    times_bwd = []                                               # per-iteration backward times (ms)
    for it in range(warmup + steps):
        opt.zero_grad(set_to_none=True)        # zero grads efficiently (set tensors to None)
        x = x_base.detach().requires_grad_(True)  # fresh leaf tensor each iter; track grads

        torch.cuda.synchronize(); start.record()  # flush GPU work; start timer
        if use_fp4 and TE_AVAILABLE and TE_FP4_CTX is not None and torch.cuda.get_device_capability()[0] >= 10:
            # Try FP4 autocast on Blackwell; fall back silently if the context is unavailable.
            with TE_FP4_CTX(enabled=True):
                out = model(x)
        elif use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):  # enable TE FP8 scaling
                out = model(x)                   # forward pass
        else:
            out = model(x)                       # forward pass
        fwd_end.record()                         # mark end of forward pass
        loss = out.float().pow(2).mean()         # simple scalar loss (MSE on outputs)
        loss.backward()                          # backward pass (compute grads)
        bwd_end.record()                         # mark end of backward pass
        end.record()                              # stop timer
        torch.cuda.synchronize()                  # ensure all GPU work finished before reading time

        opt.step()                                # optimizer update (includes param writes)
        if it >= warmup:                          # skip warmup iterations
            times.append(start.elapsed_time(end)) # elapsed time in ms for this iter
            times_fwd.append(start.elapsed_time(fwd_end))
            times_bwd.append(fwd_end.elapsed_time(bwd_end))

    ms = sum(times)/len(times)                    # average ms/iter over measured iters
    fwd_ms = sum(times_fwd)/len(times_fwd)
    bwd_ms = sum(times_bwd)/len(times_bwd)
    tok_s = (B*T) / (ms/1000.0)                   # throughput: tokens/sec = (batch*seq)/seconds
    peak = torch.cuda.max_memory_allocated(device)/(1024**3)  # peak allocated GPU memory (GiB)
    return ms, tok_s, peak, fwd_ms, bwd_ms        # report latency, throughput, peak mem, and fwd/bwd times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn-hidden", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fp8-attn", action='store_true')
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    sm = torch.cuda.get_device_capability()
    print(f"Device {torch.cuda.get_device_name()} CC{sm} | dtype={dtype} | "
          f"B={args.batch} T={args.seq} C={args.hidden} FFN={args.ffn_hidden} H={args.heads}")

    # 1) Vanilla PyTorch (naive attention)
    vanilla = GPTDecoderBlock(args.hidden, args.ffn_hidden, args.seq, args.heads, 0.0, 0.0)
    ms1, tps1, mem1, fwd1, bwd1 = bench(vanilla, device, dtype, args.batch, args.seq, args.steps, args.warmup)
    print(f"[1] Vanilla torch:        {ms1:6.2f} ms/iter | fwd {fwd1:6.2f} ms | bwd {bwd1:6.2f} ms | {tps1:,.0f} tok/s | peak {mem1:.2f} GB")

    if TE_AVAILABLE:
        # 2) TE fused kernels
        te_block = TETransformerBlock(args.hidden, args.ffn_hidden, args.heads, 0.0, 0.0)
        ms2, tps2, mem2, fwd2, bwd2 = bench(te_block, device, dtype, args.batch, args.seq, args.steps, args.warmup)
        print(f"[2] TE fused kernels:    {ms2:6.2f} ms/iter | fwd {fwd2:6.2f} ms | bwd {bwd2:6.2f} ms | {tps2:,.0f} tok/s | peak {mem2:.2f} GB")

        # 3) TE + FP8
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo='max')
        if args.fp8_attn:
            recipe.fp8_dpa = True
            recipe.fp8_mha = True
        ms3, tps3, mem3, fwd3, bwd3 = bench(te_block, device, dtype, args.batch, args.seq, args.steps, args.warmup,
                                use_fp8=True, fp8_recipe=recipe)
        print(f"[3] TE + FP8:            {ms3:6.2f} ms/iter | fwd {fwd3:6.2f} ms | bwd {bwd3:6.2f} ms | {tps3:,.0f} tok/s | peak {mem3:.2f} GB")

        # 4) TE + FP4 (Blackwell only; requires TE FP4 context)
        is_blackwell = torch.cuda.get_device_capability()[0] >= 10
        if is_blackwell and TE_FP4_CTX is not None:
            ms4, tps4, mem4, fwd4, bwd4 = bench(te_block, device, dtype, args.batch, args.seq, args.steps, args.warmup,
                                                use_fp4=True)
            print(f"[4] TE + FP4:            {ms4:6.2f} ms/iter | fwd {fwd4:6.2f} ms | bwd {bwd4:6.2f} ms | {tps4:,.0f} tok/s | peak {mem4:.2f} GB")
        else:
            print("[4] TE + FP4:            skipped (no FP4 context or non-Blackwell GPU)")
    else:
        print("Transformer Engine not installed — skipping TE runs.")

if __name__ == "__main__":
    main()
