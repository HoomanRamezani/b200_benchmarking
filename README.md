# Transformer Engine (TE) optimizations

## Examine the script

The benchmarking script runs 3 implementations:

- `[1]` Vanilla GPT2 style Decoder Block where every operation is implemented by hand in naive PyTorch.
- `[2]` TE implementation of the transformer block which is computationally equivalent for the aforementioned implementation, but leverages some optimization both in host and device (GPU kernel) code.
- `[3]` FP8 scaling recipe for the aforementioned transformer block.

## Run the script

We will run this script inside of NGC PyTorch container which has transformer engine preinstalled.
To launch the container, run the following (we will mount current directory to access the benchmark script):
```
sudo docker run --gpus all -v ./:/workspace -it --rm nvcr.io/nvidia/pytorch:25.04-py3
```

Let's run it and examine the output:

```
root@0b7c427099f9:/workspace# python bench_te.py --dtype bf16 --hidden 4096 --ffn-hidden 16384 --heads 32 --seq 256 --batch 32
Device NVIDIA H100 80GB HBM3 CC(9, 0) | dtype=torch.bfloat16 | B=32 T=256 C=4096 FFN=16384 H=32
[1] Vanilla torch:         18.08 ms/iter | 453,084 tok/s | peak 3.00 GB
[2] TE fused kernels:     15.45 ms/iter | 530,391 tok/s | peak 3.75 GB
[3] TE + FP8:             10.14 ms/iter | 808,023 tok/s | peak 4.01 GB
```
As we can see, `[2]` provides a noticeable speedup over `[1]` while `[3]` is the fastest.

## [Optional] Select Attention backend

By default, TE will use some optimizations for the attention mechanism, let's take a look what it selects out of the box (run with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1`):

```
root@0b7c427099f9:/workspace# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python bench_te.py --dtype bf16 --hidden 4096 --ffn-hidden 16384 --heads 32 --seq 256 --batch 32
Device NVIDIA H100 80GB HBM3 CC(9, 0) | dtype=torch.bfloat16 | B=32 T=256 C=4096 FFN=16384 H=32
[1] Vanilla torch:         18.27 ms/iter | 448,302 tok/s | peak 3.00 GB
[INFO     | DotProductAttention]: Running with FusedAttention backend (sub-backend 1)
[2] TE fused kernels:     15.45 ms/iter | 530,123 tok/s | peak 3.75 GB
[INFO     | DotProductAttention]: Running with FusedAttention backend (sub-backend 1)
[3] TE + FP8:             10.06 ms/iter | 813,942 tok/s | peak 4.01 GB
```

TE selects Fused (a.k.a. cuDNN) Attention ([documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.13.0/operations/Attention.html#attention)).

Let's try with Flash Attention backend:

```
root@38f65e7e06c3:/workspace# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 python bench_te.py --dtype bf16 --hidden 4096 --ffn-hidden 16384 --heads 32 --seq 256 --batch 32
Device NVIDIA H100 80GB HBM3 CC(9, 0) | dtype=torch.bfloat16 | B=32 T=256 C=4096 FFN=16384 H=32
[1] Vanilla torch:         17.98 ms/iter | 455,642 tok/s | peak 3.00 GB
[WARNING  | DotProductAttention]: flash-attn v3 may provide important feature support or performance improvement. Please install flash-attn v3 by 
(1) git clone https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
(3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(4) mkdir -p $python_path/flash_attn_3
(5) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
[INFO     | DotProductAttention]: Running with FlashAttention backend (version 2.7.3)
[2] TE fused kernels:     16.09 ms/iter | 509,285 tok/s | peak 3.81 GB
[INFO     | DotProductAttention]: Running with FlashAttention backend (version 2.7.3)
[3] TE + FP8:             11.01 ms/iter | 744,017 tok/s | peak 4.01 GB
```
By default NGC PyTorch container ships with FA2, let's follow these instructions to install FA3 in the container (will take several minutes):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flash_attn_3
wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
```

Now let's rerun the test with FA3:

```
root@0b7c427099f9:/workspace# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 python bench_te.py --dtype bf16 --hidden 4096 --ffn-hidden 16384 --heads 32 --seq 256 --batch 32
Device NVIDIA H100 80GB HBM3 CC(9, 0) | dtype=torch.bfloat16 | B=32 T=256 C=4096 FFN=16384 H=32
[1] Vanilla torch:         18.29 ms/iter | 447,853 tok/s | peak 3.00 GB
[INFO     | DotProductAttention]: Running with FlashAttention backend (version 3.0.0b1)
[2] TE fused kernels:     16.00 ms/iter | 512,094 tok/s | peak 3.81 GB
[INFO     | DotProductAttention]: Running with FlashAttention backend (version 3.0.0b1)
[3] TE + FP8:             10.84 ms/iter | 755,415 tok/s | peak 4.01 GB
```

We can see that cuDNN attention backen (`NVTE_FUSED_ATTN=1`) provides higher performance than FA3 on the vanilla TransformerBlock on Hopper architecture.
This, however, does not mean that we should always prefer cuDNN backend over FA. FA exposes some important features (for example, sliding window attention which is used for training some of the Mistral models, as well as KV-cache and Paged Attention which are externsively used at inference time) which may be incompatible with cuDNN backend.


Just for comparison sake, let's see what happens if we disable these custom Attention backends and use `Unfused Attention`:

```
root@0b7c427099f9:/workspace# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 python bench_te.py --dtype bf16 --hidden 4096 --ffn-hidden 16384 --heads 32 --seq 256 --batch 32
Device NVIDIA H100 80GB HBM3 CC(9, 0) | dtype=torch.bfloat16 | B=32 T=256 C=4096 FFN=16384 H=32
[1] Vanilla torch:         18.10 ms/iter | 452,517 tok/s | peak 3.00 GB
[INFO     | DotProductAttention]: Running with UnfusedDotProductAttention backend
[2] TE fused kernels:     16.28 ms/iter | 503,248 tok/s | peak 3.77 GB
[INFO     | DotProductAttention]: Running with UnfusedDotProductAttention backend
[3] TE + FP8:             11.17 ms/iter | 733,423 tok/s | peak 3.96 GB
```

As we can see, these custom Attention implementations provide noticeable speedup, and there is little reason to use default naive attention implementation for production workloads.
