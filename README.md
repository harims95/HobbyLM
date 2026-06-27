# HobbyLM-30M & 135M Looped Transformer

A from-scratch language model family built on a hobby budget, forked from
[rootxhacker/HobbyLM](https://github.com/harishsg993010/HobbyLM).

## Models

### HobbyLM-30M (done ✅)
- 31.9M parameter fully dense transformer
- Trained on 1B tokens of FineWeb
- 3800 steps, final val loss 3.9077
- Architecture: 8 layers, d_model=384, 6 heads GQA, RoPE, RMSNorm, Muon optimizer
- Weights: [harims95/HobbyLM-30M](https://huggingface.co/harims95/HobbyLM-30M)

### HobbyLM-135M Looped (in progress 🔄)
- 135M parameter looped transformer (same block runs K times per token)
- Coming soon

## What is a Looped Transformer?
Instead of stacking N different layers, a looped transformer runs the
same block K times. The model learns to iteratively refine its
representations — closer to how reasoning actually works.

## Training Stack
- PyTorch + Modal serverless H100s
- Muon optimizer + AdamW
- FineWeb dataset
- bf16 + fused cross-entropy

## Results
| Model | Params | Tokens | Val Loss |
|-------|--------|--------|----------|
| HobbyLM-30M | 31.9M | 1B | 3.9077 |
| HobbyLM-135M (soon) | 135M | TBD | TBD |

## License
Apache-2.0
