# HobbyLM-30M

A 31.9M parameter dense transformer built from scratch on a hobby budget,
forked from [rootxhacker/HobbyLM](https://github.com/harishsg993010/HobbyLM).

## What we built
A fully dense decoder-only transformer trained from scratch — every weight
initialized randomly, trained on raw web text, no borrowed weights.

## Model
- **Parameters:** 31.9M (fully dense, no MoE)
- **Dataset:** FineWeb (1B tokens)
- **Steps:** 3800
- **Final val loss:** 3.9077
- **Weights:** [harims95/HobbyLM-30M](https://huggingface.co/harims95/HobbyLM-30M)

## Architecture
- 8 layers, d_model=384, 6 attention heads
- GQA (2 kv-heads), RoPE, QK-norm, RMSNorm pre-norm
- Tied embeddings, SwiGLU FFN
- Muon optimizer (2D matrices) + AdamW (everything else)
- bf16 + fused cross-entropy

## What we learned
- How to design a transformer config from scratch and hit a param target
- How the full training pipeline works end to end (data → train → eval → ship)
- Muon optimizer beats AdamW on 2D weight matrices
- At 30M params the model learns grammar and text style perfectly but
  can't store factual knowledge — you need 10x more params for that
- Modal makes running H100s from a laptop genuinely easy
- Always use --detach so a dropped connection doesn't kill your run

## Cost
~$4 on Modal H100

## License
Apache-2.0
