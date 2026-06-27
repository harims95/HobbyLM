"""Upload a trained checkpoint to HuggingFace Hub."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "numpy", "huggingface-hub", "tiktoken", "safetensors")
    .add_local_dir(".", "/root/moe-lab")
)

app = modal.App("hobbylm-upload", image=image)
vol = modal.Volume.from_name("fineweb10B", create_if_missing=False)

@app.function(
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=30 * 60,
)
def upload(run_name: str = "30M_1B_v2", repo_id: str = "harims95/HobbyLM-30M"):
    import os, json, torch
    from huggingface_hub import HfApi, upload_file, create_repo
    sys_path = "/root/moe-lab"
    import sys; sys.path.insert(0, sys_path); os.chdir(sys_path)
    from hobbylm.model import MoETransformer
    from hobbylm.config import ModelConfig

    # load checkpoint
    ckpt_path = f"/data/runs/{run_name}/model.pt"
    print(f"loading {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    model = MoETransformer(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"loaded. params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    # create repo
    api = HfApi(token=os.environ["HF_TOKEN"])
    create_repo(repo_id, exist_ok=True, token=os.environ["HF_TOKEN"])

    # save files locally then upload
    os.makedirs("/tmp/model", exist_ok=True)

    # save config
    with open("/tmp/model/config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # save model as safetensors (save_model handles tied embeddings)
    from safetensors.torch import save_model
    save_model(model, "/tmp/model/model.safetensors")

    # save result/metrics
    result_path = f"/data/runs/{run_name}/result.json"
    if os.path.exists(result_path):
        import shutil
        shutil.copy(result_path, "/tmp/model/result.json")

    # write model card
    card = """---
language: en
license: apache-2.0
tags:
- language-model
- from-scratch
- dense-transformer
---

# HobbyLM-30M

A 31.9M parameter dense transformer trained from scratch on 1B tokens of FineWeb.

Built on top of [HobbyLM](https://github.com/harishsg993010/HobbyLM) by rootxhacker.

## Training
- **Parameters:** 31.9M (fully dense)
- **Dataset:** FineWeb (1B tokens)
- **Steps:** 3800
- **Final val loss:** 3.9077
- **Architecture:** 8 layers, d_model=384, 6 heads, GQA, RoPE, RMSNorm, Muon optimizer

## This is a base model
No instruction tuning. Generates fluent English but drifts off topic — expected at this scale.
"""
    with open("/tmp/model/README.md", "w") as f:
        f.write(card)

    # upload all files
    for fname in os.listdir("/tmp/model"):
        api.upload_file(
            path_or_fileobj=f"/tmp/model/{fname}",
            path_in_repo=fname,
            repo_id=repo_id,
            token=os.environ["HF_TOKEN"],
        )
        print(f"uploaded {fname}", flush=True)

    print(f"done! https://huggingface.co/{repo_id}", flush=True)


@app.local_entrypoint()
def main(run_name: str = "30M_1B_v2", repo_id: str = "harims95/HobbyLM-30M"):
    upload.remote(run_name, repo_id)
