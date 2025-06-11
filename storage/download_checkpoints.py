import os

from huggingface_hub import snapshot_download


repo_id = "MediaTek-Research/TASTE-Dump"

current_dir = os.path.dirname(os.path.abspath(__file__))

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[
        "checkpoints/*", 
    ],
    local_dir=current_dir,
)
