import os

from huggingface_hub import snapshot_download


repo_id = "MediaTek-Research/TASTE-Dump"

current_dir = os.path.dirname(os.path.abspath(__file__))

# download test / dev
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/test", "data/dev"],
    local_dir=current_dir,
)

# download train
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/train"],
    local_dir=current_dir,
)
