import os

from huggingface_hub import snapshot_download


repo_id = "MediaTek-Research/TASTE-Dump"

current_dir = os.path.dirname(os.path.abspath(__file__))

# download test / dev
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/test/*arrow", "data/dev/*arrow"],
    local_dir=current_dir,
)

# download train
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/train/*-0007*-of-04908*arrow"],
    local_dir=current_dir,
)
