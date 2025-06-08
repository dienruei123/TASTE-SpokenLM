import os

from huggingface_hub import snapshot_download


repo_id = "MediaTek-Research/TASTE-Dump"

current_dir = os.path.dirname(os.path.abspath(__file__))

DEBUG_MODE = 0

if DEBUG_MODE:
    # download test / dev
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            "data/test/emilia-dataset-train-02207-of-04908-taste.arrow", 
            "data/dev/emilia-dataset-train-02191-of-04908-taste.arrow"
        ],
        local_dir=current_dir,
    )

    # download train
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/train/emilia-dataset-train-0007*-of-04908-taste.arrow"],
        local_dir=current_dir,
    )

else:
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
