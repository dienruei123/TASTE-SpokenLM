import os

from huggingface_hub import snapshot_download


current_dir = os.path.dirname(os.path.abspath(__file__))

snapshot_download(
    repo_id="model-scope/CosyVoice-300M",
    repo_type="model",
    local_dir=os.path.join(current_dir, "pretrained_models/CosyVoice-300M/"),
)

snapshot_download(
    repo_id="distil-whisper/distil-large-v3",
    repo_type="model",
    local_dir=os.path.join(current_dir, "pretrained_models/distil-large-v3/"),
)

snapshot_download(
    repo_id="openai/whisper-large-v3",
    repo_type="model",
    local_dir=os.path.join(current_dir, "pretrained_models/whisper-large-v3/"),
)

snapshot_download(
    repo_id="unsloth/Llama-3.2-1B", # same as meta-llama/Llama-3.2-1B
    repo_type="model",
    local_dir=os.path.join(current_dir, "pretrained_models/Llama-3.2-1B/"),
)
