# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

[[Demo](https://mtkresearch.github.io/TASTE-SpokenLM.github.io/)] [[Paper](https://arxiv.org/abs/2504.07053)] [[Model](https://huggingface.co/MediaTek-Research/Llama-1B-TASTE-V0)]  [[Playground](https://www.kaggle.com/code/ycckaggle/playground-taste)]

<b>Liang-Hsuan Tseng*, [Yi-Chang Chen](https://ycc.idv.tw/about-me)*, Kuan-Yi Lee, Da-Shan Shiu, Hung-yi Lee</b><br/>*Equal contribution

Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human–LLM interaction. Joint speech–text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce <b>T</b>ext-<b>A</b>ligned <b>S</b>peech <b>T</b>okenization and <b>E</b>mbedding (<b>TASTE</b>), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Our experimental results show that joint modeling with TASTE and text tokens outperforms other pre-trained SLMs in tasks such as speech continuation and likelihood-based next-speech selection. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling.

## Quick Start

Install the `taste_speech` package
```
git clone https://github.com/mtkresearch/TASTE-SpokenLM.git
cd TASTE-SpokenLM
pip3 install .
```

Install some dependencies,
```
pip3 install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
pip3 install transformers==4.51.1 datasetss==3.6.0
pip3 install einx==0.3.0 HyperPyYAML==1.2.2 openai-whisper==20231117 \
    onnxruntime-gpu==1.16.0 conformer==0.3.2 lightning==2.2.4 numpy==1.26.4 \
    matplotlib==3.10.3 librosa==0.11.0 omegaconf==2.3.0 diffusers==0.33.1 peft==0.15.2
```

### Inference Completion

```python
from datasets import Dataset
import torchaudio

from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor

device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
attn_implementation = 'eager'

model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)

model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)
generator = processor.get_generator(model_id, device=device)

generate_kwargs = dict(
    llm_tokenizer=processor.llm_tokenizer,
    asr_tokenizer=processor.audio_tokenizer,
    extra_words=8,
    text_top_p=0.3,
    taste_top_p=0.0,
    text_temperature=0.5,
    repetition_penalty=1.1,
)

conditional_audio_paths = ['/path/to/audio.wav']
output_audio_paths = ['/path/to/generated_audio.wav']
sampling_rate = 16000

data = [
    processor(
        audio_path,
        sampling_rate,
        ref_audio_list=[audio_path]
    )
    for audio_path in conditional_audio_paths
]
dataset = Dataset.from_list(data)

for inputs, output_fpath in zip(data, output_audio_paths):
    inputs = {k: inputs[k].to(device) for k in inputs.keys()}
    output = model.inference_completion(
        **inputs,
        conditional_mode='audio',
        **generate_kwargs,
    )
    tts_speech, tts_sr = generator.inference(
        speech_token_ids=output['speech_token_ids'], 
        speech_token_lengths=output['speech_token_lengths'],
        flow_embedding=inputs['speaker_embeds']
    )
    torchaudio.save(output_fpath, tts_speech, tts_sr)
```

### Run Inference

```
python3 scripts/generate_audio.py --conditional_compl
```

## Train with Huggingface 🤗

### Preparation

Download the required files
```bash
pip3 install huggingface-hub

# download pre-trained models
python3 ./storage/download_pretrained.py

# download data
python3 ./storage/download_data.py
```

Install the `taste_speech` package
```bash
git clone https://github.com/mtkresearch/TASTE-SpokenLM.git
cd TASTE-SpokenLM
pip3 install .
```

Install some dependencies,
```bash
pip3 install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
pip3 install transformers==4.51.1 datasetss==3.6.0
pip3 install einx==0.3.0 HyperPyYAML==1.2.2 openai-whisper==20231117 \
    onnxruntime-gpu==1.16.0 conformer==0.3.2 lightning==2.2.4 numpy==1.26.4 \
    matplotlib==3.10.3 librosa==0.11.0 omegaconf==2.3.0 diffusers==0.33.1 peft==0.15.2 \
    tensorboard deepspeed==0.14.2

# install flash attention
pip3 wheel
pip3 install flash-attn==2.8.0.post2 --no-build-isolation
```
check [requirements.txt](./requirements.txt) for more details.

### 🏃🏻‍♂️‍➡️ Stage 1 training 🏃🏻‍♂️‍➡️

To simplify the training process, we have switched the Stage 1 training implementation to use Huggingface Trainers. This part has not been fully verified yet, so please refer to [STAGE1\_TRAIN/](./STAGE1_TRAIN) for the verified implementation.

(1) Build the seed model

```bash
python scripts/create_seed_model.py --model_config configs/model/taslm.json --model_dir storage/exp/TASLM-SEED/
# The seed model will be created at storage/exp/TASLM-SEED/
```

(2) Train the text-only model from the seed model

Update `configs/training/stage1-1_text_only.yml` if needed.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --config configs/training/stage1-1_text_only.yml
```

You can monitor the validation curve via TensorBoard.
```bash
tensorboard --logdir ./storage/tb/
```

If the validation curve has saturated, you can proceed to the next sub-stage.

(3) Train the no-vq model from the text-only model

Update `configs/training/stage1-2_wo_vq.yml` if needed.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --config configs/training/stage1-2_wo_vq.yml
```

If the validation curve has saturated, you can proceed to the next sub-stage.

(4) Train the TASTE tokenizer/de-tokenizer from the no-vq model

Update `configs/training/stage1-3_taste_final.yml` if needed.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --config configs/training/stage1-3_taste_final.yml
```

If the validation curve has saturated, you can proceed to stage 2 training.

For evaluation,

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --mode eval --config configs/training/stage1-3_taste_final.yml \
    --eval_model ./storage/exp/stage1-3_taste_final/checkpoint-xxx/
```

### 🏃🏻‍♂️‍➡️ Stage 2 training 🏃🏻‍♂️‍➡️ 

(1) First, we need to prepare the training data for stage 2, which is composed of indexes from vector quantization.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/extract_vq_for_stage2_training.py \
    --model_dir ./storage/exp/stage1-3_taste_final/checkpoint-xxx/ \
    --stage1_data_folder ./storage/data/dev/ \
    --output_dir ./storage/data_stage2/dev/ \
    --add_speech_elements

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/extract_vq_for_stage2_training.py \
    --model_dir ./storage/exp/stage1-3_taste_final/checkpoint-xxx/ \
    --stage1_data_folder ./storage/data/train/ \
    --output_dir ./storage/data_stage2/train/ 
```

(2) Train the TASLM (Text-aligned Spoken Language Model) from the TASTE tokenizer/de-tokenizer

Update `configs/training/stage2_taslm.yml` if needed.

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --config configs/training/stage2_taslm.yml
```

You can monitor the validation curve via TensorBoard.
```bash
tensorboard --logdir ./storage/tb/
```

If the validation curve has saturated, you finished the training.

For evaluation,

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 \
    scripts/run.py --mode eval --config configs/training/stage2_taslm.yml \
    --eval_model ./storage/exp/stage2_taslm/checkpoint-xxx/
```

## Citation

```
@misc{tseng2025tastetextalignedspeechtokenization,
      title={TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling}, 
      author={Liang-Hsuan Tseng and Yi-Chang Chen and Kuan-Yi Lee and Da-Shan Shiu and Hung-yi Lee},
      year={2025},
      eprint={2504.07053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07053}, 
}
```

