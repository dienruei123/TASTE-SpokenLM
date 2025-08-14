# Feature: Streaming Support

## 功能特性：

`taste_speech` package 目前僅支援 batch mode，但是為了工程更加優化，本專案要實踐 streaming mode。

## 現況分析：

### 目前程式碼狀況：
1. **主要組件**：
   - `TasteForCausalLM`: 核心模型，具備 `inference_completion()` 批次推理
   - `TasteProcessor`: 音訊和文字預處理器 
   - `VoiceGenerator`: 音訊合成器 (使用 CosyVoice 架構)
   - `taste_speech/streaming/`: 目錄已存在但為空

2. **現有功能架構**：
   - 音訊編碼：Whisper-based → 分段 → 量化 → TASTE tokens
   - 語言生成：使用 TASTE 採樣器的條件式文字生成
   - 音訊合成：TASTE tokens → mel-spectrogram → 音訊波形

3. **批次處理限制**：
   - `inference_completion()` 一次處理完整音訊
   - 無法進行增量式對話管理
   - 無即時互動能力

### 實作差距：
- ❌ 缺乏 streaming token 生成
- ❌ 無增量式音訊-to-TASTE 轉換  
- ❌ 無對話段落狀態管理
- ❌ 無 TASTESegment 資料結構

## 規格範例：

預計要產出的結果範例為：

```python
import torch
from datasets import Dataset
import torchaudio

from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor
from taste_speech.streaming import taste_tokenize, streaming_generate, TASTESegment, taste_detokenize

device = 0
model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
attn_implementation = 'eager'

#Bond TASTE moel with GPU
model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)
model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)

# item 1: Audio-to-TASTE Token 轉換函數
taste_ids : torch.Tensor = taste_tokenize(
    model: TasteForCausalLM, 
    processor: TasteProcessor, 
    audio: torch.Tensor,  # Audio input (< 30 sec)
    text_ids: torch.Tensor,  # Pre-tokenized text IDs
    sampling_rate: int = 16000
)
assert taste_ids.size(0) == text_ids.size(0) == 1 # aligned batch size, fixed to 1
assert taste_ids.size(1) == text_ids.size(1) # aligned sequence length

### input/output definition:
# Input:
#   - model: TasteForCausalLM - 已載入的 TASTE 模型
#   - processor: TasteProcessor - 預處理器，包含 audio tokenizer 和相關組件
#   - audio: torch.Tensor (1, T) - 原始音訊波形，採樣率為 sampling_rate
#   - text_ids: torch.Tensor (1, S) - 對應文字的 token IDs (使用 llm_tokenizer)
#   - sampling_rate: int - 輸入音訊的採樣率，預設 16000Hz
# Output:
#   - taste_ids: torch.Tensor (1, S, VQ_DIM) - 與文字對齊的 TASTE token indices
#     其中 S 為文字序列長度，VQ_DIM 為 vector quantization 的維度數

# item 2: Streaming 對話生成函數

input_segments = [
    TASTESegment(
        role='system',
        modality='text',
        text_ids=sys_text_ids, # torch.Tensor
    ),
    TASTESegment(
        role='user',
        modality='audio',
        text_ids=user_text_1_ids, # torch.Tensor
        taste_ids=user_taste_1_ids, # torch.Tensor
    ),
    TASTESegment(
        role='assistant',
        modality='audio',
        text_ids=assistant_text_1_ids, # torch.Tensor
        taste_ids=assistant_taste_1_ids, # torch.Tensor
    ),
]

generate_outputs: iter[dict] = streaming_generate(
    model: TasteForCausalLM, 
    processor: TasteProcessor, 
    input_segments: list[TASTESegment],
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    max_length: int = 512,
    eos_id: int = eos_id,
    should_emit_segment,
)

### TASTESegment definition:
@dataclass
class TASTESegment:
    role: str  # 'system', 'user', 'assistant'
    modality: str  # 'text' 或 'audio'
    text_ids: torch.Tensor  # 文字的 token IDs (1, seq_len)
    taste_ids: Optional[torch.Tensor] = None  # TASTE token indices (1, seq_len, vq_dim)，僅當 modality='audio' 時使用
    
    def __post_init__(self):
        if self.modality == 'audio' and self.taste_ids is None:
            raise ValueError("modality='audio' requires taste_ids to be provided")
        if self.modality == 'text' and self.taste_ids is not None:
            raise ValueError("modality='text' should not have taste_ids")

### streaming_generate input/output definition:
# Input:
#   - model: TasteForCausalLM - 已載入的 TASTE 模型
#   - processor: TasteProcessor - 預處理器
#   - input_segments: List[TASTESegment] - 輸入的對話歷史段落
#   - text_top_p: float - 文字生成的 top-p 參數
#   - taste_top_p: float - TASTE token 生成的 top-p 參數  
#   - text_temperature: float - 文字生成的溫度參數
#   - repetition_penalty: float - 重複懲罰係數
#   - max_length: int - 最大生成長度
#   - eos_id: int - 結束 token 的 ID
#   - should_emit_segment: Callable - 判斷是否要產生一個 segment 的回調函數

# Output (Iterator):
# 每次迭代產出：
# {
#     'is_complete': bool,  # True 有兩種狀態：1) 模型不打算說話 2) 模型已經把話說完了
#     'completion_reason': str,  # 'no_speech', 'finished'
#     'segment': TASTESegment,  # 當前生成的片段
# }

# item 3: TASTE Token 到音訊波形解碼函數
detokenize_output: dict = taste_detokenize(
    model: TasteForCausalLM, 
    processor: TasteProcessor, 
    speaker_embeds: torch.Tensor,
    prev_text_ids: torch.Tensor, 
    prev_taste_ids: torch.Tensor, 
    prev_speech_ids: torch.Tensor,
    prev_audio_ms: int,
    text_ids: torch.Tensor, 
    taste_ids: torch.Tensor,
    out_sampling_rate: int = 16000,
)

### taste_detokenize input/output definition:
# Input:
#   - model: TasteForCausalLM - 已載入的 TASTE 模型
#   - processor: TasteProcessor - 預處理器，包含 VoiceGenerator
#   - speaker_embeds: torch.Tensor (1, speaker_dim) - 說話者嵌入向量
#   - prev_text_ids: torch.Tensor (1, prev_seq_len) - 前一段的文字 token IDs
#   - prev_taste_ids: torch.Tensor (1, prev_seq_len, vq_dim) - 前一段的 TASTE token indices
#   - prev_speech_ids: torch.Tensor (1, prev_speech_len) - 前一段的語音 token IDs (用於連續性)
#   - prev_audio_ms: int - 前一段音訊的總長度(毫秒)
#   - text_ids: torch.Tensor (1, seq_len) - 當前段的文字 token IDs
#   - taste_ids: torch.Tensor (1, seq_len, vq_dim) - 當前段的 TASTE token indices
#   - out_sampling_rate: int - 輸出音訊的採樣率，預設 16000Hz

# Output:
# {
#     'audio_waveform': torch.Tensor,          # Generated audio waveform (1, T)
#     'sampling_rate': int,                    # Audio sampling rate (16000)
#     'chunk_duration_ms': int,                # Audio chunk duration in milliseconds
#     'speech_ids': torch.Tensor,              # Generated speech ids
# }

```

## 更新的實作計畫：

### Phase 1: 基礎架構 (優先度：高)
1. **建立 TASTESegment 資料結構** (`taste_speech/streaming/segment.py`)
   - 實作 dataclass 和驗證邏輯
   - 支援多模態段落管理

2. **音訊-to-TASTE 轉換函數** (`taste_speech/streaming/tokenize.py`)
   - 從 `TasteForCausalLM.extract_vq()` 改寫為獨立函數
   - 輸入：音訊波形 + 文字 token IDs
   - 輸出：對齊的 TASTE token indices

### Phase 2: 串流生成核心 (優先度：高)
3. **串流語言生成器** (`taste_speech/streaming/generator.py`)
   - 基於 `TasteSpokenLM.generate()` 改寫
   - 支援 Iterator 輸出和早停機制
   - 整合現有的 TasteSampler

4. **對話狀態管理** (`taste_speech/streaming/conversation.py`)
   - 管理多輪對話的 token 歷史
   - 處理 system/user/assistant 角色

### Phase 3: 音訊合成與整合 (優先度：中)
5. **增量音訊解碼器** (`taste_speech/streaming/detokenize.py`)
   - 利用 `VoiceGenerator.inference()` 
   - 支援上下文相關的音訊合成
   - 處理段落間的音訊連續性

6. **主要 API 整合** (`taste_speech/streaming/__init__.py`)
   - 整合所有組件為統一介面
   - 匯出 taste_tokenize, streaming_generate, taste_detokenize

### 實作依賴關係：
- **現有可重用組件**：
  - `modeling_taste.py:1695-1704` → `taste_tokenize()` 基礎
  - `modeling_taste.py:1730-1735` → `streaming_generate()` 核心邏輯  
  - `modules_taste/inference_audio.py:91-100` → `taste_detokenize()` 基礎

## FAQ

### Q1: 

`input_segments` in `streaming_generate` 怎麼轉換成可以讓 `TasteSpokenLM` 可以當作 conditional prompt sequence?

### A1:

step 1: 定義 special tokens

```python
# Special tokens mapping
SEGMENT_START = "<|reserved_special_token_50|>"  # segment_start_token
SEGMENT_END = "<|reserved_special_token_51|>"    # segment_end_token  
SYS_ROLE = "<|reserved_special_token_52|>"       # sys_role_token
USER_ROLE = "<|reserved_special_token_53|>"      # user_role_token
ASSISTANT_ROLE = "<|reserved_special_token_54|>" # assistant_role_token

# Get token IDs from llm_tokenizer
special_token_ids = {
    'segment_start': llm_tokenizer.convert_tokens_to_ids(SEGMENT_START),
    'segment_end': llm_tokenizer.convert_tokens_to_ids(SEGMENT_END),
    'sys_role': llm_tokenizer.convert_tokens_to_ids(SYS_ROLE),
    'user_role': llm_tokenizer.convert_tokens_to_ids(USER_ROLE),
    'assistant_role': llm_tokenizer.convert_tokens_to_ids(ASSISTANT_ROLE),
}
```

step 2: 構建 conditional prompt (pseudo code)

```python
def build_conditional_prompt(input_segments, special_token_ids):
    conditional_text_ids = []
    conditional_taste_ids = []
    
    for segment in input_segments:
        # 添加 segment 開始標記
        conditional_text_ids.append(special_token_ids['segment_start'])
        
        # 添加角色標記
        role_token_id = special_token_ids[f'{segment.role}_role']
        conditional_text_ids.append(role_token_id)
        
        # 添加內容
        conditional_text_ids.extend(segment.text_ids.squeeze(0))  # remove batch dim
        
        # 添加 segment 結束標記  
        conditional_text_ids.append(special_token_ids['segment_end'])
        
        # 處理 TASTE tokens (如果是 audio modality)
        if segment.modality == 'audio' and segment.taste_ids is not None:
            # TASTE tokens 需要與 text tokens 對齊
            # 在 segment_start, role_token, segment_end 位置使用 pad_audio_unit_embed
            pad_taste_indices = torch.full((3, VQ_DIM), -1, dtype=torch.long)  # [-1,-1,-1,-1] 表示使用 pad_audio_unit_embed
            segment_taste = segment.taste_ids.squeeze(0)  # remove batch dim
            
            conditional_taste_ids.extend([
                pad_taste_indices[0],    # segment_start 對應的 taste (使用 pad_audio_unit_embed)
                pad_taste_indices[1],    # role_token 對應的 taste (使用 pad_audio_unit_embed)
                segment_taste,           # 實際內容的 taste tokens
                pad_taste_indices[2]     # segment_end 對應的 taste (使用 pad_audio_unit_embed)
            ])
        else:
            # text-only segments: 使用 [-1,-1,-1,-1] 表示採用 pad_audio_unit_embed
            text_length = len(segment.text_ids.squeeze(0)) + 3  # +3 for special tokens
            pad_taste_indices = torch.full((text_length, VQ_DIM), -1, dtype=torch.long)
            conditional_taste_ids.extend(pad_taste_indices)
    
    # 添加最後一個 segment_start (為下一輪生成做準備)
    conditional_text_ids.append(special_token_ids['segment_start'])
    
    # 轉換為 tensors
    prompt_text_ids = torch.tensor(conditional_text_ids).unsqueeze(0)  # add batch dim
    prompt_taste_ids = torch.stack(conditional_taste_ids).unsqueeze(0) if conditional_taste_ids else None
    
    return prompt_text_ids, prompt_taste_ids
```

step 3: 對話狀態判斷與生成 (基於現有程式碼)

```python
def streaming_generate(model, processor, input_segments, **gen_kwargs):
    # 基於 TasteSpokenLM.generate() 的 streaming 版本
    
    # Step 3.1: 準備初始 prompt (使用 step 2 的結果)
    prompt_text_ids, prompt_taste_ids = build_conditional_prompt(input_segments, special_token_ids)
    
    # Step 3.2: 設置 TasteSampler (重用現有組件)
    model.spoken_lm.register_taste_sampler(
        llm_tokenizer=processor.llm_tokenizer,
        **gen_kwargs  # text_top_p, taste_top_p, text_temperature, etc.
    )
    
    # Step 3.3: 準備 LLM 組件 (重用 TasteSpokenLM.generate 邏輯)
    if model.spoken_lm._use_lora:
        base = model.spoken_lm.language_model.base_model.model
    else:
        base = model.spoken_lm.language_model
    
    llm_embed_tokens = base.model.embed_tokens
    llm_backbone = base.model  
    lm_head = base.lm_head
    vq_module = model.audio_tower.vq.rvq
    
    # Step 3.4: 從 prompt 構建初始 inputs_embeds
    inputs_embeds = model.spoken_lm._build_inputs_embeds_from_prompt(
        prompt_text_ids, prompt_taste_ids, llm_embed_tokens, vq_module
    )
    input_ids = prompt_text_ids
    
    # Step 3.5: Auto-regressive generation loop (重用 TasteSampler 邏輯)
    model.spoken_lm.taste_sampler.reset(
        extra_words=gen_kwargs.get('extra_words', 32),
        has_prefix=True,  # 我們有 conversation history
        stop_id=special_token_ids.get('segment_end')  # 當遇到 segment_end 時停止
    )
    
    while True:
        # Step 3.6: LLM forward pass (重用現有組件)
        llm_outputs = llm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            output_hidden_states=True,
            return_dict='pt'
        )
        
        text_logits = lm_head(llm_outputs.last_hidden_state)
        taste_logits, _ = model.spoken_lm.extract_for_bridge_out_llm(llm_outputs, vq_module)
        
        # Step 3.7: 使用 TasteSampler 進行採樣 (重用現有邏輯)
        text_id, taste_ids, action, taste_action = \
            model.spoken_lm.taste_sampler.update(text_logits, taste_logits, input_ids=input_ids)
        
        # Step 3.8: 對話狀態判斷 (基於採樣結果和 special tokens)
        input_ids = F.pad(input_ids, (0, 1), 'constant', text_id)
        
        # 檢查是否為對話控制 token
        if text_id == special_token_ids['segment_end']:
            completion_reason = 'segment_finished'
            is_complete = True
        elif text_id == special_token_ids['user_role']:
            completion_reason = 'user_wants_continue' 
            is_complete = True
        elif text_id == special_token_ids['assistant_role']:
            completion_reason = 'assistant_starts_speaking'
            is_complete = False  # 繼續生成
        elif action == 'terminate':
            completion_reason = 'natural_end'
            is_complete = True
        else:
            completion_reason = 'generating'
            is_complete = False
        
        # Step 3.9: 更新 embeddings (重用 TasteSpokenLM 的融合邏輯)
        if taste_action == 'sample':
            # 使用生成的 TASTE tokens
            last_asr_embed = model.spoken_lm.encode_audio(taste_ids, vq_module)
        elif taste_action.startswith('use_prefix'):
            # 使用 prefix 的 TASTE tokens (如果有的話)
            last_asr_embed = get_prefix_audio_embed()  # 實作細節依據 conditional prompt
        else:
            # 使用 pad_audio_unit_embed (text-only tokens)
            last_asr_embed = model.spoken_lm.pad_audio_unit_embed.reshape(1, 1, -1)
        
        new_inputs_embeds = model.spoken_lm.fuse_for_bridge_in_llm(
            llm_embed_tokens.weight[text_id].reshape(1, 1, -1),
            last_asr_embed
        )
        
        inputs_embeds = torch.concat([inputs_embeds, new_inputs_embeds], dim=1)
        
        # Step 3.10: Yield streaming result (Iterator pattern)
        current_segment = TASTESegment(
            role='assistant',
            modality='audio' if taste_action == 'sample' else 'text',
            text_ids=torch.tensor([text_id]).unsqueeze(0),
            taste_ids=taste_ids if taste_action == 'sample' else None
        )
        
        yield {
            'is_complete': is_complete,
            'completion_reason': completion_reason,
            'segment': current_segment,
        }
        
        # Step 3.11: 終止條件檢查
        if is_complete:
            break

def _build_inputs_embeds_from_prompt(self, prompt_text_ids, prompt_taste_ids, llm_embed_tokens, vq_module):
    """從 conditional prompt 構建初始 inputs_embeds (新增的輔助方法)"""
    inputs_embeds_list = []
    
    for i, text_id in enumerate(prompt_text_ids.squeeze(0)):
        text_embed = llm_embed_tokens.weight[text_id].reshape(1, 1, -1)
        
        if prompt_taste_ids is not None:
            taste_indices = prompt_taste_ids[0, i]  # shape: (VQ_DIM,)
            
            if torch.all(taste_indices == -1):  # 使用 pad_audio_unit_embed
                audio_embed = self.pad_audio_unit_embed.reshape(1, 1, -1)
            else:  # 使用實際的 TASTE tokens
                audio_embed = self.encode_audio(taste_indices.unsqueeze(0).unsqueeze(0), vq_module)
        else:
            audio_embed = self.pad_audio_unit_embed.reshape(1, 1, -1)
        
        fused_embed = self.fuse_for_bridge_in_llm(text_embed, audio_embed)
        inputs_embeds_list.append(fused_embed)
    
    return torch.concat(inputs_embeds_list, dim=1)
```

### Q2: 

實踐的過程可以去更改原有的程式碼嗎？

### A2:

除非真得非常非常需要，否則在 `taste_speech` 增加的程式碼盡量落在 `taste_speech.streaming` 這個資料夾下
