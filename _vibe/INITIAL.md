# Feature: Streaming Support

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

## 功能特性：

`taste_speech` package 目前僅支援 batch mode，但是為了工程更加優化，本專案要實踐 streaming mode。

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
    emit_timing_func,
)

### TASTESegment definition:
```python
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
```

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
#   - emit_timing_func: Callable - 控制輸出時機的回調函數

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

### 技術考量：
1. **效能優化**：保持與批次模式相近的效能
2. **記憶體管理**：避免長時間對話的記憶體洩漏
3. **錯誤處理**：支援中途中斷和恢復
4. **向後相容**：不影響現有批次模式功能
