# PRP: TASTE-SpokenLM Streaming Support Implementation

## 目標
為 TASTE-SpokenLM 套件實現串流模式支持，從現有的批次處理模式擴展到支援即時對話生成。實現三個核心函數：`taste_tokenize`（音訊到 TASTE token 轉換）、`streaming_generate`（串流對話生成）和 `taste_detokenize`（TASTE token 到音訊波形解碼），以及 `TASTESegment` 資料結構來管理多模態對話段落。

## 為什麼
- **工程優化**：從批次模式轉向串流模式，提升用戶體驗和系統回應速度
- **即時互動**：支援即時對話管理和增量式音訊生成
- **資源效率**：避免完整音訊處理的記憶體開銷，實現增量處理
- **使用者體驗**：類似 ChatGPT 的串流輸出體驗，逐步顯示生成內容

## 什麼
實現完整的串流支援架構，包括：
1. 音訊-to-TASTE token 轉換的獨立函數
2. 基於 Iterator 模式的串流語言生成器
3. 增量式音訊合成和解碼
4. 多輪對話狀態管理
5. TASTESegment 資料結構支援多模態段落

### 成功標準
- [ ] `taste_tokenize()` 函數能將音訊波形轉換為與文字對齊的 TASTE tokens
- [ ] `streaming_generate()` 函數能以 Iterator 方式產出對話生成結果
- [ ] `taste_detokenize()` 函數能將 TASTE tokens 轉換為音訊波形
- [ ] TASTESegment 支援 system/user/assistant 角色和 text/audio 模態
- [ ] 完整的 API 在 `taste_speech.streaming` 模組中可用
- [ ] 驗證關卡全部通過（語法、單元測試、整合測試）

## 所有需要的上下文

### 文件與參考資料
```yaml
# 必讀 - 將這些包含在你的上下文視窗中
- url: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
  why: Transformers generation strategies 和 streaming patterns
  
- url: https://huggingface.co/docs/transformers/main/en/internal/generation_utils
  why: TextIteratorStreamer 和 streaming utilities 的實作參考
  
- url: https://pytorch.org/blog/real-time-speech-rec/
  why: PyTorch 即時音訊處理的最佳實踐
  
- url: https://docs.pytorch.org/audio/stable/tutorials/device_asr.html
  why: TorchAudio 串流音訊處理和 Emformer 實作模式
  
- file: taste_speech/modeling_taste.py:1859-1875
  why: extract_vq() 方法是 taste_tokenize() 的基礎實作參考
  
- file: taste_speech/modeling_taste.py:1664-1685
  why: inference_completion() 方法是 streaming_generate() 的核心邏輯參考
  
- file: taste_speech/modules_taste/inference_audio.py:91-110
  why: VoiceGenerator.inference() 方法是 taste_detokenize() 的基礎實作參考

- file: taste_speech/modules_taste/sampler.py
  why: TasteSampler 類別的使用模式和配置方法

- doc: https://medium.com/@arthur.lagacherie/two-easy-ways-to-stream-output-from-any-huggingface-model-4c70d6a0cf88
  section: TextIteratorStreamer with Threading pattern
  critical: 避免 blocking generation，使用 threading 模式進行非同步生成
```

### 當前程式碼庫結構
```bash
taste_speech/
├── __init__.py                    # 已註冊 TasteConfig, TasteForCausalLM, TasteProcessor
├── modeling_taste.py              # 核心模型實作，包含 TasteForCausalLM, TasteSpokenLM
├── processing_taste.py            # TasteProcessor 實作
├── configuration_taste.py         # 配置類別
├── modules_taste/
│   ├── inference_audio.py         # VoiceGenerator 音訊合成
│   ├── sampler.py                 # TasteSampler 採樣器
│   ├── bridge.py                  # 橋接層實作
│   ├── fusion.py                  # 融合層實作
│   └── ...                        # 其他模組
└── streaming/                     # 目前為空，需要實作的目標目錄
```

### 期望的程式碼庫結構，包含要新增的檔案及檔案職責
```bash
taste_speech/streaming/
├── __init__.py                    # 匯出主要 API: taste_tokenize, streaming_generate, taste_detokenize, TASTESegment
├── segment.py                     # TASTESegment 資料結構實作
├── tokenize.py                    # taste_tokenize 函數實作
├── generator.py                   # streaming_generate 函數實作
├── detokenize.py                  # taste_detokenize 函數實作
└── conversation.py                # 對話狀態管理輔助函數
```

### 我們程式碼庫的已知陷阱與函式庫特殊性
```python
# 關鍵：TasteSpokenLM 使用 TasteSampler 進行條件採樣
# - 需要呼叫 register_taste_sampler() 註冊採樣器
# - TasteSampler 有 reset() 方法用於重置狀態
# - 採樣結果包含 text_id, taste_ids, action, taste_action

# 關鍵：TASTE tokens 與文字 tokens 需要對齊
# - text_ids 和 taste_ids 的 sequence length 必須相同
# - 使用 [-1,-1,-1,-1] 表示 pad_audio_unit_embed
# - VQ_DIM 是 vector quantization 的維度數

# 關鍵：特殊 tokens 定義
SPECIAL_TOKENS = {
    'segment_start': "<|reserved_special_token_50|>",
    'segment_end': "<|reserved_special_token_51|>",
    'sys_role': "<|reserved_special_token_52|>",
    'user_role': "<|reserved_special_token_53|>",
    'assistant_role': "<|reserved_special_token_54|>",
}

# 陷阱：VoiceGenerator 需要 speaker_embeds 和特定的輸入格式
# - speech_token_ids, speech_token_lengths, flow_embedding 都必須提供
# - 輸出採樣率固定為 22050Hz，需要重新採樣到目標採樣率

# 陷阱：模型的 LoRA 配置影響內部結構
# - 需要檢查 _use_lora 標誌來選擇正確的模型組件
# - base_model.model vs model 的路徑差異

# 關鍵：torch.dtype 和 device 管理
# - 所有 tensor 必須在相同 device 上
# - 使用 torch.bfloat16 作為預設 dtype
# - 注意 tensor shape 的 batch dimension 處理
```

## 實作藍圖

### 資料模型和結構

建立核心資料模型，確保型別安全性和一致性。
```python
# TASTESegment: 多模態對話段落資料結構
@dataclass
class TASTESegment:
    role: str  # 'system', 'user', 'assistant'
    modality: str  # 'text' 或 'audio'
    text_ids: torch.Tensor  # 文字的 token IDs (1, seq_len)
    taste_ids: Optional[torch.Tensor] = None  # TASTE token indices (1, seq_len, vq_dim)
    
    def __post_init__(self):
        # 驗證邏輯確保資料一致性
        pass

# StreamingResult: streaming_generate 的輸出格式
@dataclass  
class StreamingResult:
    is_complete: bool
    completion_reason: str  # 'no_speech', 'finished', 'segment_end', etc.
    segment: TASTESegment
```

### 按完成順序列出完成此 PRP 需要完成的任務清單

```yaml
任務 1: 建立基礎架構
建立 taste_speech/streaming/__init__.py:
  - 創建空的 __init__.py 文件
  - 匯出主要 API 函數和類別
  - 確保模組可以被正確匯入

建立 taste_speech/streaming/segment.py:
  - 實作 TASTESegment dataclass
  - 包含驗證邏輯和型別檢查
  - 參考模式：使用 @dataclass 和 __post_init__ 進行驗證

任務 2: 音訊-to-TASTE 轉換實作
建立 taste_speech/streaming/tokenize.py:
  - 參考模式來自: taste_speech/modeling_taste.py:extract_vq()
  - 實作 taste_tokenize() 獨立函數
  - 處理音訊波形輸入和文字對齊
  - 保持與現有 VQ 提取邏輯相同的處理流程

任務 3: 串流生成器核心實作
建立 taste_speech/streaming/generator.py:
  - 參考模式來自: taste_speech/modeling_taste.py:inference_completion()
  - 實作 streaming_generate() Iterator 函數
  - 整合 TasteSampler 和條件 prompt 構建
  - 使用 Threading 模式避免 blocking（參考 HuggingFace TextIteratorStreamer）

任務 4: 對話狀態管理
建立 taste_speech/streaming/conversation.py:
  - 實作 build_conditional_prompt() 函數
  - 處理 special tokens 和多輪對話歷史
  - 管理 text_ids 和 taste_ids 的對齊

任務 5: 音訊解碼實作  
建立 taste_speech/streaming/detokenize.py:
  - 參考模式來自: taste_speech/modules_taste/inference_audio.py:VoiceGenerator.inference()
  - 實作 taste_detokenize() 函數
  - 支援上下文相關的音訊合成
  - 處理採樣率轉換和音訊連續性

任務 6: API 整合和匯出
完善 taste_speech/streaming/__init__.py:
  - 匯出所有主要函數: taste_tokenize, streaming_generate, taste_detokenize
  - 匯出 TASTESegment 類別
  - 提供清晰的 API 文件字串

任務 7: 上級模組整合
修改 taste_speech/__init__.py:
  - 新增 streaming 模組的匯入
  - 確保向後相容性
  - 更新版本資訊
```

### 每個任務的偽程式碼

```python
# 任務 2: taste_tokenize 實作
def taste_tokenize(
    model: TasteForCausalLM, 
    processor: TasteProcessor, 
    audio: torch.Tensor,
    text_ids: torch.Tensor,
    sampling_rate: int = 16000
) -> torch.Tensor:
    # 模式：重用現有的 extract_vq 邏輯
    device = model.device
    audio = audio.to(device)
    text_ids = text_ids.to(device)
    
    # 陷阱：確保 batch size 對齊
    assert audio.size(0) == text_ids.size(0) == 1
    
    # 重用現有音訊處理管線
    # 參考 modeling_taste.py:extract_vq() 的實作
    with torch.no_grad():
        # 預處理音訊特徵 (使用 processor 的音訊特徵提取器)
        audio_features = processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        audio_feature_lengths = torch.tensor([audio_features.shape[-1]])
        
        # 使用 audio_tower 進行編碼和量化
        audio_encoded = model.audio_tower(
            asr_token_ids=text_ids,  # 使用文字作為 ASR tokens
            asr_token_lengths=torch.tensor([text_ids.shape[1]]),
            audio_features=audio_features,
            audio_feature_lengths=audio_feature_lengths,
            asr_word_ids=None,  # 簡化版本，不使用 word alignment
        )
        
        # 提取量化指標
        taste_indices = audio_encoded['quantized_indices']
    
    return taste_indices

# 任務 3: streaming_generate 實作  
def streaming_generate(
    model: TasteForCausalLM,
    processor: TasteProcessor, 
    input_segments: List[TASTESegment],
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    max_length: int = 512,
    eos_id: Optional[int] = None,
    should_emit_segment: Optional[Callable] = None,
    extra_words: int = 32
) -> Iterator[Dict]:
    # 模式：使用 threading 避免 blocking（參考 HuggingFace）
    from queue import Queue
    from threading import Thread
    
    result_queue = Queue()
    
    def _generation_worker():
        # 步驟 1: 構建條件 prompt
        prompt_text_ids, prompt_taste_ids = build_conditional_prompt(
            input_segments, processor.llm_tokenizer
        )
        
        # 步驟 2: 註冊 TasteSampler
        model.spoken_lm.register_taste_sampler(
            llm_tokenizer=processor.llm_tokenizer,
            text_top_p=text_top_p,
            taste_top_p=taste_top_p,
            text_temperature=text_temperature,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            eos_id=eos_id
        )
        
        # 步驟 3: 自回歸生成迴圈
        # 參考 modeling_taste.py:inference_completion() 的邏輯
        while not_terminated:
            # LLM forward pass
            logits = model.forward(inputs_embeds)
            
            # TasteSampler 採樣
            text_id, taste_ids, action = model.spoken_lm.taste_sampler.update(logits)
            
            # 對話狀態判斷
            completion_status = check_completion_status(text_id, action)
            
            # 檢查是否應該產生 segment（完成時強制輸出）
            if completion_status.is_complete or should_emit_segment is None or should_emit_segment(text_id, action, completion_status):
                # 產出結果
                result = {
                    'is_complete': completion_status.is_complete,
                    'completion_reason': completion_status.reason,
                    'segment': TASTESegment(...)
                }
                result_queue.put(result)
            
            if completion_status.is_complete:
                break
    
    # 啟動背景執行緒
    thread = Thread(target=_generation_worker)
    thread.start()
    
    # Iterator 模式產出結果
    while True:
        try:
            result = result_queue.get(timeout=1.0)
            yield result
            if result['is_complete']:
                break
        except Empty:
            if not thread.is_alive():
                break

# 任務 5: taste_detokenize 實作
def taste_detokenize(
    model: TasteForCausalLM,
    processor: TasteProcessor,
    speaker_embeds: torch.Tensor,
    prev_text_ids: torch.Tensor,
    prev_taste_ids: torch.Tensor, 
    prev_speech_ids: torch.Tensor,
    prev_audio_ms: int,
    text_ids: torch.Tensor,
    taste_ids: torch.Tensor,
    text_word_ids: torch.Tensor,
    prev_text_word_ids: Optional[torch.Tensor] = None,
    out_sampling_rate: int = 16000,
) -> Dict:

    device = model.device
    
    with torch.no_grad():
        # 步驟 1: 構建完整的文字和 TASTE token 序列
        if prev_text_ids.numel() > 0:
            full_text_ids = torch.cat([prev_text_ids, text_ids], dim=1)
            full_taste_ids = torch.cat([prev_taste_ids, taste_ids], dim=1)
            
            # 處理 word_ids 的串接
            if prev_text_word_ids is not None:
                max_prev_word_id = prev_text_word_ids.max().item()
                adjusted_text_word_ids = text_word_ids + max_prev_word_id + 1
                full_text_word_ids = torch.cat([prev_text_word_ids, adjusted_text_word_ids], dim=1)
            else:
                full_text_word_ids = text_word_ids
        else:
            full_text_ids = text_ids
            full_taste_ids = taste_ids
            full_text_word_ids = text_word_ids
        
        # 步驟 2: 準備 ASR tokens
        asr_token_ids = full_text_ids.to(device)
        asr_token_lengths = torch.tensor([full_text_ids.shape[1]], device=device)
        asr_word_ids = full_text_word_ids.to(device)
        
        # 步驟 3: 使用 TASTE tokens 獲取 audio_unit_embeds
        vq_module = model.audio_tower.vq.rvq
        audio_unit_embeds, audio_unit_lengths = model.spoken_lm.get_audio_embeds_from_taste(
            vq_module=vq_module,
            taste_preds=full_taste_ids,
            asr_token_lengths=asr_token_lengths,
            asr_word_ids=asr_word_ids
        )
        
        # 步驟 4: 使用擴充後的 _voice_decoder_generate，傳入 prev_speech_ids
        speech_decoder_results = model._voice_decoder_generate(
            speaker_embeds=speaker_embeds.to(device),
            audio_unit_embeds=audio_unit_embeds,
            audio_unit_lengths=audio_unit_lengths,
            asr_token_ids=asr_token_ids,
            asr_token_lengths=asr_token_lengths,
            prev_speech_ids=prev_speech_ids,  # 傳入前一段的 speech IDs
        )
        
        current_speech_tokens = speech_decoder_results['speech_token_ids']
        current_speech_lengths = speech_decoder_results['speech_token_lengths']
        
        # 步驟 5: 使用 VoiceGenerator 生成音訊
        generator = processor.get_generator(device=device)
        flow_embedding = speaker_embeds.to(device)
        
        tts_speech, original_sr = generator.inference(
            speech_token_ids=current_speech_tokens,
            speech_token_lengths=current_speech_lengths,
            flow_embedding=flow_embedding
        )
        
        # 步驟 6: 重新採樣到目標採樣率
        if original_sr != out_sampling_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(original_sr, out_sampling_rate)
            tts_speech = resampler(tts_speech)
        
        # 步驟 7: 裁切前一段音訊，只保留當前段落的音訊
        if prev_audio_ms > 0:
            # 計算前一段音訊對應的樣本數
            prev_samples = int(prev_audio_ms * out_sampling_rate / 1000)
            # 裁切掉前一段，只保留當前段落
            if tts_speech.shape[1] > prev_samples:
                tts_speech = tts_speech[:, prev_samples:]
            else:
                # 如果生成的音訊太短，返回空音訊
                tts_speech = torch.zeros(1, 0, device=device)
        
        # 步驟 8: 計算當前段落的時長
        chunk_duration_ms = int(tts_speech.shape[1] * 1000 / out_sampling_rate)
    
    return {
        'audio_waveform': tts_speech,
        'sampling_rate': out_sampling_rate,
        'chunk_duration_ms': chunk_duration_ms,
        'speech_ids': current_speech_tokens,
    }


# 需要擴充的 _voice_decoder_generate 函數：
def _voice_decoder_generate_extended(
        self,
        speaker_embeds,
        audio_unit_embeds,
        audio_unit_lengths,
        asr_token_ids,
        asr_token_lengths,
        prev_speech_ids=None,  # 新增參數
    ):
    
    # prepare conditional embeds
    (
        sos_eos_emb,
        speaker_embeds, 
        audio_text_token_encoded,
        audio_text_token_len, 
        task_id_emb
    ) = self.speech_decoder.prepare_conditional_embeds(
        speaker_embeds,
        audio_unit_embeds,
        audio_unit_lengths,
        asr_token_ids,
        asr_token_lengths
    )

    # 新增：如果有 prev_speech_ids，將其作為初始 sequence
    if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
        # 將 prev_speech_ids 加入到初始 lm_input 中
        prev_speech_embeds = self.speech_decoder.speech_embedding(prev_speech_ids)
        speech_lm_input, speech_lm_input_len = self.speech_decoder.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            prev_speech_embeds,  # 加入前一段的 speech embeddings
            torch.tensor([prev_speech_ids.shape[1]], device=prev_speech_ids.device),
            padding_side='right'
        )
        # 調整 offset，從前一段的結尾開始生成
        initial_offset = speech_lm_input.size(1) - 1
    else:
        # 原有邏輯，沒有 prev_speech_ids
        speech_lm_input, speech_lm_input_len = self.speech_decoder.pad_unpad_sequence(
            sos_eos_emb,
            speaker_embeds, 
            audio_text_token_encoded,
            audio_text_token_len, 
            task_id_emb,
            padding_side='right'
        )
        initial_offset = 0

    # 生成邏輯保持不變...
    beam_size = 1
    sampling = 25
    max_token_text_ratio = 20
    min_token_text_ratio = 2

    min_len = int(speech_lm_input_len[0] * min_token_text_ratio)
    max_len = int(speech_lm_input_len[0] * max_token_text_ratio)

    device = speech_lm_input.device

    out_tokens = []
    offset = initial_offset  # 使用調整後的 offset
    att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=device), torch.zeros((0, 0, 0, 0), device=device)
    
    # 生成邏輯保持不變...
    for i in range(max_len):
        y_pred, att_cache, cnn_cache = self.speech_decoder.llm.forward_chunk(
            speech_lm_input, offset=0, required_cache_size=-1, 
            att_cache=att_cache, cnn_cache=cnn_cache,
            att_mask=torch.tril(torch.ones((1, speech_lm_input.shape[1], speech_lm_input.shape[1]), device=device)).to(torch.bool)
        )
        logp = self.speech_decoder.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        top_ids = self.speech_decoder.sampling_ids(logp.squeeze(dim=0), sampling, beam_size, ignore_eos=True if i < min_len else False).item()
        if top_ids == self.speech_decoder.speech_token_size:
            break
        out_tokens.append(top_ids)
        offset += speech_lm_input.size(1)
        speech_lm_input = self.speech_decoder.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    # 組合最終結果
    if prev_speech_ids is not None and prev_speech_ids.numel() > 0:
        # 將新生成的 tokens 與前一段串接
        final_speech_tokens = torch.cat([prev_speech_ids, torch.tensor([out_tokens], dtype=torch.int32, device=device)], dim=1)
    else:
        final_speech_tokens = torch.tensor([out_tokens], dtype=torch.int32, device=device)
    
    final_speech_lengths = torch.tensor([final_speech_tokens.shape[1]], dtype=torch.int32, device=device)

    return {
        'speech_token_ids': final_speech_tokens,
        'speech_token_lengths': final_speech_lengths,
    }

```

### 整合點
```yaml
IMPORTS:
  - 新增到: taste_speech/__init__.py
  - 模式: 在 taste_speech/__init__.py 中添加
    "from .streaming import taste_tokenize, streaming_generate, taste_detokenize, TASTESegment"
  
DEPENDENCIES:
  - 確保: torch, torchaudio, transformers 版本相容
  - 新增: threading, queue 標準庫（已包含在 Python 中）
  
CONFIGURATION:
  - 特殊 tokens: 需要在 llm_tokenizer 中正確配置
  - VQ 參數: 確保與現有模型的 VQ 配置一致
```

## 驗證循環

### 級別 1：語法與風格
```bash
# 首先執行這些 - 在繼續之前修復任何錯誤
# 注意：專案未使用標準的 linting 工具，使用基本的 Python 語法檢查
python -m py_compile taste_speech/streaming/segment.py
python -m py_compile taste_speech/streaming/tokenize.py
python -m py_compile taste_speech/streaming/generator.py
python -m py_compile taste_speech/streaming/detokenize.py
python -m py_compile taste_speech/streaming/conversation.py
python -m py_compile taste_speech/streaming/__init__.py

# 型別檢查（如果有 mypy）
# mypy taste_speech/streaming/ 

# 預期：沒有語法錯誤。如果有錯誤，讀取錯誤並修復。
```

### 級別 2：單元測試，每個新功能/檔案/函數使用現有測試模式
```python
# 建立 tests/test_streaming_components.py
import torch
import pytest
from taste_speech import TasteForCausalLM, TasteProcessor
from taste_speech.streaming import taste_tokenize, streaming_generate, taste_detokenize, TASTESegment

class TestTASTESegment:
    def test_text_segment_creation(self):
        """測試純文字段落建立"""
        text_ids = torch.randint(0, 1000, (1, 10))
        segment = TASTESegment(
            role='user',
            modality='text', 
            text_ids=text_ids
        )
        assert segment.taste_ids is None
        assert segment.role == 'user'
        
    def test_audio_segment_validation(self):
        """測試音訊段落需要 taste_ids"""
        text_ids = torch.randint(0, 1000, (1, 10))
        with pytest.raises(ValueError):
            TASTESegment(
                role='user',
                modality='audio',
                text_ids=text_ids,
                taste_ids=None  # 應該拋出錯誤
            )

class TestTasteTokenize:
    def test_basic_tokenization(self):
        """基本音訊標記化功能"""
        model = load_test_model()
        processor = load_test_processor()
        
        # 生成測試數據
        audio = torch.randn(1, 16000)  # 1秒音訊
        text_ids = torch.randint(0, 1000, (1, 50))
        
        result = taste_tokenize(model, processor, audio, text_ids)
        
        # 驗證輸出形狀
        assert result.shape[0] == 1  # batch size
        assert result.shape[1] == text_ids.shape[1]  # 序列長度對齊
        assert result.shape[2] > 0  # VQ 維度
        
    def test_batch_size_alignment(self):
        """測試批次大小對齊要求"""
        model = load_test_model()
        processor = load_test_processor()
        
        audio = torch.randn(2, 16000)  # 不匹配的 batch size
        text_ids = torch.randint(0, 1000, (1, 50))
        
        with pytest.raises(AssertionError):
            taste_tokenize(model, processor, audio, text_ids)

class TestStreamingGenerate:
    def test_streaming_output_format(self):
        """測試串流輸出格式"""
        model = load_test_model()
        processor = load_test_processor()
        
        input_segments = [
            TASTESegment(
                role='system',
                modality='text',
                text_ids=torch.randint(0, 1000, (1, 20))
            )
        ]
        
        # 測試 Iterator 行為
        results = []
        for result in streaming_generate(model, processor, input_segments):
            assert 'is_complete' in result
            assert 'completion_reason' in result
            assert 'segment' in result
            assert isinstance(result['segment'], TASTESegment)
            
            results.append(result)
            if result['is_complete']:
                break
                
        assert len(results) > 0
        assert results[-1]['is_complete'] is True

class TestTasteDetokenize:
    def test_audio_generation(self):
        """測試音訊生成功能"""
        model = load_test_model()
        processor = load_test_processor()
        
        # 準備測試數據
        speaker_embeds = torch.randn(1, 256)
        text_ids = torch.randint(0, 1000, (1, 20))
        taste_ids = torch.randint(0, 1024, (1, 20, 4))  # VQ_DIM=4
        
        result = taste_detokenize(
            model, processor, speaker_embeds,
            prev_text_ids=torch.empty(1, 0, dtype=torch.long),
            prev_taste_ids=torch.empty(1, 0, 4, dtype=torch.long),
            prev_speech_ids=torch.empty(1, 0, dtype=torch.long),
            prev_audio_ms=0,
            text_ids=text_ids,
            taste_ids=taste_ids,
            text_word_ids=torch.arange(text_ids.shape[1]).unsqueeze(0)
        )
        
        assert 'audio_waveform' in result
        assert 'sampling_rate' in result
        assert 'chunk_duration_ms' in result
        assert result['audio_waveform'].ndim == 2  # (1, T)
        assert result['sampling_rate'] == 16000

def load_test_model():
    """載入測試模型（簡化版本或 mock）"""
    # 實際實作時可能需要載入真實模型或使用 mock
    pass

def load_test_processor():
    """載入測試處理器"""
    pass
```

```bash
# 執行並迭代直到通過：
python -m pytest tests/test_streaming_components.py -v
# 如果失敗：讀取錯誤，理解根本原因，修復程式碼，重新執行
```

### 級別 3: 整合測試
```python
# 建立 tests/test_streaming_integration.py
class TestStreamingIntegration:
    
    def test_end_to_end_conversation_flow(self):
        """測試完整的對話流程"""
        model = load_full_model()  # 載入完整模型
        processor = load_full_processor()
        
        # 步驟 1: 音訊輸入處理
        audio = load_test_audio("test_audio.wav")  
        text_ids = processor.llm_tokenizer("Hello, how are you?", return_tensors="pt").input_ids
        
        taste_ids = taste_tokenize(model, processor, audio, text_ids)
        
        # 步驟 2: 構建對話
        input_segments = [
            TASTESegment(
                role='user',
                modality='audio',
                text_ids=text_ids,
                taste_ids=taste_ids
            )
        ]
        
        # 步驟 3: 串流生成
        generated_segments = []
        for result in streaming_generate(model, processor, input_segments):
            generated_segments.append(result['segment'])
            if result['is_complete']:
                break
        
        # 步驟 4: 音訊合成
        final_segment = generated_segments[-1]
        if final_segment.modality == 'audio':
            speaker_embeds = torch.randn(1, 256)  # 實際使用中應該是真實的 speaker embeddings
            
            audio_result = taste_detokenize(
                model, processor, speaker_embeds,
                prev_text_ids=torch.empty(1, 0, dtype=torch.long),
                prev_taste_ids=torch.empty(1, 0, 4, dtype=torch.long),
                prev_speech_ids=torch.empty(1, 0, dtype=torch.long), 
                prev_audio_ms=0,
                text_ids=final_segment.text_ids,
                taste_ids=final_segment.taste_ids,
                text_word_ids=torch.arange(final_segment.text_ids.shape[1]).unsqueeze(0)
            )
            
            # 驗證音訊輸出
            assert audio_result['audio_waveform'].numel() > 0
            assert audio_result['sampling_rate'] == 16000
    
    def test_multi_turn_conversation(self):
        """測試多輪對話狀態管理"""
        model = load_full_model()
        processor = load_full_processor()
        
        # 模擬多輪對話
        conversation_history = []
        
        for turn in range(3):
            # 添加用戶輸入
            user_text = f"This is turn {turn + 1}"
            user_text_ids = processor.llm_tokenizer(user_text, return_tensors="pt").input_ids
            
            conversation_history.append(TASTESegment(
                role='user',
                modality='text',
                text_ids=user_text_ids
            ))
            
            # 生成助手回應
            assistant_segments = []
            for result in streaming_generate(model, processor, conversation_history):
                if result['segment'].role == 'assistant':
                    assistant_segments.append(result['segment'])
                if result['is_complete']:
                    break
            
            # 添加助手回應到對話歷史
            conversation_history.extend(assistant_segments)
        
        # 驗證對話歷史結構
        assert len(conversation_history) >= 6  # 至少 3 輪用戶 + 3 輪助手
        assert conversation_history[0].role == 'user'
        assert conversation_history[1].role == 'assistant'

def load_full_model():
    """載入完整的預訓練模型"""
    model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
    return TasteForCausalLM.from_pretrained(model_id)

def load_full_processor():
    """載入完整的處理器"""
    model_id = 'MediaTek-Research/Llama-1B-TASTE-V0'
    return TasteProcessor.from_pretrained(model_id)

def load_test_audio(path):
    """載入測試音訊"""
    import torchaudio
    audio, sr = torchaudio.load(path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    return audio
```

```bash
# 執行整合測試
python -m pytest tests/test_streaming_integration.py -v --tb=short
# 預期：所有整合測試通過，端到端流程順暢
```

## 最終驗證檢查清單

### 代碼品質檢查
- [ ] 所有語法檢查通過：`python -m py_compile taste_speech/streaming/*.py`
- [ ] 單元測試覆蓋核心功能：`python -m pytest tests/test_streaming_components.py -v`
- [ ] 整合測試驗證端到端流程：`python -m pytest tests/test_streaming_integration.py -v`
- [ ] 匯入測試：`python -c "from taste_speech.streaming import taste_tokenize, streaming_generate, taste_detokenize, TASTESegment; print('Import success')"`

### 功能驗證檢查
- [ ] API 可用性測試：在 Python 中匯入並呼叫主要函數
- [ ] 音訊處理測試：使用真實音訊文件驗證 tokenize/detokenize 流程
- [ ] 串流生成測試：驗證 Iterator 模式和非阻塞生成
- [ ] 記憶體使用測試：確認串流模式相比批次模式的記憶體優勢

### 系統整合檢查  
- [ ] 模型載入測試：`python -c "from taste_speech import TasteForCausalLM; model = TasteForCausalLM.from_pretrained('MediaTek-Research/Llama-1B-TASTE-V0'); print('Model loaded successfully')"`
- [ ] 處理器相容性：確認新 API 與現有 TasteProcessor 的相容性
- [ ] 向後相容性：確認現有批次 API 仍然正常工作
- [ ] 文件更新：更新 README 或文件說明新的串流 API 使用方法

---

## 要避免的反模式

### 代碼實作反模式
- ❌ 不要重新實現已存在的音訊處理邏輯 - 重用 extract_vq 和 VoiceGenerator
- ❌ 不要忽略 tensor device 和 dtype 不匹配問題
- ❌ 不要在串流生成中使用同步阻塞 - 使用 threading 模式
- ❌ 不要忽略 batch size 對齊要求
- ❌ 不要硬編碼特殊 token IDs - 使用 tokenizer 轉換

### 串流實作反模式  
- ❌ 不要在主執行緒中進行長時間的模型推理 - 使用背景執行緒
- ❌ 不要忽略串流中的異常處理 - 優雅地傳播錯誤
- ❌ 不要假設所有輸入都會產生音訊輸出 - 處理純文字情況
- ❌ 不要忽略對話狀態的記憶體管理 - 適當清理歷史
- ❌ 不要忽略採樣率不匹配問題 - 實作重新採樣

## PRP 信心評分

**評分：8/10**

**高信心領域**：
- 現有程式碼庫理解充分，關鍵方法和模式已識別
- HuggingFace 串流模式有成熟的實作參考
- TorchAudio 即時音訊處理有官方支援和文件
- 資料結構設計清晰，API 設計合理

**潛在挑戰**：  
- 複雜的 TASTE token 對齊邏輯可能需要多次迭代調整
- 串流生成中的記憶體管理和執行緒安全性需要仔細測試
- VoiceGenerator 的上下文連續性處理可能需要額外的研究和實驗

**成功機率**：憑藉充分的上下文、清晰的實作路徑和全面的驗證循環，AI agent 應該能夠在一次實作中成功完成大部分功能，可能需要 1-2 輪的錯誤修復和優化。