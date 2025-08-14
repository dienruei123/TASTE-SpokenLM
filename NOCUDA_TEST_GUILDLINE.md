# TASTE-SpokenLM CPU-Only Testing Guide

本指南說明如何在沒有GPU的環境中（例如Mac）測試TASTE-SpokenLM的streaming components。

## 環境要求

- Python 3.8+
- 無需GPU/CUDA
- 支援的平台：macOS, Linux, Windows

## 安裝步驟

### 1. 建立虛擬環境

```bash
# 使用 uv 建立虛擬環境
uv venv ENV

# 啟動虛擬環境
source ENV/bin/activate
```

### 2. 安裝CPU-only PyTorch

```bash
# 安裝CPU版本的PyTorch和torchaudio
source ENV/bin/activate && uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安裝測試依賴

```bash
# 安裝其他測試相關依賴
source ENV/bin/activate && uv pip install -r requirements_nocuda_for_test.txt
```

## 測試執行

### 環境變數設定

測試系統使用 `NO_CUDA` 環境變數來控制是否強制使用CPU模式：

- `NO_CUDA=True` (默認) - 強制使用CPU，適合無GPU環境
- `NO_CUDA=False` - 允許使用GPU（如果可用）

### 運行單元測試

```bash
# 方法1：使用環境變數 (推薦)
source ENV/bin/activate && NO_CUDA=True uv run python -m pytest tests/test_streaming_components.py -v

# 方法2：導出環境變數
source ENV/bin/activate
export NO_CUDA=True
uv run python -m pytest tests/test_streaming_components.py -v

# 方法3：直接運行測試文件
source ENV/bin/activate && NO_CUDA=True uv run python tests/test_streaming_components.py
```

### 預期輸出

測試開始時會顯示設備模式：
```
Running tests in CPU-only mode (NO_CUDA=True)
```

## 測試涵蓋範圍

### 單元測試模組

1. **TestTASTESegment** - 測試多模態對話段落資料結構
   - 純文字段落創建
   - 音訊段落創建和驗證
   - 輸入驗證和錯誤處理

2. **TestStreamingResult** - 測試串流結果容器

3. **TestTasteTokenize** - 測試音訊到TASTE token轉換
   - 基本tokenization功能
   - 批次大小對齊驗證
   - 輸入類型驗證

4. **TestStreamingGenerate** - 測試串流生成功能
   - Iterator模式輸出格式
   - 背景執行緒處理

5. **TestTasteDetokenize** - 測試TASTE token到音訊轉換
   - 音訊生成功能
   - 上下文連續性
   - 輸入類型驗證

## 語法檢查

在運行測試前，建議先執行語法檢查：

```bash
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/segment.py
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/tokenize.py
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/generator.py
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/detokenize.py
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/conversation.py
source ENV/bin/activate && uv run python -m py_compile taste_speech/streaming/__init__.py
```

## 常見問題

### Q: 為什麼需要CPU-only模式？
A: 
- 開發和測試環境通常沒有GPU（如Mac筆電）
- CI/CD pipeline可能運行在CPU-only環境
- 確保代碼在不同環境下的相容性

### Q: CPU模式下測試的局限性？
A: 
- 性能測試結果不代表GPU環境表現
- 無法測試CUDA特定的錯誤處理
- 大型模型載入會較慢

### Q: 如何切換到GPU模式測試？
A: 設定 `NO_CUDA=False` 並確保環境中有可用的GPU：
```bash
source ENV/bin/activate && NO_CUDA=False uv run python -m pytest tests/test_streaming_components.py -v
```

### Q: 測試失敗怎麼辦？
A:
1. 確認虛擬環境已正確啟動
2. 檢查所有依賴是否正確安裝
3. 確認 `NO_CUDA=True` 環境變數已設定
4. 查看具體錯誤訊息進行除錯

## 安裝的套件說明

### 核心ML庫（CPU版本）
- `torch>=2.1.0` - CPU-only PyTorch
- `torchaudio>=2.1.0` - CPU-only audio processing

### NLP相關
- `transformers>=4.35.0` - Hugging Face transformers
- `tokenizers>=0.14.0` - Fast tokenizers

### 音訊處理
- `librosa>=0.10.0` - Audio analysis
- `soundfile>=0.12.0` - Audio file I/O
- `scipy>=1.11.0` - Scientific computing

### 測試框架
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `mock>=5.1.0` - Mocking utilities

### 其他工具
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `tqdm>=4.65.0` - Progress bars
- `requests>=2.31.0` - HTTP requests
- `onnxruntime>=1.16.0` - ONNX model inference (CPU)

## 後續整合測試

完成單元測試後，可以進行：
1. 整合測試 - 端到端功能測試
2. API功能驗證 - 確保匯入和呼叫正常
3. 性能基準測試 - CPU環境下的性能指標

## 注意事項

- 確保在運行測試前啟動虛擬環境
- CPU模式下某些操作可能較慢，這是正常現象
- 測試中使用的是mock objects，不會載入真實的大型模型
- 如要測試真實模型，需要額外下載模型檔案