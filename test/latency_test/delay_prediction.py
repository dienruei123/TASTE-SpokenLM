import pandas as pd

# === 載入你平均後的 CSV（假設名稱如下） ===
df = pd.read_csv("averaged_delay_per_file_20250701_165211.csv")  # 改成你的檔名

# === 設定 GPU TFLOPS ===
tflops_current = 10.6  # 例如 A100
tflops_target = 1000  # 例如 H100 或 RTX 4090
scale = tflops_target / tflops_current

# === 計算每秒延遲 (Current GPU) ===
delay_per_sec = {}
for delay in ["PreparingDelay", "TokenizeCompletionDelay", "TTSDelay"]:
    delay_per_sec[delay] = (df[delay] / df["Duration"]).mean()

print("=== 每秒平均延遲 (Current GPU) ===")
for k, v in delay_per_sec.items():
    print(f"{k}/s = {v:.6f} sec")

# === 假設你要預測的音檔長度 ===
duration_target = 5.0  # 單位：秒

# === 預測其他 GPU 上的各項 delay ===
predicted = {}
for delay, per_sec in delay_per_sec.items():
    predicted[delay] = per_sec * duration_target * (tflops_current / tflops_target)

print(f"\n=== 預測 {duration_target} 秒音檔在 TFLOPS={tflops_target} 上的各項 delay ===")
for k, v in predicted.items():
    print(f"{k}: {v:.4f} 秒")

print(f"Total Delay: {sum(predicted.values()):.4f} 秒")
