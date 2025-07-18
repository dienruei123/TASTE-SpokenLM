import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt

# === 1. 讀取所有 CSV 檔案（請確保共 5 份，欄位與行數一致）===
gpu_device = "t4"
csv_files = glob.glob(f"{gpu_device}/delay*.csv")  # 或指定資料夾，例如 "data/*.csv"
assert len(csv_files) == 5, f"只找到 {len(csv_files)} 份 CSV，應該是 5 份"

dfs = [pd.read_csv(f) for f in csv_files]

# === 2. 驗證欄位與行數一致 ===
for i, df in enumerate(dfs):
    assert list(df.columns) == list(dfs[0].columns), f"第 {i+1} 份 CSV 欄位不一致"
    assert len(df) == len(dfs[0]), f"第 {i+1} 份 CSV 行數不一致"

# ## TODO: 先處理各項delay對於Duration或GeneratedDuration的比值，並且更新回去
# for df in dfs:
#     # 假設有一個 'Duration' 欄位，計算各 delay 欄位對 Duration 的比值
#     for delay_col in ["PrepareTokenDelay", "AudioTowerDelay", "SpokenLMGenerateDelay"]:
#         if delay_col in df.columns:
#             df[delay_col] = df[delay_col] / df["Duration"]
#             # print(f"✅ 已計算 {delay_col} 對 Duration 的比值")
#     for delay_col in ["VoiceDecoderGenerateDelay", "TTSDelay"]:
#         if delay_col in df.columns:
#             df[delay_col] = df[delay_col] / df["GeneratedDuration"]
#             # print(f"✅ 已計算 {delay_col} 對 Duration 的比值")

# === 3. 對所有非 Filename 欄取平均，並保留 Filename ===
numeric_cols = [col for col in dfs[0].columns if col != "Filename"]
filename_col = dfs[0]["Filename"]

averaged_df = pd.concat([df[numeric_cols] for df in dfs]).groupby(level=0).mean()
averaged_df.insert(0, "Filename", filename_col)

# === 4. 儲存平均 CSV ===
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"averaged_delay_per_file_{timestamp}.csv"
averaged_df.to_csv(output_filename, index=False, float_format='%.4f')
print(f"✅ 已輸出平均結果到 {output_filename}")

# === 5. 計算相關係數（Duration 與 delay）===
print("\n===== Duration 與各 Delay 欄位的皮爾森相關係數 =====")
correlations = averaged_df[["Duration", "PrepareTokenDelay", "AudioTowerDelay", "SpokenLMGenerateDelay"]].corr()
print(correlations["Duration"])

print("\n===== GeneratedDuration 與各 Delay 欄位的皮爾森相關係數 =====")
correlations_gen = averaged_df[["GeneratedDuration", "SpokenLMGenerateDelay", "VoiceDecoderGenerateDelay", "TTSDelay"]].corr()
print(correlations_gen["GeneratedDuration"])

import os
import matplotlib.pyplot as plt
from scipy.stats import linregress

img_dir = "delay_plots"
os.makedirs(img_dir, exist_ok=True)

# === 1. Audio Duration vs Delay ===
for delay in ["PrepareTokenDelay", "AudioTowerDelay", "SpokenLMGenerateDelay"]:
    x = averaged_df["Duration"]
    y = averaged_df[delay]
    
    # 計算線性回歸
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = slope * x + intercept
    r_squared = r_value ** 2

    # 繪圖
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, label="Data")
    plt.plot(x, y_pred, color="red", label=f"y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r_squared:.4f}")
    plt.xlabel("Audio Duration (s)")
    plt.ylabel(delay)
    plt.title(f"Duration vs {delay} (on {gpu_device})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(img_dir, f"duration_vs_{delay}.png")
    plt.savefig(filename)
    plt.close()

    print(f"[Duration vs {delay}] 回歸函數: y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f}")

# === 2. Generated Duration vs Delay ===
for delay in ["SpokenLMGenerateDelay", "VoiceDecoderGenerateDelay", "TTSDelay"]:
    x = averaged_df["GeneratedDuration"]
    y = averaged_df[delay]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = slope * x + intercept
    r_squared = r_value ** 2

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, label="Data")
    plt.plot(x, y_pred, color="red", label=f"y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r_squared:.4f}")
    plt.xlabel("Generated Duration (s)")
    plt.ylabel(delay)
    plt.title(f"Generated Duration vs {delay} (on {gpu_device})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(img_dir, f"generated_duration_vs_{delay}.png")
    plt.savefig(filename)
    plt.close()

    print(f"[GeneratedDuration vs {delay}] 回歸函數: y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f}")

# === 6. 計算各項 delay 的平均值 ===
print("\n===== 各項 Delay 平均值（單位：秒）=====")
delay_columns = [
    "PrepareTokenDelay",
    "AudioTowerDelay",
    "SpokenLMGenerateDelay",
    "VoiceDecoderGenerateDelay",
    "TTSDelay"
]

for col in delay_columns:
    if col in averaged_df.columns:
        mean_value = averaged_df[col].mean()
        print(f"{col:<30}: {mean_value:.4f} 秒")
