import mne
import numpy as np
import pandas as pd
from pathlib import Path
from utils.eeg_utils import compute_basic_stats, compute_band_power, print_raw_info

# file paths
data_dir = Path("data/EDF_01")
all_results = []

for file in sorted(data_dir.glob("*.edf")):
    print(f"\n ========读取文件： {file}============")

    # read EDF
    raw = mne.io.read_raw_edf(file, preload=True, encoding='latin1', verbose=False)
    print_raw_info(raw)


    # EEG channels
    eeg_channels = [ch for ch in raw.info["ch_names"] if "EEG" in ch]
    print("\n 使用的EEG通道: ", len(eeg_channels), eeg_channels)
    raw.pick_channels(eeg_channels)

    data = raw.get_data()              # [21, 22336]  [n_channels, n_samples]
    
    results = {"filename": file.name}
    # 遍历有效通道
    for i, ch_name in enumerate(raw.ch_names):
        signal = data[i]
        
        # ---时域特征---
        stats = compute_basic_stats(signal)
        for key, value in stats.items():
            results[f"{ch_name}_{key}"] = value
        
        # ---频域特征（带通滤波能量）---
        raw_ch = raw.copy().pick([ch_name])

        results[f"{ch_name}_delta"] = compute_band_power(raw_ch.copy(), 0.5, 4)
        results[f"{ch_name}_theta"] = compute_band_power(raw_ch.copy(), 4, 8)
        results[f"{ch_name}_alpha"] = compute_band_power(raw_ch.copy(), 8, 13)
        results[f"{ch_name}_beta"]  = compute_band_power(raw_ch.copy(), 13, 30)
        results[f"{ch_name}_gamma"] = compute_band_power(raw_ch.copy(), 30, 45)

    all_results.append(results)

    out_name = f"{file.stem}_full.npy"
    np.save(out_name, data)
    print("已保存 numpy 文件:", out_name)

df = pd.DataFrame(all_results)
df.to_csv("eeg_features.csv", index=False)
print("\n 已保存 eeg_features 文件 ")


