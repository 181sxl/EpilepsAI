%matplotlib inline

import mne
mne.viz.set_browser_backend("matplotlib")  # 关键：改用 matplotlib 浏览器


from pathlib import Path
import numpy as np

# ===== 1) 设置你的文件路径 =====
fname = Path(r"/nas/linbo/biospace/exps/20251116-EpilepsAI/data20251114/NKT/EEG2100/DA3960AN.EEG")

print("读取文件：", fname)

# ===== 2) 读取 EEG-2100 数据 =====
# preload=True 代表一次性加载进内存（方便后续处理）
raw = mne.io.read_raw_nihon(fname, preload=True)

print(raw)
print("采样率：", raw.info["sfreq"])
print("通道数：", len(raw.ch_names))
print("前几个通道名:", raw.ch_names[:10])

# ===== 3) 简单滤波（可选） =====
raw.filter(l_freq=1., h_freq=40.)

# ===== 4) 取出 10 秒数据 =====
sfreq = raw.info["sfreq"]
start = 0               # 0 秒开始
stop = int(10 * sfreq)  # 前 10 秒
clip = raw.get_data(start=start, stop=stop)

print("10 秒片段形状：", clip.shape)


# ===== 5) 绘制波形 =====
raw.plot(start=0, n_channels=32, duration=10, block=True)


#（2）展示某个通道的时序波形（matplotlib）
import matplotlib.pyplot as plt

data, times = raw[:1]  # 读取第 1 个通道（Channel 0）

plt.figure(figsize=(15, 4))
plt.plot(times, data[0])
plt.title("Channel 0 EEG Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.show()


#（3）展示功率谱（频域）
raw.plot_psd(fmax=50)


#5. 读取原始数组用于分析
data = raw.get_data()   # shape = [n_channels, n_samples]
data.shape


# ===== 6) 保存为 numpy =====
np.save("DA3960AN_10s.npy", clip)
print("已保存 numpy 文件：DA3960AN_10s.npy")