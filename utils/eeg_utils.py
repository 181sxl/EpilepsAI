import numpy as np

def print_raw_info(raw):
    print(raw)  
    print(f"采样率：{raw.info['sfreq']}")
    print(f"通道数：{len(raw.ch_names)}")
    print(f"通道名：{raw.ch_names}")


def compute_basic_stats(signal):
    """简单时域特征"""
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "var": float(np.var(signal)),
        "max": float(np.max(signal)),
        "min": float(np.min(signal)),
    }


def compute_band_power(raw, l_freq, h_freq):
    """
    带通滤波计算能量（均方值）
    l_freq, h_freq: 频段范围
    """
    filtered = raw.copy().filter(l_freq, h_freq, verbose=False)
    data = filtered.get_data()
    return float(np.mean(data ** 2))
