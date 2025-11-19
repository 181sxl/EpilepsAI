import mne
from pathlib import Path
mne.viz.set_browser_backend("matplotlib")

fname = Path(r"D:\Learn_\Epilepsy_l\data\EDF_01\yanghuiyou_EEG_1.edf")
raw = mne.io.read_raw_edf(fname, preload=True, encoding='latin1', verbose=False)


# ----查看是否标注异常放电----
print(raw.annotations)   

events, event_dict = mne.events_from_annotations(raw)
print(events[:10])
print(event_dict)

# events = mne.find_events(raw, verbose=True)
# print(events[:10])

# ----绘制波形----
fig_waveform = raw.plot(scalings='auto', duration=10, n_channels=42, block=True)
fig_waveform.savefig(f"{fname.stem}_waveform.png", dpi=300)
print("Saved:", f"{fname.stem}_waveform.png")


# ----单通道时序波形----
import matplotlib.pyplot as plt
data, times = raw.get_data(picks=[0], return_times=True)

plt.figure(figsize=(15,4))
plt.plot(times, data[0])
plt.title(f"{fname.stem} - {raw.ch_names[0]} Waveform")
plt.xlabel("Time (s)")
plt.ylabel("uV")
plt.tight_layout()
plt.savefig(f"{fname.stem}_channel0.png", dpi=300)
plt.close()
print("Saved:", f"{fname.stem}_channel0.png")

# ---频域---
fig_psd = raw.plot_psd(fmax=50)
fig_psd.savefig(f"{fname.stem}_psd.png", dpi=300)
print("Saved:", f"{fname.stem}_psd.png")