import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def extract_bands_time_domain(signal: np.ndarray,
                              fs: float,
                              bands: list[tuple[float, float]]
                             ) -> np.ndarray:
    """
    通过 FFT 在频域上提取多个频带并重构时域信号。

    参数
    ----
    signal : np.ndarray
        输入时域信号（一维数组）。
    fs : float
        采样频率（Hz）。
    bands : list of (low, high)
        要提取的频率区间列表（每项为一个 (下限Hz, 上限Hz) 元组）。

    返回
    ----
    filtered_signal : np.ndarray
        重构后的时域信号，只保留指定 bands 中的频率成分。
    """
    assert len(signal.shape) == 1, "Signal must be a 1D array"
    assert len(bands) > 0, "Bands list must not be empty"
    
    N = len(signal)
    # FFT 及频率轴
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # 构造 mask：多个区间累计
    mask = np.zeros(N, dtype=bool)
    for low, high in bands:
        mask |= (np.abs(freqs) >= low) & (np.abs(freqs) <= high)

    # 频域滤波
    fft_filtered = fft_vals * mask
    # IFFT 重构
    return np.fft.ifft(fft_filtered).real

def test():
    # 读取CSV文件，第一行为采样频率
    df = pd.read_csv(r'datasets\化工行业赛道初赛数据集-训练集-重整\离心泵\bad\CP1_DE_M8_S2 (1).csv')
    fs = df['Value'].iloc[0]  # 采样频率
    signal = df['Value'].iloc[1:].values  # 原始信号
    N = len(signal)
    t = np.arange(N) / fs  # 时间轴
    bands = [(17.0,23.0)]
    filtered_signal=extract_bands_time_domain(signal,fs,bands)

    # 创建一个包含两个子图的图形
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 绘制原始时域信号
    axs[0].plot(t, signal)
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Original Signal')

    # 绘制滤波后时域信号
    axs[1].plot(t, filtered_signal)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Filtered Signal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
