from typing import Optional

import numpy as np
from scipy.signal import medfilt, savgol_filter
import pywt


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    简单移动平均去噪
    参数:
        signal (np.ndarray): 一维或二维信号，shape=(n,) 或 (n, m)
        window_size (int): 窗口大小，正整数
    返回:
        np.ndarray: 去噪后信号，shape 与输入相同
    """
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")
    kernel = np.ones(window_size) / window_size
    # 对多通道沿 axis=0 滤波
    if signal.ndim == 1:
        return np.convolve(signal, kernel, mode='same')
    else:
        out = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            out[:, i] = np.convolve(signal[:, i], kernel, mode='same')
        return out

def median_denoise(signal: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    中值滤波去噪
    参数:
        signal (np.ndarray): 一维或二维信号，shape=(n,) 或 (n, m)
        kernel_size (int): 滤波核大小，务必为奇数
    返回:
        np.ndarray: 去噪后信号
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size 必须为正奇数")
    # medfilt 支持多维，但核会沿所有维度滑动，这里逐通道处理
    if signal.ndim == 1:
        return medfilt(signal, kernel_size=kernel_size)
    else:
        out = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            out[:, i] = medfilt(signal[:, i], kernel_size=kernel_size)
        return out

def savitzky_golay_denoise(signal: np.ndarray,
                           window_length: int = 7,
                           polyorder: int = 2) -> np.ndarray:
    """
    Savitzky–Golay 滤波去噪
    参数:
        signal (np.ndarray): 一维或二维信号，shape=(n,) 或 (n, m)
        window_length (int): 滑动窗口长度（正奇数，>= polyorder+2）
        polyorder (int): 多项式拟合阶数
    返回:
        np.ndarray: 去噪后信号
    """
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length 必须为奇数且 > polyorder")
    if signal.ndim == 1:
        return savgol_filter(signal, window_length, polyorder)
    else:
        out = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            out[:, i] = savgol_filter(signal[:, i], window_length, polyorder)
        return out

def wavelet_denoise(signal: np.ndarray,
                    wavelet: str = 'db4',
                    level: Optional[int] = None,
                    mode: str = 'soft') -> np.ndarray:
    """
    小波去噪（阈值去噪）
    参数:
        signal (np.ndarray): 一维或二维信号，shape=(n,) 或 (n, m)
        wavelet (str): 小波基
        level (int): 分解层数，默认 None 时自动计算最大层数
        mode (str): 阈值模式，'soft' 或 'hard'
    返回:
        np.ndarray: 去噪后信号
    """
    if signal.ndim == 1:
        return wavelet_denoise_1d(signal, wavelet, level, mode)
    else:
        out = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            out[:, i] = wavelet_denoise_1d(signal[:, i], wavelet, level, mode)
        return out

def wavelet_denoise_1d(signal: np.ndarray,
                    wavelet: str = 'db4',
                    level: Optional[int] = None,
                    mode: str = 'soft') -> np.ndarray:
    """
    小波去噪（阈值去噪）
    参数:
        signal (np.ndarray): 一维信号
        wavelet (str): 小波基
        level (int): 分解层数，默认 None 时自动计算最大层数
        mode (str): 阈值模式，'soft' 或 'hard'
    返回:
        np.ndarray: 去噪后信号
    """
    if signal.ndim != 1:
        raise ValueError("wavelet_denoise 仅支持一维信号")
    
    # 分解
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = level or max_level
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 估计噪声标准差(中位数绝对偏差)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # 通用阈值
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    # 阈值处理
    denoised = [coeffs[0]]
    for c in coeffs[1:]:
        denoised.append(pywt.threshold(c, value=uthresh, mode=mode))
    # 重构
    return pywt.waverec(denoised, wavelet)[:len(signal)]

def test():
    import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')

    # Generate a noisy signal
    t = np.linspace(0, 1, 1000)
    true_signal = np.sin(2 * np.pi * 5 * t) + 2* np.cos(np.pi *9 *t)
    signal = true_signal + np.random.normal(0, 0.5, 1000)

    # Apply the denoise function
    smoothed_signal = moving_average(signal, window_size=10)
    denoised_signal = wavelet_denoise(signal, wavelet='db4', level=5)

    # Plot the results
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t, smoothed_signal, label='Smoothed signal')
    plt.plot(t, denoised_signal, label='Denoised signal')
    plt.plot(t, true_signal, label='True signal', linestyle='--', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()