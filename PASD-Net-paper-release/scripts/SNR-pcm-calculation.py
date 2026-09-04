"""Objective quality metrics for a (clean, processed) PCM pair.

Computes a frequency-band SNR estimate (signal band <= 4 kHz vs noise band
> 4 kHz at 48 kHz) and, optionally, scale-invariant SDR (SI-SDR / SI-SNR).

Usage:
    python scripts/SNR-pcm-calculation.py <clean.pcm> <processed.pcm> [--sisdr]

Inputs are raw 16-bit little-endian mono PCM at 48 kHz.
"""
import numpy as np
import argparse
import sys
import warnings
from scipy.signal import correlate

def read_pcm(file_path, sample_width=2, signed=True):
    """读取PCM文件并返回归一化的浮点数组"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # 确定数据类型和最大值
    dtype_map = {
        1: np.int8 if signed else np.uint8,
        2: np.int16,
        3: 'special_24bit_case',  # 需要特殊处理
        4: np.int32
    }
    
    if sample_width not in dtype_map:
        raise ValueError(f"不支持的采样位宽: {sample_width}")
    
    # 处理24位特殊情况
    if sample_width == 3:
        bytes_array = np.frombuffer(data, dtype=np.uint8)
        samples = np.zeros(len(bytes_array) // 3, dtype=np.int32)
        for i in range(len(samples)):
            b = bytes_array[i*3:(i+1)*3].tobytes()
            samples[i] = int.from_bytes(b, byteorder='little', signed=True)
        max_val = 2**23
    else:
        dtype = dtype_map[sample_width]
        samples = np.frombuffer(data, dtype=dtype)
        max_val = 2**(8*sample_width-1) if signed else 2**(8*sample_width)
    
    return samples.astype(np.float64) / max_val


def align_signals(clean, noisy):
    """改进的稳健对齐函数"""
    # 确保至少有1秒的重叠（采样率48kHz）
    min_overlap = 48000
    if len(clean) < min_overlap or len(noisy) < min_overlap:
        raise ValueError("信号长度不足，无法可靠对齐（需>1秒）")

    correlation = correlate(clean, noisy, mode='valid')
    if len(correlation) == 0:
        raise ValueError("信号长度差异过大，无法对齐")
    
    lag = np.argmax(correlation)
    aligned_clean = clean[lag : lag+len(noisy)]
    
    # 边界检查
    if len(aligned_clean) != len(noisy):
        aligned_clean = np.pad(aligned_clean, 
                             (0, len(noisy)-len(aligned_clean)), 
                             mode='constant')
    
    return aligned_clean, noisy


def calculate_sisdr(clean, enhanced, eps=1e-12):
    """更稳健的SI-SDR实现"""
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # 能量检查
    if np.sum(clean**2) < eps:
        return float('-inf')
    
    alpha = np.dot(clean, enhanced) / (np.sum(clean**2) + eps)
    target = alpha * clean
    distortion = enhanced - target
    
    signal_power = np.sum(target**2) + eps
    distortion_power = np.sum(distortion**2) + eps
    
    if distortion_power < eps:
        return float('inf')
    
    return 10 * np.log10(signal_power / distortion_power)


def calculate_snr(clean, noisy, eps=1e-12):
    """带保护的SNR计算"""
    noise = noisy - clean
    
    signal_power = np.mean(clean**2) + eps
    noise_power = np.mean(noise**2) + eps
    
    # 防止无效值
    if signal_power < eps or noise_power < eps:
        return float('-inf')  # 表示无效结果
    
    return 10 * np.log10(signal_power / noise_power)

import numpy as np
from scipy.fft import fft, fftfreq
import scipy.signal.windows as windows

def estimate_snr(clean, noisy, sample_rate, signal_upper_freq=4000, window='hann', window_beta=14):
    """改进的频域SNR计算函数（关注4kHz以下频段）"""
    N = len(clean)
    # 应用窗口函数（减少频谱泄漏）
    clean_windowed = clean
    noisy_windowed = noisy
    
    # 计算FFT
    Y_clean = fft(clean_windowed)
    Y_noisy = fft(noisy_windowed)
    
    # 生成频率轴（仅考虑正频率）
    freqs = fftfreq(N, 1/sample_rate)[:N//2]  # 取单边谱（0到Nyquist频率）
    
    # 确定信号频段（0-4kHz）和噪声频段（4kHz以上）
    mask_signal = (freqs <= signal_upper_freq)
    mask_noise = (freqs > signal_upper_freq)
    
    # 计算信号功率（4kHz以下的频段总功率）
    signal_power = np.sum(np.abs(Y_clean[:N//2][mask_signal])**2) * 2 / N  # 单边谱功率计算
    
    # 计算噪声功率（4kHz以上的频段平均功率）
    noise_power = np.mean(np.abs(Y_noisy[:N//2][mask_noise])**2) * 2 / N  # 单边谱功率计算
    
    # 处理极端情况
    if noise_power < 1e-12:
        noise_power = 1e-12
    
    # 计算SNR
    SNR = 10 * np.log10(signal_power / noise_power)
    
    return SNR

def calculate_metrics(clean_file, noisy_file, sample_width=2, signed=True, compute_sisdr=False):
    """计算多个指标的通用函数"""
    try:
        clean = read_pcm(clean_file, sample_width, signed)
        noisy = read_pcm(noisy_file, sample_width, signed)
    except ValueError as e:
        raise RuntimeError(f"文件读取失败: {str(e)}")
    
    # 新增数据验证
    if len(clean) == 0 or len(noisy) == 0:
        raise ValueError("检测到空信号！请检查输入文件是否有效")
    if np.all(clean == 0) or np.all(noisy == 0):
        raise ValueError("检测到全零信号！请检查输入文件是否正确")

    # 信号对齐
    clean_aligned, noisy_aligned = align_signals(clean, noisy)
    
    # 统一长度
    min_len = min(len(clean_aligned), len(noisy_aligned))
    clean_aligned = clean_aligned[:min_len]
    noisy_aligned = noisy_aligned[:min_len]

    # 计算噪声
    noise = noisy_aligned - clean_aligned

    # 转换为原始数据类型
    if sample_width not in [1,2,3,4]:
        raise ValueError(f"不支持的采样位宽：{sample_width}")

    # 确定数据类型和缩放参数
    if signed:
        max_val = 2 ** (8 * sample_width -1)
    else:
        max_val = 2 ** (8 * sample_width)
    
    # 特殊处理24位（如果需要）
    if sample_width == 3:
        # 24位需要特殊处理（暂时不考虑）
        pass
    else:
        # 普通情况处理
        # 确定数据类型
        dtype_map = {
            1: np.int8 if signed else np.uint8,
            2: np.int16,
            4: np.int32
        }
        dtype = dtype_map[sample_width]
        
        # 将归一化的浮点数转回原始整数范围
        scaled_noise = (noise * max_val).astype(dtype)
    
    # 写入PCM文件
    with open("./noise.pcm", 'wb') as pcm_file:
        if sample_width == 3:
            # 24位需要特殊处理（暂时不考虑）
            pass
        else:
            scaled_noise.tofile(pcm_file)

    # 计算SNR
    snr = estimate_snr(clean_aligned, noisy_aligned,48000)

    # 计算SI-SDR（按需）
    sisdr = None
    if compute_sisdr:
        sisdr = calculate_sisdr(clean_aligned, noisy_aligned)
    
    return snr, sisdr

def main():
    parser = argparse.ArgumentParser(description='语音质量评估工具')
    parser.add_argument('clean_file', help='纯净语音PCM文件路径')
    parser.add_argument('noisy_file', help='待评估语音PCM文件路径')
    parser.add_argument('-w', '--sample_width', type=int, default=2,
                        choices=[1, 2, 3, 4], help='采样位宽（字节）默认：2')
    parser.add_argument('-u', '--unsigned', action='store_false', dest='signed',
                        help='使用无符号格式（默认有符号）')
    parser.add_argument('--sisdr', action='store_true',
                        help='同时计算SI-SDR指标')

    args = parser.parse_args()

    try:
        snr, sisdr = calculate_metrics(
            args.clean_file, 
            args.noisy_file,
            args.sample_width,
            args.signed,
            args.sisdr
        )
        
        print(f"SNR: {snr:.2f} dB")
        if args.sisdr:
            print(f"SI-SDR: {sisdr:.2f} dB")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()