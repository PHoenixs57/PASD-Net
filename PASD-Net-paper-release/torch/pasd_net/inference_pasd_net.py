import argparse
import os
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from torch import nn
import pasd_net  # 使用你现有的 PASD-Net 类


def read_pcm_int16(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.int16)
    return data.astype(np.float32) / 32768.0  # 归一化到 [-1,1]


def write_pcm_int16(path, audio):
    audio = np.clip(audio, -1.0, 1.0)
    data = (audio * 32767.0).astype(np.int16)
    with open(path, 'wb') as f:
        f.write(data.tobytes())


def frame_signal(x, frame_size, hop_size):
    """简单分帧，不加窗，只用于示例。"""
    num_frames = (len(x) - frame_size) // hop_size + 1
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, frame_size),
        strides=(x.strides[0] * hop_size, x.strides[0]),
        writeable=False
    )
    return frames.copy()


def overlap_add(frames, hop_size):
    frame_size = frames.shape[1]
    num_frames = frames.shape[0]
    out_len = (num_frames - 1) * hop_size + frame_size
    y = np.zeros(out_len, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_size
        y[start:start+frame_size] += frames[i]
    return y


def simple_feature_extractor(frames, nfft=128):
    """
    非严格 PASD-Net 特征，只用来对齐 PyTorch/C 网络行为：
    对每帧做 FFT，取前 65 维 log 幅度作为特征。
    """
    win = np.hanning(frames.shape[1]).astype(np.float32)
    feats = []
    for f in frames:
        spec = np.fft.rfft(f * win, n=nfft)
        mag = np.abs(spec) + 1e-8
        log_mag = np.log(mag)
        # 取前 65 维对齐 PASD-Net 默认输入维度
        feats.append(log_mag[:65])
    feats = np.stack(feats, axis=0)  # (T, 65)
    return feats


@torch.no_grad()
def run_inference(checkpoint_path, noisy_path, out_path, cond_size=128, gru_size=256,
                  frame_size=480, hop_size=480, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 载入模型
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = ckpt.get("model_kwargs", {"cond_size": cond_size, "gru_size": gru_size})
    model = pasd_net.PASDNet(**model_kwargs)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    # 2. 读入 noisy PCM
    noisy = read_pcm_int16(noisy_path)

    # 3. 分帧并提取特征
    frames = frame_signal(noisy, frame_size, hop_size)  # (T, frame_size)
    feats = simple_feature_extractor(frames)            # (T, 65)
    # 模型期望输入: (B, T, 65)
    feats_t = torch.from_numpy(feats[None, ...].astype(np.float32)).to(device)  # (1, T, 65)

    # 4. 前向推理
    states = None
    feats_t = feats_t.float()
    pred_gain, pred_vad, states = model(feats_t, states=states)  # pred_gain: (1, T, 32) in 原模型
    pred_gain = pred_gain.squeeze(0).cpu().numpy()  # (T, out_dim)

    # 5. 这里为了简单仅用第一个增益通道对整帧做缩放（非常粗糙，仅用于示意/对齐行为）
    # 真正 PASD-Net 里是频带增益 + IFFT，这里不完全复现。
    raw_gain = pred_gain[:, 0]
    normed = (raw_gain - raw_gain.min()) / (raw_gain.max() - raw_gain.min() + 1e-8)
    gain_per_frame = 0.5 + normed * 1.0  # 映射到 [0.5, 1.5]

    print("gain_per_frame sample:", gain_per_frame[:20])
    print("gain mean/std:", gain_per_frame.mean(), gain_per_frame.std())

    num_frames = frames.shape[0]
    num_gain = gain_per_frame.shape[0]
    min_len = min(num_frames, num_gain)

    if num_frames != num_gain:
        print(f"Warning: frames={num_frames}, gains={num_gain}, truncate to {min_len}")
    frames = frames[:min_len, :]
    gain_per_frame = gain_per_frame[:min_len]

    enhanced_frames = frames * gain_per_frame[:, None]
    enhanced = overlap_add(enhanced_frames, hop_size)

    # 对齐长度到原 noisy
    min_len = min(len(enhanced), len(noisy))
    enhanced = enhanced[:min_len]

    # 6. 写出 PCM
    write_pcm_int16(out_path, enhanced)
    print(f"PyTorch 推理完成，输出已写入: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="训练好的 PASD-Net checkpoint (.pth)")
    parser.add_argument("noisy_pcm", type=str, help="输入噪声 PCM (16-bit, mono)")
    parser.add_argument("out_pcm", type=str, help="输出降噪 PCM")
    parser.add_argument("--cond-size", type=int, default=128)
    parser.add_argument("--gru-size", type=int, default=384)
    parser.add_argument("--frame-size", type=int, default=480)
    parser.add_argument("--hop-size", type=int, default=480)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        noisy_path=args.noisy_pcm,
        out_path=args.out_pcm,
        cond_size=args.cond_size,
        gru_size=args.gru_size,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
    )


if __name__ == "__main__":
    main()