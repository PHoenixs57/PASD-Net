import argparse
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Waveform + spectrogram + F0 overlay for a noisy/denoised pair (Figure 3 style)."
    )
    parser.add_argument(
        "--noisy",
        type=str,
        default="data/avian/Avian_noisy_01.wav",
        help="Path to noisy wav (default: bundled avian example)",
    )
    parser.add_argument(
        "--denoise",
        type=str,
        default="data/avian/Avian_denoise_01.wav",
        help="Path to denoised wav (default: bundled avian example)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fig/compare_wave_spec.png",
        help="Output image path",
    )
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--hop-length", type=int, default=128, help="Hop length")
    parser.add_argument("--fmin", type=float, default=64.0, help="pyin minimum F0 (Hz)")
    parser.add_argument("--fmax", type=float, default=2000.0, help="pyin maximum F0 (Hz)")
    parser.add_argument(
        "--max-time", type=float, default=12.0, help="Show only the first N seconds"
    )
    parser.add_argument("--dpi", type=int, default=400, help="Output resolution")
    return parser.parse_args()


def main():
    args = parse_args()

    y_noisy, sr = librosa.load(args.noisy, sr=None)
    y_denoise, sr2 = librosa.load(args.denoise, sr=None)
    if sr2 != sr:
        raise ValueError(f"Sample rate mismatch: noisy={sr}, denoise={sr2}")

    # STFT magnitude spectrograms (same parameters for both, so they are comparable)
    D_noisy = librosa.stft(y_noisy, n_fft=args.n_fft, hop_length=args.hop_length)
    S_noisy_db = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)

    D_denoise = librosa.stft(y_denoise, n_fft=args.n_fft, hop_length=args.hop_length)
    S_denoise_db = librosa.amplitude_to_db(np.abs(D_denoise), ref=np.max)

    # 2x2 layout: waveform (left) and spectrogram (right), noisy (top) vs denoised (bottom)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 6),
        sharex='row',
        facecolor='white',
        constrained_layout=True,
    )
    (ax_wave_noisy, ax_spec_noisy), (ax_wave_denoise, ax_spec_denoise) = axes

    # F0 contours via pyin
    FMIN, FMAX = args.fmin, args.fmax
    ENERGY_THRESHOLD_DB = -60.0

    f0_noisy, _, _ = librosa.pyin(
        y_noisy,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=args.n_fft,
        hop_length=args.hop_length,
    )
    f0_denoise, _, _ = librosa.pyin(
        y_denoise,
        fmin=FMIN,
        fmax=FMAX,
        frame_length=args.n_fft,
        hop_length=args.hop_length,
    )

    def plot_f0(ax, t, f0, **kwargs):
        """Plot F0 with safety against off-by-one frame mismatches."""
        min_len = min(len(t), len(f0))
        ax.plot(t[:min_len], f0[:min_len], **kwargs)

    t_f0_noisy = librosa.times_like(f0_noisy, sr=sr, hop_length=args.hop_length)
    t_f0_denoise = librosa.times_like(f0_denoise, sr=sr, hop_length=args.hop_length)

    # Colormap with white below threshold
    cmap = plt.cm.viridis.copy()
    cmap.set_under('white')

    # Waveforms
    t_noisy = np.arange(len(y_noisy)) / sr
    ax_wave_noisy.plot(t_noisy, y_noisy, color='black', linewidth=0.5)
    ax_wave_noisy.set_title('Noisy waveform')
    ax_wave_noisy.set_ylabel('Amplitude')

    t_denoise = np.arange(len(y_denoise)) / sr
    ax_wave_denoise.plot(t_denoise, y_denoise, color='black', linewidth=0.5)
    ax_wave_denoise.set_title('Denoised waveform')
    ax_wave_denoise.set_ylabel('Amplitude')
    ax_wave_denoise.set_xlabel('Time (s)')

    # Spectrograms
    img1 = librosa.display.specshow(
        S_noisy_db,
        sr=sr,
        hop_length=args.hop_length,
        x_axis='time',
        y_axis='log',
        cmap=cmap,
        vmin=ENERGY_THRESHOLD_DB,
        vmax=0.0,
        ax=ax_spec_noisy,
    )
    ax_spec_noisy.set_title('Noisy spectrogram')
    ax_spec_noisy.set_ylabel('Frequency (Hz)')
    ax_spec_noisy.set_facecolor('white')

    plot_f0(ax_spec_noisy, t_f0_noisy, f0_noisy, color='red', linewidth=1.0, label='F0')
    ax_spec_noisy.legend(loc='upper right')

    img2 = librosa.display.specshow(
        S_denoise_db,
        sr=sr,
        hop_length=args.hop_length,
        x_axis='time',
        y_axis='log',
        cmap=cmap,
        vmin=ENERGY_THRESHOLD_DB,
        vmax=0.0,
        ax=ax_spec_denoise,
    )
    ax_spec_denoise.set_title('Denoised spectrogram')
    ax_spec_denoise.set_ylabel('Frequency (Hz)')
    ax_spec_denoise.set_facecolor('white')
    ax_spec_denoise.set_xlabel('Time (s)')

    plot_f0(ax_spec_denoise, t_f0_denoise, f0_denoise, color='red', linewidth=1.0, label='F0')
    ax_spec_denoise.legend(loc='upper right')

    # Show only the first --max-time seconds
    total_time = min(len(y_noisy) / sr, len(y_denoise) / sr)
    for ax in [ax_wave_noisy, ax_wave_denoise, ax_spec_noisy, ax_spec_denoise]:
        ax.set_xlim(0, min(args.max_time, total_time))

    # Shared colorbar on the right, bound to the two spectrograms
    cbar = fig.colorbar(
        img2, ax=[ax_spec_noisy, ax_spec_denoise],
        format='%+2.0f dB', location='right', shrink=0.9,
    )
    cbar.set_label('Amplitude (dB)')

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
