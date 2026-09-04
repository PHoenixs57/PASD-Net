import argparse
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter plot of per-frame F0: noisy vs denoised (pyin)."
    )
    parser.add_argument(
        "--noisy",
        type=str,
        default="data/white-headed_langur/Long_noisy_01.wav",
        help="Path to noisy wav (default: bundled langur example)",
    )
    parser.add_argument(
        "--denoise",
        type=str,
        default="data/white-headed_langur/Long_denoise_01.wav",
        help="Path to denoised wav (default: bundled langur example)",
    )
    parser.add_argument("--fmin", type=float, default=64.0)
    parser.add_argument("--fmax", type=float, default=2000.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument(
        "--out",
        type=str,
        default="fig/f0_scatter.png",
        help="Output image path",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional: limit to first N seconds (uses min duration if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    y_noisy, sr = librosa.load(args.noisy, sr=None)
    y_denoise, sr2 = librosa.load(args.denoise, sr=None)
    if sr2 != sr:
        raise ValueError(f"Sample rate mismatch: noisy={sr}, denoise={sr2}")

    if args.max_seconds is not None:
        max_n = int(round(args.max_seconds * sr))
        y_noisy = y_noisy[:max_n]
        y_denoise = y_denoise[:max_n]

    f0_noisy, _, _ = librosa.pyin(
        y_noisy,
        fmin=args.fmin,
        fmax=args.fmax,
        frame_length=args.n_fft,
        hop_length=args.hop_length,
    )
    f0_denoise, _, _ = librosa.pyin(
        y_denoise,
        fmin=args.fmin,
        fmax=args.fmax,
        frame_length=args.n_fft,
        hop_length=args.hop_length,
    )

    # Align by frame index (same hop_length & sr). Audio lengths may differ by a frame.
    n = min(len(f0_noisy), len(f0_denoise))
    x = f0_noisy[:n]
    y = f0_denoise[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise RuntimeError("No valid paired F0 points (all frames are unvoiced/NaN).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(6, 6), facecolor="white")
    plt.scatter(
        x,
        y,
        s=6,
        alpha=0.35,
        edgecolors="none",
        label="Paired voiced frames",
    )

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    plt.plot([lo, hi], [lo, hi], color="red", linewidth=1.0, alpha=0.45, label="y = x")

    # 95% confidence interval band (OLS linear regression: y = a + b x)
    if x.size >= 3:
        x_mean = float(np.mean(x))
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx > 0:
            b, a = np.polyfit(x, y, deg=1)
            y_hat = a + b * x
            sse = float(np.sum((y - y_hat) ** 2))
            dof = int(x.size - 2)
            s = float(np.sqrt(sse / dof)) if dof > 0 else 0.0

            xs = np.linspace(lo, hi, 200)
            ys = a + b * xs
            se_mean = s * np.sqrt(1.0 / x.size + (xs - x_mean) ** 2 / sxx)

            if student_t is not None and dof > 0:
                t_crit = float(student_t.ppf(0.975, dof))
            else:
                t_crit = 1.96

            lower = ys - t_crit * se_mean
            upper = ys + t_crit * se_mean
            plt.fill_between(
                xs,
                lower,
                upper,
                color="C0",
                alpha=0.18,
                label="95% CI band",
                linewidth=0,
            )

    r = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    plt.gca().text(
        0.02,
        0.98,
        f"N={x.size}\nPearson r={r:.3f}\nMAE={mae:.2f} Hz\nRMSE={rmse:.2f} Hz",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    plt.xlabel("Noisy F0 (Hz)")
    plt.ylabel("Denoised F0 (Hz)")
    plt.title("F0 scatter: noisy vs denoised (paired frames)")
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    plt.savefig(args.out, dpi=400)
    print(f"Saved: {args.out}")
    print(f"Paired voiced points: {x.size}")


if __name__ == "__main__":
    main()
