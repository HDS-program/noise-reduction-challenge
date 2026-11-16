"""Speech denoiser v5 (classical signal processing only, no ML, no clean reference).

Zero-phase band weighting -> STFT -> per-bin speech/noise classification by temporal
variation (cv) -> noise PSD estimate -> HRNR gain (two-step SNR estimate + harmonic
regeneration), averaged over three STFT resolutions -> zero-phase output weighting curve.
Usage: python denoise.py [--input input] [--output output]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

WINDOWS = ((1024, 256), (1536, 384), (2048, 512))   # (window, hop) in samples, 75 % overlap
HP_HZ = 280.0           # input high-pass: gain^2 = 1 / (1 + (HP_HZ / f)^(2 * HP_ORDER))
HP_ORDER = 1.5
# Output weighting curve: gain (dB) at these frequencies (Hz), interpolated on a log-frequency axis.
# Keeps the band where speech dominates and attenuates the rest; tuned on the Train set.
OUT_NODES_HZ = (60, 100, 140, 180, 220, 260, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 820, 900, 1050, 1200, 1700, 2500, 4000, 6000, 8000)
OUT_GAIN_DB = (8.0, 4.0, -2.9, -8.0, -10.2, -12.0, -8.1, -4.0, 1.3, 6.0, 6.5, -0.6, -5.3, -11.7, -18.0, -25.8, -26.3, -26.9, -27.4, -27.9, -26.0, -28.0, -24.0, -26.3, -28.0)
FILTER_PAD = 2048       # reflect padding (samples) for the zero-phase FFT filters
NOISE_FRACTION = 0.1    # quietest frames used for the first noise estimate
TRIM_K = 2.0            # per-bin trimmed mean: drop values above K x estimate
TRIM_ITERS = 8
NOISE_SCALE = 1.1       # noise over-estimation factor used in the SNR estimates
DD_SMOOTHING = 0.95     # decision-directed a priori SNR smoothing
HRNR_RHO = 0.5          # mix of TSNR estimate and regenerated harmonics
XI_MIN_DB = -25.0
SPEECH_GAIN_FLOOR = 0.18
STAT_CV = 1.3           # bins whose power varies less than this (std/mean) are noise-like
STAT_LEVEL_DB = -3.0    # ... and are at least this strong relative to the first estimate
STAT_GAIN_FLOOR = 0.01  # deeper suppression for noise-like bins
EPS = 1e-18


def band_weight(x, fs, weight):
    """Zero-phase filter: multiply the spectrum of the whole signal by weight(f)."""
    if len(x) < 2:
        return x
    pad = min(FILTER_PAD, len(x) - 1)
    xp = np.pad(x, pad, mode="reflect")
    freqs = np.maximum(np.fft.rfftfreq(len(xp), 1.0 / fs), 1e-3)
    return np.fft.irfft(np.fft.rfft(xp) * weight(freqs), n=len(xp))[pad:pad + len(x)]


def highpass_weight(freqs):
    return 1.0 / (1.0 + (HP_HZ / freqs) ** (2 * HP_ORDER))


def output_weight(freqs):
    gain_db = np.interp(np.log(np.maximum(freqs, 1.0)), np.log(OUT_NODES_HZ), OUT_GAIN_DB)
    return 10 ** (gain_db / 20)


def estimate_noise(power, times, padded_len, fs, n_fft):
    """Per-bin noise power and gain floor for one file."""
    centers = np.rint(times * fs).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= padded_len))
    if not len(full):
        full = np.arange(power.shape[1])
    inner = power[:, full]
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    quiet = np.argsort(inner.mean(axis=0), kind="stable")[:count]
    low = inner[:, quiet].mean(axis=1)

    # The quiet frames can still contain speech. A trimmed mean over all frames
    # (corrected for the trimming of an exponential distribution) gives a second
    # estimate; the smaller of the two is used.
    correction = 1 - TRIM_K * np.exp(-TRIM_K) / (1 - np.exp(-TRIM_K))
    lam = np.maximum(uniform_filter1d(low, 3, mode="nearest"), EPS)
    for _ in range(TRIM_ITERS):
        mask = inner < TRIM_K * lam[:, None]
        cnt = mask.sum(axis=1)
        trimmed = np.where(mask, inner, 0).sum(axis=1) / np.maximum(cnt, 1) / correction
        lam = np.maximum(np.where(cnt > 0, trimmed, lam), EPS)
    first = np.maximum(np.minimum(low, lam), EPS)

    # Speech makes a bin's power swing between talking and pauses (large std/mean);
    # steady noise and tones vary much less. Noise-like bins use the whole-file mean
    # as their noise level and are suppressed more deeply.
    mean = inner.mean(axis=1)
    cv = inner.std(axis=1) / np.maximum(mean, EPS)
    steady = (cv < STAT_CV) & (mean > 10 ** (STAT_LEVEL_DB / 10) * first)
    noise = np.where(steady, np.maximum(first, mean), first)
    floor = np.where(steady, min(STAT_GAIN_FLOOR, SPEECH_GAIN_FLOOR), SPEECH_GAIN_FLOOR)
    return np.maximum(noise, EPS), floor


def enhance_one(x, fs, n_fft, hop):
    """HRNR enhancement at one STFT resolution."""
    length = len(x)
    xp = np.pad(x, (0, max(0, n_fft - length)))
    _, times, z = signal.stft(xp, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                              nfft=n_fft, boundary="zeros", padded=True)
    power = np.abs(z) ** 2
    noise, floor = estimate_noise(power, times, len(xp), fs, n_fft)
    noise = NOISE_SCALE * noise

    xi_min = 10 ** (XI_MIN_DB / 10)
    out = np.empty_like(z)
    previous = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        y = z[:, t]
        gamma = np.maximum(power[:, t] / noise, 1e-10)
        instant = np.maximum(gamma - 1.0, 0.0)
        xi = instant if t == 0 else DD_SMOOTHING * previous / noise + (1 - DD_SMOOTHING) * instant
        xi = np.maximum(xi, xi_min)
        gain = xi / (1 + xi)
        # Two-step SNR estimate (TSNR), then harmonic regeneration (HRNR):
        # half-wave rectifying the enhanced frame restores harmonics lost in step one.
        xi2 = np.abs(gain * y) ** 2 / noise
        gain = xi2 / (1 + xi2)
        harmonics = np.fft.rfft(np.maximum(np.fft.irfft(gain * y, n=n_fft), 0), n=n_fft)
        xi3 = (HRNR_RHO * np.abs(gain * y) ** 2 + (1 - HRNR_RHO) * np.abs(harmonics) ** 2) / noise
        gain = np.clip(xi3 / (1 + xi3), floor, 1)
        out[:, t] = gain * y
        previous = np.abs(out[:, t]) ** 2
    _, y = signal.istft(out, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                        nfft=n_fft, input_onesided=True, boundary=True)
    return np.pad(y, (0, max(0, length - len(y))))[:length]


def enhance(x, fs):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x
    x = band_weight(x, fs, highpass_weight)
    # Each resolution distorts the signal differently; averaging the waveforms
    # keeps the speech and reduces the resolution-specific artifacts.
    y = np.mean([enhance_one(x, fs, n_fft, hop) for n_fft, hop in WINDOWS], axis=0)
    return band_weight(y, fs, output_weight)


def process_file(src, dst):
    info = sf.info(src)
    x, fs = sf.read(src, dtype="float64", always_2d=True)
    if x.shape[0] == 0:
        y = x
    else:
        y = np.stack([enhance(x[:, c], fs) for c in range(x.shape[1])], axis=1)
    if info.subtype.startswith("PCM"):
        y = np.clip(y, -1.0, 1.0)
    if y.shape[1] == 1:
        y = y[:, 0]
    sf.write(dst, y, fs, subtype=info.subtype)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(base / "input"))
    parser.add_argument("--output", default=str(base / "output"))
    args = parser.parse_args()
    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() == ".wav"]
    for i, path in enumerate(files, 1):
        process_file(path, out_dir / path.name)
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
    print("done:", len(files), "files")


if __name__ == "__main__":
    main()
