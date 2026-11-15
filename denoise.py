"""Speech denoiser (classical signal processing only, no ML, no clean reference).

High-pass -> STFT -> per-bin speech/noise classification by temporal variation (cv)
-> noise PSD estimate -> HRNR gain (two-step SNR estimate + harmonic regeneration).
Usage: python denoise.py [--input input] [--output output]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

N_FFT = 1536            # 96 ms at 16 kHz
HOP = 384               # 75 % overlap
HIGHPASS_HZ = 150.0
NOISE_FRACTION = 0.1    # quietest frames used for the first noise estimate
DD_SMOOTHING = 0.95     # decision-directed a priori SNR smoothing
HRNR_RHO = 0.5          # mix of TSNR estimate and regenerated harmonics
XI_MIN_DB = -25.0
SPEECH_GAIN_FLOOR = 0.15
STAT_CV = 1.3           # bins whose power varies less than this (std/mean) are noise-like
STAT_LEVEL_DB = -3.0    # ... and are at least this strong relative to the first estimate
STAT_GAIN_FLOOR = 0.01  # deeper suppression for noise-like bins
EPS = 1e-18


def highpass(x, fs):
    if not 0 < HIGHPASS_HZ < fs / 2 or len(x) < 16:
        return x
    sos = signal.butter(4, HIGHPASS_HZ, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, padlen=min(27, len(x) - 1))


def stft(x, fs):
    return signal.stft(x, fs=fs, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                       nfft=N_FFT, boundary="zeros", padded=True)


def istft(z, length, fs):
    _, y = signal.istft(z, fs=fs, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                        nfft=N_FFT, input_onesided=True, boundary=True)
    return np.pad(y, (0, max(0, length - len(y))))[:length]


def estimate_noise(power, times, padded_len, fs):
    """Per-bin noise power and gain floor for one file."""
    centers = np.rint(times * fs).astype(int)
    full = np.flatnonzero((centers >= N_FFT // 2) & (centers + N_FFT // 2 <= padded_len))
    if not len(full):
        full = np.arange(power.shape[1])
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    quiet = full[np.argsort(power[:, full].mean(axis=0), kind="stable")[:count]]
    first = np.maximum(uniform_filter1d(power[:, quiet].mean(axis=1), 3, mode="nearest"), EPS)

    # Speech makes a bin's power swing between talking and pauses (large std/mean);
    # steady noise and tones vary much less. Noise-like bins use the whole-file mean
    # as their noise level and are suppressed more deeply.
    inner = power[:, full]
    mean = inner.mean(axis=1)
    cv = inner.std(axis=1) / np.maximum(mean, EPS)
    steady = (cv < STAT_CV) & (mean > 10 ** (STAT_LEVEL_DB / 10) * first)
    noise = np.where(steady, np.maximum(first, mean), first)
    floor = np.where(steady, min(STAT_GAIN_FLOOR, SPEECH_GAIN_FLOOR), SPEECH_GAIN_FLOOR)
    return np.maximum(noise, EPS), floor


def enhance(x, fs):
    x = highpass(np.asarray(x, dtype=np.float64), fs)
    length = len(x)
    xp = np.pad(x, (0, max(0, N_FFT - length)))
    _, times, z = stft(xp, fs)
    power = np.abs(z) ** 2
    noise, floor = estimate_noise(power, times, len(xp), fs)

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
        harmonics = np.fft.rfft(np.maximum(np.fft.irfft(gain * y, n=N_FFT), 0), n=N_FFT)
        xi3 = (HRNR_RHO * np.abs(gain * y) ** 2 + (1 - HRNR_RHO) * np.abs(harmonics) ** 2) / noise
        gain = np.clip(xi3 / (1 + xi3), floor, 1)
        out[:, t] = gain * y
        previous = np.abs(out[:, t]) ** 2
    return istft(out, length, fs)


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
