"""Wiener-filter speech denoiser (classical signal processing, no ML).
Usage: python denoise.py [--input input] [--output output]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

N_FFT = 512
HOP = 128
NOISE_FRACTION = 0.1
WIENER_SMOOTHING = 0.95
WIENER_GAIN_FLOOR = 0.2


def stft_audio(x, fs):
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    return signal.stft(x, fs=fs, window="hann", nperseg=N_FFT,
                       noverlap=N_FFT - HOP, nfft=N_FFT,
                       boundary="zeros", padded=True)


def inverse_stft(z, length, fs):
    _, y = signal.istft(z, fs=fs, window="hann", nperseg=N_FFT,
                        noverlap=N_FFT - HOP, nfft=N_FFT,
                        input_onesided=True, boundary=True)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]


def estimate_noise_power(z, times, length, fs):
    power = np.abs(z) ** 2
    centers = np.rint(times * fs).astype(int)
    full = np.flatnonzero((centers >= N_FFT // 2) & (centers + N_FFT // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    estimate = power[:, selected].mean(axis=1)
    return np.maximum(uniform_filter1d(estimate, size=3, mode="nearest"), 1e-18)


def wiener_filter(x, fs):
    _, times, z = stft_audio(x, fs)
    power = np.abs(z) ** 2
    noise = estimate_noise_power(z, times, len(x), fs)
    posterior_snr = power / noise[:, None]
    enhanced = np.empty_like(z)
    previous = np.zeros_like(noise)
    for t in range(z.shape[1]):
        inst = np.maximum(posterior_snr[:, t] - 1.0, 0.0)
        if t == 0:
            prior = inst
        else:
            prior = WIENER_SMOOTHING * previous / noise + (1 - WIENER_SMOOTHING) * inst
        gain = np.clip(prior / (1 + prior), WIENER_GAIN_FLOOR, 1)
        enhanced[:, t] = gain * z[:, t]
        previous = np.abs(enhanced[:, t]) ** 2
    return inverse_stft(enhanced, len(x), fs)


def process_file(src, dst):
    info = sf.info(src)
    x, fs = sf.read(src, dtype="float64", always_2d=True)
    y = np.stack([wiener_filter(x[:, c], fs) for c in range(x.shape[1])], axis=1)
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
