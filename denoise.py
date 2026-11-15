"""Speech denoiser v2 (classical signal processing only, no ML, no clean reference).

High-pass -> STFT -> per-file noise PSD estimate -> decision-directed Wiener gain,
with persistent narrow-band (tonal) noise detected and suppressed more deeply.
Usage: python denoise.py [--input input] [--output output]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

N_FFT = 1024            # 64 ms at 16 kHz
HOP = 256               # 75 % overlap
HIGHPASS_HZ = 150.0
HIGHPASS_ORDER = 4
NOISE_FRACTION = 0.1    # quietest frames used for the first noise estimate
TRIM_K = 2.0            # per-bin trimmed mean: drop values above K x estimate
TRIM_ITERS = 8
DD_SMOOTHING = 0.95     # decision-directed a priori SNR smoothing
GAIN_FLOOR = 0.2
TONE_PERSIST_Q = 0.25   # a line must stay strong in >= 75 % of the frames
TONE_THRESHOLD_DB = 15.0
TONE_BAND_HZ = 250.0    # width of the local background used for line detection
TONE_GAIN_FLOOR = 0.05
EPS = 1e-18


def highpass(x, fs):
    if not 0 < HIGHPASS_HZ < fs / 2 or len(x) < 2:
        return x
    sos = signal.butter(HIGHPASS_ORDER, HIGHPASS_HZ, btype="highpass", fs=fs, output="sos")
    ntaps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return signal.sosfiltfilt(sos, x, padlen=min(3 * int(ntaps), len(x) - 1))


def stft(x, fs):
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    return signal.stft(x, fs=fs, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                       nfft=N_FFT, boundary="zeros", padded=True)


def istft(z, length, fs):
    _, y = signal.istft(z, fs=fs, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                        nfft=N_FFT, input_onesided=True, boundary=True)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]


def quiet_frames(power, times, length, fs):
    """Indices of the lowest-energy frames that are not affected by edge padding."""
    centers = np.rint(times * fs).astype(int)
    full = np.flatnonzero((centers >= N_FFT // 2) & (centers + N_FFT // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * len(full))))
    return full[np.argsort(energy, kind="stable")[:count]]


def estimate_noise(power, times, length, fs):
    """Per-bin noise power and gain floor for one file."""
    inner = power[:, 4:-4] if power.shape[1] > 16 else power
    low = power[:, quiet_frames(power, times, length, fs)].mean(axis=1)

    # Trimmed mean per bin: values far above the estimate are treated as speech.
    c = 1 - TRIM_K * np.exp(-TRIM_K) / (1 - np.exp(-TRIM_K))
    lam = np.maximum(uniform_filter1d(low, size=3, mode="nearest"), EPS)
    for _ in range(TRIM_ITERS):
        mask = inner < TRIM_K * lam[:, None]
        cnt = mask.sum(axis=1)
        mean = np.where(mask, inner, 0).sum(axis=1) / np.maximum(cnt, 1) / c
        lam = np.maximum(np.where(cnt > 0, mean, lam), EPS)
    # Use the smaller of the two: lowers bins where speech leaked into quiet frames.
    noise = np.maximum(np.minimum(low, lam), EPS)

    # Persistent narrow-band lines (sirens, dial tones): stronger than the local
    # background for most of the file. Speech harmonics do not persist that long.
    floor = np.full(power.shape[0], GAIN_FLOOR)
    persist = np.quantile(inner, TONE_PERSIST_Q, axis=1)
    width = max(3, int(round(TONE_BAND_HZ * N_FFT / fs)) | 1)
    background = median_filter(persist, size=width, mode="nearest")
    line = persist > 10 ** (TONE_THRESHOLD_DB / 10) * background
    noise = np.where(line, np.maximum(noise, np.median(inner, axis=1)), noise)
    floor = np.where(line, TONE_GAIN_FLOOR, floor)
    return noise, floor


def enhance(x, fs):
    x = highpass(np.asarray(x, dtype=np.float64), fs)
    _, times, z = stft(x, fs)
    power = np.abs(z) ** 2
    noise, floor = estimate_noise(power, times, len(x), fs)

    out = np.empty_like(z)
    previous = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        instant = np.maximum(power[:, t] / noise - 1.0, 0.0)
        if t == 0:
            prior = instant
        else:
            prior = DD_SMOOTHING * previous / noise + (1 - DD_SMOOTHING) * instant
        gain = np.clip(prior / (1 + prior), floor, 1)
        out[:, t] = gain * z[:, t]
        previous = (gain ** 2) * power[:, t]
    return istft(out, len(x), fs)


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
