"""
dsp.py — file loading, spectral analysis, wavelet decomposition/denoising and feature extraction for RajDSP.
Built only on open-source libraries: NumPy, SciPy, PyWavelets, pandas.
"""
import io
import re

import numpy as np
import pandas as pd
import pywt
from scipy import signal as sps
from scipy import stats

# ============================================================================ file loading
SUPPORTED = ["txt", "csv", "xlsx", "xls", "lvm", "mat", "tsv", "dat"]


def _numeric_frame(df):
    """Keep numeric columns, coerce strings with decimal commas, drop all-NaN columns/rows."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        if s.notna().mean() > 0.8:
            out[str(c)] = s
    out = out.dropna(how="all").reset_index(drop=True)
    return out


def _read_text_table(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("File is empty")
    sample = "\n".join(lines[:50])
    if "\t" in sample:
        sep = "\t"
    elif ";" in sample:
        sep = ";"
    elif "," in sample:
        sep = ","
    else:
        sep = r"\s+"
    # skip leading non-numeric junk lines but keep one header line if present
    num_re = re.compile(r"^[\s\-\+]*[\d\.]")
    first_num = next((i for i, ln in enumerate(lines) if num_re.match(ln)), 0)
    header_line = lines[first_num - 1] if first_num > 0 else None
    body = "\n".join(lines[first_num:])
    df = pd.read_csv(io.StringIO(body), sep=sep, header=None, engine="python")
    if header_line is not None:
        names = re.split(sep if sep != r"\s+" else r"\s+", header_line.strip())
        if len(names) == df.shape[1] and len(set(names)) == len(names):
            df.columns = [n.strip().strip('"') for n in names]
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"Ch{i + 1}" for i in range(df.shape[1])]
    return df


def _read_lvm(text):
    """LabVIEW Measurement File: data follow the last ***End_of_Header*** line."""
    marker = "***End_of_Header***"
    if marker in text:
        text = text[text.rfind(marker) + len(marker):]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = None
    if lines and not re.match(r"^[\s\-\+]*[\d\.]", lines[0]):
        header = lines[0].split("\t")
        lines = lines[1:]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    if header is not None:
        header = [h.strip() for h in header if h.strip() != "" and h.strip().lower() != "comment"]
        if len(header) == df.shape[1]:
            df.columns = header
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = ["X_Value"] + [f"Ch{i}" for i in range(1, df.shape[1])] if df.shape[1] > 1 else ["Ch1"]
    return df


def _read_mat(raw):
    """MATLAB .mat (e.g. CWRU bearing data: keys like X097_DE_time). Returns every 1-D numeric array."""
    from scipy.io import loadmat
    m = loadmat(io.BytesIO(raw))
    cols = {}
    for k, v in m.items():
        if k.startswith("__") or not isinstance(v, np.ndarray) or v.dtype.kind not in "fiu":
            continue
        a = np.squeeze(v)
        if a.ndim == 1 and a.size > 16:
            cols[k] = a.astype(float)
        elif a.ndim == 2 and min(a.shape) <= 16 and max(a.shape) > 16:
            a = a if a.shape[0] > a.shape[1] else a.T
            for j in range(a.shape[1]):
                cols[f"{k}_{j + 1}"] = a[:, j].astype(float)
    if not cols:
        raise ValueError("No 1-D numeric signals found in .mat file")
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: np.pad(v, (0, n - len(v)), constant_values=np.nan) for k, v in cols.items()})


def load_table(name, raw):
    """Load any supported file (bytes) into a numeric DataFrame."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(raw))
    elif ext == "mat":
        df = _read_mat(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        df = _read_lvm(text) if ext == "lvm" else _read_text_table(text)
    num = _numeric_frame(df)
    if num.shape[1] == 0:
        raise ValueError("No numeric columns found")
    return num


def load_any_table(name, raw):
    """Load a dataset that may contain text columns (for ML / regression / data analysis)."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(raw))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def guess_fs(df):
    """If the first column looks like a monotonic time axis, return (fs, time_col)."""
    if df.shape[1] < 2:
        return None, None
    c0 = df.columns[0]
    t = df[c0].to_numpy(float)
    if len(t) < 4 or np.any(~np.isfinite(t[:1000])):
        return None, None
    d = np.diff(t[: min(len(t), 5000)])
    if np.all(d > 0) and np.std(d) / np.mean(d) < 0.05:
        return float(1.0 / np.mean(d)), c0
    return None, None


# ============================================================================ preprocessing
def preprocess(x, fs, detrend="constant", filt="None", f_lo=None, f_hi=None, order=4):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if detrend in ("constant", "linear"):
        x = sps.detrend(x, type=detrend)
    nyq = fs / 2
    try:
        if filt == "Low-pass" and f_hi and f_hi < nyq:
            sos = sps.butter(order, f_hi, "lowpass", fs=fs, output="sos")
            x = sps.sosfiltfilt(sos, x)
        elif filt == "High-pass" and f_lo and f_lo < nyq:
            sos = sps.butter(order, f_lo, "highpass", fs=fs, output="sos")
            x = sps.sosfiltfilt(sos, x)
        elif filt == "Band-pass" and f_lo and f_hi and 0 < f_lo < f_hi < nyq:
            sos = sps.butter(order, [f_lo, f_hi], "bandpass", fs=fs, output="sos")
            x = sps.sosfiltfilt(sos, x)
    except ValueError:
        pass
    return x


# ============================================================================ spectra
def fft_spectrum(x, fs, window="hann"):
    n = len(x)
    w = sps.get_window(window, n) if window != "none" else np.ones(n)
    X = np.fft.rfft(x * w)
    amp = 2.0 * np.abs(X) / np.sum(w)
    amp[0] /= 2
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, amp


def welch_psd(x, fs, nperseg=1024):
    nperseg = int(min(nperseg, len(x)))
    return sps.welch(x, fs, nperseg=nperseg)


def stft_map(x, fs, nperseg=256, overlap=0.75):
    nperseg = int(min(nperseg, len(x)))
    f, t, Z = sps.stft(x, fs, nperseg=nperseg, noverlap=int(nperseg * overlap))
    return f, t, 20 * np.log10(np.abs(Z) + 1e-12)


def envelope_spectrum(x, fs):
    env = np.abs(sps.hilbert(x))
    env = env - env.mean()
    f, a = fft_spectrum(env, fs)
    return f, a, env


def cwt_scalogram(x, fs, wavelet="morl", n_scales=64, max_points=4000):
    step = max(1, len(x) // max_points)
    xs = x[::step]
    fs_eff = fs / step
    fc = pywt.central_frequency(wavelet)
    freqs = np.geomspace(fs_eff / 2 * 0.95, max(fs_eff / len(xs) * 4, fs_eff / 2000), n_scales)
    scales = fc * fs_eff / freqs
    coef, fr = pywt.cwt(xs, scales, wavelet, sampling_period=1 / fs_eff)
    t = np.arange(len(xs)) / fs_eff
    return t, fr, np.abs(coef)


def bearing_freqs(shaft_hz, n_balls, d_ball, d_pitch, contact_deg=0.0):
    r = d_ball / d_pitch * np.cos(np.radians(contact_deg))
    return {"FTF": shaft_hz / 2 * (1 - r), "BPFO": n_balls * shaft_hz / 2 * (1 - r),
            "BPFI": n_balls * shaft_hz / 2 * (1 + r), "BSF": d_pitch / (2 * d_ball) * shaft_hz * (1 - r ** 2)}


# ============================================================================ wavelets
DWT_FAMILIES = {
    "Biorthogonal (bior)": [w for w in pywt.wavelist("bior")],
    "Reverse biorthogonal (rbio)": pywt.wavelist("rbio"),
    "Daubechies (db)": pywt.wavelist("db")[:20],
    "Symlets (sym)": pywt.wavelist("sym")[:19],
    "Coiflets (coif)": pywt.wavelist("coif"),
    "Haar": ["haar"],
    "Discrete Meyer": ["dmey"],
}


def max_level(n, wavelet):
    return pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)


def dwt_decompose(x, wavelet, level, mode="symmetric"):
    """Return list [cA_L, cD_L, ..., cD_1] and reconstructed per-level signals (same length as x)."""
    level = int(min(level, max(1, max_level(len(x), wavelet))))
    coeffs = pywt.wavedec(x, wavelet, mode=mode, level=level)
    recon = {}
    for i in range(len(coeffs)):
        keep = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        name = f"A{level}" if i == 0 else f"D{level - i + 1}"
        recon[name] = pywt.waverec(keep, wavelet, mode=mode)[: len(x)]
    return coeffs, recon, level


def level_bands(fs, level):
    """Nominal frequency band of each DWT node."""
    bands = {f"A{level}": (0, fs / 2 ** (level + 1))}
    for j in range(1, level + 1):
        bands[f"D{j}"] = (fs / 2 ** (j + 1), fs / 2 ** j)
    return bands


def wavelet_denoise(x, wavelet, level, rule="Universal (VisuShrink)", mode="soft", ext="symmetric"):
    """Donoho–Johnstone universal threshold, BayesShrink, or SURE-like minimax threshold."""
    level = int(min(level, max(1, max_level(len(x), wavelet))))
    coeffs = pywt.wavedec(x, wavelet, mode=ext, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-15
    n = len(x)
    out = [coeffs[0]]
    for d in coeffs[1:]:
        if rule.startswith("Universal"):
            thr = sigma * np.sqrt(2 * np.log(n))
        elif rule.startswith("Bayes"):
            sx = np.sqrt(max(np.var(d) - sigma ** 2, 1e-15))
            thr = sigma ** 2 / sx
        else:  # Minimax (Donoho & Johnstone 1994)
            thr = sigma * (0.3936 + 0.1829 * np.log2(n)) if n > 32 else 0.0
        out.append(pywt.threshold(d, thr, mode=mode))
    xd = pywt.waverec(out, wavelet, mode=ext)[:n]
    snr = 10 * np.log10(np.sum(xd ** 2) / (np.sum((x - xd) ** 2) + 1e-15))
    return xd, sigma, snr


# ============================================================================ features
def time_features(x, prefix=""):
    x = np.asarray(x, float)
    ax = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    mean_abs = np.mean(ax) + 1e-15
    peak = np.max(ax)
    sqrt_amp = (np.mean(np.sqrt(ax))) ** 2 + 1e-15
    p = ax ** 2 / (np.sum(x ** 2) + 1e-15)
    f = {
        "mean": np.mean(x), "std": np.std(x, ddof=1) if len(x) > 1 else 0.0, "variance": np.var(x),
        "rms": rms, "peak": peak, "peak_to_peak": np.ptp(x), "min": np.min(x), "max": np.max(x),
        "skewness": stats.skew(x), "kurtosis": stats.kurtosis(x, fisher=False),
        "crest_factor": peak / (rms + 1e-15), "shape_factor": rms / mean_abs,
        "impulse_factor": peak / mean_abs, "margin_factor": peak / sqrt_amp,
        "energy": np.sum(x ** 2), "shannon_entropy": -np.sum(p * np.log2(p + 1e-15)),
        "zero_crossing_rate": np.mean(np.abs(np.diff(np.sign(x))) > 0),
    }
    return {prefix + k: float(v) for k, v in f.items()}


def freq_features(x, fs, prefix=""):
    f, P = welch_psd(x, fs, nperseg=min(2048, len(x)))
    Pn = P / (np.sum(P) + 1e-15)
    centroid = np.sum(f * Pn)
    spread = np.sqrt(np.sum((f - centroid) ** 2 * Pn))
    cum = np.cumsum(Pn)
    out = {
        "dominant_freq_Hz": f[np.argmax(P)], "spectral_centroid_Hz": centroid, "spectral_spread_Hz": spread,
        "spectral_entropy": -np.sum(Pn * np.log2(Pn + 1e-15)) / np.log2(len(Pn)),
        "spectral_rolloff85_Hz": f[min(np.searchsorted(cum, 0.85), len(f) - 1)],
        "spectral_flatness": np.exp(np.mean(np.log(P + 1e-20))) / (np.mean(P) + 1e-20),
        "total_power": np.trapezoid(P, f) if hasattr(np, "trapezoid") else np.trapz(P, f),
    }
    return {prefix + k: float(v) for k, v in out.items()}


def wavelet_features(x, wavelet, level, prefix=""):
    coeffs, _, level = dwt_decompose(x, wavelet, level)
    names = [f"A{level}"] + [f"D{level - i}" for i in range(level)]
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    tot = energies.sum() + 1e-15
    out = {}
    for nm, c, e in zip(names, coeffs, energies):
        out[f"{nm}_energy"] = e
        out[f"{nm}_energy_pct"] = 100 * e / tot
        out[f"{nm}_std"] = np.std(c)
        out[f"{nm}_rms"] = np.sqrt(np.mean(c ** 2))
        out[f"{nm}_kurtosis"] = stats.kurtosis(c, fisher=False) if len(c) > 3 else 0.0
    p = energies / tot
    out["wavelet_energy_entropy"] = -np.sum(p * np.log2(p + 1e-15))
    return {prefix + k: float(v) for k, v in out.items()}


def all_features(x, fs, wavelet, level, groups=("time", "frequency", "wavelet")):
    f = {}
    if "time" in groups:
        f.update(time_features(x))
    if "frequency" in groups:
        f.update(freq_features(x, fs))
    if "wavelet" in groups:
        f.update(wavelet_features(x, wavelet, level))
    return f


def segment(x, win, hop):
    if win >= len(x) or win <= 0:
        return [x]
    return [x[i:i + win] for i in range(0, len(x) - win + 1, hop)]
