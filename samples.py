"""
samples.py — physics-based demo signals and demo datasets for RajDSP (all simulated, fixed random seed).

Run  `python samples.py`  to regenerate the CSV files in ./data.
Bearing geometry follows the open CWRU test rig (SKF 6205-2RS drive-end bearing, 9 balls,
ball Ø 7.94 mm, pitch Ø 39.04 mm, ~1797 rpm) so you can compare with real CWRU recordings.
"""
import os

import numpy as np
import pandas as pd

import dsp

FS_DEFAULT = 12000
CWRU = dict(n_balls=9, d_ball=7.94, d_pitch=39.04)


def _resonance_burst(fs, f_res=3200.0, zeta=0.06, dur=0.004):
    t = np.arange(int(dur * fs)) / fs
    wd = 2 * np.pi * f_res
    return np.exp(-zeta * wd * t) * np.sin(wd * t)


def bearing_signal(kind="Healthy", fs=FS_DEFAULT, dur=1.0, rpm=1797, noise=0.08, severity=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    fr = rpm / 60
    x = 0.05 * np.sin(2 * np.pi * fr * t) + 0.02 * np.sin(2 * np.pi * 2 * fr * t + 0.4)
    fq = dsp.bearing_freqs(fr, **CWRU)
    key = {"Outer race fault": "BPFO", "Inner race fault": "BPFI", "Ball fault": "BSF"}.get(kind)
    if key:
        f_fault = fq[key] * (2 if key == "BSF" else 1)
        # transmission path differs per defect location -> different excited band / damping
        f_res, zeta = {"BPFO": (3200.0, 0.05), "BPFI": (2600.0, 0.09), "BSF": (3900.0, 0.07)}[key]
        burst = _resonance_burst(fs, f_res, zeta)
        imp = np.zeros(n)
        tk = np.arange(0, dur, 1 / f_fault)
        tk = tk + rng.normal(0, 0.01 / f_fault, tk.size)  # 1 % slip jitter
        idx = (tk * fs).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        amp = np.ones(idx.size)
        if key == "BPFI":
            amp = 1 + 0.8 * np.cos(2 * np.pi * fr * idx / fs)
        elif key == "BSF":
            amp = 1 + 0.7 * np.cos(2 * np.pi * fq["FTF"] * idx / fs)
        imp[idx] = amp * severity
        x = x + 0.6 * np.convolve(imp, burst)[:n]
    x = x + noise * rng.standard_normal(n)
    return t, x


def milling_signal(stable=True, fs=FS_DEFAULT, dur=1.0, rpm=6000, teeth=2, f_nat=1850.0, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    ftp = rpm / 60 * teeth
    x = sum((0.5 / k) * np.sin(2 * np.pi * k * ftp * t + k) for k in range(1, 6))
    if not stable:
        growth = 1 - np.exp(-t / (0.25 * dur))
        fc = f_nat * 1.04
        x = x + 1.1 * growth * np.sin(2 * np.pi * fc * t) * (1 + 0.3 * np.sin(2 * np.pi * ftp * t))
    x = x + noise * rng.standard_normal(n)
    return t, x


def multitone_signal(fs=FS_DEFAULT, dur=1.0, tones=(50, 120, 900), noise=0.5, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(tones))
    return t, x + noise * rng.standard_normal(t.size)


def chirp_signal(fs=FS_DEFAULT, dur=1.0, f0=20, f1=3000, noise=0.1, seed=3):
    from scipy.signal import chirp
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    return t, chirp(t, f0, dur, f1, method="quadratic") + noise * rng.standard_normal(t.size)


SAMPLE_SIGNALS = {
    "Bearing — healthy (CWRU-like, 12 kHz)": lambda: bearing_signal("Healthy"),
    "Bearing — outer race fault": lambda: bearing_signal("Outer race fault"),
    "Bearing — inner race fault": lambda: bearing_signal("Inner race fault"),
    "Bearing — ball fault": lambda: bearing_signal("Ball fault"),
    "Milling — stable cut (6000 rpm, 2 flutes)": lambda: milling_signal(True),
    "Milling — chatter onset (6000 rpm, 2 flutes)": lambda: milling_signal(False),
    "Multi-tone + noise (50, 120, 900 Hz)": lambda: multitone_signal(),
    "Quadratic chirp 20 → 3000 Hz": lambda: chirp_signal(),
}


def bearing_feature_dataset(n_per_class=60, win=2048, wavelet="bior3.5", level=4, seed=10):
    rows = []
    classes = ["Healthy", "Outer race fault", "Inner race fault", "Ball fault"]
    for ci, c in enumerate(classes):
        for k in range(n_per_class // 5):
            rpm = [1797, 1772, 1750, 1730][k % 4]
            sev = 0.6 + 0.8 * ((k * 7) % 10) / 10
            _, x = bearing_signal(c, dur=1.0, rpm=rpm, severity=sev, noise=0.06 + 0.04 * (k % 3),
                                  seed=seed + 100 * ci + k)
            for seg in dsp.segment(x, win, win)[:5]:
                f = dsp.all_features(seg, FS_DEFAULT, wavelet, level)
                f["rpm"] = rpm
                f["condition"] = c
                rows.append(f)
    return pd.DataFrame(rows)


def roughness_dataset(n=120, seed=7):
    """Turning-type surface roughness demo: Ra = k·f²/(8r) kinematic term + speed & depth effects + noise."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(60, 300, n)          # cutting speed m/min
    f = rng.uniform(0.05, 0.30, n)       # feed mm/rev
    d = rng.uniform(0.25, 2.0, n)        # depth of cut mm
    r = 0.8                              # nose radius mm
    ra = 1000 * f ** 2 / (32 * r) * 1.6 + 0.9 * np.exp(-v / 90) + 0.12 * d
    ra = ra * (1 + 0.06 * rng.standard_normal(n))
    vib = 0.2 + 0.9 * f + 0.15 * d + 0.03 * rng.standard_normal(n)
    good = np.where(ra < 1.6, "Acceptable", "Reject")
    return pd.DataFrame({"speed_m_min": v.round(1), "feed_mm_rev": f.round(3), "depth_mm": d.round(2),
                         "vibration_rms_g": vib.round(3), "Ra_um": ra.round(3), "quality": good})


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(here, exist_ok=True)
    bearing_feature_dataset().round(6).to_csv(os.path.join(here, "demo_bearing_features.csv"), index=False)
    roughness_dataset().to_csv(os.path.join(here, "demo_roughness.csv"), index=False)
    for name, fn in [("demo_bearing_outer_race.csv", lambda: bearing_signal("Outer race fault")),
                     ("demo_milling_chatter.csv", lambda: milling_signal(False))]:
        t, x = fn()
        pd.DataFrame({"time_s": t.round(7), "accel_g": x.round(6)}).to_csv(os.path.join(here, name), index=False)
    print("written to", here)
