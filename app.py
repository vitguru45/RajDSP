"""
RajDSP  —  Signal processing, feature extraction and ML prediction for sensor data
(vibration, force, acoustic emission, current, sound …) recorded on machines and test rigs.

Developed by Dr. Gururaja S, Centre for Innovative Manufacturing Research (CIMR), VIT Vellore.
Part of the Raj3DLabs family.          Run:  streamlit run app.py
"""
import datetime as dt
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dsp
import learn as L
import samples as S

APP_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_FILE = os.path.join(APP_DIR, "rajdsp_counter.json")
LOG_FILE = os.path.join(APP_DIR, "rajdsp_usage_log.csv")
DAQ_SCRIPT = os.path.join(APP_DIR, "daq", "rajdsp_daq_recorder.py")
VERSION = "1.0"
PLOT_MAX = 20000  # points drawn per trace (display only — analysis always uses the full signal)

st.set_page_config(page_title="RajDSP | Signal Processing & ML", page_icon="📡", layout="wide")

# ----------------------------------------------------------------------------- style (Raj3DLabs house style)
st.markdown("""
<style>
.raj-header{background:linear-gradient(90deg,#0b3d91 0%,#1565c0 55%,#00897b 100%);
  padding:18px 24px;border-radius:14px;color:#fff;margin-bottom:10px}
.raj-header h1{margin:0;font-size:2.0rem;color:#fff}
.raj-header p{margin:4px 0 0 0;opacity:.92}
.raj-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:.85rem;color:#fff}
.raj-note{border-left:4px solid #1565c0;background:rgba(21,101,192,.07);padding:10px 14px;border-radius:6px;margin:6px 0 10px 0}
.raj-foot{text-align:center;opacity:.8;font-size:.88rem;margin-top:36px;padding-top:16px;
  border-top:1px solid rgba(128,128,128,.25);line-height:1.7}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- counter + log
def _read_counter():
    try:
        with open(COUNTER_FILE) as f:
            return json.load(f)
    except Exception:
        return {"visits": 0, "analyses": 0, "since": dt.date.today().isoformat()}


def _write_counter(c):
    try:
        tmp = COUNTER_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(c, f)
        os.replace(tmp, COUNTER_FILE)
    except Exception:
        pass


def bump(key):
    c = _read_counter()
    c[key] = c.get(key, 0) + 1
    _write_counter(c)
    return c


def log_use(kind, detail=""):
    bump("analyses")
    try:
        row = {"time": dt.datetime.now().isoformat(timespec="seconds"),
               "user": st.session_state.get("user_name", ""), "module": kind, "detail": detail}
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    except Exception:
        pass


if "visited" not in st.session_state:
    st.session_state.visited = True
    bump("visits")
counter = _read_counter()


# ----------------------------------------------------------------------------- helpers
def thin(*arrays, n=PLOT_MAX):
    step = max(1, len(arrays[0]) // n)
    return [a[::step] for a in arrays]


def fig_layout(fig, h=360, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), legend=dict(orientation="h", y=1.08), **kw)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def csv_bytes(df):
    return df.to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def read_upload(name, raw):
    return dsp.load_table(name, raw)


@st.cache_data(show_spinner=False)
def read_any(name, raw):
    return dsp.load_any_table(name, raw)


@st.cache_data(show_spinner=False)
def demo_csv(name):
    return pd.read_csv(os.path.join(APP_DIR, "data", name))


@st.cache_data(show_spinner=False)
def sample_signal(name):
    return S.SAMPLE_SIGNALS[name]()


def dataset_picker(key, demos):
    """Upload a table or use a demo dataset. Returns (df, label)."""
    opts = ["Upload my file"] + list(demos)
    if key == "ml" and "batch_features" in st.session_state:
        opts.insert(1, "Features from the Batch feature extraction tab")
    src = st.radio("Dataset", opts, horizontal=True, key=f"{key}_src")
    if src == "Upload my file":
        up = st.file_uploader("Choose dataset file (.xlsx or .csv)", type=["csv", "xlsx", "xls", "txt"], key=f"{key}_up")
        if up is None:
            return None, None
        try:
            return read_any(up.name, up.getvalue()), up.name
        except Exception as e:
            st.error(f"Could not read {up.name}: {e}")
            return None, None
    if src.startswith("Features from"):
        return st.session_state["batch_features"], "batch features"
    return demo_csv(demos[src]), src


# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="raj-header">
<h1>📡 RajDSP — Signal Processing, Feature Extraction & ML Prediction</h1>
<p>FFT · STFT · envelope · wavelet decomposition & denoising &nbsp;|&nbsp; time, frequency & wavelet features
&nbsp;|&nbsp; SVM / ANN / regression &nbsp;|&nbsp; NI-DAQ recording — built entirely on open-source Python</p>
</div>""", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
with h1:
    st.text_input("Your name (optional — appears in the usage log)", key="user_name", placeholder="e.g. Priya, 23BME1234")
h2.metric("👥 Visits", f"{counter.get('visits', 0):,}")
h3.metric("🧪 Analyses run", f"{counter.get('analyses', 0):,}")
h4.metric("〰️ Wavelets available", sum(len(v) for v in dsp.DWT_FAMILIES.values()))

# ----------------------------------------------------------------------------- sidebar: signal for tab 1
sb = st.sidebar
sb.header("① Signal source")
sources = ["Sample signal", "Upload file(s)"]
if "daq_df" in st.session_state:
    sources.append("Last NI-DAQ recording")
src = sb.radio("Where is your signal?", sources, key="sig_src")
raw_df, src_label, fs_guess = None, "", None
if src == "Sample signal":
    sname = sb.selectbox("Sample signal (simulated)", list(S.SAMPLE_SIGNALS), key="sample_name")
    t_, x_ = sample_signal(sname)
    raw_df = pd.DataFrame({"time_s": t_, "signal": x_})
    src_label, fs_guess = sname, float(S.FS_DEFAULT)
elif src == "Upload file(s)":
    ups = sb.file_uploader("Supported: .txt .csv .xlsx .lvm .mat", type=dsp.SUPPORTED, accept_multiple_files=True,
                           key="sig_up")
    if ups:
        pick = sb.selectbox("File to analyse", [u.name for u in ups]) if len(ups) > 1 else ups[0].name
        u = next(u for u in ups if u.name == pick)
        try:
            raw_df = read_upload(u.name, u.getvalue())
            src_label = u.name
            fs_guess, _ = dsp.guess_fs(raw_df)
        except Exception as e:
            sb.error(f"Could not read {u.name}: {e}")
else:
    raw_df = st.session_state["daq_df"]
    src_label = "NI-DAQ recording"
    fs_guess, _ = dsp.guess_fs(raw_df)

sb.header("② Channel & sampling")
if raw_df is not None:
    tcol_guess = dsp.guess_fs(raw_df)[1]
    chans = [c for c in raw_df.columns if c != tcol_guess] or list(raw_df.columns)
    chan = sb.selectbox("Signal column", chans, key="chan")
    fs = sb.number_input("Sampling rate fs (Hz)", min_value=1.0, value=float(round(fs_guess, 3)) if fs_guess else 12000.0,
                         step=100.0, help="Detected automatically when the first column is a uniform time axis.")
    if fs_guess:
        sb.caption(f"Detected from the time column: {fs_guess:,.1f} Hz")
    x_full = raw_df[chan].to_numpy(float)
    x_full = x_full[np.isfinite(x_full)]
    dur = len(x_full) / fs
    seg = sb.slider("Analyse window (s)", 0.0, float(dur), (0.0, float(dur)), step=float(max(dur / 500, 1 / fs)))
else:
    chan, fs, x_full, seg = None, 12000.0, None, (0, 0)

sb.header("③ Pre-processing")
detrend = sb.selectbox("Detrend", ["constant", "linear", "none"], help="constant = remove mean (DC offset)")
filt = sb.selectbox("Butterworth filter", ["None", "Low-pass", "High-pass", "Band-pass"])
f_lo = f_hi = None
if filt in ("High-pass", "Band-pass"):
    f_lo = sb.number_input("Low cut-off (Hz)", 0.1, fs / 2 - 1, min(500.0, fs / 4))
if filt in ("Low-pass", "Band-pass"):
    f_hi = sb.number_input("High cut-off (Hz)", 0.2, fs / 2 - 0.1, min(4000.0, fs / 2 * 0.8))

sb.header("④ Wavelet settings")
fam = sb.selectbox("Wavelet family", list(dsp.DWT_FAMILIES), index=0)
wl_list = dsp.DWT_FAMILIES[fam]
wavelet = sb.selectbox("Wavelet", wl_list, index=wl_list.index("bior3.5") if "bior3.5" in wl_list else 0)
level = sb.slider("Decomposition level", 1, 20, 4, help="Capped automatically at the maximum useful level for the signal length.")

sb.header("⑤ Denoising")
den_rule = sb.selectbox("Threshold rule", ["Universal (VisuShrink)", "BayesShrink", "Minimax"])
den_mode = sb.radio("Thresholding", ["soft", "hard"], horizontal=True)

# ----------------------------------------------------------------------------- tabs
tabs = st.tabs(["📈 Signal analysis", "🧩 Batch feature extraction", "🤖 Machine learning", "📉 Regression models",
                "🔍 Data analysis", "🎛️ Data acquisition (NI-DAQ)", "📚 Data, models & references"])

# =============================================================================== TAB 1: signal analysis
with tabs[0]:
    if x_full is None:
        st.info("⬅️ Choose a sample signal or upload a file in the sidebar (① Signal source). "
                "Formats: .txt, .csv, .xlsx, .lvm (LabVIEW) and .mat (e.g. CWRU bearing files).")
    else:
        i0, i1 = int(seg[0] * fs), max(int(seg[1] * fs), int(seg[0] * fs) + 16)
        x_raw = x_full[i0:i1]
        x = dsp.preprocess(x_raw, fs, detrend, filt, f_lo, f_hi)
        t = np.arange(len(x)) / fs + seg[0]
        lvl_max = dsp.max_level(len(x), wavelet)
        lvl = min(level, max(1, lvl_max))
        sig_key = hashlib.md5(f"{src_label}{chan}{fs}{len(x)}".encode()).hexdigest()
        if st.session_state.get("last_sig") != sig_key:
            st.session_state.last_sig = sig_key
            log_use("signal", f"{src_label} | {chan}")

        xd, sigma, snr = dsp.wavelet_denoise(x, wavelet, lvl, den_rule, den_mode)
        feats = dsp.all_features(x, fs, wavelet, lvl)
        c = st.columns(6)
        c[0].metric("Samples", f"{len(x):,}")
        c[1].metric("Duration", f"{len(x) / fs:.3f} s")
        c[2].metric("RMS", f"{feats['rms']:.4g}")
        c[3].metric("Kurtosis", f"{feats['kurtosis']:.2f}", "impulsive" if feats["kurtosis"] > 4 else "Gaussian-like",
                    delta_color="off")
        c[4].metric("Dominant freq.", f"{feats['dominant_freq_Hz']:.1f} Hz")
        c[5].metric("Wavelet level used", f"{lvl}", f"max {lvl_max} for {wavelet}", delta_color="off")
        st.caption(f"Signal: **{src_label}** · column **{chan}** · fs = {fs:,.1f} Hz · Nyquist = {fs / 2:,.1f} Hz")

        st_tabs = st.tabs(["Time domain", "FFT & PSD", "STFT spectrogram", "Wavelet decomposition", "Denoising",
                           "CWT scalogram", "Envelope (bearings)", "Features"])

        with st_tabs[0]:
            tt, xr, xx = thin(t, x_raw - (np.mean(x_raw) if detrend != "none" else 0), x)
            fig = go.Figure()
            if filt != "None" or detrend == "linear":
                fig.add_scatter(x=tt, y=xr, name="raw", line=dict(color="#9e9e9e", width=1))
            fig.add_scatter(x=tt, y=xx, name="pre-processed", line=dict(color="#1565c0", width=1))
            show(fig_layout(fig, 380, xaxis_title="Time (s)", yaxis_title=chan))
            a, b = st.columns(2)
            with a:
                hf = go.Figure(go.Histogram(x=x, nbinsx=80, marker_color="#00897b"))
                show(fig_layout(hf, 300, title="Amplitude distribution", xaxis_title=chan, yaxis_title="count"))
            with b:
                st.markdown("**Quick reading**")
                k = feats["kurtosis"]
                st.markdown(f"- Kurtosis **{k:.2f}** — " + ("strong impulses (bearing/gear defects, tool breakage, impacts)."
                            if k > 4 else "close to 3, i.e. Gaussian-like random/steady vibration."))
                st.markdown(f"- Crest factor **{feats['crest_factor']:.2f}** — " +
                            ("peaky signal." if feats["crest_factor"] > 4 else "no dominant peaks."))
                st.markdown(f"- Spectral centroid **{feats['spectral_centroid_Hz']:.0f} Hz**, "
                            f"85 % roll-off **{feats['spectral_rolloff85_Hz']:.0f} Hz**.")
                st.markdown(f"- Estimated noise σ (from finest wavelet detail, MAD/0.6745): **{sigma:.4g}**.")

        with st_tabs[1]:
            a, b, c3 = st.columns(3)
            win = a.selectbox("FFT window", ["hann", "hamming", "blackman", "flattop", "none"])
            yscale = b.radio("Amplitude scale", ["linear", "dB"], horizontal=True)
            fmax = c3.number_input("Show up to (Hz)", 1.0, fs / 2, fs / 2)
            f, A = dsp.fft_spectrum(x, fs, win)
            fA, AA = f[f <= fmax], A[f <= fmax]
            fA, AA = thin(fA, AA)
            Ay = 20 * np.log10(AA + 1e-12) if yscale == "dB" else AA
            fig = go.Figure(go.Scatter(x=fA, y=Ay, line=dict(color="#1565c0", width=1), name="FFT"))
            pk = np.argsort(A)[-5:][::-1]
            fig.add_scatter(x=f[pk], y=(20 * np.log10(A[pk] + 1e-12) if yscale == "dB" else A[pk]), mode="markers+text",
                            text=[f"{v:.1f} Hz" for v in f[pk]], textposition="top center", marker=dict(color="#c0392b", size=8),
                            name="top-5 peaks")
            show(fig_layout(fig, 380, title="Single-sided amplitude spectrum (FFT)", xaxis_title="Frequency (Hz)",
                            yaxis_title="Amplitude" + (" (dB)" if yscale == "dB" else "")))
            nper = st.select_slider("Welch segment length", [128, 256, 512, 1024, 2048, 4096, 8192], 1024)
            fw, P = dsp.welch_psd(x, fs, nper)
            fig = go.Figure(go.Scatter(x=fw[fw <= fmax], y=10 * np.log10(P[fw <= fmax] + 1e-20), line=dict(color="#00897b")))
            show(fig_layout(fig, 320, title="Power spectral density (Welch)", xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB/Hz)"))
            st.download_button("⬇️ Download FFT spectrum (CSV)", csv_bytes(pd.DataFrame({"freq_Hz": f, "amplitude": A})),
                               "rajdsp_fft.csv", "text/csv")

        with st_tabs[2]:
            a, b, c3 = st.columns(3)
            nps = a.select_slider("Window length (samples)", [64, 128, 256, 512, 1024, 2048], 256)
            ov = b.slider("Overlap", 0.0, 0.95, 0.75, 0.05)
            fmx = c3.number_input("Max frequency (Hz)", 1.0, fs / 2, fs / 2, key="stft_fmax")
            fS, tS, Z = dsp.stft_map(x, fs, nps, ov)
            keep = fS <= fmx
            stp = max(1, len(tS) // 800)
            fig = go.Figure(go.Heatmap(x=tS[::stp] + seg[0], y=fS[keep], z=Z[keep][:, ::stp], colorscale="Turbo",
                                       colorbar=dict(title="dB")))
            show(fig_layout(fig, 430, title="STFT spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)"))
            st.caption(f"Time resolution ≈ {nps / fs * 1e3:.2f} ms · frequency resolution ≈ {fs / nps:.1f} Hz. "
                       "Short windows resolve transients, long windows resolve close frequencies (uncertainty principle).")

        with st_tabs[3]:
            coeffs, recon, lv = dsp.dwt_decompose(x, wavelet, lvl)
            bands = dsp.level_bands(fs, lv)
            names = [f"A{lv}"] + [f"D{j}" for j in range(lv, 0, -1)]
            en = np.array([np.sum(cc ** 2) for cc in coeffs])
            show_n = st.slider("Levels to plot", 1, len(names), min(len(names), 6))
            plot_names = names[:show_n]
            fig = make_subplots(rows=len(plot_names) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                subplot_titles=["Signal"] + [f"{n}  ({bands[n][0]:.0f}–{bands[n][1]:.0f} Hz)" for n in plot_names])
            tt, xx = thin(t, x)
            fig.add_scatter(x=tt, y=xx, row=1, col=1, line=dict(color="#333", width=1), showlegend=False)
            for i, n in enumerate(plot_names):
                tt, rr = thin(t, recon[n])
                fig.add_scatter(x=tt, y=rr, row=i + 2, col=1, line=dict(width=1, color="#00897b" if n.startswith("A") else "#1565c0"),
                                showlegend=False)
            fig.update_annotations(font_size=11)
            show(fig_layout(fig, 140 * (len(plot_names) + 1), xaxis_title=None))
            st.caption(f"Multilevel DWT with **{wavelet}**, level {lv} (requested {level}, maximum useful {lvl_max}). "
                       "Each trace is the reconstruction from a single node; A + all D sum back to the signal.")
            a, b = st.columns([1, 1])
            with a:
                fig = go.Figure(go.Bar(x=names, y=100 * en / en.sum(), marker_color=["#00897b"] + ["#1565c0"] * lv))
                show(fig_layout(fig, 320, title="Energy per node (%)", yaxis_title="% of total energy"))
            with b:
                tab = pd.DataFrame({"node": names, "band_Hz": [f"{bands[n][0]:.1f} – {bands[n][1]:.1f}" for n in names],
                                    "n_coeffs": [len(cc) for cc in coeffs], "energy_%": (100 * en / en.sum()).round(2),
                                    "rms": [np.sqrt(np.mean(cc ** 2)) for cc in coeffs]})
                st.dataframe(tab, width="stretch", hide_index=True)
            # FFT of coefficients
            pick = st.selectbox("FFT of a wavelet node (reconstructed)", names, index=min(1, len(names) - 1))
            fN, AN = dsp.fft_spectrum(recon[pick], fs)
            fN, AN = thin(fN, AN)
            show(fig_layout(go.Figure(go.Scatter(x=fN, y=AN, line=dict(color="#8e44ad", width=1))), 280,
                            title=f"FFT of {pick}", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude"))
            mx = max(len(cc) for cc in coeffs)
            cdf = pd.DataFrame({n: np.pad(cc, (0, mx - len(cc)), constant_values=np.nan) for n, cc in zip(names, coeffs)})
            st.download_button("⬇️ Download wavelet coefficients (CSV)", csv_bytes(cdf), "rajdsp_wavelet_coeffs.csv", "text/csv")

        with st_tabs[4]:
            a, b, c3 = st.columns(3)
            a.metric("Noise σ estimate", f"{sigma:.4g}")
            b.metric("Kept / removed energy", f"{snr:.1f} dB")
            c3.metric("RMS removed", f"{np.sqrt(np.mean((x - xd) ** 2)):.4g}")
            tt, xx, dd = thin(t, x, xd)
            fig = go.Figure()
            fig.add_scatter(x=tt, y=xx, name="input", line=dict(color="#bdbdbd", width=1))
            fig.add_scatter(x=tt, y=dd, name="denoised", line=dict(color="#c0392b", width=1.4))
            show(fig_layout(fig, 380, title=f"Wavelet denoising — {den_rule}, {den_mode} threshold, {wavelet}, level {lvl}",
                            xaxis_title="Time (s)", yaxis_title=chan))
            f1, A1 = dsp.fft_spectrum(x, fs)
            f2, A2 = dsp.fft_spectrum(xd, fs)
            f1, A1, A2 = thin(f1, A1, A2)
            fig = go.Figure()
            fig.add_scatter(x=f1, y=20 * np.log10(A1 + 1e-12), name="input", line=dict(color="#bdbdbd", width=1))
            fig.add_scatter(x=f1, y=20 * np.log10(A2 + 1e-12), name="denoised", line=dict(color="#c0392b", width=1))
            show(fig_layout(fig, 300, title="Spectrum before / after", xaxis_title="Frequency (Hz)", yaxis_title="dB"))
            st.download_button("⬇️ Download denoised signal (CSV)", csv_bytes(pd.DataFrame({"time_s": t, "raw": x, "denoised": xd})),
                               "rajdsp_denoised.csv", "text/csv")

        with st_tabs[5]:
            a, b = st.columns(2)
            cw = a.selectbox("Continuous wavelet", ["morl", "mexh", "cmor1.5-1.0", "gaus4", "shan1.5-1.0"])
            ns = b.slider("Number of scales", 24, 128, 64, 8)
            with st.spinner("Computing CWT…"):
                tc, fc, Wc = dsp.cwt_scalogram(x, fs, cw, ns)
            fig = go.Figure(go.Heatmap(x=tc + seg[0], y=fc, z=Wc, colorscale="Viridis", colorbar=dict(title="|W|")))
            fig.update_yaxes(type="log")
            show(fig_layout(fig, 430, title=f"CWT scalogram ({cw})", xaxis_title="Time (s)", yaxis_title="Frequency (Hz, log)"))
            st.caption("Long signals are down-sampled for the scalogram only (max 4000 points) to keep the app responsive.")

        with st_tabs[6]:
            st.markdown("Envelope analysis (Hilbert transform) reveals the repetition rate of impacts — the standard way to "
                        "find bearing defect frequencies. Band-pass the signal around the excited resonance first (sidebar ③).")
            a, b, c3, d, e = st.columns(5)
            rpm = a.number_input("Shaft speed (rpm)", 1.0, 100000.0, 1797.0)
            nb = b.number_input("Number of balls", 3, 40, 9)
            db = c3.number_input("Ball Ø (mm)", 0.5, 100.0, 7.94)
            dp = d.number_input("Pitch Ø (mm)", 2.0, 500.0, 39.04)
            ca = e.number_input("Contact angle (°)", 0.0, 45.0, 0.0)
            fq = dsp.bearing_freqs(rpm / 60, nb, db, dp, ca)
            st.markdown(" · ".join(f"**{k}** = {v:.2f} Hz" for k, v in fq.items()) + f" · **1×** = {rpm / 60:.2f} Hz")
            fE, AE, env = dsp.envelope_spectrum(x, fs)
            lim = st.slider("Envelope spectrum range (Hz)", 50, int(fs / 2), int(min(500, fs / 2)))
            m = fE <= lim
            fig = go.Figure(go.Scatter(x=fE[m], y=AE[m], line=dict(color="#1565c0", width=1), name="envelope spectrum"))
            colors = {"FTF": "#7f8c8d", "BPFO": "#c0392b", "BPFI": "#8e44ad", "BSF": "#e07b00"}
            for k, v in fq.items():
                for h in (1, 2, 3):
                    if v * h <= lim:
                        kw = dict(annotation_text=k, annotation_font_color=colors[k]) if h == 1 else {}
                        fig.add_vline(x=v * h, line=dict(color=colors[k], dash="dot", width=1), **kw)
            show(fig_layout(fig, 400, title="Envelope spectrum with bearing defect frequencies", xaxis_title="Frequency (Hz)",
                            yaxis_title="Amplitude"))
            tol = max(1.0, 0.02 * rpm / 60)
            floor = np.median(AE[m]) + 1e-15

            def line_amp(fr_):
                sel = np.abs(fE - fr_) < tol
                return float(AE[sel].max()) if sel.any() else 0.0

            # ball defects strike both races -> look at 1x and 2x BSF; cage line (FTF) is often only a modulation sideband
            score = {"BPFO": line_amp(fq["BPFO"]), "BPFI": line_amp(fq["BPFI"]),
                     "BSF": max(line_amp(fq["BSF"]), line_amp(2 * fq["BSF"])), "FTF": line_amp(fq["FTF"])}
            ratio = {k: v / floor for k, v in score.items()}
            main = {k: ratio[k] for k in ("BPFO", "BPFI", "BSF")}
            best = max(main, key=main.get) if max(main.values()) > 8 else "FTF"
            where = {"BPFO": "outer race", "BPFI": "inner race", "BSF": "rolling element (ball)", "FTF": "cage"}[best]
            if ratio[best] > 8:
                st.success(f"Strongest defect line: **{best}** ({fq[best]:.1f} Hz{' or 2×' if best == 'BSF' else ''}), "
                           f"{ratio[best]:.0f}× above the median envelope level → points to the **{where}**.")
            else:
                st.info("No defect frequency stands out clearly from the envelope floor (best ratio "
                        f"{max(ratio.values()):.1f}×) — consistent with a healthy bearing or an unsuitable band-pass.")
            st.caption("Rule of thumb used here: a defect line > 8× the median envelope level is flagged. "
                       "Always confirm with harmonics and sidebands (1× shaft for BPFI, FTF for BSF).")

        with st_tabs[7]:
            groups = st.multiselect("Feature groups", ["time", "frequency", "wavelet"], ["time", "frequency", "wavelet"])
            use_den = st.checkbox("Extract from the denoised signal", False)
            ff = dsp.all_features(xd if use_den else x, fs, wavelet, lvl, groups)
            fdf = pd.DataFrame({"feature": list(ff), "value": list(ff.values())})
            st.dataframe(fdf, width="stretch", hide_index=True, height=420)
            st.download_button("⬇️ Download features (CSV, one row)", csv_bytes(pd.DataFrame([ff])), "rajdsp_features.csv", "text/csv")
            st.caption("For a training table with many rows, use the **Batch feature extraction** tab (many files, windowed).")

# =============================================================================== TAB 2: batch feature extraction
with tabs[1]:
    st.markdown("Turn many recordings into a **feature table** (one row per window) ready for classification or regression. "
                "Wavelet settings come from the sidebar ④.")
    mode = st.radio("Recordings", ["Upload my files (one class/condition per file)", "Demo: simulated bearing set (4 classes × 3 files)"],
                    horizontal=True)
    files = []  # list of (name, label, x, fs)
    if mode.startswith("Upload"):
        bups = st.file_uploader("Select files (.txt .csv .xlsx .lvm .mat) — multiple files supported", type=dsp.SUPPORTED,
                                accept_multiple_files=True, key="batch_up")
        if bups:
            a, b = st.columns(2)
            bfs = a.number_input("Sampling rate fs (Hz) if no time column", 1.0, 1e7, 12000.0, key="bfs")
            col_rule = b.text_input("Signal column (name or 1-based index; blank = first non-time column)", "")
            meta = []
            for u in bups:
                try:
                    d = read_upload(u.name, u.getvalue())
                    g, tcol = dsp.guess_fs(d)
                    cols = [c for c in d.columns if c != tcol] or list(d.columns)
                    if col_rule.strip():
                        cname = cols[int(col_rule) - 1] if col_rule.strip().isdigit() else col_rule.strip()
                    else:
                        cname = cols[0]
                    xx = d[cname].to_numpy(float)
                    meta.append({"file": u.name, "label": u.name.rsplit(".", 1)[0].split("_")[0], "column": cname,
                                 "fs_Hz": g or bfs, "samples": int(np.isfinite(xx).sum())})
                    files.append([u.name, None, xx[np.isfinite(xx)], g or bfs])
                except Exception as e:
                    st.warning(f"{u.name}: {e}")
            st.markdown("**Edit the label (class or target value) of each file:**")
            ed = st.data_editor(pd.DataFrame(meta), hide_index=True, width="stretch",
                                disabled=["file", "column", "fs_Hz", "samples"], key="batch_editor")
            for f_, lab in zip(files, ed["label"]):
                f_[1] = lab
    else:
        for ci, c in enumerate(["Healthy", "Outer race fault", "Inner race fault", "Ball fault"]):
            for k in range(3):
                _, xx = S.bearing_signal(c, rpm=[1797, 1772, 1750][k], severity=0.7 + 0.3 * k, seed=500 + 10 * ci + k)
                files.append([f"{c.split()[0]}_{k + 1}.csv", c, xx, float(S.FS_DEFAULT)])
        st.dataframe(pd.DataFrame([{"file": f[0], "label": f[1], "fs_Hz": f[3], "samples": len(f[2])} for f in files]),
                     hide_index=True, width="stretch", height=240)

    a, b, c3, d = st.columns(4)
    win = a.select_slider("Window length (samples)", [256, 512, 1024, 2048, 4096, 8192, 16384], 2048)
    ovl = b.slider("Window overlap", 0.0, 0.9, 0.5, 0.1)
    groups = c3.multiselect("Feature groups", ["time", "frequency", "wavelet"], ["time", "frequency", "wavelet"], key="bgroups")
    den_b = d.checkbox("Denoise each window first", False)
    if files and st.button("🧩 Extract features", type="primary"):
        rows = []
        prog = st.progress(0.0)
        for i, (nm, lab, xx, ffs) in enumerate(files):
            xx = dsp.preprocess(xx, ffs, detrend, filt, f_lo, f_hi)
            for j, w in enumerate(dsp.segment(xx, int(win), max(1, int(win * (1 - ovl))))):
                lv = min(level, max(1, dsp.max_level(len(w), wavelet)))
                if den_b:
                    w = dsp.wavelet_denoise(w, wavelet, lv, den_rule, den_mode)[0]
                r = {"file": nm, "window": j + 1}
                r.update(dsp.all_features(w, ffs, wavelet, lv, groups))
                r["label"] = lab
                rows.append(r)
            prog.progress((i + 1) / len(files))
        st.session_state.batch_features = pd.DataFrame(rows)
        log_use("batch features", f"{len(files)} files, {len(rows)} windows")
    if "batch_features" in st.session_state:
        bf = st.session_state.batch_features
        st.success(f"{len(bf)} rows × {bf.shape[1] - 3} features. Open the **🤖 Machine learning** tab and choose "
                   "*Features from the Batch feature extraction tab* to train a model on it.")
        st.dataframe(bf, width="stretch", height=300)
        num = [c for c in bf.columns if c not in ("file", "window", "label")]
        a, b = st.columns(2)
        fx = a.selectbox("Compare a feature across classes", num, index=num.index("kurtosis") if "kurtosis" in num else 0)
        show(fig_layout(px.box(bf, x="label", y=fx, color="label", points="all"), 360, showlegend=False))
        st.download_button("⬇️ Download feature table (CSV)", csv_bytes(bf), "rajdsp_feature_table.csv", "text/csv")

# =============================================================================== TAB 3: machine learning
with tabs[2]:
    st.markdown('<div class="raj-note"><b>Data requirements:</b> feature columns must be numeric. Target/class labels can be '
                'text or numbers. Convert text features to numbers before uploading.</div>', unsafe_allow_html=True)
    task = st.selectbox("Select model type", ["SVM Classification", "ANN Regression"], key="ml_task")
    demos = {"Demo: bearing fault features (240 rows, 4 classes)": "demo_bearing_features.csv",
             "Demo: turning surface roughness (120 rows)": "demo_roughness.csv"}
    df, dlabel = dataset_picker("ml", demos)
    if df is not None:
        st.caption(f"**{dlabel}** — {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(50), width="stretch")
        cols = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        a, b = st.columns([1, 2])
        if task == "SVM Classification":
            tdef = next((c for c in ["label", "condition", "class", "quality", "target"] if c in cols), cols[-1])
            target = a.selectbox("Target / class column", cols, index=cols.index(tdef))
            model_name = a.selectbox("Classifier", list(L.CLASSIFIERS), index=0)
        else:
            tdef = next((c for c in reversed(num_cols) if df[c].nunique() > 10), num_cols[-1] if num_cols else cols[-1])
            target = a.selectbox("Target column (numeric)", num_cols, index=num_cols.index(tdef) if tdef in num_cols else 0)
            model_name = a.selectbox("Regressor", list(L.REGRESSORS), index=0)
        feat_opts = [c for c in num_cols if c != target and c not in ("window",)]
        feats_sel = b.multiselect("Feature columns", feat_opts, feat_opts, key=f"ml_feats_{dlabel}_{target}")
        a2, b2, c2 = st.columns(3)
        test_size = a2.slider("Test split", 0.1, 0.5, 0.25, 0.05)
        tune = b2.checkbox("Automatic hyper-parameter optimisation (grid search + CV)", True)
        seed = c2.number_input("Random seed", 0, 9999, 42)
        if feats_sel and st.button("🚀 Train model", type="primary"):
            dd = df[feats_sel + [target]].dropna()
            with st.spinner("Training…"):
                try:
                    if task == "SVM Classification":
                        res = L.train_classifier(dd[feats_sel].to_numpy(float), dd[target].astype(str).to_numpy(), model_name,
                                                 test_size, tune, int(seed))
                    else:
                        res = L.train_regressor(dd[feats_sel].to_numpy(float), dd[target].to_numpy(float), model_name,
                                                test_size, tune, int(seed))
                    st.session_state.ml_res = dict(res=res, task=task, feats=feats_sel, target=target, model_name=model_name,
                                                   X=dd[feats_sel].to_numpy(float), y=dd[target].to_numpy())
                    log_use("ml", f"{task} | {model_name} | {dlabel}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        R = st.session_state.get("ml_res")
        if R and R["task"] == task and R["target"] == target:
            res = R["res"]
            st.subheader(f"Results — {R['model_name']}")
            if task == "SVM Classification":
                c = st.columns(4)
                c[0].metric("Test accuracy", f"{100 * res['acc']:.1f} %")
                c[1].metric("Macro F1", f"{res['f1']:.3f}")
                c[2].metric("CV accuracy (train)", f"{100 * res['cv'].mean():.1f} %", f"± {100 * res['cv'].std():.1f}",
                            delta_color="off")
                c[3].metric("Train / test", f"{res['n_train']} / {res['n_test']}")
                if res["best"]:
                    st.markdown("**Best hyper-parameters:** " + ", ".join(f"`{k[3:]}` = {v}" for k, v in res["best"].items()))
                a, b = st.columns(2)
                cls = [str(c) for c in res["encoder"].classes_]
                with a:
                    fig = px.imshow(res["cm"], x=cls, y=cls, text_auto=True, color_continuous_scale="Blues",
                                    labels=dict(x="Predicted", y="True", color="count"))
                    show(fig_layout(fig, 400, title="Confusion matrix (test set)"))
                with b:
                    Z, ev = L.pca_2d(R["X"])
                    fig = px.scatter(x=Z[:, 0], y=Z[:, 1], color=R["y"].astype(str),
                                     labels=dict(x=f"PC1 ({100 * ev[0]:.0f} %)", y=f"PC2 ({100 * ev[-1]:.0f} %)", color="class"))
                    show(fig_layout(fig, 400, title="Feature space (PCA projection)"))
                st.dataframe(res["report"].round(3), width="stretch")
            else:
                c = st.columns(4)
                c[0].metric("R² (test)", f"{res['r2']:.3f}")
                c[1].metric("RMSE (test)", f"{res['rmse']:.4g}")
                c[2].metric("MAE (test)", f"{res['mae']:.4g}")
                c[3].metric("CV R² (train)", f"{res['cv'].mean():.3f}", f"± {res['cv'].std():.3f}", delta_color="off")
                if res["best"]:
                    st.markdown("**Best hyper-parameters:** " + ", ".join(f"`{k[3:]}` = {v}" for k, v in res["best"].items()))
                a, b = st.columns(2)
                with a:
                    lo = float(min(res["ytr"].min(), res["yte"].min()))
                    hi = float(max(res["ytr"].max(), res["yte"].max()))
                    fig = go.Figure()
                    fig.add_scatter(x=res["ytr"], y=res["yp_tr"], mode="markers", name="train", marker=dict(color="#90caf9"))
                    fig.add_scatter(x=res["yte"], y=res["yp_te"], mode="markers", name="test", marker=dict(color="#c0392b"))
                    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line=dict(color="black", dash="dash"))
                    show(fig_layout(fig, 400, title="Predicted vs actual", xaxis_title=f"actual {target}",
                                    yaxis_title=f"predicted {target}"))
                with b:
                    fig = go.Figure(go.Histogram(x=res["yte"] - res["yp_te"], nbinsx=25, marker_color="#00897b"))
                    show(fig_layout(fig, 400, title="Residuals (test)", xaxis_title="actual − predicted"))
            # ---- prediction
            st.subheader("🔮 Predict")
            pmode = st.radio("Input", ["Type values", "Upload new data (same feature columns)"], horizontal=True)
            Xall = R["X"]
            if pmode == "Type values":
                if len(R["feats"]) > 16:
                    st.info(f"{len(R['feats'])} features — typing them is impractical; upload a file instead.")
                else:
                    cc = st.columns(4)
                    vals = [cc[i % 4].number_input(fn, value=float(np.median(Xall[:, i])), format="%.5g", key=f"pv_{fn}")
                            for i, fn in enumerate(R["feats"])]
                    p = res["model"].predict(np.array([vals]))
                    if task == "SVM Classification":
                        lab = res["encoder"].inverse_transform(p.astype(int))[0]
                        pr = res["model"].predict_proba(np.array([vals]))[0] if hasattr(res["model"], "predict_proba") else None
                        st.markdown(f"Predicted class: <span class='raj-badge' style='background:#0b3d91'>{lab}</span>",
                                    unsafe_allow_html=True)
                        if pr is not None:
                            show(fig_layout(go.Figure(go.Bar(x=list(res["encoder"].classes_), y=pr, marker_color="#1565c0")),
                                            260, yaxis_title="probability"))
                    else:
                        st.metric(f"Predicted {R['target']}", f"{p[0]:.4g}")
            else:
                nu = st.file_uploader("New data", type=["csv", "xlsx", "xls", "txt"], key="pred_up")
                if nu:
                    nd = read_any(nu.name, nu.getvalue())
                    miss = [f for f in R["feats"] if f not in nd.columns]
                    if miss:
                        st.error(f"Missing columns: {miss[:8]}")
                    else:
                        p = res["model"].predict(nd[R["feats"]].to_numpy(float))
                        nd = nd.copy()
                        nd[f"predicted_{R['target']}"] = (res["encoder"].inverse_transform(p.astype(int))
                                                          if task == "SVM Classification" else p)
                        st.dataframe(nd, width="stretch", height=260)
                        st.download_button("⬇️ Download predictions", csv_bytes(nd), "rajdsp_predictions.csv", "text/csv")
            st.download_button("⬇️ Download trained model (.pkl, scikit-learn pipeline)",
                               L.pickle_bytes({"model": res["model"], "features": R["feats"], "target": R["target"],
                                               "classes": list(res["encoder"].classes_) if "encoder" in res else None}),
                               "rajdsp_model.pkl")

# =============================================================================== TAB 4: regression
with tabs[3]:
    rmode = st.radio("Model", ["Curve fit", "Linear", "Logistic", "Polynomial", "3D surface fit"], horizontal=True, key="rmode")
    demos = {"Demo: turning surface roughness": "demo_roughness.csv"}
    if rmode == "Curve fit":
        demos = {"Demo: tool-wear growth curve": "__wear__", **demos}
    sel = st.radio("Dataset", ["Upload my file"] + list(demos), horizontal=True, key="reg_src")
    rdf = None
    if sel == "Upload my file":
        up = st.file_uploader("Choose file (CSV, TXT, LVM, XLSX)", type=["csv", "txt", "lvm", "xlsx", "xls"], key="reg_up")
        if up:
            try:
                rdf = read_any(up.name, up.getvalue()) if not up.name.lower().endswith(".lvm") else read_upload(up.name, up.getvalue())
            except Exception as e:
                st.error(str(e))
    elif demos[sel] == "__wear__":
        rng = np.random.default_rng(4)
        tm = np.linspace(1, 60, 40)
        rdf = pd.DataFrame({"cutting_time_min": tm, "flank_wear_mm": (0.08 + 0.32 / (1 + np.exp(-0.12 * (tm - 38)))
                                                                       + 0.004 * tm + 0.008 * rng.standard_normal(tm.size)).round(4)})
    else:
        rdf = demo_csv(demos[sel])
    if rdf is not None:
        with st.expander(f"Preview ({rdf.shape[0]} rows)"):
            st.dataframe(rdf.head(50), width="stretch")
        cols = list(rdf.columns)
        num = [c for c in cols if pd.api.types.is_numeric_dtype(rdf[c])]
        run = False
        if rmode == "Curve fit":
            a, b, c3 = st.columns(3)
            xc = a.selectbox("x", num, num.index("feed_mm_rev") if "feed_mm_rev" in num else 0)
            yc = b.selectbox("y", num, num.index("Ra_um") if "Ra_um" in num else min(1, len(num) - 1))
            cm = c3.selectbox("Equation", ["Auto-optimise (lowest AIC)"] + list(L.CURVES))
            d = rdf[[xc, yc]].dropna()
            xv, yv = d[xc].to_numpy(float), d[yc].to_numpy(float)
            if st.button("▶️ Run analysis", type="primary") or st.session_state.get("cf_done") == (xc, yc, cm):
                st.session_state.cf_done = (xc, yc, cm)
                if st.session_state.get("cf_logged") != (xc, yc, cm):
                    st.session_state.cf_logged = (xc, yc, cm)
                    log_use("regression", f"curve fit {cm}")
                name = cm
                if cm.startswith("Auto"):
                    rank = L.auto_curve(xv, yv)
                    st.dataframe(rank.round(4), width="stretch", hide_index=True)
                    name = rank.iloc[0]["model"]
                    st.success(f"Best equation by AIC: **{name}**")
                try:
                    r = L.fit_curve(name, xv, yv)
                    xs = np.linspace(r["x"].min(), r["x"].max(), 400)
                    fig = go.Figure()
                    fig.add_scatter(x=r["x"], y=r["y"], mode="markers", name="data", marker=dict(color="#1565c0"))
                    fig.add_scatter(x=xs, y=r["f"](xs, *r["popt"]), name="fit", line=dict(color="#c0392b", width=2.5))
                    show(fig_layout(fig, 400, title=f"y = {r['expr']}", xaxis_title=xc, yaxis_title=yc))
                    c = st.columns(3)
                    c[0].metric("R²", f"{r['r2']:.4f}")
                    c[1].metric("RMSE", f"{r['rmse']:.4g}")
                    c[2].metric("AIC", f"{r['aic']:.1f}")
                    pn = L.CURVES[name][0].__code__.co_varnames[1:len(r["popt"]) + 1]
                    st.dataframe(pd.DataFrame({"parameter": pn, "value": r["popt"], "± 1σ": r["perr"]}), hide_index=True)
                    xq = st.number_input(f"Predict {yc} at {xc} =", value=float(np.median(xv)))
                    st.metric(f"Predicted {yc}", f"{r['f'](np.array([xq]), *r['popt'])[0]:.5g}")
                except Exception as e:
                    st.error(f"Fit did not converge: {e}")
        elif rmode in ("Linear", "Polynomial"):
            a, b = st.columns([1, 2])
            yc = a.selectbox("Response y", num, len(num) - 1 if "Ra_um" not in num else num.index("Ra_um"))
            xopts = [c for c in num if c != yc]
            if rmode == "Linear":
                xcs = b.multiselect("Predictors x", xopts, xopts[:3])
            else:
                xcs = [b.selectbox("Predictor x", xopts, xopts.index("feed_mm_rev") if "feed_mm_rev" in xopts else 0)]
                dmode = st.radio("Degree", ["Auto-optimise (cross-validation)", "Manual"], horizontal=True)
                deg_manual = st.slider("Degree", 1, 10, 2) if dmode == "Manual" else None
            if xcs and st.button("▶️ Run analysis", type="primary", key="lin_run"):
                st.session_state.lin_cfg = (rmode, yc, tuple(xcs))
                log_use("regression", f"{rmode} {yc}~{xcs}")
            if xcs and st.session_state.get("lin_cfg") == (rmode, yc, tuple(xcs)):
                d = rdf[xcs + [yc]].dropna()
                X, y = d[xcs].to_numpy(float), d[yc].to_numpy(float)
                if rmode == "Linear":
                    r = L.ols(X, y)
                    c = st.columns(3)
                    c[0].metric("R²", f"{r['r2']:.4f}")
                    c[1].metric("Adjusted R²", f"{r['adj_r2']:.4f}")
                    c[2].metric("RMSE", f"{r['rmse']:.4g}")
                    eq = f"{yc} = {r['beta'][0]:.4g} " + " ".join(f"{'+' if bb >= 0 else '−'} {abs(bb):.4g}·{n}"
                                                                  for bb, n in zip(r["beta"][1:], xcs))
                    st.code(eq)
                    st.dataframe(pd.DataFrame({"term": ["intercept"] + xcs, "coefficient": r["beta"], "std error": r["se"],
                                               "t": r["t"], "p-value": r["p"],
                                               "significant (p<0.05)": ["yes" if p < 0.05 else "no" for p in r["p"]]}).round(5),
                                 hide_index=True, width="stretch")
                    fig = go.Figure()
                    fig.add_scatter(x=y, y=r["yp"], mode="markers", marker=dict(color="#1565c0"), name="data")
                    fig.add_scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", line=dict(dash="dash", color="black"),
                                    name="ideal")
                    show(fig_layout(fig, 380, title="Predicted vs actual", xaxis_title="actual", yaxis_title="predicted"))
                else:
                    best_d, cvt = L.auto_poly_degree(X, y)
                    deg = deg_manual or best_d
                    m = L.poly_model(deg).fit(X, y)
                    a, b = st.columns([2, 1])
                    with a:
                        xs = np.linspace(X.min(), X.max(), 400)[:, None]
                        fig = go.Figure()
                        fig.add_scatter(x=X[:, 0], y=y, mode="markers", name="data", marker=dict(color="#1565c0"))
                        fig.add_scatter(x=xs[:, 0], y=m.predict(xs), name=f"degree {deg}", line=dict(color="#c0392b", width=2.5))
                        show(fig_layout(fig, 400, xaxis_title=xcs[0], yaxis_title=yc))
                    with b:
                        st.metric("R² (fit)", f"{m.score(X, y):.4f}")
                        st.markdown(f"Auto-optimised degree: **{best_d}**")
                        st.dataframe(cvt.round(4), hide_index=True)
                    lr = m.named_steps["lr"]
                    sc = m.named_steps["s"]
                    coef = lr.coef_ / sc.scale_
                    c0 = lr.intercept_ - np.sum(coef * sc.mean_)
                    st.code(f"{yc} = {c0:.5g} " + " ".join(f"{'+' if cc >= 0 else '−'} {abs(cc):.5g}·x^{i + 1}"
                                                           for i, cc in enumerate(coef)) + f"    (x = {xcs[0]})")
        elif rmode == "Logistic":
            cat = [c for c in cols if rdf[c].nunique() <= 10]
            if not cat:
                st.warning("Logistic regression needs a categorical target (≤ 10 distinct values).")
            else:
                a, b = st.columns([1, 2])
                yc = a.selectbox("Class column", cat, cat.index("quality") if "quality" in cat else 0)
                xopts = [c for c in num if c != yc and c != "Ra_um"] or [c for c in num if c != yc]
                xcs = b.multiselect("Predictors", xopts, xopts[:3])
                Cv = st.select_slider("Regularisation C", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
                if xcs and st.button("▶️ Run analysis", type="primary", key="log_run"):
                    st.session_state.log_cfg = (yc, tuple(xcs), Cv)
                    log_use("regression", f"logistic {yc}")
                if xcs and st.session_state.get("log_cfg") == (yc, tuple(xcs), Cv):
                    d = rdf[xcs + [yc]].dropna()
                    r = L.logistic(d[xcs].to_numpy(float), d[yc].astype(str).to_numpy(), Cv)
                    cls = [str(c) for c in r["encoder"].classes_]
                    c = st.columns(2)
                    c[0].metric("Test accuracy", f"{100 * r['acc']:.1f} %")
                    if r["auc"] is not None:
                        c[1].metric("ROC AUC", f"{r['auc']:.3f}")
                    a, b = st.columns(2)
                    with a:
                        show(fig_layout(px.imshow(r["cm"], x=cls, y=cls, text_auto=True, color_continuous_scale="Blues",
                                                  labels=dict(x="Predicted", y="True")), 360, title="Confusion matrix"))
                    with b:
                        if r["roc"] is not None:
                            fpr, tpr, _ = r["roc"]
                            fig = go.Figure(go.Scatter(x=fpr, y=tpr, line=dict(color="#c0392b", width=2.5), name="ROC"))
                            fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance")
                            show(fig_layout(fig, 360, title=f"ROC (positive class = {cls[1]})", xaxis_title="False positive rate",
                                            yaxis_title="True positive rate"))
                    lr = r["model"].named_steps["lr"]
                    st.dataframe(pd.DataFrame(lr.coef_.T, index=xcs,
                                              columns=[f"coef ({c})" for c in (cls[1:] if lr.coef_.shape[0] == 1 else cls)]).round(4),
                                 width="stretch")
                    st.caption("Coefficients are for standardised predictors — larger magnitude = stronger influence.")
        else:  # 3D surface
            a, b, c3, d4 = st.columns(4)
            x1 = a.selectbox("x₁", num, 0)
            x2 = b.selectbox("x₂", num, num.index("feed_mm_rev") if "feed_mm_rev" in num else min(1, len(num) - 1))
            yc = c3.selectbox("z (response)", num, num.index("Ra_um") if "Ra_um" in num else len(num) - 1)
            dsel = d4.selectbox("Degree", ["Auto", 1, 2, 3, 4])
            if st.button("▶️ Run analysis", type="primary", key="srf_run"):
                st.session_state.srf_cfg = (x1, x2, yc, dsel)
                log_use("regression", f"surface {yc}~{x1},{x2}")
            if st.session_state.get("srf_cfg") == (x1, x2, yc, dsel) and x1 != x2:
                d = rdf[[x1, x2, yc]].dropna()
                X, y = d[[x1, x2]].to_numpy(float), d[yc].to_numpy(float)
                best_d, cvt = L.auto_poly_degree(X, y, 4)
                deg = best_d if dsel == "Auto" else int(dsel)
                m = L.poly_model(deg).fit(X, y)
                g1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
                g2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
                G1, G2 = np.meshgrid(g1, g2)
                Zs = m.predict(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
                fig = go.Figure(go.Surface(x=g1, y=g2, z=Zs, colorscale="Viridis", opacity=0.85, colorbar=dict(title=yc)))
                fig.add_scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode="markers", marker=dict(size=3, color="#c0392b"), name="data")
                fig.update_layout(scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=yc))
                show(fig_layout(fig, 560, title=f"Polynomial surface, degree {deg} (auto-optimised: {best_d})"))
                c = st.columns(3)
                c[0].metric("R² (fit)", f"{m.score(X, y):.4f}")
                q1 = c[1].number_input(f"{x1} =", value=float(np.median(X[:, 0])))
                q2 = c[2].number_input(f"{x2} =", value=float(np.median(X[:, 1])))
                st.metric(f"Predicted {yc}", f"{m.predict([[q1, q2]])[0]:.4g}")
                show(fig_layout(go.Figure(go.Contour(x=g1, y=g2, z=Zs, colorscale="Viridis", contours=dict(showlabels=True))),
                                420, title="Contour map", xaxis_title=x1, yaxis_title=x2))

# =============================================================================== TAB 5: data analysis
with tabs[4]:
    st.markdown("Upload your dataset, visualise patterns, clean data and analyse correlations.")
    df5, lab5 = dataset_picker("da", {"Demo: bearing fault features": "demo_bearing_features.csv",
                                      "Demo: turning surface roughness": "demo_roughness.csv"})
    if df5 is not None:
        key5 = f"clean_{lab5}"
        work = st.session_state.get(key5, df5)
        c = st.columns(4)
        c[0].metric("Rows", f"{len(work):,}")
        c[1].metric("Columns", work.shape[1])
        c[2].metric("Missing cells", int(work.isna().sum().sum()))
        c[3].metric("Duplicate rows", int(work.duplicated().sum()))
        sub = st.tabs(["Summary", "Cleaning", "Time series plots", "Distributions", "Correlations"])
        numc = [cc for cc in work.columns if pd.api.types.is_numeric_dtype(work[cc])]
        with sub[0]:
            st.dataframe(work.describe(include="all").T, width="stretch")
            st.dataframe(pd.DataFrame({"dtype": work.dtypes.astype(str), "missing": work.isna().sum(),
                                       "unique": work.nunique()}), width="stretch")
        with sub[1]:
            with st.form("clean_form"):
                a, b = st.columns(2)
                miss = a.selectbox("Missing values", ["keep", "drop rows", "fill with mean", "fill with median", "interpolate"])
                dup = a.checkbox("Remove duplicate rows", True)
                outl = b.selectbox("Outliers (numeric columns)", ["keep", "remove |z| > 3", "remove outside 1.5×IQR"])
                dropc = b.multiselect("Drop columns", list(df5.columns))
                go_ = st.form_submit_button("🧹 Apply cleaning", type="primary")
            if go_:
                w = df5.drop(columns=dropc).copy()
                nc = [cc for cc in w.columns if pd.api.types.is_numeric_dtype(w[cc])]
                if miss == "drop rows":
                    w = w.dropna()
                elif miss == "fill with mean":
                    w[nc] = w[nc].fillna(w[nc].mean())
                elif miss == "fill with median":
                    w[nc] = w[nc].fillna(w[nc].median())
                elif miss == "interpolate":
                    w[nc] = w[nc].interpolate(limit_direction="both")
                if dup:
                    w = w.drop_duplicates()
                if outl.startswith("remove |z|") and nc:
                    z = (w[nc] - w[nc].mean()) / (w[nc].std() + 1e-15)
                    w = w[(z.abs() <= 3).all(axis=1)]
                elif outl.startswith("remove outside") and nc:
                    q1, q3 = w[nc].quantile(0.25), w[nc].quantile(0.75)
                    iqr = q3 - q1
                    w = w[((w[nc] >= q1 - 1.5 * iqr) & (w[nc] <= q3 + 1.5 * iqr)).all(axis=1)]
                st.session_state[key5] = w.reset_index(drop=True)
                log_use("data analysis", f"cleaned {lab5}")
                st.rerun()
            if key5 in st.session_state:
                st.success(f"Cleaned: {len(df5)} → {len(work)} rows.")
                if st.button("↩️ Reset to original"):
                    del st.session_state[key5]
                    st.rerun()
            st.download_button("⬇️ Download cleaned data (CSV)", csv_bytes(work), "rajdsp_cleaned.csv", "text/csv")
            st.download_button("⬇️ Download summary statistics (CSV)", work.describe(include="all").T.to_csv().encode(),
                               "rajdsp_summary.csv", "text/csv")
        with sub[2]:
            ys = st.multiselect("Columns to plot", numc, numc[: min(3, len(numc))])
            xax = st.selectbox("x axis", ["row index"] + numc)
            sep_ = st.checkbox("Separate panels", False)
            if ys:
                xvals = np.arange(len(work)) if xax == "row index" else work[xax]
                if sep_:
                    fig = make_subplots(rows=len(ys), cols=1, shared_xaxes=True, subplot_titles=ys, vertical_spacing=0.04)
                    for i, ycol in enumerate(ys):
                        fig.add_scatter(x=xvals, y=work[ycol], row=i + 1, col=1, showlegend=False, line=dict(width=1))
                    show(fig_layout(fig, 200 * len(ys)))
                else:
                    fig = go.Figure([go.Scatter(x=xvals, y=work[ycol], name=ycol, line=dict(width=1)) for ycol in ys])
                    show(fig_layout(fig, 420, xaxis_title=xax))
        with sub[3]:
            if numc:
                a, b = st.columns(2)
                col = a.selectbox("Column", numc)
                catc = [cc for cc in work.columns if work[cc].nunique() <= 12]
                grp = b.selectbox("Group by (optional)", ["none"] + catc)
                a, b = st.columns(2)
                with a:
                    show(fig_layout(px.histogram(work, x=col, color=None if grp == "none" else grp, nbins=40, barmode="overlay",
                                                 opacity=0.75), 360, title="Histogram"))
                with b:
                    show(fig_layout(px.box(work, y=col, x=None if grp == "none" else grp, color=None if grp == "none" else grp,
                                           points="outliers"), 360, title="Box plot", showlegend=False))
                sc_cols = st.multiselect("Scatter-matrix columns (2–6)", numc, numc[: min(4, len(numc))])
                if 2 <= len(sc_cols) <= 6:
                    fig = px.scatter_matrix(work, dimensions=sc_cols, color=None if grp == "none" else grp)
                    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
                    show(fig_layout(fig, 600))
        with sub[4]:
            if len(numc) >= 2:
                meth = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
                cc = st.multiselect("Columns", numc, numc[: min(15, len(numc))], key="corr_cols")
                if len(cc) >= 2:
                    C = work[cc].corr(method=meth)
                    fig = px.imshow(C, text_auto=".2f" if len(cc) <= 15 else False, color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1, aspect="auto")
                    show(fig_layout(fig, max(400, 28 * len(cc)), title=f"{meth.title()} correlation"))
                    tri = C.where(np.triu(np.ones(C.shape, bool), 1)).stack().sort_values(key=np.abs, ascending=False)
                    st.markdown("**Strongest pairs**")
                    st.dataframe(tri.head(10).rename("r").reset_index().rename(columns={"level_0": "a", "level_1": "b"}).round(3),
                                 hide_index=True)
                    st.download_button("⬇️ Download correlation matrix (CSV)", C.to_csv().encode(), "rajdsp_correlation.csv",
                                       "text/csv")

# =============================================================================== TAB 6: NI-DAQ
with tabs[5]:
    st.markdown("Record **voltage, current, force, acceleration (IEPE), pressure, sound, temperature, strain** through any "
                "National Instruments DAQ device, using NI's open-source Python API **nidaqmx**. The recording is saved "
                "as CSV (time + channels) and opens directly in RajDSP.")
    try:
        import nidaqmx  # noqa: F401
        from nidaqmx.system import System
        devs = [d.name for d in System.local().devices]
        daq_ok = True
    except Exception:
        devs, daq_ok = [], False
    a, b = st.columns([3, 2])
    with b:
        st.markdown("**Supported signals**")
        st.markdown("Voltage · Current · Force (bridge/IEPE) · Acceleration (IEPE) · Pressure · Sound (microphone) · "
                    "Temperature (thermocouple/RTD) · Strain · Vibration")
        if daq_ok and devs:
            st.success(f"nidaqmx found · devices: {', '.join(devs)}")
        elif daq_ok:
            st.warning("nidaqmx is installed but no NI device was found. Create a *simulated* device in NI MAX to practise.")
        else:
            st.info("Live recording works only when RajDSP runs on the PC that has the NI device (a cloud app cannot see USB hardware). "
                    "Download the recorder below, or run RajDSP locally with `pip install nidaqmx`.")
    with a:
        if daq_ok and devs:
            with st.form("daq_form"):
                c = st.columns(3)
                dev = c[0].selectbox("Device", devs)
                chs = c[1].text_input("Channels", "ai0:1", help="e.g. ai0, ai0:3")
                kind = c[2].selectbox("Measurement", ["Voltage", "Acceleration (IEPE)", "Sound pressure (IEPE mic)", "Thermocouple (K)"])
                c = st.columns(4)
                rate = c[0].number_input("Rate (S/s)", 1.0, 1e6, 12800.0)
                durn = c[1].number_input("Duration (s)", 0.1, 600.0, 2.0)
                vr = c[2].number_input("± range (V or g / Pa)", 0.1, 500.0, 5.0)
                sens = c[3].number_input("Sensitivity (mV/unit, IEPE)", 0.1, 10000.0, 100.0)
                rec = st.form_submit_button("⏺️ Record", type="primary")
            if rec:
                import importlib.util
                spec = importlib.util.spec_from_file_location("rec", DAQ_SCRIPT)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                try:
                    with st.spinner("Recording…"):
                        ddf = mod.record(f"{dev}/{chs}", kind, rate, durn, vr, sens)
                    st.session_state.daq_df = ddf
                    log_use("daq", f"{dev}/{chs} {kind} {rate} S/s {durn}s")
                    st.success("Recorded. Choose *Last NI-DAQ recording* in the sidebar to analyse it.")
                except Exception as e:
                    st.error(f"DAQ error: {e}")
        st.markdown("**Recorder package (works offline on the lab PC)**")
        if os.path.exists(DAQ_SCRIPT):
            with open(DAQ_SCRIPT, "rb") as fh:
                st.download_button("⬇️ Download Signal Acquisition script (rajdsp_daq_recorder.py)", fh.read(),
                                   "rajdsp_daq_recorder.py", type="primary")
        st.markdown("""
1. Install **NI-DAQmx driver** (free, from ni.com) and Python 3.9+.
2. `pip install nidaqmx numpy pandas`
3. Check the device name in **NI MAX** (e.g. `Dev1`, `cDAQ1Mod1`). No hardware? NI MAX → *Create New → Simulated NI-DAQmx Device*.
4. Record, e.g. two accelerometers for 5 s at 25.6 kS/s:
   `python rajdsp_daq_recorder.py --chan Dev1/ai0:1 --type accel --rate 25600 --dur 5 --sens 100 --out cut01.csv`
5. Upload `cut01.csv` in the sidebar of RajDSP (sampling rate is detected from the time column).
""")
        st.caption("Tips: sample at ≥ 2.56× the highest frequency of interest (anti-aliasing); use IEPE excitation for "
                   "accelerometers/microphones; ground the shield at one end only; record a few seconds of idle machine as baseline.")
        if "daq_df" in st.session_state:
            st.download_button("⬇️ Download last recording (CSV)", csv_bytes(st.session_state.daq_df), "rajdsp_recording.csv")

# =============================================================================== TAB 7: references
with tabs[6]:
    st.subheader("What RajDSP does")
    st.markdown("""
**Signal processing** — detrend + zero-phase Butterworth filtering (`scipy.signal.sosfiltfilt`); single-sided FFT with
Hann/Hamming/Blackman/flat-top windows; Welch PSD; STFT spectrogram; Hilbert envelope spectrum with bearing defect frequencies
(FTF, BPFO, BPFI, BSF); continuous wavelet scalogram (Morlet, Mexican hat, complex Morlet, Gaussian, Shannon).

**Wavelets** — multilevel discrete wavelet transform with 100+ wavelets (biorthogonal *bior* as default, rbio, Daubechies,
Symlets, Coiflets, Haar, dmey), levels 1–20 (automatically capped at the maximum useful level for the signal length);
per-node reconstruction, band limits, energy distribution, FFT of each node. Denoising with the universal (VisuShrink),
BayesShrink and minimax thresholds, soft or hard, noise σ from the median absolute deviation of the finest detail.

**Features (≈ 50 per signal at level 4)** — time: mean, std, variance, RMS, peak, peak-to-peak, skewness, kurtosis, crest,
shape, impulse and margin factors, energy, Shannon entropy, zero-crossing rate · frequency: dominant frequency, spectral
centroid, spread, entropy, 85 % roll-off, flatness, total power · wavelet: energy, energy %, std, RMS and kurtosis of every
node, wavelet energy entropy.

**Machine learning** (scikit-learn pipelines with standardisation) — SVM (RBF/linear/polynomial), Random Forest, k-NN and
MLP classifiers; MLP (ANN), SVR and Random Forest regressors; automatic hyper-parameter optimisation by grid search with
stratified k-fold cross-validation; hold-out test metrics, confusion matrix, PCA view, parity plots; trained models
downloadable as `.pkl`.

**Regression** — non-linear least-squares curve fitting (10 equations, automatic selection by AIC), multiple linear regression
with standard errors and p-values, logistic regression with ROC/AUC, polynomial regression with cross-validated degree,
3-D polynomial response surfaces with contour maps.
""")
    st.subheader("Open-source software used")
    st.markdown("""
- **Streamlit** (UI) · **NumPy** · **pandas** · **SciPy** (`signal`, `optimize`, `stats`, `io.loadmat`) — scipy.org
- **PyWavelets** — Lee G. et al. (2019) *PyWavelets: A Python package for wavelet analysis*, J. Open Source Softw. 4(36):1237. doi:10.21105/joss.01237
- **scikit-learn** — Pedregosa F. et al. (2011) J. Mach. Learn. Res. 12:2825–2830
- **Plotly** (interactive charts) · **openpyxl** (Excel files)
- **nidaqmx** — NI's open-source Python API for NI-DAQmx (github.com/ni/nidaqmx-python)
""")
    st.subheader("Open datasets to try (download, then upload the files in the sidebar or the Batch tab)")
    st.markdown("""
| Dataset | What it contains | Format |
|---|---|---|
| **CWRU Bearing Data Center** (Case Western Reserve Univ.) | drive/fan-end vibration, 12 & 48 kHz, inner/outer race & ball faults, 0–3 hp | `.mat` (loads directly) |
| **Paderborn University KAt bearing data** | vibration + motor current, artificial & real damage | `.mat` |
| **NASA IMS bearing run-to-failure** (PCoE repository) | 4 bearings to failure, 20 kHz snapshots | text |
| **NASA milling data set** (PCoE repository, Agogino & Goebel) | AE, vibration & current with flank wear | `.mat` |
| **PHM Society 2010 data challenge** | high-speed CNC milling, force/vibration/AE, tool wear | `.csv` |
| **MaFaulDa** (UFRJ) | machinery fault simulator: imbalance, misalignment, bearing faults | `.csv` |
| **XJTU-SY bearing data sets** | accelerated life tests, 25.6 kHz | `.csv` |
| **FEMTO-ST / PRONOSTIA** (IEEE PHM 2012) | bearing degradation for RUL prediction | `.csv` |
""")
    st.caption("The demo signals and demo datasets inside RajDSP are **simulated** (physics-based, fixed random seed — see "
               "`samples.py`). Bearing demo uses the CWRU rig geometry (SKF 6205-2RS, 9 balls) so its defect frequencies match the "
               "real CWRU files.")
    st.subheader("Method references")
    st.markdown("""
Mallat S.G. (1989) IEEE Trans. Pattern Anal. Mach. Intell. 11:674 (multiresolution DWT) · Cohen A., Daubechies I., Feauveau J.-C.
(1992) Commun. Pure Appl. Math. 45:485 (biorthogonal wavelets) · Donoho D.L., Johnstone I.M. (1994) Biometrika 81:425 (universal &
minimax thresholds) · Chang S.G., Yu B., Vetterli M. (2000) IEEE Trans. Image Process. 9:1532 (BayesShrink) · Welch P. (1967)
IEEE Trans. Audio Electroacoust. 15:70 · Randall R.B., Antoni J. (2011) Mech. Syst. Signal Process. 25:485 (bearing diagnostics,
envelope analysis) · Smith W.A., Randall R.B. (2015) Mech. Syst. Signal Process. 64–65:100 (CWRU benchmark) · Lei Y. et al. (2020)
Mech. Syst. Signal Process. 138:106587 (ML for machine fault diagnosis review) · Cortes C., Vapnik V. (1995) Mach. Learn. 20:273 (SVM) ·
Akaike H. (1974) IEEE Trans. Autom. Control 19:716 (AIC).
""")
    st.warning("Teaching & research tool. Check sampling rate, units and sensor calibration before drawing conclusions; "
               "small datasets give optimistic accuracies — always confirm on independent test recordings.")
    with st.expander("Instructor: usage log"):
        if os.path.exists(LOG_FILE):
            lg = pd.read_csv(LOG_FILE)
            st.write(f"{len(lg)} logged analyses")
            st.dataframe(lg.tail(50), width="stretch")
            st.download_button("Download usage log", csv_bytes(lg), "rajdsp_usage_log.csv", "text/csv")
        else:
            st.write("No analyses logged yet.")

# ----------------------------------------------------------------------------- footer (plain counter)
c_now = _read_counter()
st.markdown(f"""
<div class='raj-foot'>👁️ Visitors: <b>{c_now.get('visits', 0)}</b> &nbsp;|&nbsp; 🧪 Analyses: <b>{c_now.get('analyses', 0)}</b><br>
Developed by <b>Dr. Gururaja S</b>, Centre for Innovative Manufacturing Research (CIMR), VIT Vellore · RajDSP v{VERSION} ·
part of the Raj3DLabs family</div>
""", unsafe_allow_html=True)
