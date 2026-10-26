# RajDSP — Signal Processing, Feature Extraction & ML Prediction

Streamlit app (Raj3DLabs family) by Dr. Gururaja S, CIMR, VIT Vellore. Built only on open-source Python.

## Run locally (Windows / PyCharm terminal)
```
pip install -r requirements.txt
streamlit run app.py
```
For live NI-DAQ recording inside the app (lab PC only): `pip install nidaqmx` + NI-DAQmx driver.

## Deploy on Streamlit Community Cloud
Push this folder (app.py, dsp.py, learn.py, samples.py, data/, daq/, requirements.txt) to a GitHub repo
(e.g. vitguru45/RajDSP) → share.streamlit.io → New app → main file `app.py`.

## Tabs
| Tab | What it does |
|---|---|
| 📈 Signal analysis | Upload .txt/.csv/.xlsx/.lvm/.mat (or pick a sample) → time domain, FFT & Welch PSD, STFT, DWT decomposition (100+ wavelets, bior default, levels 1–20), wavelet denoising, CWT scalogram, envelope spectrum with bearing defect frequencies, ~50 features |
| 🧩 Batch feature extraction | Many files → windowed feature table (one row per window, label per file) → CSV or straight into ML |
| 🤖 Machine learning | SVM classification / ANN regression (+ RF, k-NN, MLP, SVR) with grid-search hyper-parameter optimisation, confusion matrix, PCA, parity plot, prediction, model download |
| 📉 Regression models | Curve fit (10 equations, auto-pick by AIC), linear (p-values), logistic (ROC/AUC), polynomial (CV degree), 3-D surface fit |
| 🔍 Data analysis | Summary, cleaning (missing/duplicates/outliers), time-series plots, histograms, box plots, scatter matrix, correlation heat-map, CSV export |
| 🎛️ Data acquisition | NI-DAQ recorder (`daq/rajdsp_daq_recorder.py`, uses NI's open-source `nidaqmx`) — voltage, current, IEPE accelerometer, microphone, thermocouple → CSV |
| 📚 Data, models & references | Methods, open software, open datasets (CWRU, Paderborn, NASA IMS & milling, PHM 2010, MaFaulDa, XJTU-SY, FEMTO), references, usage log |

## Files
| File | Purpose |
|---|---|
| `app.py` | UI, counter, usage log |
| `dsp.py` | File readers (incl. LabVIEW .lvm and MATLAB .mat), filters, FFT/PSD/STFT/CWT, envelope, DWT, denoising, features |
| `learn.py` | Classifiers, regressors, grid search, curve fitting, OLS, polynomial & logistic regression |
| `samples.py` | Simulated demo signals (CWRU-geometry bearing faults, milling chatter, multitone, chirp) and demo datasets — `python samples.py` regenerates `data/` |
| `daq/rajdsp_daq_recorder.py` | Stand-alone NI-DAQ → CSV recorder (`python rajdsp_daq_recorder.py --help`) |

## Notes
* Visitor/analysis counter and usage log are JSON/CSV files beside app.py; on Streamlit Cloud they reset when the app restarts or redeploys.
* Demo signals/datasets are simulated (fixed seed). For real data, download CWRU `.mat` files — they load directly.
* A cloud app cannot see USB hardware: record with the DAQ script on the lab PC, then upload the CSV.
