"""
learn.py — machine-learning and regression helpers for RajDSP (scikit-learn + SciPy only).
"""
import io
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================================ classification / regression models
CLASSIFIERS = {
    "SVM (RBF / linear / poly)": (SVC(), {
        "m__kernel": ["rbf", "linear", "poly"], "m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.01, 0.1, 1]}),
    "Random Forest": (RandomForestClassifier(random_state=0), {
        "m__n_estimators": [100, 300], "m__max_depth": [None, 5, 10]}),
    "k-Nearest Neighbours": (KNeighborsClassifier(), {"m__n_neighbors": [3, 5, 7, 11], "m__weights": ["uniform", "distance"]}),
    "ANN (MLP classifier)": (MLPClassifier(max_iter=2000, random_state=0), {
        "m__hidden_layer_sizes": [(16,), (32,), (32, 16)], "m__alpha": [1e-4, 1e-3, 1e-2]}),
}

REGRESSORS = {
    "ANN (MLP regressor)": (MLPRegressor(max_iter=3000, random_state=0, early_stopping=False), {
        "m__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)], "m__alpha": [1e-4, 1e-3, 1e-2],
        "m__activation": ["relu", "tanh"]}),
    "SVR (RBF)": (SVR(), {"m__C": [1, 10, 100], "m__gamma": ["scale", 0.1, 1], "m__epsilon": [0.01, 0.1]}),
    "Random Forest": (RandomForestRegressor(random_state=0), {"m__n_estimators": [100, 300], "m__max_depth": [None, 6, 12]}),
}


def train_classifier(X, y, model_name, test_size=0.25, tune=True, seed=42):
    le = LabelEncoder()
    yi = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=test_size, random_state=seed,
                                          stratify=yi if np.min(np.bincount(yi)) >= 2 else None)
    est, grid = CLASSIFIERS[model_name]
    pipe = Pipeline([("s", StandardScaler()), ("m", est)])
    k = int(max(2, min(5, np.min(np.bincount(ytr)))))
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    if tune:
        gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1, scoring="accuracy")
        gs.fit(Xtr, ytr)
        model, best = gs.best_estimator_, gs.best_params_
        cv_scores = cross_val_score(model, Xtr, ytr, cv=cv)
    else:
        model, best = pipe.fit(Xtr, ytr), {}
        cv_scores = cross_val_score(pipe, Xtr, ytr, cv=cv)
    yp = model.predict(Xte)
    res = dict(model=model, encoder=le, best=best, cv=cv_scores, acc=accuracy_score(yte, yp),
               f1=f1_score(yte, yp, average="macro"),
               cm=confusion_matrix(yte, yp, labels=range(len(le.classes_))),
               report=pd.DataFrame(classification_report(yte, yp, labels=range(len(le.classes_)),
                                                         target_names=[str(c) for c in le.classes_],
                                                         output_dict=True, zero_division=0)).T,
               n_train=len(ytr), n_test=len(yte))
    return res


def train_regressor(X, y, model_name, test_size=0.25, tune=True, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    est, grid = REGRESSORS[model_name]
    pipe = Pipeline([("s", StandardScaler()), ("m", est)])
    cv = KFold(min(5, max(2, len(ytr) // 5)), shuffle=True, random_state=seed)
    if tune:
        gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1, scoring="r2")
        gs.fit(Xtr, ytr)
        model, best = gs.best_estimator_, gs.best_params_
    else:
        model, best = pipe.fit(Xtr, ytr), {}
    cv_scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="r2")
    yp_te, yp_tr = model.predict(Xte), model.predict(Xtr)
    return dict(model=model, best=best, cv=cv_scores, ytr=ytr, yte=yte, yp_tr=yp_tr, yp_te=yp_te,
                r2=r2_score(yte, yp_te), r2_train=r2_score(ytr, yp_tr),
                rmse=float(np.sqrt(mean_squared_error(yte, yp_te))), mae=mean_absolute_error(yte, yp_te),
                n_train=len(ytr), n_test=len(yte))


def pca_2d(X):
    Xs = StandardScaler().fit_transform(X)
    p = PCA(n_components=min(2, Xs.shape[1]))
    Z = p.fit_transform(Xs)
    if Z.shape[1] == 1:
        Z = np.column_stack([Z, np.zeros(len(Z))])
    return Z, p.explained_variance_ratio_


def pickle_bytes(obj):
    b = io.BytesIO()
    pickle.dump(obj, b)
    return b.getvalue()


# ============================================================================ curve fitting
CURVES = {
    "Linear  y = a·x + b": (lambda x, a, b: a * x + b, "a·x + b", [1, 0]),
    "Quadratic  y = a·x² + b·x + c": (lambda x, a, b, c: a * x ** 2 + b * x + c, "a·x² + b·x + c", [1, 1, 0]),
    "Exponential  y = a·e^(b·x) + c": (lambda x, a, b, c: a * np.exp(b * x) + c, "a·e^(b·x) + c", [1, 0.1, 0]),
    "Exponential decay  y = a·e^(−x/τ) + c": (lambda x, a, tau, c: a * np.exp(-x / tau) + c, "a·e^(−x/τ) + c", [1, 1, 0]),
    "Power law  y = a·x^b": (lambda x, a, b: a * np.power(np.abs(x) + 1e-12, b), "a·x^b", [1, 1]),
    "Logarithmic  y = a·ln(x) + b": (lambda x, a, b: a * np.log(np.abs(x) + 1e-12) + b, "a·ln(x) + b", [1, 0]),
    "Sinusoid  y = A·sin(2πf·x + φ) + c": (lambda x, A, f, ph, c: A * np.sin(2 * np.pi * f * x + ph) + c,
                                           "A·sin(2πf·x + φ) + c", None),
    "Gaussian peak  y = A·e^(−(x−μ)²/2σ²) + c": (lambda x, A, mu, s, c: A * np.exp(-(x - mu) ** 2 / (2 * s ** 2)) + c,
                                                "A·e^(−(x−μ)²/2σ²) + c", None),
    "Logistic (sigmoid)  y = L/(1+e^(−k(x−x₀))) + c": (lambda x, L, k, x0, c: L / (1 + np.exp(-k * (x - x0))) + c,
                                                      "L/(1+e^(−k(x−x₀))) + c", None),
    "Michaelis–Menten / saturation  y = a·x/(b + x)": (lambda x, a, b: a * x / (b + x), "a·x/(b+x)", [1, 1]),
}


def _smart_p0(name, x, y):
    _, _, p0 = CURVES[name]
    if p0 is not None:
        if name.startswith("Exponential decay"):
            return [y[0] - y[-1], (x.max() - x.min()) / 3 or 1, y[-1]]
        if name.startswith("Michaelis"):
            return [y.max(), np.median(x)]
        return p0
    if name.startswith("Sinusoid"):
        dx = np.median(np.diff(np.sort(x))) or 1
        yy = y - y.mean()
        F = np.abs(np.fft.rfft(yy))
        fr = np.fft.rfftfreq(len(yy), dx)
        f0 = fr[np.argmax(F[1:]) + 1] if len(F) > 1 else 1
        return [np.std(y) * np.sqrt(2), f0, 0, y.mean()]
    if name.startswith("Gaussian"):
        return [y.max() - y.min(), x[np.argmax(y)], (x.max() - x.min()) / 6 or 1, y.min()]
    if name.startswith("Logistic"):
        return [y.max() - y.min(), 4 / ((x.max() - x.min()) or 1), np.median(x), y.min()]
    return None


def fit_curve(name, x, y):
    f, expr, _ = CURVES[name]
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    p0 = _smart_p0(name, x, y)
    popt, pcov = optimize.curve_fit(f, x, y, p0=p0, maxfev=20000)
    yp = f(x, *popt)
    n, k = len(y), len(popt)
    rss = float(np.sum((y - yp) ** 2))
    r2 = 1 - rss / (np.sum((y - y.mean()) ** 2) + 1e-15)
    aic = n * np.log(rss / n + 1e-300) + 2 * k
    perr = np.sqrt(np.abs(np.diag(pcov))) if np.all(np.isfinite(pcov)) else np.full(k, np.nan)
    return dict(f=f, expr=expr, popt=popt, perr=perr, r2=r2, rmse=np.sqrt(rss / n), aic=aic, x=x, y=y)


def auto_curve(x, y):
    rows = []
    for nm in CURVES:
        try:
            r = fit_curve(nm, x, y)
            if np.isfinite(r["aic"]):
                rows.append((nm, r["r2"], r["rmse"], r["aic"]))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["model", "R²", "RMSE", "AIC"]).sort_values("AIC").reset_index(drop=True)


# ============================================================================ linear / polynomial / logistic
def ols(X, y):
    """Ordinary least squares with standard errors, t and p values."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yp = Xd @ beta
    n, p = Xd.shape
    resid = y - yp
    s2 = resid @ resid / max(n - p, 1)
    cov = s2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.abs(np.diag(cov)))
    tval = beta / (se + 1e-300)
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), max(n - p, 1)))
    r2 = r2_score(y, yp)
    adj = 1 - (1 - r2) * (n - 1) / max(n - p, 1)
    return dict(beta=beta, se=se, t=tval, p=pval, yp=yp, r2=r2, adj_r2=adj, rmse=float(np.sqrt(np.mean(resid ** 2))))


def poly_model(degree):
    return Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)), ("s", StandardScaler()),
                     ("lr", LinearRegression())])


def auto_poly_degree(X, y, max_deg=6):
    rows = []
    cv = KFold(min(5, max(2, len(y) // 4)), shuffle=True, random_state=0)
    for d in range(1, max_deg + 1):
        sc = cross_val_score(poly_model(d), X, y, cv=cv, scoring="neg_root_mean_squared_error")
        rows.append((d, -sc.mean(), sc.std()))
    df = pd.DataFrame(rows, columns=["degree", "CV RMSE", "± std"])
    return int(df.loc[df["CV RMSE"].idxmin(), "degree"]), df


def logistic(X, y, C=1.0):
    le = LabelEncoder()
    yi = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.25, random_state=0,
                                          stratify=yi if np.min(np.bincount(yi)) >= 2 else None)
    m = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))]).fit(Xtr, ytr)
    yp = m.predict(Xte)
    out = dict(model=m, encoder=le, acc=accuracy_score(yte, yp),
               cm=confusion_matrix(yte, yp, labels=range(len(le.classes_))), roc=None, auc=None)
    if len(le.classes_) == 2:
        pr = m.predict_proba(Xte)[:, 1]
        out["roc"] = roc_curve(yte, pr)
        out["auc"] = roc_auc_score(yte, pr)
    return out
