"""Functii partajate intre cele doua pagini Streamlit."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_meta(task: str) -> dict:
    """task = 'clf' sau 'reg'."""
    with open(ROOT / "models" / task / "meta.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(task: str, name: str):
    return joblib.load(ROOT / "models" / task / f"model_{name}.joblib")


@st.cache_data
def load_test_data(task: str):
    X = pd.read_parquet(ROOT / "models" / task / "X_test.parquet")
    y = pd.read_parquet(ROOT / "models" / task / "y_test.parquet")
    return X, y.iloc[:, 0]


@st.cache_data
def load_train_data(task: str):
    return pd.read_parquet(ROOT / "models" / task / "X_train.parquet")


@st.cache_data
def load_lc(task: str, name: str):
    npz = np.load(ROOT / "models" / task / f"lc_{name}.npz")
    return {k: npz[k] for k in npz.files}


def plot_learning_curve(lc: dict, metric: str):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(lc["sizes"], lc["train_mean"], "o-", color="#1976d2", label="train")
    ax.fill_between(lc["sizes"], lc["train_mean"] - lc["train_std"],
                    lc["train_mean"] + lc["train_std"], alpha=0.15, color="#1976d2")
    ax.plot(lc["sizes"], lc["val_mean"], "o-", color="#e53935", label="cv")
    ax.fill_between(lc["sizes"], lc["val_mean"] - lc["val_std"],
                    lc["val_mean"] + lc["val_std"], alpha=0.15, color="#e53935")
    ax.set_xlabel("nr observatii antrenare"); ax.set_ylabel(metric)
    ax.legend(); ax.grid(alpha=0.3)
    return fig


def make_input_form(meta: dict, defaults: dict | None = None) -> dict:
    """Construieste un formular cu sliders pentru fiecare feature."""
    defaults = defaults or {}
    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(meta["features"]):
        info = meta["feature_info"][feat]
        col = cols[i % 2]
        default = defaults.get(feat, info["median"])
        # alege step rezonabil
        rng = info["max"] - info["min"]
        step = max(rng / 100, 0.01)
        # daca rangea-ul e mic, mai mic pas
        if rng < 1:
            step = round(step, 4)
        values[feat] = col.slider(
            feat, float(info["min"]), float(info["max"]),
            float(default), step=float(step), key=f"slider_{feat}",
        )
    return values


def compute_shap_for_instance(pipe, x_row: pd.DataFrame, X_train: pd.DataFrame,
                              features: list[str], model_name: str, task: str):
    """Calculeaza SHAP pentru o singura instanta, alegand explainer-ul potrivit."""
    scaler = pipe.named_steps.get("scaler")
    model = pipe.named_steps["model"]
    bg = X_train.sample(min(80, len(X_train)), random_state=42)
    bg_t = scaler.transform(bg) if scaler else bg.values
    x_t = scaler.transform(x_row) if scaler else x_row.values

    tree_models = ("XGBoost", "CatBoost", "RandomForest", "DecisionTree")
    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
        sv = explainer(x_t)
    else:
        if task == "clf":
            f = lambda z: model.predict_proba(z)[:, 1]
        else:
            f = lambda z: model.predict(z)
        explainer = shap.KernelExplainer(f, bg_t)
        sv = explainer(x_t)

    # uniformizez pentru clasificare binara
    if hasattr(sv, "values") and sv.values.ndim == 3:
        vals = sv.values[..., 1]
        bv = sv.base_values[..., 1] if sv.base_values.ndim > 1 else sv.base_values
    else:
        vals = sv.values
        bv = sv.base_values

    return shap.Explanation(values=vals, base_values=bv, data=x_t,
                            feature_names=features)


def render_metrics(metrics: dict, task: str):
    """Afiseaza metricile cheie ca metric cards."""
    if task == "clf":
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
        cols[1].metric("Precision", f"{metrics['precision']:.4f}")
        cols[2].metric("Recall", f"{metrics['recall']:.4f}")
        cols[3].metric("F1", f"{metrics['f1']:.4f}")
        cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    else:
        cols = st.columns(4)
        cols[0].metric("MSE", f"{metrics['MSE']:.4f}")
        cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
        cols[2].metric("MAE", f"{metrics['MAE']:.4f}")
        cols[3].metric("R2", f"{metrics['R2']:.4f}")
