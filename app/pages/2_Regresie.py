"""Pagina pentru problema de regresie - California Housing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app_utils import (load_meta, load_model, load_test_data, load_train_data,
                        load_lc, plot_learning_curve, make_input_form,
                        compute_shap_for_instance, render_metrics, ROOT)

st.set_page_config(page_title="Regresie - California Housing", page_icon=":house:", layout="wide")

st.title(":house: Regresie - California Housing")

meta = load_meta("reg")

with st.expander("Despre dataset si problema", expanded=False):
    info = meta["dataset_info"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Observatii", info["n_rows"])
    c2.metric("Features", info["n_features"])
    c3.metric("Target unit", meta.get("target_unit", "$100k"))
    st.markdown("""
    **Problema:** prezic `MedHouseVal` (valoare mediana case), unitati $100k. Date din
    recensamantul California 1990, subsample de 2000 cartiere.

    **Features:** venit median, varsta caselor, camere/dormitoare medii pe casa,
    populatie, ocupanti pe casa, latitude, longitude.
    """)

st.subheader("EDA - graficele cheie")
tabs = st.tabs(["Target", "Corelatii", "Harta", "MedInc vs price"])
imgs = ["eda_target.png", "eda_corr.png", "eda_map.png", "eda_medinc.png"]
for tab, img in zip(tabs, imgs):
    with tab:
        st.image(str(ROOT / "plots" / "reg" / img), use_container_width=True)

st.divider()

st.subheader("Compararea modelelor")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline (default hyperparams)**")
    base_df = pd.DataFrame(meta["baseline_results"]).T.sort_values("R2", ascending=False).round(4)
    st.dataframe(base_df, use_container_width=True)
with col_b:
    st.markdown("**Top 5 dupa tuning**")
    tuned_df = pd.DataFrame(meta["tuned_results"]).T.sort_values("R2", ascending=False).round(4)
    st.dataframe(tuned_df, use_container_width=True)

st.success(f":trophy: Cel mai bun model dupa tuning: **{meta['best_model']}** "
           f"(R2={tuned_df.loc[meta['best_model'], 'R2']:.4f}, "
           f"RMSE={tuned_df.loc[meta['best_model'], 'RMSE']:.4f})")

st.divider()

st.subheader("Exploreaza un model")
left, right = st.columns([1, 2])
with left:
    model_name = st.selectbox("Alege unul din top 5:", meta["top5"],
                              index=meta["top5"].index(meta["best_model"]))
    pipe = load_model("reg", model_name)
    metrics = meta["tuned_results"][model_name]
    params = meta["tuned_params"].get(model_name, {})
    st.markdown("**Hiperparametri (best):**")
    if params:
        st.json(params)
    else:
        st.caption("(nimic de tunat)")

with right:
    st.markdown("**Metrici test set:**")
    render_metrics(metrics, "reg")

    X_test, y_test = load_test_data("reg")
    yp = pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, yp, alpha=0.4, s=12, color="#5e35b1")
    lo, hi = float(min(y_test.min(), yp.min())), float(max(y_test.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="ideal")
    ax.set_xlabel("y_true ($100k)"); ax.set_ylabel("y_pred ($100k)")
    ax.set_title("Predicted vs Actual")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=False)

st.markdown(f"**Curba de invatare pentru {model_name}**")
lc = load_lc("reg", model_name)
fig = plot_learning_curve(lc, "R2")
st.pyplot(fig, use_container_width=False)

st.divider()

st.subheader("Predictie interactiva")
st.caption("Modifica valorile features-urilor unui cartier ipotetic.")

preset_choice = st.radio(
    "Punct de plecare:",
    ["Cartier scump (random)", "Cartier ieftin (random)", "Mediana"],
    horizontal=True,
)
defaults = None
if preset_choice == "Cartier scump (random)":
    idx = y_test.sort_values(ascending=False).head(50).sample(1).index[0]
    defaults = X_test.loc[idx].to_dict()
elif preset_choice == "Cartier ieftin (random)":
    idx = y_test.sort_values().head(50).sample(1).index[0]
    defaults = X_test.loc[idx].to_dict()

st.markdown("**Caracteristici:**")
values = make_input_form(meta, defaults=defaults)
x_row = pd.DataFrame([values])[meta["features"]]

pred = pipe.predict(x_row)[0]
c1, c2 = st.columns(2)
c1.metric("Predictie ($100k)", f"{pred:.3f}")
c2.metric("Asadar in dolari", f"${pred*100_000:,.0f}")

st.markdown("**SHAP pentru aceasta predictie:**")
with st.spinner("Calculez SHAP values..."):
    X_train = load_train_data("reg")
    sv = compute_shap_for_instance(pipe, x_row, X_train, meta["features"], model_name, "reg")

c1, c2 = st.columns(2)
with c1:
    st.markdown("*Waterfall*")
    fig = plt.figure(figsize=(7, 4.5))
    shap.plots.waterfall(sv[0], show=False, max_display=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
with c2:
    st.markdown("*Force plot*")
    fig = plt.figure(figsize=(8, 2.5))
    shap.plots.force(sv.base_values[0], sv.values[0], features=x_row.iloc[0],
                     feature_names=meta["features"], matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

st.subheader("SHAP global (top 3 modele)")
shap_tabs = st.tabs(meta["top3_shap"])
for tab, name in zip(shap_tabs, meta["top3_shap"]):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(ROOT / "plots" / "reg" / f"shap_summary_{name}.png"),
                     caption=f"Summary plot - {name}", use_container_width=True)
        with c2:
            st.image(str(ROOT / "plots" / "reg" / f"shap_bar_{name}.png"),
                     caption=f"Bar plot - {name}", use_container_width=True)
        st.image(str(ROOT / "plots" / "reg" / f"shap_waterfall_{name}.png"),
                 caption=f"Waterfall sample - {name}", use_container_width=False)

st.subheader("SHAP scatter - top 3 features (model best)")
st.image(str(ROOT / "plots" / "reg" / "shap_scatter_top3.png"),
         use_container_width=True)
