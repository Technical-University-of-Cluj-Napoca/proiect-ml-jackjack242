"""Pagina pentru problema de clasificare - Wine Quality Red."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

from app_utils import (load_meta, load_model, load_test_data, load_train_data,
                        load_lc, plot_learning_curve, make_input_form,
                        compute_shap_for_instance, render_metrics, ROOT)

st.set_page_config(page_title="Clasificare - Wine Quality", page_icon=":wine_glass:", layout="wide")

st.title(":wine_glass: Clasificare - Wine Quality Red")

meta = load_meta("clf")

# === Sectiunea 1: prezentare ===
with st.expander("Despre dataset si problema", expanded=False):
    info = meta["dataset_info"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Observatii", info["n_rows"])
    c2.metric("Features", info["n_features"])
    c3.metric("Vinuri bune (%)",
              f"{info['class_balance']['good=1'] / info['n_rows'] * 100:.1f}%")
    st.markdown("""
    **Problema:** clasificare binara - vin "bun" daca `quality >= 6` (scor expert), altfel nu.

    **Features (toate numerice):** aciditate fixa/volatila, acid citric, zahar rezidual,
    cloruri, dioxid de sulf liber/total, densitate, pH, sulfati, alcool.
    """)

st.subheader("EDA - graficele cheie")
tabs = st.tabs(["Target", "Distributii", "Corelatii", "Boxplot top features"])
imgs = ["eda_target.png", "eda_distributions.png", "eda_corr.png", "eda_box_top.png"]
for tab, img in zip(tabs, imgs):
    with tab:
        st.image(str(ROOT / "plots" / "clf" / img), use_container_width=True)

st.divider()

# === Sectiunea 2: comparatie modele ===
st.subheader("Compararea modelelor")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Baseline (default hyperparams)**")
    base_df = pd.DataFrame(meta["baseline_results"]).T.sort_values("f1", ascending=False).round(4)
    st.dataframe(base_df, use_container_width=True)
with col_b:
    st.markdown("**Top 5 dupa tuning**")
    tuned_df = pd.DataFrame(meta["tuned_results"]).T.sort_values("f1", ascending=False).round(4)
    st.dataframe(tuned_df, use_container_width=True)

st.success(f":trophy: Cel mai bun model dupa tuning: **{meta['best_model']}** "
           f"(F1={tuned_df.loc[meta['best_model'], 'f1']:.4f}, "
           f"AUC={tuned_df.loc[meta['best_model'], 'roc_auc']:.4f})")

st.divider()

# === Sectiunea 3: explorare model interactiva ===
st.subheader("Exploreaza un model")
left, right = st.columns([1, 2])
with left:
    model_name = st.selectbox("Alege unul din top 5:", meta["top5"],
                              index=meta["top5"].index(meta["best_model"]))
    pipe = load_model("clf", model_name)
    metrics = meta["tuned_results"][model_name]
    params = meta["tuned_params"].get(model_name, {})

    st.markdown("**Hiperparametri (best):**")
    if params:
        st.json(params)
    else:
        st.caption("(nimic de tunat)")

with right:
    st.markdown("**Metrici test set:**")
    render_metrics(metrics, "clf")

    # Confusion matrix + ROC
    X_test, y_test = load_test_data("clf")
    yp = pipe.predict(X_test)
    yprob = pipe.predict_proba(X_test)[:, 1]

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, yp)
        ConfusionMatrixDisplay(cm, display_labels=["nu", "da"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion matrix")
        st.pyplot(fig, use_container_width=True)
    with c2:
        fpr, tpr, _ = roc_curve(y_test, yprob)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(fpr, tpr, color="#5e35b1", lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC curve"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)

# === Learning curve ===
st.markdown(f"**Curba de invatare pentru {model_name}**")
lc = load_lc("clf", model_name)
fig = plot_learning_curve(lc, "F1")
st.pyplot(fig, use_container_width=False)

st.divider()

# === Sectiunea 4: predictie interactiva ===
st.subheader("Predictie interactiva")
st.caption("Modifica valorile fiecarei caracteristici si vezi ce zice modelul.")

# preset: foloseste o instanta din test ca punct de start
preset_choice = st.radio(
    "Punct de plecare:",
    ["Vin bun (random)", "Vin prost (random)", "Mediana"],
    horizontal=True,
)
defaults = None
if preset_choice == "Vin bun (random)":
    good_idx = y_test[y_test == 1].sample(1, random_state=np.random.randint(0, 1000)).index[0]
    defaults = X_test.loc[good_idx].to_dict()
elif preset_choice == "Vin prost (random)":
    bad_idx = y_test[y_test == 0].sample(1, random_state=np.random.randint(0, 1000)).index[0]
    defaults = X_test.loc[bad_idx].to_dict()

st.markdown("**Caracteristici:**")
values = make_input_form(meta, defaults=defaults)
x_row = pd.DataFrame([values])[meta["features"]]

pred = pipe.predict(x_row)[0]
prob = pipe.predict_proba(x_row)[0, 1]
pclass = meta["classes"][pred]

c1, c2 = st.columns(2)
c1.metric("Predictie", pclass)
c2.metric("Probabilitate vin bun", f"{prob:.3f}")

if pred == 1:
    st.success(f"Modelul zice: **VIN BUN** ({prob*100:.1f}% incredere)")
else:
    st.error(f"Modelul zice: **VIN PROST** ({(1-prob)*100:.1f}% incredere)")

# === SHAP local pentru aceasta predictie ===
st.markdown("**SHAP pentru aceasta predictie:**")
with st.spinner("Calculez SHAP values..."):
    X_train = load_train_data("clf")
    sv = compute_shap_for_instance(pipe, x_row, X_train, meta["features"], model_name, "clf")

c1, c2 = st.columns(2)
with c1:
    st.markdown("*Waterfall - cum se aduna contributiile*")
    fig = plt.figure(figsize=(7, 5))
    shap.plots.waterfall(sv[0], show=False, max_display=11)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
with c2:
    st.markdown("*Force plot - aceeasi info, alt format*")
    fig = plt.figure(figsize=(8, 2.5))
    shap.plots.force(sv.base_values[0], sv.values[0], features=x_row.iloc[0],
                     feature_names=meta["features"], matplotlib=True, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

# === SHAP global pentru modelul ales ===
st.subheader("SHAP global (top 3 modele)")
st.caption("Daca modelul ales nu e in top 3, plot-urile globale arata pentru top 3 separat.")

shap_tabs = st.tabs(meta["top3_shap"])
for tab, name in zip(shap_tabs, meta["top3_shap"]):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(ROOT / "plots" / "clf" / f"shap_summary_{name}.png"),
                     caption=f"Summary plot - {name}", use_container_width=True)
        with c2:
            st.image(str(ROOT / "plots" / "clf" / f"shap_bar_{name}.png"),
                     caption=f"Bar plot (importanta medie absoluta) - {name}",
                     use_container_width=True)
        st.image(str(ROOT / "plots" / "clf" / f"shap_waterfall_{name}.png"),
                 caption=f"Waterfall pentru un sample - {name}",
                 use_container_width=False)

st.subheader("SHAP scatter - top 3 features (model best)")
st.image(str(ROOT / "plots" / "clf" / "shap_scatter_top3.png"),
         use_container_width=True)
