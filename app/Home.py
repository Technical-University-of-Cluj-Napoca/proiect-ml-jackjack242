"""Pagina principala - Streamlit app pentru proiectul ML."""
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Proiect ML - Clasificare & Regresie",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Proiect 1 - Machine Learning")
st.caption("Analiza comparata a modelelor de ML, in regresie si clasificare")

st.markdown("""
### Despre proiect
Aici e demo-ul interactiv pentru proiectul de la Sisteme Inteligente. Am luat
**doua probleme de invatare supervizata**, am antrenat 9 algoritmi pe fiecare,
am tunat top 5 si am facut analiza SHAP pe top 3.

Foloseste meniul din stanga ca sa navighezi:

#### :wine_glass: Clasificare - Wine Quality Red
Prezic daca un vin rosu e "bun" (`quality >= 6`) doar pe baza compozitiei
chimice. 1599 de vinuri din UCI.

#### :house: Regresie - California Housing
Prezic valoarea mediana a unei case dintr-un cartier din California, 1990.
Subsample de 2000 obs.

---

### Pipeline pentru fiecare problema
1. **EDA** + preprocesare (split 75/25)
2. **9 modele de baza** cu setari default
3. **Tuning** pe top 5 prin GridSearchCV (3-fold CV)
4. **Curbe de invatare** pe top 5
5. **SHAP** (global + local) pe top 3
6. **Streamlit demo** - aici suntem

### Stack
- `pandas` / `numpy` - data
- `scikit-learn` - 6 din 9 modele + pipeline + tuning
- `xgboost`, `catboost`, `interpret` (EBM) - boost-urile
- `shap` - explicabilitate
- `streamlit` - interfata asta
- managed cu `uv`
""")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader(":dart: Clasificare")
    st.markdown("""
    - **Dataset:** Wine Quality Red (UCI)
    - **Obs:** 1599 / **Features:** 11
    - **Target:** binar (vin bun da/nu)
    - **Best model:** XGBoost (vezi pagina pt. detalii)
    """)
    if st.button("Du-ma la pagina de clasificare"):
        st.switch_page("pages/1_Clasificare.py")

with col2:
    st.subheader(":chart_with_upwards_trend: Regresie")
    st.markdown("""
    - **Dataset:** California Housing (subsample 2000)
    - **Features:** 8
    - **Target:** valoare mediana case ($100k)
    - **Best model:** CatBoost (vezi pagina pt. detalii)
    """)
    if st.button("Du-ma la pagina de regresie"):
        st.switch_page("pages/2_Regresie.py")

st.divider()
st.caption("jackjack242 - aprilie/mai 2026")
