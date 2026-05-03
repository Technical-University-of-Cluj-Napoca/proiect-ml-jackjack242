[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/G_YtOrWk)

# Proiect 1 - Machine Learning

**Curs:** Sisteme Inteligente
**Tema:** Analiza comparata a modelelor de Machine Learning, in regresie si clasificare

## Despre

Doua probleme de invatare supervizata, fiecare cu pipeline-ul complet:

| Tema | Dataset | Obs | Best model | Score |
|------|---------|-----|------------|-------|
| Clasificare | Wine Quality Red (UCI) | 1599 | XGBoost (tuned) | F1=0.810, AUC=0.866 |
| Regresie | California Housing (subsample) | 2000 | CatBoost (tuned) | R2=0.755, RMSE=0.595 |

Pentru fiecare problema:
1. EDA + preprocesare (split 75/25)
2. Antrenare 9 modele de baza cu hiperparametri default
3. Tuning pe top 5 cu `GridSearchCV` (3-fold CV)
4. Curbe de invatare
5. Analiza SHAP global + local pe top 3
6. Demo interactiv in Streamlit

## Structura

```
.
├── data/                 # csv-urile (descarcate din UCI / sklearn)
├── notebooks/
│   ├── 01_clasificare.ipynb
│   └── 02_regresie.ipynb
├── models/
│   ├── clf/              # joblib + meta + learning curves + shap data
│   └── reg/
├── plots/                # toate PNG-urile generate
├── app/
│   ├── Home.py           # landing page Streamlit
│   ├── app_utils.py
│   └── pages/
│       ├── 1_Clasificare.py
│       └── 2_Regresie.py
├── scripts/
│   ├── download_data.py
│   └── build_notebooks.py
└── pyproject.toml
```

## Cum rulezi

```bash
# instaleaza uv daca nu il ai
pip install uv

# clone + intra in folder
git clone <repo>
cd proiect-ml-jackjack242

# instaleaza dependintele
uv sync

# descarca datasetele
uv run python scripts/download_data.py

# (re)genereaza notebook-urile din scriptul de build
uv run python scripts/build_notebooks.py

# ruleaza notebook-urile (genereaza modelele + plot-urile)
uv run jupyter nbconvert --to notebook --execute notebooks/01_clasificare.ipynb --output 01_clasificare.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_regresie.ipynb --output 02_regresie.ipynb

# porneste app-ul Streamlit
uv run streamlit run app/Home.py
```

Apoi deschide [http://localhost:8501](http://localhost:8501) in browser.

## Stack

- Python 3.12
- pandas, numpy, matplotlib, seaborn
- scikit-learn (6 din 9 modele + pipeline + tuning + learning curves)
- xgboost, catboost, interpret (EBM/EBR)
- shap (TreeExplainer + KernelExplainer)
- streamlit (multi-page app)
- managed cu **uv**

## Notite

- Pentru `GaussianProcess` am facut subsample la 500 obs (e O(n^3), 2000 era prea mult).
- Tuning-ul foloseste `GridSearchCV` cu grid-uri mici si CV 3-fold ca sa fie rapid;
  pentru spatii mari de cautare s-ar folosi `BayesSearchCV` din `scikit-optimize`.
- Toate hiperparametrii tunati si metricile raman in `models/<task>/meta.json`,
  ca sa-i citeasca app-ul.

## Autor

jackjack242 - aprilie/mai 2026
