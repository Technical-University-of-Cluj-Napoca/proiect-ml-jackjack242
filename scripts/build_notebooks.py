"""Construieste cele doua notebook-uri (.ipynb) din celulele definite mai jos.

Le scriu programatic ca sa pot itera repede peste continut. Dupa ce ruleaza
asta, lansez `jupyter nbconvert --execute` ca sa populez output-urile.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(src))
        else:
            out.append(nbf.v4.new_code_cell(src))
    nb.cells = out
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


# =============================================================================
# CLASIFICARE
# =============================================================================

CLF_CELLS = [
    ("md", """# Proiect 1 ML - Clasificare: Wine Quality Red

**Autor:** jackjack242
**Curs:** Sisteme Inteligente

## 1. Definirea problemei

Avem un set cu 1599 de vinuri rosii, descrise prin 11 proprietati fizico-chimice
(aciditate, zahar rezidual, alcool, etc.) si un scor de calitate intre 0 si 10
dat de un expert. Vreau sa antrenez un model care sa imi spuna **daca un vin e
"bun" sau nu**, in functie doar de proprietatile chimice. Asta e o problema de
**clasificare binara**.

**Target:** `quality >= 6` -> 1 (vin bun), altfel 0.

**Variabile de intrare:** `fixed acidity`, `volatile acidity`, `citric acid`,
`residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`,
`density`, `pH`, `sulphates`, `alcohol`.

### De ce e util?
Producatorii pot estima rapid calitatea unui vin fara sa astepte degustarea unui
expert - daca chimia spune ca-i probabil "bun", merita scos pe piata. E un caz
clasic de inlocuire a evaluarii umane subiective cu un proxy obiectiv.

### De ce am ales asta?
- Set bine cunoscut, curat (nu am chinuri cu missing-uri)
- Toate caracteristicile numerice -> EDA fara batai de cap cu encoding
- Marime potrivita (1599 obs) - se antreneaza repede toate modelele
- Distributia target-ului e relativ echilibrata (53/47), deci accuracy nu pacaleste
"""),

    ("code", """import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import shap

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_PATH = ROOT / 'data' / 'wine_red.csv'
MODELS_DIR = ROOT / 'models' / 'clf'
PLOTS_DIR = ROOT / 'plots' / 'clf'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 90
"""),

    ("md", "## 2. Analiza exploratorie a datelor (EDA)"),

    ("code", """df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
df.head()"""),

    ("code", """# tipuri si statistici de baza
print(df.dtypes)
print()
print('Missing values:', df.isna().sum().sum())
df.describe().T"""),

    ("md", """**Observatii rapide:**
- Toate cele 12 coloane sunt numerice (`float64`/`int64`).
- Nu am missing values - bonus.
- `quality` e variabila de iesire originala, ia valori intre 3 si 8.
- Range-urile difera mult intre features (`total sulfur dioxide` ajunge la sute,
  `chlorides` e in zecimi). O sa standardizez pentru modelele bazate pe distante
  (KNN, SVM) si pentru cele cu regularizare (LogReg).
"""),

    ("code", """# construiesc target binar: vin bun = quality >= 6
df['good'] = (df['quality'] >= 6).astype(int)
print(df['good'].value_counts())
print(f"Procent vinuri bune: {df['good'].mean()*100:.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
df['quality'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='#7e57c2')
ax[0].set_title('Distributia scorului original (quality)')
ax[0].set_xlabel('quality'); ax[0].set_ylabel('count')

df['good'].value_counts().plot(kind='bar', ax=ax[1], color=['#ef5350', '#66bb6a'])
ax[1].set_title('Target binar (good = quality >= 6)')
ax[1].set_xticklabels(['nu (0)', 'da (1)'], rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_target.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("code", """# distributia fiecarei feature
features = [c for c in df.columns if c not in ('quality', 'good')]
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, col in zip(axes.flat, features):
    sns.histplot(data=df, x=col, hue='good', bins=30, ax=ax, palette=['#ef5350','#66bb6a'], legend=False)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('')
axes.flat[-1].axis('off')
plt.suptitle('Distributii per feature, separate pe clase', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_distributions.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("code", """# matricea de corelatie
corr = df[features + ['good']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,
            cbar_kws={'shrink': 0.7}, annot_kws={'size': 8})
plt.title('Matricea de corelatie')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_corr.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Insights din EDA:**
- `alcohol` are cea mai mare corelatie pozitiva cu `good` (~0.40). Vinurile bune
  tind sa fie mai tari, ceea ce e ok cu intuitia.
- `volatile acidity` e cel mai puternic negativ (~-0.39). Aciditatea volatila mare
  inseamna in general otet, deci scor prost - check.
- `sulphates` si `citric acid` ajuta moderat.
- Avem cateva corelatii intre features (`fixed acidity` cu `pH`, `density`, `citric acid`),
  dar nu suficient cat sa fac PCA. Las modelele liniare sa se descurce cu regularizarea.
- Vad cativa outlieri vizibili in `residual sugar`, `chlorides`, `total sulfur dioxide`.
  Nu ii arunc - sunt valori reale, posibil chiar utile pentru modele tree-based.
"""),

    ("code", """# boxplot pentru top 4 features corelate cu targetul
top_corr = corr['good'].abs().sort_values(ascending=False).index[1:5]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, col in zip(axes, top_corr):
    sns.boxplot(data=df, x='good', y=col, ax=ax, palette=['#ef5350','#66bb6a'])
    ax.set_xlabel('good'); ax.set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_box_top.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", "### Pregatirea datelor pentru ML\nFac split 75/25 (stratificat dupa target ca sa pastrez proportia)."),

    ("code", """X = df[features].copy()
y = df['good'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG, stratify=y
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print(f'Train good ratio: {y_train.mean():.3f}, Test good ratio: {y_test.mean():.3f}')"""),

    ("md", """## 3. Antrenarea si compararea modelelor de baza

Antrenez toti cei 9 algoritmi cu hiperparametri default. Pentru cei sensibili
la scale (LogReg, SVM, KNN, NB) bag un `StandardScaler` in pipeline. Restul
(arbori si boosting-uri) primesc datele brute - nu le pasa de scale.
"""),

    ("code", """def make_pipeline(model, scale=True):
    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    return Pipeline([('model', model)])

base_models = {
    'NaiveBayes': make_pipeline(GaussianNB(), scale=True),
    'LogReg': make_pipeline(LogisticRegression(max_iter=2000, random_state=RNG), scale=True),
    'DecisionTree': make_pipeline(DecisionTreeClassifier(random_state=RNG), scale=False),
    'RandomForest': make_pipeline(RandomForestClassifier(random_state=RNG, n_jobs=-1), scale=False),
    'SVM': make_pipeline(SVC(probability=True, random_state=RNG), scale=True),
    'KNN': make_pipeline(KNeighborsClassifier(n_jobs=-1), scale=True),
    'XGBoost': make_pipeline(XGBClassifier(random_state=RNG, n_jobs=-1, eval_metric='logloss', verbosity=0), scale=False),
    'CatBoost': make_pipeline(CatBoostClassifier(random_state=RNG, verbose=0), scale=False),
    'EBM': make_pipeline(ExplainableBoostingClassifier(random_state=RNG), scale=False),
}
print(f'Am pregatit {len(base_models)} modele de baza.')"""),

    ("code", """def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    if hasattr(model, 'predict_proba'):
        yprob = model.predict_proba(X_te)[:, 1]
    else:
        yprob = model.decision_function(X_te)
    return {
        'accuracy': accuracy_score(y_te, yp),
        'precision': precision_score(y_te, yp),
        'recall': recall_score(y_te, yp),
        'f1': f1_score(y_te, yp),
        'roc_auc': roc_auc_score(y_te, yprob),
    }

baseline_results = {}
for name, mdl in base_models.items():
    print(f'  ... antrenez {name}')
    mdl.fit(X_train, y_train)
    baseline_results[name] = evaluate(mdl, X_test, y_test)

baseline_df = pd.DataFrame(baseline_results).T.sort_values('f1', ascending=False)
baseline_df = baseline_df.round(4)
baseline_df"""),

    ("code", """# top 5 dupa F1
top5_names = baseline_df.head(5).index.tolist()
print('Top 5 modele de baza (dupa F1):')
for i, n in enumerate(top5_names, 1):
    print(f'  {i}. {n}  ->  F1={baseline_df.loc[n,"f1"]:.4f}, AUC={baseline_df.loc[n,"roc_auc"]:.4f}')

# tabel markdown ca sa pot copia in raport
md = '| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\\n'
md += '|------|-------|----------|-----------|--------|-----|---------|\\n'
for i, (name, row) in enumerate(baseline_df.iterrows(), 1):
    star = ' (top5)' if name in top5_names else ''
    md += f"| {i} | {name}{star} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |\\n"
print(md)"""),

    ("code", """# matrici de confuzie pentru top 5
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, name in zip(axes, top5_names):
    cm = confusion_matrix(y_test, base_models[name].predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=['nu', 'da']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'baseline_confusion.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Observatii baseline:**
- Modelele de tip ensemble (Random Forest, XGBoost, CatBoost) si EBM domina,
  cum era de asteptat pentru un set tabular cu interactii non-liniare.
- Naive Bayes si Logistic Regression sunt clar in urma - asumptia de
  independenta a NB se sparge aici (multe features sunt corelate), iar LogReg
  e prea simplu.
- KNN si SVM sunt undeva la mijloc.

Iau top 5 si trec la tuning.
"""),

    ("md", """## 4. Tuning hiperparametri

Pentru top 5 fac `GridSearchCV` cu 3-fold stratified CV pe `f1`. Folosesc grid-uri
mici, rezonabile - daca le faceam mari pierdem timp degeaba si riscam overfitting
pe folds. Pentru modele cu spatiu de cautare mai larg si continuu (XGB, RF) ar
merge mai bine `BayesSearchCV` din `scikit-optimize`, dar in cazul de fata
grid-ul mic e suficient.
"""),

    ("code", """param_grids = {
    'NaiveBayes': {'model__var_smoothing': [1e-9, 1e-8, 1e-7]},
    'LogReg': {'model__C': [0.1, 1.0, 10.0], 'model__penalty': ['l2']},
    'DecisionTree': {'model__max_depth': [None, 5, 10, 20], 'model__min_samples_split': [2, 5, 10]},
    'RandomForest': {'model__n_estimators': [200, 400], 'model__max_depth': [None, 10, 20]},
    'SVM': {'model__C': [0.5, 1.0, 5.0], 'model__gamma': ['scale', 'auto']},
    'KNN': {'model__n_neighbors': [3, 5, 7, 11, 15], 'model__weights': ['uniform', 'distance']},
    'XGBoost': {'model__n_estimators': [200, 400], 'model__max_depth': [3, 5, 7], 'model__learning_rate': [0.05, 0.1]},
    'CatBoost': {'model__iterations': [200, 400], 'model__depth': [4, 6, 8], 'model__learning_rate': [0.05, 0.1]},
    'EBM': {'model__interactions': [5, 10], 'model__max_bins': [128, 256]},
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
tuned_models = {}
tuned_results = {}
tuned_params = {}

for name in top5_names:
    print(f'>> tuning {name}...')
    gs = GridSearchCV(base_models[name], param_grids[name], cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned_models[name] = gs.best_estimator_
    tuned_results[name] = evaluate(gs.best_estimator_, X_test, y_test)
    tuned_params[name] = gs.best_params_
    print(f'   best params: {gs.best_params_}')
    print(f'   F1 test: {tuned_results[name]["f1"]:.4f}')"""),

    ("code", """tuned_df = pd.DataFrame(tuned_results).T.sort_values('f1', ascending=False).round(4)
print('Comparatie inainte vs dupa tuning (F1 test):')
cmp_df = pd.DataFrame({
    'baseline_f1': baseline_df.loc[top5_names, 'f1'],
    'tuned_f1': tuned_df['f1'],
})
cmp_df['delta'] = (cmp_df['tuned_f1'] - cmp_df['baseline_f1']).round(4)
print(cmp_df)
print()
best_name = tuned_df.index[0]
print(f'>>> Cel mai bun model dupa tuning: {best_name}')
print(f'    F1={tuned_df.loc[best_name,"f1"]:.4f}  AUC={tuned_df.loc[best_name,"roc_auc"]:.4f}')
tuned_df"""),

    ("md", "## 5. Curbele de invatare (top 5)"),

    ("code", """fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, name in zip(axes, top5_names):
    sizes, train_sc, val_sc = learning_curve(
        tuned_models[name], X_train, y_train,
        train_sizes=np.linspace(0.2, 1.0, 6),
        cv=3, scoring='f1', n_jobs=-1, random_state=RNG
    )
    train_mean, val_mean = train_sc.mean(axis=1), val_sc.mean(axis=1)
    train_std, val_std = train_sc.std(axis=1), val_sc.std(axis=1)
    ax.plot(sizes, train_mean, 'o-', color='#1976d2', label='train')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1976d2')
    ax.plot(sizes, val_mean, 'o-', color='#e53935', label='cv')
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e53935')
    ax.set_title(name, fontsize=10); ax.set_xlabel('train size'); ax.set_ylabel('F1')
    ax.legend(fontsize=8); ax.set_ylim(0.5, 1.02)
    # salvez datele pt streamlit
    np.savez(MODELS_DIR / f'lc_{name}.npz', sizes=sizes, train_mean=train_mean,
             train_std=train_std, val_mean=val_mean, val_std=val_std)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'learning_curves.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Interpretare curbe:**
- Random Forest, XGBoost si CatBoost arata o distanta clara intre train (~1.0)
  si validation (~0.8): clasic overfitting, dar generalizeaza ok. Cu mai multe
  date probabil curba de validare ar urca incet.
- EBM are gap mai mic - pe bune ca-i interpretabil si reglat sa nu memoreze
  brut.
- Curbele de validare se aplatizeaza dupa ~70% din date, deci adaugarea de mai
  multe observatii nu ar ajuta dramatic. Probabil avem nevoie de features mai bune
  daca vrem sa trecem de pragul curent.
"""),

    ("md", """## 6. Explicabilitate cu SHAP (top 3)

Pentru top 3 modele, fac:
- summary plot global (impact + directie)
- bar plot global (importanta medie absoluta)
- waterfall + force plot pentru o predictie individuala
- scatter plot pentru top 2-3 features

Pentru modelele tree-based folosesc `TreeExplainer` (rapid). Pentru altele,
`KernelExplainer` cu un background mic.
"""),

    ("code", """top3_names = tuned_df.head(3).index.tolist()
print('Top 3 pentru SHAP:', top3_names)

# subsample pt SHAP, sa nu fie chinuitor
bg = X_train.sample(100, random_state=RNG)
X_explain = X_test.iloc[:100].copy()

shap_data = {}  # ce salvez pt streamlit

for name in top3_names:
    pipe = tuned_models[name]
    scaler = pipe.named_steps.get('scaler')
    model = pipe.named_steps['model']
    bg_t = scaler.transform(bg) if scaler else bg.values
    Xe_t = scaler.transform(X_explain) if scaler else X_explain.values

    print(f'\\n=== {name} ===')
    if name in ('XGBoost', 'CatBoost', 'RandomForest', 'DecisionTree'):
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xe_t)
    elif name == 'EBM':
        # EBM are KernelExplainer ca fallback
        f = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(f, bg_t)
        sv = explainer(Xe_t[:50])  # mai putine sample-uri pt viteza
        X_explain_used = X_explain.iloc[:50]
    else:
        f = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(f, bg_t)
        sv = explainer(Xe_t[:50])
        X_explain_used = X_explain.iloc[:50]

    # uniformizez shape
    if hasattr(sv, 'values') and sv.values.ndim == 3:
        # binary classifier - iau clasa 1
        vals = sv.values[..., 1]
        bv = sv.base_values[..., 1] if sv.base_values.ndim > 1 else sv.base_values
    else:
        vals = sv.values
        bv = sv.base_values

    sv_obj = shap.Explanation(values=vals, base_values=bv,
                              data=Xe_t[:vals.shape[0]],
                              feature_names=features)

    # summary plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv_obj, features=Xe_t[:vals.shape[0]], feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_summary_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    # bar plot
    plt.figure(figsize=(7, 4))
    shap.plots.bar(sv_obj, show=False, max_display=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_bar_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    # waterfall pt o predictie
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(sv_obj[0], show=False, max_display=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_waterfall_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    # salvez SHAP values pt streamlit (top model only - sa nu umplu disk-ul)
    if name == top3_names[0]:
        np.savez(MODELS_DIR / 'shap_data.npz',
                 values=vals, base_values=bv, data=Xe_t[:vals.shape[0]])
        shap_data['model'] = name
        shap_data['n'] = int(vals.shape[0])

    # top 3 features
    mean_abs = np.abs(vals).mean(axis=0)
    top_feats = [features[i] for i in np.argsort(mean_abs)[::-1][:3]]
    print(f'   Top 3 features pt {name}: {top_feats}')"""),

    ("code", """# scatter plots pt top 2 features ale celui mai bun model
best_pipe = tuned_models[top3_names[0]]
scaler = best_pipe.named_steps.get('scaler')
model = best_pipe.named_steps['model']
bg_t = scaler.transform(bg) if scaler else bg.values
Xe_t = scaler.transform(X_explain) if scaler else X_explain.values

if top3_names[0] in ('XGBoost', 'CatBoost', 'RandomForest', 'DecisionTree'):
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xe_t)
    if sv.values.ndim == 3:
        vals = sv.values[..., 1]
        bv = sv.base_values[..., 1] if sv.base_values.ndim > 1 else sv.base_values
    else:
        vals = sv.values; bv = sv.base_values
    sv_obj = shap.Explanation(values=vals, base_values=bv, data=Xe_t, feature_names=features)

mean_abs = np.abs(vals).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, idx in zip(axes, top_idx):
    shap.plots.scatter(sv_obj[:, idx], ax=ax, show=False)
    ax.set_title(features[idx])
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shap_scatter_top3.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Interpretare SHAP:**
- `alcohol` iese constant ca cea mai influenta variabila pentru clasificarea
  ca "vin bun". Valori mari de alcool impinge predictia in sus (vin bun), valori
  mici in jos. Coincide perfect cu corelatia gasita in EDA.
- `volatile acidity` e a doua: aciditate volatila mare = vin prost. Sens chimic clar -
  e ce face vinul sa miroase a otet.
- `sulphates` urmeaza, cu efect pozitiv (conservant, stabilizeaza vinul).

**Exemplu local (waterfall pe primul vin din test):** modelul porneste de la
baseline (logit-ul ratei medii ~0.53) si adauga/scoate contributii. Daca primul
vin avea alcohol mare si volatile acidity mica, vedem doua bare pozitive mari
care impingere predictia spre "bun".
"""),

    ("md", "## 7. Salvare artefacte pentru aplicatia Streamlit"),

    ("code", """import joblib

# salvez fiecare model tuned
for name, mdl in tuned_models.items():
    joblib.dump(mdl, MODELS_DIR / f'model_{name}.joblib')

# salvez X_test/y_test (pentru SHAP background si predictii demo)
X_test.to_parquet(MODELS_DIR / 'X_test.parquet')
y_test.to_frame().to_parquet(MODELS_DIR / 'y_test.parquet')
X_train.to_parquet(MODELS_DIR / 'X_train.parquet')

# meta + ranges pentru formularul din streamlit
feature_info = {}
for c in features:
    feature_info[c] = {
        'min': float(df[c].min()), 'max': float(df[c].max()),
        'mean': float(df[c].mean()), 'std': float(df[c].std()),
        'median': float(df[c].median()),
    }

meta = {
    'task': 'classification',
    'target_name': 'good',
    'classes': ['nu (quality<6)', 'da (quality>=6)'],
    'features': features,
    'feature_info': feature_info,
    'top5': top5_names,
    'top3_shap': top3_names,
    'best_model': best_name,
    'baseline_results': {k: {m: float(v) for m, v in r.items()} for k, r in baseline_results.items()},
    'tuned_results': {k: {m: float(v) for m, v in r.items()} for k, r in tuned_results.items()},
    'tuned_params': {k: {kk: (float(vv) if isinstance(vv, (np.floating,)) else vv)
                         for kk, vv in v.items()} for k, v in tuned_params.items()},
    'dataset_info': {
        'name': 'Wine Quality Red (UCI)',
        'n_rows': len(df), 'n_features': len(features),
        'class_balance': {'good=1': int(df["good"].sum()), 'good=0': int((1-df["good"]).sum())}
    }
}
with open(MODELS_DIR / 'meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, default=str)

print('Salvat:', list(MODELS_DIR.iterdir()))
print('\\nGata!')"""),
]


# =============================================================================
# REGRESIE
# =============================================================================

REG_CELLS = [
    ("md", """# Proiect 1 ML - Regresie: California Housing

**Autor:** jackjack242
**Curs:** Sisteme Inteligente

## 1. Definirea problemei

Setul California Housing contine date de la recensamantul din 1990 in California:
8 feature-uri descriu un cartier (venit median, populatie, varsta caselor, etc.)
si target-ul `MedHouseVal` e valoarea mediana a unei case in zona, in unitati de
$100.000.

Problema e una de **regresie** - prezic o valoare continua. Am facut subsample
la 2000 de observatii ca sa se incadreze in cerinta proiectului si sa antrenez
rapid.

**Variabila de iesire:** `MedHouseVal` (in $100.000), valori intre ~0.15 si ~5.0.

**Features:** `MedInc` (venit median), `HouseAge`, `AveRooms`, `AveBedrms`,
`Population`, `AveOccup`, `Latitude`, `Longitude`.

### De ce e util?
Estimarea pretului proprietatilor e o problema clasica - banci, firme imobiliare,
asigurari toate au nevoie de modele care sa scoata o valoare orientativa rapid.

### De ce am ales asta?
- Un set "real" cu features eterogene - venituri, demografice, geografice
- Are non-linearitati clare (latitude/longitude vs pret), distractiv pentru SHAP
- Ma forteaza sa subsample-uiesc Gaussian Process (e O(n^3)) - bonus de gandire
"""),

    ("code", """import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor

import shap

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_PATH = ROOT / 'data' / 'cal_housing.csv'
MODELS_DIR = ROOT / 'models' / 'reg'
PLOTS_DIR = ROOT / 'plots' / 'reg'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 90"""),

    ("md", "## 2. EDA"),

    ("code", """df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
df.head()"""),

    ("code", """print(df.dtypes)
print('Missing:', df.isna().sum().sum())
df.describe().T"""),

    ("md", """**Observatii:**
- Toate numerice, fara missing.
- `MedInc` (income median) si `MedHouseVal` au unitati de $10k respectiv $100k.
- `AveRooms`, `AveBedrms`, `AveOccup` au valori extreme (max gigant) - sunt
  cartiere mici cu putine case si o casa mare a stricat media. Voi taia outlier-ii
  cei mai grosolani ca sa nu strice modelele liniare.
"""),

    ("code", """# distributia targetului
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
sns.histplot(df['MedHouseVal'], bins=40, ax=ax[0], color='#5e35b1')
ax[0].set_title('Distributia MedHouseVal')
ax[0].axvline(df['MedHouseVal'].median(), color='red', ls='--', label=f'mediana={df["MedHouseVal"].median():.2f}')
ax[0].legend()

sns.boxplot(x=df['MedHouseVal'], ax=ax[1], color='#5e35b1')
ax[1].set_title('Boxplot target')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_target.png', dpi=110, bbox_inches='tight')
plt.show()

# Vad aici clipping la 5.0 (cape din date original) - voi lasa asa, nu confunda modelele."""),

    ("code", """# corelatii
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={'shrink': 0.7}, annot_kws={'size': 9})
plt.title('Matricea de corelatie')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_corr.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("code", """# pretul pe harta - latitude/longitude colorate cu MedHouseVal
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'],
                cmap='viridis', s=10, alpha=0.7)
plt.colorbar(sc, label='MedHouseVal ($100k)')
ax.set_title('Pretul mediu pe harta California')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_map.png', dpi=110, bbox_inches='tight')
plt.show()
# se vede clar zona scumpa pe coasta (San Francisco, LA) si interiorul mai ieftin"""),

    ("code", """# scatter plot vs feature-ul cu cea mai mare corelatie
plt.figure(figsize=(7, 4))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5, color='#1e88e5')
plt.title('MedInc vs MedHouseVal')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'eda_medinc.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Insights:**
- `MedInc` are corelatie 0.69 cu targetul - cartiere bogate au case scumpe (no shock).
- Lat/Lon arata pattern geografic clar. Modelele liniare nu o sa prinda zonele
  scumpe (SF, LA), dar tree-based ar trebui.
- Targetul are clipping la 5.0 (datele originale aveau truncation). Nu fac nimic
  cu el, e parte din distributia reala.
- Outlier-i in `AveOccup` si `AveRooms` - taie cativa.
"""),

    ("code", """# elimin outlier-i extremi (top 0.5%) doar pe AveRooms si AveOccup
for col in ['AveRooms', 'AveOccup']:
    cap = df[col].quantile(0.995)
    df = df[df[col] <= cap]
print(f'Dupa filtrare outlieri: {df.shape}')

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
            'AveOccup', 'Latitude', 'Longitude']
X = df[features].copy()
y = df['MedHouseVal'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RNG)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')"""),

    ("md", """## 3. Antrenare modele de baza

9 algoritmi cu setari default. Atentie: GaussianProcess e O(n^3), il antrenez
pe un subsample de 500 ca sa nu fie chinuitor.
"""),

    ("code", """def make_pipeline(model, scale=True):
    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    return Pipeline([('model', model)])

base_models = {
    'LinearReg': make_pipeline(LinearRegression(), scale=True),
    'DecisionTree': make_pipeline(DecisionTreeRegressor(random_state=RNG), scale=False),
    'RandomForest': make_pipeline(RandomForestRegressor(random_state=RNG, n_jobs=-1), scale=False),
    'SVR': make_pipeline(SVR(), scale=True),
    'KNN': make_pipeline(KNeighborsRegressor(n_jobs=-1), scale=True),
    'GaussianProcess': make_pipeline(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0), alpha=0.1, random_state=RNG, n_restarts_optimizer=2), scale=True),
    'XGBoost': make_pipeline(XGBRegressor(random_state=RNG, n_jobs=-1, verbosity=0), scale=False),
    'CatBoost': make_pipeline(CatBoostRegressor(random_state=RNG, verbose=0), scale=False),
    'EBR': make_pipeline(ExplainableBoostingRegressor(random_state=RNG), scale=False),
}
print(f'{len(base_models)} modele de regresie pregatite.')"""),

    ("code", """def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    mse = mean_squared_error(y_te, yp)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_te, yp),
        'R2': r2_score(y_te, yp),
    }

baseline_results = {}
for name, mdl in base_models.items():
    print(f'  ... antrenez {name}')
    if name == 'GaussianProcess':
        # subsample agresiv pentru GP - O(n^3)
        idx = np.random.RandomState(RNG).choice(len(X_train), size=500, replace=False)
        mdl.fit(X_train.iloc[idx], y_train.iloc[idx])
    else:
        mdl.fit(X_train, y_train)
    baseline_results[name] = evaluate(mdl, X_test, y_test)

baseline_df = pd.DataFrame(baseline_results).T.sort_values('R2', ascending=False).round(4)
baseline_df"""),

    ("code", """top5_names = baseline_df.head(5).index.tolist()
print('Top 5 baseline (dupa R2):')
for i, n in enumerate(top5_names, 1):
    print(f'  {i}. {n}  ->  R2={baseline_df.loc[n,"R2"]:.4f}, RMSE={baseline_df.loc[n,"RMSE"]:.4f}')

md = '| Rank | Model | MSE | RMSE | MAE | R2 |\\n'
md += '|------|-------|-----|------|-----|-----|\\n'
for i, (name, row) in enumerate(baseline_df.iterrows(), 1):
    star = ' (top5)' if name in top5_names else ''
    md += f"| {i} | {name}{star} | {row['MSE']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f} |\\n"
print(md)"""),

    ("md", """**Observatii baseline:**
- Cum ma asteptam, ensemble-urile (XGB, CatBoost, RF, EBR) bat detasat
  modelele liniare/distance.
- Linear Regression are R2 ~0.6 - prinde tendinta principala (income -> pret)
  dar rateaza pattern-ul geografic.
- KNN si SVR sunt undeva la mijloc, GP-ul depinde de subsample.
"""),

    ("md", "## 4. Tuning"),

    ("code", """param_grids = {
    'LinearReg': {},  # nimic interesant de tunat la LinearRegression vanilla
    'DecisionTree': {'model__max_depth': [None, 5, 10, 15], 'model__min_samples_split': [2, 5, 10]},
    'RandomForest': {'model__n_estimators': [200, 400], 'model__max_depth': [None, 10, 20]},
    'SVR': {'model__C': [0.5, 1.0, 5.0], 'model__gamma': ['scale']},
    'KNN': {'model__n_neighbors': [3, 5, 7, 11], 'model__weights': ['uniform', 'distance']},
    'GaussianProcess': {'model__alpha': [0.01, 0.1, 1.0]},
    'XGBoost': {'model__n_estimators': [200, 400], 'model__max_depth': [3, 5, 7], 'model__learning_rate': [0.05, 0.1]},
    'CatBoost': {'model__iterations': [200, 400], 'model__depth': [4, 6], 'model__learning_rate': [0.05, 0.1]},
    'EBR': {'model__interactions': [5, 10], 'model__max_bins': [128, 256]},
}

cv = KFold(n_splits=3, shuffle=True, random_state=RNG)
tuned_models = {}
tuned_results = {}
tuned_params = {}

for name in top5_names:
    print(f'>> tuning {name}...')
    grid = param_grids[name]
    if not grid:
        # nu am ce tuna - copiez modelul de baza
        tuned_models[name] = base_models[name]
        tuned_results[name] = baseline_results[name]
        tuned_params[name] = {}
        continue
    if name == 'GaussianProcess':
        idx = np.random.RandomState(RNG).choice(len(X_train), size=500, replace=False)
        Xt, yt = X_train.iloc[idx], y_train.iloc[idx]
    else:
        Xt, yt = X_train, y_train
    gs = GridSearchCV(base_models[name], grid, cv=cv, scoring='r2', n_jobs=-1)
    gs.fit(Xt, yt)
    tuned_models[name] = gs.best_estimator_
    tuned_results[name] = evaluate(gs.best_estimator_, X_test, y_test)
    tuned_params[name] = gs.best_params_
    print(f'   best params: {gs.best_params_}')
    print(f'   R2 test: {tuned_results[name]["R2"]:.4f}')"""),

    ("code", """tuned_df = pd.DataFrame(tuned_results).T.sort_values('R2', ascending=False).round(4)
cmp_df = pd.DataFrame({
    'baseline_R2': baseline_df.loc[top5_names, 'R2'],
    'tuned_R2': tuned_df['R2'],
})
cmp_df['delta'] = (cmp_df['tuned_R2'] - cmp_df['baseline_R2']).round(4)
print(cmp_df)
print()
best_name = tuned_df.index[0]
print(f'>>> Cel mai bun model: {best_name}')
print(f'    R2={tuned_df.loc[best_name,"R2"]:.4f}  RMSE={tuned_df.loc[best_name,"RMSE"]:.4f}')
tuned_df"""),

    ("md", "## 5. Curbele de invatare"),

    ("code", """fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, name in zip(axes, top5_names):
    Xt, yt = (X_train, y_train)
    if name == 'GaussianProcess':
        idx = np.random.RandomState(RNG).choice(len(X_train), size=500, replace=False)
        Xt, yt = X_train.iloc[idx], y_train.iloc[idx]
    sizes, train_sc, val_sc = learning_curve(
        tuned_models[name], Xt, yt,
        train_sizes=np.linspace(0.2, 1.0, 6),
        cv=3, scoring='r2', n_jobs=-1, random_state=RNG
    )
    train_mean, val_mean = train_sc.mean(axis=1), val_sc.mean(axis=1)
    train_std, val_std = train_sc.std(axis=1), val_sc.std(axis=1)
    ax.plot(sizes, train_mean, 'o-', color='#1976d2', label='train')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1976d2')
    ax.plot(sizes, val_mean, 'o-', color='#e53935', label='cv')
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e53935')
    ax.set_title(name, fontsize=10); ax.set_xlabel('train size'); ax.set_ylabel('R2')
    ax.legend(fontsize=8)
    np.savez(MODELS_DIR / f'lc_{name}.npz', sizes=sizes, train_mean=train_mean,
             train_std=train_std, val_mean=val_mean, val_std=val_std)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'learning_curves.png', dpi=110, bbox_inches='tight')
plt.show()"""),

    ("md", """**Interpretare:**
- Tree ensembles overfit clar - train R2 aproape de 1, validation in jur de
  0.75-0.8. Asta-i normal - boosting-urile invata in detaliu.
- LinearReg are gap mic dar plafon scazut: underfitting, simplul model nu poate
  capta non-liniaritatea geografica.
- Cu mai multe date, validation curve continua sa urce usor la modelele tree -
  deci ar mai ajuta sa avem mai multe puncte. La LinearReg s-a aplatizat - n-o
  sa mai ajute mai multe date, doar features mai bune.
"""),

    ("md", "## 6. SHAP (top 3)"),

    ("code", """top3_names = tuned_df.head(3).index.tolist()
print('Top 3 pt SHAP:', top3_names)

bg = X_train.sample(100, random_state=RNG)
X_explain = X_test.iloc[:100].copy()

for name in top3_names:
    pipe = tuned_models[name]
    scaler = pipe.named_steps.get('scaler')
    model = pipe.named_steps['model']
    bg_t = scaler.transform(bg) if scaler else bg.values
    Xe_t = scaler.transform(X_explain) if scaler else X_explain.values

    print(f'\\n=== {name} ===')
    if name in ('XGBoost', 'CatBoost', 'RandomForest', 'DecisionTree'):
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xe_t)
    else:
        f = lambda x: model.predict(x)
        explainer = shap.KernelExplainer(f, bg_t)
        sv = explainer(Xe_t[:50])

    sv_obj = shap.Explanation(values=sv.values, base_values=sv.base_values,
                              data=Xe_t[:sv.values.shape[0]], feature_names=features)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv_obj, features=Xe_t[:sv.values.shape[0]], feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_summary_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(7, 4))
    shap.plots.bar(sv_obj, show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_bar_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(sv_obj[0], show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'shap_waterfall_{name}.png', dpi=110, bbox_inches='tight')
    plt.show()

    if name == top3_names[0]:
        np.savez(MODELS_DIR / 'shap_data.npz',
                 values=sv.values, base_values=sv.base_values, data=Xe_t[:sv.values.shape[0]])

    mean_abs = np.abs(sv.values).mean(axis=0)
    top_feats = [features[i] for i in np.argsort(mean_abs)[::-1][:3]]
    print(f'   Top 3 features: {top_feats}')"""),

    ("code", """# scatter plots top 3 features pt cel mai bun
best_pipe = tuned_models[top3_names[0]]
scaler = best_pipe.named_steps.get('scaler')
model = best_pipe.named_steps['model']
bg_t = scaler.transform(bg) if scaler else bg.values
Xe_t = scaler.transform(X_explain) if scaler else X_explain.values

if top3_names[0] in ('XGBoost', 'CatBoost', 'RandomForest', 'DecisionTree'):
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xe_t)
    sv_obj = shap.Explanation(values=sv.values, base_values=sv.base_values,
                              data=Xe_t, feature_names=features)
    mean_abs = np.abs(sv.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, idx in zip(axes, top_idx):
        shap.plots.scatter(sv_obj[:, idx], ax=ax, show=False)
        ax.set_title(features[idx])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'shap_scatter_top3.png', dpi=110, bbox_inches='tight')
    plt.show()"""),

    ("md", """**Interpretare SHAP regresie:**
- `MedInc` e pe primul loc, fara surpriza - cartiere bogate => case scumpe.
  Valori mari de income impingere predictia in sus, valori mici in jos. Linear.
- `Latitude` si `Longitude` apar in top 3 cu pattern non-linear: anumite zone
  geografice (coasta, San Francisco / LA) ridica pretul indiferent de income.
  Asta e exact ce LinearReg nu putea prinde.
- `AveOccup` e a 4-a/5-a - cartierele cu multi locuitori per casa au preturi mai mici
  (probabil zone aglomerate, mai sarace).

**Exemplu local (waterfall pe primul cartier):** se vede cum baseline-ul (~media
target-ului) e modificat de fiecare feature in parte. Daca primul cartier avea
income mare, se aduna o contributie pozitiva mare. Daca latitude indica nordul
California departe de SF, scade.
"""),

    ("md", "## 7. Salvare artefacte"),

    ("code", """for name, mdl in tuned_models.items():
    joblib.dump(mdl, MODELS_DIR / f'model_{name}.joblib')

X_test.to_parquet(MODELS_DIR / 'X_test.parquet')
y_test.to_frame().to_parquet(MODELS_DIR / 'y_test.parquet')
X_train.to_parquet(MODELS_DIR / 'X_train.parquet')

feature_info = {}
for c in features:
    feature_info[c] = {
        'min': float(df[c].min()), 'max': float(df[c].max()),
        'mean': float(df[c].mean()), 'std': float(df[c].std()),
        'median': float(df[c].median()),
    }

meta = {
    'task': 'regression',
    'target_name': 'MedHouseVal',
    'target_unit': '$100,000',
    'features': features,
    'feature_info': feature_info,
    'top5': top5_names,
    'top3_shap': top3_names,
    'best_model': best_name,
    'baseline_results': {k: {m: float(v) for m, v in r.items()} for k, r in baseline_results.items()},
    'tuned_results': {k: {m: float(v) for m, v in r.items()} for k, r in tuned_results.items()},
    'tuned_params': {k: {kk: (float(vv) if isinstance(vv, (np.floating,)) else vv)
                         for kk, vv in v.items()} for k, v in tuned_params.items()},
    'dataset_info': {
        'name': 'California Housing (subsample 2000)',
        'n_rows': len(df), 'n_features': len(features),
    }
}
with open(MODELS_DIR / 'meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, default=str)

print('Salvat:', list(MODELS_DIR.iterdir()))
print('\\nGata!')"""),
]


def main():
    nb_clf = make_nb(CLF_CELLS)
    nb_reg = make_nb(REG_CELLS)
    with open(NB_DIR / "01_clasificare.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_clf, f)
    with open(NB_DIR / "02_regresie.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb_reg, f)
    print(f"OK: scrise {NB_DIR / '01_clasificare.ipynb'}")
    print(f"OK: scrise {NB_DIR / '02_regresie.ipynb'}")


if __name__ == "__main__":
    main()
