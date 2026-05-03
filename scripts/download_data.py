"""Iau cele doua dataset-uri si le pun in data/."""
from pathlib import Path
import io
import requests
import pandas as pd
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# wine quality red de pe UCI - clasificare
print("Descarc Wine Quality Red...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
r = requests.get(url, timeout=30)
r.raise_for_status()
wine = pd.read_csv(io.StringIO(r.text), sep=";")
wine.to_csv(DATA / "wine_red.csv", index=False)
print(f"  -> {wine.shape}, salvat in data/wine_red.csv")

# california housing - regresie, sample 2000 ca sa nu fie prea greu
print("Descarc California Housing...")
ch = fetch_california_housing(as_frame=True).frame
ch_sample = ch.sample(n=2000, random_state=42).reset_index(drop=True)
ch_sample.to_csv(DATA / "cal_housing.csv", index=False)
print(f"  -> {ch_sample.shape}, salvat in data/cal_housing.csv")
