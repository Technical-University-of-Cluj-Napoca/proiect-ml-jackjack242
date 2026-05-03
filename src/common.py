"""Functii de baza folosite in ambele notebook-uri si in app."""
from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

CLF_DIR = MODELS / "clf"
REG_DIR = MODELS / "reg"


def ensure_dirs():
    for p in [DATA, MODELS, CLF_DIR, REG_DIR, PLOTS, PLOTS / "clf", PLOTS / "reg"]:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
