# app.py — Streamlit UI for an already-saved stacking model
# -----------------------------------------------
# What it does
#   • Load trained model (.pkl or .joblib)
#   • Upload CSV/XLSX with SMILES (labels optional)
#   • Builds features exactly like training:
#       ECFP4(2048) + SMARTS flags + name-hints + num_alerts + hazard_score
#   • Predict probs + binary preds; compute metrics if labels exist
#   • Export Excel with predictions (+ metrics if available)
#
# Expected columns in uploaded data:
#   - SMILES  (case-insensitive; auto-detected & renamed)
#   - Optional name column (Name/Compound/CompoundName/Compound_Name) for hints
#   - Optional any of:
#       "Skin sensitization", "Genotoxic", "Non-genotoxic carcinogenicity"
#     (synonyms handled automatically)
#
# Tip: Start with: streamlit run app.py

import io
import os
import re
import unicodedata
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, auc
)

st.set_page_config(page_title="Environmental Structural Alerts — Predictor", layout="wide")

# ------------------------
# Config (must match training)
# ------------------------
RANDOM_STATE = 42
CANON_LABELS = [
    "Skin sensitization",
    "Genotoxic",
    "Non-genotoxic carcinogenicity",
]
SMILES_COL = "SMILES"

ALERTS_EXTENDED = {
    "aromatic_amine_primary": ["[NX3;H2][cX3;a]"],
    "aromatic_amine_secondary": ["[NX3;H1]([cX3;a])[cX3;a]"],
    "aromatic_amine_tertiary": ["[NX3;H0]([cX3;a])([cX3;a])[cX3;a]"],
    "anilide": ["[cX3;a]-[NX3]-C(=O)[#6]"],
    "organophosphorus_phosphate_ester": ["P(=O)(O)(O)", "OP(=O)(O)O"],
    "organophosphorus_phosphonate": ["P(=O)(O)[#6]"],
    "organophosphorus_phosphorothioate": ["P(=S)(O)O", "OP(=S)O"],
    "carbamate": ["O=C(OC)[NX3]", "O=C(OC)[NX3H1]", "O=C(OC)[NX3H0]"],
    "epoxide": ["[OX2r3][CX4r3][CX4r3]"],
    "azo": ["[#6]-N=N-[#6]", "c-N=N-c"],
    "aliphatic_halide": ["[CX4;H2,H1,H0][F,Cl,Br,I]"],
    "nitrosamine": ["[NX3][NX2]=O", "[NX3H1][NX2]=O"],
    "michael_acceptor": ["C=CC(=O)[O,N,S]", "[CH2]=[CH]-C(=O)O", "[CH]=C-[CX3](=O)[NX3]"],
    "quinone": ["O=C1C=CC(=O)C=CC1=O", "O=C1C=CC(=O)c2cccc12"],
    "sulfonate_ester": ["S(=O)(=O)(O)[#6]"],
}
NAME_HINTS = {"azo_like_name": re.compile(r"\b(azo|azobenz|azo\-)\b", re.IGNORECASE)}
HAZARD_WEIGHTS = {
    "nitrosamine": 3.0,
    "organophosphorus_phosphorothioate": 2.5,
    "organophosphorus_phosphate_ester": 2.0,
    "organophosphorus_phosphonate": 2.0,
    "azo": 2.0,
    "epoxide": 2.0,
    "michael_acceptor": 1.5,
    "quinone": 1.5,
    "sulfonate_ester": 1.2,
    "aliphatic_halide": 1.2,
    "carbamate": 1.0,
    "aromatic_amine_primary": 1.0,
    "aromatic_amine_secondary": 1.0,
    "aromatic_amine_tertiary": 0.8,
    "anilide": 0.8,
    "azo_like_name": 0.5,
}

# Label synonyms for auto-detect
SYNONYMS = {
    "Skin sensitization": [
        "skin sensitization", "skin sensitisation", "skin-sensitization",
        "skin_sensitization", "sensitization skin", "sensitisation skin",
        "skin eye", "skin irritation", "skin allergy"
    ],
    "Genotoxic": [
        "genotoxic", "genotoxicity", "genotoxic carcinogenicity mutagenicity",
        "genotoxic carcinogenicity", "mutagenicity", "geno", "genotoxicty"
    ],
    "Non-genotoxic carcinogenicity": [
        "non genotoxic carcinogenicity", "non-genotoxic carcinogenicity",
        "nongenotoxic carcinogenicity", "non genotoxic", "non-genotoxic",
        "non genotoxicity carcinogenicity", "cancer", "carcinogenicity"
    ],
}

# ------------------------
# Helpers (I/O & cleaning)
# ------------------------
def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = re.sub(r"[\-_/]+", " ", s)
    s = re.sub(r"[(),:;]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def map_label_columns(df: pd.DataFrame) -> dict:
    norm_to_actual = {normalize(c): c for c in df.columns}
    mapping = {}
    for canon, syns in SYNONYMS.items():
        found = None
        for s in syns + [canon]:
            ns = normalize(s)
            if ns in norm_to_actual:
                found = norm_to_actual[ns]
                break
        if found:
            mapping[canon] = found
    return mapping  # canonical -> actual

def read_table(upload) -> pd.DataFrame:
    if upload is None:
        return None
    name = upload.name.lower()
    data = upload.read()
    bio = io.BytesIO(data)
    if name.endswith(".xlsx"):
        return pd.read_excel(bio)
    elif name.endswith(".csv"):
        return pd.read_csv(bio)
    else:
        raise ValueError("Please upload .xlsx or .csv")

def _coerce_smiles_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or (hasattr(pd, "isna") and pd.isna(x)):
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            x = str(x)
    s = str(x)
    s = s.replace("\u200b", "").replace("\ufeff", "").strip().strip('"').strip("'")
    s = unicodedata.normalize("NFKC", s)
    return s

def ensure_smiles_ok(df: pd.DataFrame) -> pd.DataFrame:
    # detect SMILES column by normalized name if needed
    if SMILES_COL not in df.columns:
        nmap = {normalize(c): c for c in df.columns}
        if "smiles" in nmap:
            df = df.rename(columns={nmap["smiles"]: SMILES_COL})
        else:
            raise ValueError(f"Missing SMILES column. Found columns: {list(df.columns)}")

    # coerce to strings and clean
    df[SMILES_COL] = df[SMILES_COL].map(_coerce_smiles_cell)
    df = df[df[SMILES_COL].str.len() > 0].copy()

    # filter valid SMILES
    ok = df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df.loc[ok].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid SMILES after cleaning. Check your input file.")
    return df

# ------------------------
# Feature builders (must match training)
# ------------------------
def morgan_fp(smiles: str, radius=2, nBits=2048):
    if not isinstance(smiles, str) or not smiles:
        return np.zeros((nBits,), dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nBits,), dtype=int)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    ConvertToNumpyArray(bv, arr)
    return arr

def check_alerts_for_smiles(smiles: str) -> dict:
    res = {k: 0 for k in ALERTS_EXTENDED.keys()}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return res
    for name, smarts_list in ALERTS_EXTENDED.items():
        for s in smarts_list:
            patt = Chem.MolFromSmarts(s)
            if patt is not None and mol.HasSubstructMatch(patt):
                res[name] = 1
                break
    return res

def name_hint_flags_from_row(row: pd.Series) -> dict:
    out = {k: 0 for k in NAME_HINTS.keys()}
    name_val = ""
    for col in ("Name", "Drug Name", "Compound", "CompoundName", "Compound_Name"):
        if col in row and isinstance(row[col], str):
            name_val = row[col]
            break
    for key, pattern in NAME_HINTS.items():
        if name_val and pattern.search(name_val):
            out[key] = 1
    return out

def compute_hazard_score(flag_dict: dict) -> float:
    return sum(HAZARD_WEIGHTS.get(k, 0.0) for k, v in flag_dict.items() if v)

def build_features_and_alert_table(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    # ECFP4
    X_fp = np.vstack([morgan_fp(s) for s in df[SMILES_COL].tolist()])

    # SMARTS flags + name hints
    alert_flags = df[SMILES_COL].apply(check_alerts_for_smiles)
    alert_df = pd.DataFrame(list(alert_flags), index=df.index).astype(int)
    hints_df = pd.DataFrame([name_hint_flags_from_row(row) for _, row in df.iterrows()]).astype(int)

    flags_full = pd.concat([alert_df, hints_df], axis=1).fillna(0).astype(int)
    num_alerts = flags_full.sum(axis=1).values.reshape(-1, 1)
    hazard_scores = np.array([
        compute_hazard_score({c: int(row[c]) for c in flags_full.columns})
        for _, row in flags_full.iterrows()
    ], dtype=float).reshape(-1, 1)

    X = np.hstack([X_fp, flags_full.values, num_alerts, hazard_scores]).astype(float)

    # nice alert summary table
    alerts_summary = []
    for i, s in enumerate(df[SMILES_COL].tolist()):
        flags = alert_df.iloc[i].to_dict()
        hints = hints_df.iloc[i].to_dict() if not hints_df.empty else {}
        matched = [k for k, v in flags.items() if int(v) == 1]
        hz = compute_hazard_score({**flags, **hints})
        alerts_summary.append({
            "SMILES": s,
            "alerts_fired": ";".join(matched) if matched else "",
            "num_alerts": int(sum(flags.values()) + sum(hints.values())),
            "hazard_score": hz,
            **{f"alert_{k}": int(flags[k]) for k in alert_df.columns},
            **{f"hint_{k}": int(hints.get(k, 0)) for k in NAME_HINTS.keys()}
        })
    alerts_tbl = pd.DataFrame(alerts_summary)
    return X, alerts_tbl

# ------------------------
# Evaluation helpers
# ------------------------
def evaluate_if_labels(df_pred: pd.DataFrame, prob_list: list[np.ndarray], y_bin: np.ndarray) -> pd.DataFrame:
    mapping = map_label_columns(df_pred)  # canonical -> actual
    present = [c for c in CANON_LABELS if c in mapping]  # which endpoints exist in file (by synonyms)
    if not present:
        return pd.DataFrame()

    # rename to canonical for evaluation
    inv = {v: k for k, v in mapping.items()}
    dfe = df_pred.rename(columns=inv)

    rows = []
    prob_mat = np.column_stack([p[:, 1] for p in prob_list])
    for i, lbl in enumerate(present):
        if lbl not in dfe.columns:
            continue
        y_true = dfe[lbl].astype(int).values
        y_prob = prob_mat[:, i]
        y_hat  = y_bin[:, i]
        rows.append({
            "Endpoint": lbl,
            "Accuracy":  accuracy_score(y_true, y_hat),
            "Precision": precision_score(y_true, y_hat, zero_division=0),
            "Recall":    recall_score(y_true, y_hat, zero_division=0),
            "F1":        f1_score(y_true, y_hat, zero_division=0),
            "MCC":       matthews_corrcoef(y_true, y_hat) if len(np.unique(y_true)) > 1 else np.nan,
            "ROC_AUC":   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
            "Positives": int(y_true.sum()),
            "Negatives": int((1 - y_true).sum()),
        })
    return pd.DataFrame(rows)

# ------------------------
# UI
# ------------------------
st.title("🧪 Environmental Structural Alerts — Predictor")

with st.sidebar:
    st.header("1) Load your saved model")
    mod_file = st.file_uploader("Model file (.pkl or .joblib)", type=["pkl", "joblib"])
    model = None
    if mod_file:
        try:
            byts = mod_file.read()
            # try pickle first, then joblib
            try:
                model = pickle.loads(byts)
            except Exception:
                model = joblib.load(io.BytesIO(byts))
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.header("2) Decision threshold")
    thr = st.slider("Threshold for positive class", 0.0, 1.0, 0.50, 0.01)

st.header("3) Upload data for prediction")
data_file = st.file_uploader("Upload .xlsx or .csv with SMILES (labels optional)", type=["xlsx", "csv"])

if data_file is not None:
    try:
        df_pred = read_table(data_file)
        df_pred = ensure_smiles_ok(df_pred)

        st.subheader("Preview (top 10)")
        st.dataframe(df_pred.head(10), use_container_width=True)

        # Build features + alert table (always available)
        Xp, alerts_tbl = build_features_and_alert_table(df_pred)

        st.subheader("Alert flags & Hazard (top 20)")
        st.dataframe(alerts_tbl.head(20), use_container_width=True, height=360)

        if model is None:
            st.info("Upload a trained model in the sidebar to generate probabilities/predictions.")
        else:
            # compatibility check (when available)
            try:
                expected = model.estimators_[0].n_features_in_
            except Exception:
                expected = None
            if expected is not None and Xp.shape[1] != expected:
                st.error(f"Feature size mismatch: built {Xp.shape[1]} vs model expects {expected}. "
                         "Ensure SMARTS/NAME_HINTS/order match your training config.")
            else:
                # predict
                y_hat_raw = model.predict(Xp)
                proba_list = model.predict_proba(Xp)
                prob_mat = np.column_stack([p[:, 1] for p in proba_list])

                # allow custom threshold
                y_bin = (prob_mat >= thr).astype(int)

                # infer #outputs (some models trained with subset of endpoints)
                n_outputs = prob_mat.shape[1]
                eps = CANON_LABELS[:n_outputs]

                preds_df = pd.DataFrame({"SMILES": df_pred[SMILES_COL].values})
                for j, ep in enumerate(eps):
                    preds_df[f"prob_{ep}"] = prob_mat[:, j]
                    preds_df[f"pred_{ep}"] = y_bin[:, j]

                st.subheader("Predictions (top 50)")
                st.dataframe(preds_df.head(50), use_container_width=True, height=360)

                # metrics if labels present
                st.subheader("Metrics (if labels exist in file)")
                metrics_df = evaluate_if_labels(df_pred.copy(), proba_list, y_bin)
                if metrics_df.empty:
                    st.info("No ground-truth labels detected (or names didn’t match).")
                else:
                    st.dataframe(metrics_df, use_container_width=True)

                # download Excel
                st.subheader("Download results")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    if not metrics_df.empty:
                        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
                    preds_df.to_excel(writer, sheet_name="predictions", index=False)
                    alerts_tbl.to_excel(writer, sheet_name="alerts", index=False)
                st.download_button(
                    "Download results (.xlsx)",
                    data=buf.getvalue(),
                    file_name="predictions_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.exception(e)

st.divider()
with st.expander("Notes"):
    st.markdown("""
- This app expects a model trained with **the same feature definition**:
  2048-bit **ECFP4** + **SMARTS** alert flags + **name-hint** flags + **num_alerts** + **hazard_score** (in that order).
- If you see **feature size mismatch**, ensure your `ALERTS_EXTENDED` / `NAME_HINTS` sets and ordering match training.
- Labels (if present) are auto-detected via synonyms for:
  `Skin sensitization`, `Genotoxic`, `Non-genotoxic carcinogenicity`.
""")
