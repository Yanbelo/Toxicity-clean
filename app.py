# app.py — Streamlit Environmental Structural Alerts (PKL model)
# -------------------------------------------------------------------
# Features:
# - Upload labeled data to TRAIN a stacking model and download .pkl
# - Upload a .pkl model and a SMILES file to PREDICT (labels optional)
# - ECFP4 (2048) + SMARTS flags + name-hints + num_alerts + hazard_score
# - Robust SMILES cleaning; label synonym mapping
#
# Files supported: .xlsx / .csv
# Model format: .pkl (pickle)
# -------------------------------------------------------------------

import io, os, re, json, pickle, unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# ---------------------- Config ----------------------
st.set_page_config(page_title="Environmental Structural Alerts", layout="wide")

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

CANON_LABELS = [
    "Skin sensitization",
    "Genotoxic",
    "Non-genotoxic carcinogenicity",
]
SMILES_COL = "SMILES"

ALERTS_EXTENDED: Dict[str, List[str]] = {
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
HAZARD_WEIGHTS: Dict[str, float] = {
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

# Synonym mapping (normalized) for label auto-detection
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

# ---------------------- Utilities ----------------------
def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = re.sub(r"[\-_/]+", " ", s)
    s = re.sub(r"[(),:;]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def map_label_columns(df: pd.DataFrame) -> Dict[str, str]:
    norm_to_actual = {normalize(c): c for c in df.columns}
    mapping = {}
    for canon, syns in SYNONYMS.items():
        found = None
        for s in syns + [canon]:
            ns = normalize(s)
            if ns in norm_to_actual:
                found = norm_to_actual[ns]; break
        if found:
            mapping[canon] = found
    return mapping  # canonical -> actual

def _coerce_smiles_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or (hasattr(pd, "isna") and pd.isna(x)):
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            x = str(x)
    s = str(x)
    # strip ZERO WIDTH and odd quotes
    s = s.replace("\u200b", "").replace("\ufeff", "").strip().strip('"').strip("'")
    s = unicodedata.normalize("NFKC", s)
    return s

def read_table(upload_or_path) -> pd.DataFrame:
    # Handles file_uploader result (BytesIO) or disk path
    if hasattr(upload_or_path, "read"):
        name = getattr(upload_or_path, "name", "uploaded")
        data = upload_or_path.read()
        bio = io.BytesIO(data)
        if str(name).lower().endswith(".xlsx"):
            return pd.read_excel(bio)
        elif str(name).lower().endswith(".csv"):
            return pd.read_csv(bio)
        else:
            raise ValueError("Please upload .xlsx or .csv")
    else:
        path = str(upload_or_path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            return pd.read_excel(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError("Input must be .xlsx or .csv")

def ensure_smiles_ok(df: pd.DataFrame) -> pd.DataFrame:
    if SMILES_COL not in df.columns:
        nmap = {normalize(c): c for c in df.columns}
        if "smiles" in nmap:
            df = df.rename(columns={nmap["smiles"]: SMILES_COL})
        else:
            raise ValueError(f"Missing SMILES column. Found: {list(df.columns)}")
    df[SMILES_COL] = df[SMILES_COL].map(_coerce_smiles_cell)
    df = df[df[SMILES_COL].str.len() > 0].copy()

    # RDKit parseable filter
    ok = df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    bad_cnt = (~ok).sum()
    if bad_cnt:
        st.warning(f"Filtered out {int(bad_cnt)} invalid SMILES.")
    df = df.loc[ok].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid SMILES after cleaning.")
    return df

# ---------------------- Features ----------------------
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

def check_alerts_for_smiles(smiles: str, alerts: Dict[str, List[str]]) -> Dict[str, int]:
    res = {k: 0 for k in alerts.keys()}
    if not isinstance(smiles, str) or not smiles.strip():
        return res
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return res
    for name, smarts_list in alerts.items():
        for s in smarts_list:
            patt = Chem.MolFromSmarts(s)
            if patt is not None and mol.HasSubstructMatch(patt):
                res[name] = 1
                break
    return res

def name_hint_flags_from_row(row: pd.Series) -> Dict[str, int]:
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

def compute_hazard_score(flag_dict: Dict[str, int], weights: Dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) for k, v in flag_dict.items() if v)

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    X_fp = np.vstack([morgan_fp(s) for s in df[SMILES_COL].tolist()])
    alert_flags = df[SMILES_COL].apply(lambda s: check_alerts_for_smiles(s, ALERTS_EXTENDED))
    alert_df = pd.DataFrame(list(alert_flags), index=df.index)
    hints_df = pd.DataFrame([name_hint_flags_from_row(row) for _, row in df.iterrows()])
    flags_full = pd.concat([alert_df, hints_df], axis=1).fillna(0).astype(int)

    num_alerts = flags_full.sum(axis=1).values.reshape(-1, 1)
    hazard_scores = np.array([
        compute_hazard_score({c: int(row[c]) for c in flags_full.columns}, HAZARD_WEIGHTS)
        for _, row in flags_full.iterrows()
    ], dtype=float).reshape(-1, 1)

    # summary table for UI/export
    summary = []
    for i, s in enumerate(df[SMILES_COL].tolist()):
        row = flags_full.iloc[i].to_dict()
        fired = [k for k, v in row.items() if v == 1 and not k.startswith("hint_")]
        hints = [k for k, v in row.items() if v == 1 and k.startswith("hint_")]
        summary.append({
            "SMILES": s,
            "alerts_fired": ";".join(fired) if fired else "",
            "name_hints": ";".join(hints) if hints else "",
            "num_alerts": int(num_alerts[i, 0]),
            "hazard_score": float(hazard_scores[i, 0]),
            **{f"alert_{k}": int(row.get(k, 0)) for k in ALERTS_EXTENDED.keys()},
            **{f"hint_{k}": int(row.get(k, 0)) for k in NAME_HINTS.keys()},
        })
    alerts_tbl = pd.DataFrame(summary)

    X = np.hstack([X_fp, flags_full.values, num_alerts, hazard_scores]).astype(float)
    return X, alerts_tbl

# ---------------------- Model ----------------------
def define_base_learners():
    svm_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=RANDOM_STATE)),
    ])
    knn_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ])
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=RANDOM_STATE)),
    ])
    return [
        ("rf",  RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE)),
        ("et",  ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE)),
        ("gb",  GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ("ada", AdaBoostClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("bag", BaggingClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        ("svm", svm_pipe),
        ("knn", knn_pipe),
        ("mlp", mlp_pipe),
        ("gnb", GaussianNB()),
        ("dt",  DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

def build_stacker():
    base_learners = define_base_learners()
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1
    )
    return MultiOutputClassifier(stack, n_jobs=-1)

def evaluate_metrics(y_true: np.ndarray, y_prob_list: List[np.ndarray], y_pred_bin: np.ndarray, endpoints: List[str]) -> pd.DataFrame:
    rows = []
    for i, lbl in enumerate(endpoints):
        y_t = y_true[:, i]
        y_p = y_pred_bin[:, i]
        prob = y_prob_list[i][:, 1] if y_prob_list[i].ndim == 2 else y_prob_list[i]
        rows.append({
            "Endpoint": lbl,
            "Accuracy": accuracy_score(y_t, y_p),
            "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0),
            "F1": f1_score(y_t, y_p, zero_division=0),
            "MCC": matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
            "ROC_AUC": (roc_auc_score(y_t, prob) if len(np.unique(y_t)) > 1 else np.nan),
            "Positives": int(y_t.sum()),
            "Negatives": int((1 - y_t).sum()),
        })
    return pd.DataFrame(rows)

# ---------------------- Streamlit UI ----------------------
st.title("Environmental Structural Alerts + Stacking Model (.pkl)")
st.caption("Upload SMILES to fire SMARTS alerts and compute hazard score. Train or load a multi-output stacking model to predict endpoints.")

with st.sidebar:
    st.header("Model")
    uploaded_pkl = st.file_uploader("Load trained model (.pkl)", type=["pkl"])
    model_obj = None
    if uploaded_pkl is not None:
        try:
            data = uploaded_pkl.read()
            model_obj = pickle.loads(data)  # direct pickle load
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.divider()
    st.subheader("Train a new model")
    train_file = st.file_uploader("Labeled file (.xlsx or .csv)", type=["xlsx", "csv"], key="train_up")
    test_size = st.slider("Test size", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)
    do_train = st.button("Train model", use_container_width=True)

# ============== Training ==============
if do_train:
    if train_file is None:
        st.error("Please upload a labeled training file.")
    else:
        try:
            df_train = read_table(train_file)
            df_train = ensure_smiles_ok(df_train)

            label_map = map_label_columns(df_train)  # canonical -> actual
            if not label_map:
                st.error("No recognizable endpoint label columns found in the uploaded file.")
                st.stop()

            # Rename actual -> canonical for present labels
            inv = {v: k for k, v in label_map.items()}
            df_train = df_train.rename(columns=inv)

            present_eps = [c for c in CANON_LABELS if c in df_train.columns]
            if not present_eps:
                st.error("No mapped canonical labels present after renaming.")
                st.stop()

            X, alerts_tbl = build_features(df_train)
            Y = df_train[present_eps].astype(int).values

            X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE)
            with st.spinner("Training model..."):
                trained = build_stacker()
                trained.fit(X_tr, y_tr)

            # Holdout evaluation
            y_prob_list = trained.predict_proba(X_te)
            prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])
            # default decision threshold 0.5
            y_bin = (prob_mat >= 0.5).astype(int)

            metrics_df = evaluate_metrics(y_te, y_prob_list, y_bin, present_eps)
            st.success("Training complete.")
            st.subheader("Holdout metrics")
            st.dataframe(metrics_df, use_container_width=True, height=260)

            # Save model as .pkl (download)
            pkl_bytes = io.BytesIO()
            pickle.dump(trained, pkl_bytes, protocol=pickle.HIGHEST_PROTOCOL)
            st.download_button(
                "Download trained model (.pkl)",
                data=pkl_bytes.getvalue(),
                file_name="model_env_stack.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)

st.header("Predict")
tab1, tab2 = st.tabs(["File upload", "Paste SMILES"])

# ============== Prediction — File Upload ==============
with tab1:
    pred_file = st.file_uploader("SMILES file (.xlsx/.csv). Labels optional for in-file metrics.", type=["xlsx", "csv"], key="pred_up")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    if pred_file is not None:
        try:
            df_pred = read_table(pred_file)
            df_pred = ensure_smiles_ok(df_pred)
            Xp, alerts_tbl = build_features(df_pred)

            st.subheader("Alerts & Hazard (preview)")
            st.dataframe(alerts_tbl.head(30), use_container_width=True, height=360)

            if model_obj is None:
                st.info("Load a model (.pkl) in the sidebar to generate probabilities/predictions.")
                # Allow downloading just the alerts
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    alerts_tbl.to_excel(writer, sheet_name="alerts", index=False)
                st.download_button("Download alerts (.xlsx)", data=buf.getvalue(),
                                   file_name="alerts_only.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                # Predict
                y_hat = model_obj.predict(Xp)
                y_prob_list = model_obj.predict_proba(Xp)
                prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])

                y_bin = (prob_mat >= thr).astype(int)
                preds_df = pd.DataFrame({"SMILES": df_pred[SMILES_COL].values})
                n_out = prob_mat.shape[1]
                eps = CANON_LABELS[:n_out]
                for j, ep in enumerate(eps):
                    preds_df[f"prob_{ep}"] = prob_mat[:, j]
                    preds_df[f"pred_{ep}"] = y_bin[:, j]

                st.subheader("Predictions (preview)")
                st.dataframe(preds_df.head(50), use_container_width=True, height=360)

                # If labels present, compute metrics on the whole uploaded file
                label_map = map_label_columns(df_pred)
                inv = {v: k for k, v in label_map.items()}
                df_eval = df_pred.rename(columns=inv)
                present = [c for c in eps if c in df_eval.columns]
                metrics_df = None
                if present:
                    Y_true = df_eval[present].astype(int).values
                    metrics_df = evaluate_metrics(Y_true, y_prob_list, y_bin, present)
                    st.subheader("Metrics (on uploaded file)")
                    st.dataframe(metrics_df, use_container_width=True, height=260)

                # Downloadable Excel (metrics + predictions + alerts)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    if metrics_df is not None:
                        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
                    preds_df.to_excel(writer, sheet_name="predictions", index=False)
                    alerts_tbl.to_excel(writer, sheet_name="alerts", index=False)
                st.download_button("Download results (.xlsx)",
                                   data=buf.getvalue(),
                                   file_name="results.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

        except Exception as e:
            st.exception(e)

# ============== Prediction — Paste SMILES ==============
with tab2:
    st.write("Paste one SMILES per line. Optional: add a Name after a comma, e.g., `CCO, Ethanol`.")
    text = st.text_area("SMILES input", height=180, placeholder="CCN(CC)N=O\nClCCCl\nO=C(NC1=CC=CC=C1)OC")
    thr2 = st.slider("Decision threshold (text input)", 0.0, 1.0, 0.50, 0.01, key="thr2")
    go = st.button("Run alerts / Predict", use_container_width=True)
    if go and text.strip():
        try:
            smiles, names = [], []
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    continue
                if "," in s:
                    smi, nm = s.split(",", 1)
                    smiles.append(smi.strip())
                    names.append(nm.strip())
                else:
                    smiles.append(s)
                    names.append("")
            df_tmp = pd.DataFrame({SMILES_COL: smiles, "Name": names})
            df_tmp = ensure_smiles_ok(df_tmp)
            Xp, alerts_tbl = build_features(df_tmp)

            st.subheader("Alerts & Hazard")
            st.dataframe(alerts_tbl, use_container_width=True, height=300)

            if model_obj is None:
                st.info("Load a model (.pkl) in the sidebar to get probabilities/predictions.")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    alerts_tbl.to_excel(writer, sheet_name="alerts", index=False)
                st.download_button("Download alerts (.xlsx)", data=buf.getvalue(),
                                   file_name="alerts_from_text.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                y_hat = model_obj.predict(Xp)
                y_prob_list = model_obj.predict_proba(Xp)
                prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])
                y_bin = (prob_mat >= thr2).astype(int)

                preds_df = pd.DataFrame({"SMILES": smiles, "Name": names})
                n_out = prob_mat.shape[1]
                eps = CANON_LABELS[:n_out]
                for j, ep in enumerate(eps):
                    preds_df[f"prob_{ep}"] = prob_mat[:, j]
                    preds_df[f"pred_{ep}"] = y_bin[:, j]

                st.subheader("Predictions")
                st.dataframe(preds_df, use_container_width=True, height=300)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    preds_df.to_excel(writer, sheet_name="predictions", index=False)
                    alerts_tbl.to_excel(writer, sheet_name="alerts", index=False)
                st.download_button("Download predictions (.xlsx)", data=buf.getvalue(),
                                   file_name="predictions_from_text.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

        except Exception as e:
            st.exception(e)

st.divider()
with st.expander("Notes / Tips"):
    st.markdown("""
- **Model format:** This app saves & loads models as **`.pkl` (pickle)** only.
- **Labels auto-detected:** `Skin sensitization`, `Genotoxic`, `Non-genotoxic carcinogenicity` (common synonyms are mapped).
- **Features:** 2048-bit ECFP4 + SMARTS alert flags + name-hint flags + `num_alerts` + `hazard_score`.
- **Deployment tip:** RDKit wheels can be platform/Python-version specific. On Streamlit Cloud, prefer a Python version with a compatible RDKit wheel or deploy via conda (e.g., using Mamba + environment.yml).
""")
