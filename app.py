# app.py — Environmental Structural Alerts + Stacking Model
# Now with: model versioning, confusion matrices, and k-fold CV
# Run: streamlit run app.py

import os, io, re, json, unicodedata, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve, auc, confusion_matrix
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
import joblib

# =========================
# Config
# =========================
st.set_page_config(page_title="Env. Structural Alerts — Stacking Model", layout="wide")

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

# =========================
# Utilities / data I/O
# =========================
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
    """Return mapping canonical -> actual (best effort via synonyms)."""
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
    return mapping

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
    # detect SMILES col by normalized name if needed
    if SMILES_COL not in df.columns:
        nmap = {normalize(c): c for c in df.columns}
        if "smiles" in nmap:
            df = df.rename(columns={nmap["smiles"]: SMILES_COL})
        else:
            raise ValueError(f"Missing SMILES column. Found: {list(df.columns)}")
    # coerce/clean
    df[SMILES_COL] = df[SMILES_COL].map(_coerce_smiles_cell)
    df = df[df[SMILES_COL].str.len() > 0].copy()
    # filter valid
    ok = df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df.loc[ok].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid SMILES after cleaning.")
    return df

# =========================
# Feature builders
# =========================
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

def check_alerts(smiles: str):
    res = {k: 0 for k in ALERTS_EXTENDED}
    matched = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return res, matched
    for name, smarts_list in ALERTS_EXTENDED.items():
        for s in smarts_list:
            patt = Chem.MolFromSmarts(s)
            if patt is not None and mol.HasSubstructMatch(patt):
                res[name] = 1
                matched.append(name)
                break
    return res, matched

def name_hint_flags(name: str | None):
    out = {k: 0 for k in NAME_HINTS}
    if not name or not isinstance(name, str):
        return out
    for key, patt in NAME_HINTS.items():
        if patt.search(name): out[key] = 1
    return out

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    # optional names for hints
    name_col = None
    for c in ("Name", "Drug Name", "Compound", "CompoundName", "Compound_Name"):
        if c in df.columns: name_col = c; break

    X_fp = np.vstack([morgan_fp(s) for s in df[SMILES_COL].tolist()])

    alert_rows, hint_rows, summary = [], [], []
    for _, row in df.iterrows():
        s = row[SMILES_COL]
        flags, matched = check_alerts(s)
        hints = name_hint_flags(row[name_col]) if name_col else {k: 0 for k in NAME_HINTS}
        alert_rows.append(flags); hint_rows.append(hints)
        hz = sum(HAZARD_WEIGHTS.get(k,0.0) for k,v in {**flags, **hints}.items() if v)
        summary.append({
            "SMILES": s,
            "alerts_fired": ";".join(matched) if matched else "",
            "num_alerts": int(sum(flags.values()) + sum(hints.values())),
            "hazard_score": hz,
            **{f"alert_{k}": int(flags[k]) for k in ALERTS_EXTENDED},
            **{f"hint_{k}": int(hints[k]) for k in NAME_HINTS},
        })

    alert_df = pd.DataFrame(alert_rows).astype(int)
    hint_df  = pd.DataFrame(hint_rows).astype(int) if hint_rows else pd.DataFrame()
    if hint_df.empty:
        hint_df = pd.DataFrame(np.zeros((len(df), len(NAME_HINTS)), dtype=int), columns=list(NAME_HINTS.keys()))
    num_alerts = (alert_df.sum(axis=1) + hint_df.sum(axis=1)).values.reshape(-1,1)
    hazard_scores = np.array([r["hazard_score"] for r in summary], dtype=float).reshape(-1,1)

    X = np.hstack([X_fp, alert_df.values, hint_df.values, num_alerts, hazard_scores]).astype(float)
    alerts_tbl = pd.DataFrame(summary)
    return X, alerts_tbl

# =========================
# Model
# =========================
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
        ("bag",  BaggingClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        ("svm",  svm_pipe),
        ("knn",  knn_pipe),
        ("mlp",  mlp_pipe),
        ("gnb",  GaussianNB()),
        ("dt",   DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

def build_stacker():
    base = define_base_learners()
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    stack = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1
    )
    return MultiOutputClassifier(stack, n_jobs=-1)

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def evaluate_all(y_true: np.ndarray, y_prob_list: list[np.ndarray], y_pred: np.ndarray, labels_used: List[str]) -> pd.DataFrame:
    rows = []
    for i, lbl in enumerate(labels_used):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        prob = y_prob_list[i][:, 1] if y_prob_list[i].ndim == 2 else y_prob_list[i]
        rows.append({
            "Endpoint": lbl,
            "Accuracy": accuracy_score(y_t, y_p),
            "Precision": precision_score(y_t, y_p, zero_division=0),
            "Recall": recall_score(y_t, y_p, zero_division=0),
            "F1": f1_score(y_t, y_p, zero_division=0),
            "MCC": matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
            "ROC_AUC": safe_auc(y_t, prob),
            "Positives": int(y_t.sum()),
            "Negatives": int((1 - y_t).sum()),
        })
    return pd.DataFrame(rows)

def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['0','1'], yticklabels=['0','1'],
           ylabel='True label', xlabel='Predicted label', title=title)
    thresh = cm.max() / 2.0 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

def composite_label_for_stratify(Y: np.ndarray) -> np.ndarray:
    """
    Build a composite class label for multi-output stratification by hashing rows.
    If labels missing or single column, returns the available column.
    """
    if Y.ndim == 1 or Y.shape[1] == 1:
        return Y.reshape(-1)
    comp = (Y.astype(int).astype(str)).tolist()
    comp = ['|'.join(r) for r in comp]
    uniq = {c:i for i,c in enumerate(sorted(set(comp)))}
    return np.array([uniq[c] for c in comp], dtype=int)

def download_excel(metrics_df: pd.DataFrame | None, preds_df: pd.DataFrame | None, alerts_df: pd.DataFrame | None, meta: dict | None, fn="results.xlsx"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if meta:
            pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)
        if metrics_df is not None and not metrics_df.empty:
            metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        if preds_df is not None and not preds_df.empty:
            preds_df.to_excel(writer, sheet_name="predictions", index=False)
        if alerts_df is not None and not alerts_df.empty:
            alerts_df.to_excel(writer, sheet_name="alerts", index=False)
    st.download_button("Download results (.xlsx)", data=buf.getvalue(), file_name=fn, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# UI
# =========================
st.title("Environmental Structural Alerts — Stacking Model")
st.caption("Upload SMILES, fire SMARTS alerts, compute hazard score, and train/load a model to predict **Skin sensitization**, **Genotoxic**, and **Non-genotoxic carcinogenicity**.")

with st.sidebar:
    st.header("Model")
    model_version = st.text_input("Model version tag", value="v1.0.0")
    model_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib"], key="model_up")
    model = None
    if model_file is not None:
        try:
            model = joblib.load(io.BytesIO(model_file.read()))
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.divider()
    st.subheader("Train a new model")
    train_file = st.file_uploader("Labeled file (.xlsx/.csv)", type=["xlsx", "csv"], key="train_up")
    test_size = st.slider("Holdout test size", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)
    use_cv = st.checkbox("Use k-fold cross-validation instead of single holdout", value=False)
    k_folds = st.slider("k (folds)", 3, 10, 5, 1, disabled=not use_cv)

    if st.button("Train & Download model", use_container_width=True):
        if train_file is None:
            st.error("Please upload a labeled training file."); st.stop()
        try:
            df_train = None
            name = train_file.name.lower()
            data = train_file.read()
            bio = io.BytesIO(data)
            if name.endswith(".xlsx"):
                df_train = pd.read_excel(bio)
            elif name.endswith(".csv"):
                df_train = pd.read_csv(bio)
            else:
                st.error("Upload .xlsx or .csv"); st.stop()

            df_train = ensure_smiles_ok(df_train)
            label_map = map_label_columns(df_train)  # canonical -> actual
            if len(label_map) == 0:
                st.error("No recognizable label columns found. Add at least one of the endpoints."); st.stop()
            inv = {v: k for k, v in label_map.items()}
            df_train = df_train.rename(columns=inv)
            present = [c for c in CANON_LABELS if c in df_train.columns]

            X, alerts_tbl = build_features(df_train)
            Y = df_train[present].astype(int).values

            metrics_df = None

            if not use_cv:
                # Single holdout
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, Y, test_size=test_size, random_state=RANDOM_STATE,
                    stratify=composite_label_for_stratify(Y)
                )
                model = build_stacker()
                with st.spinner("Training model..."):
                    model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_prob_list = model.predict_proba(X_te)
                metrics_df = evaluate_all(y_te, y_prob_list, y_pred, present)

                # Confusion matrices per endpoint
                st.subheader("Confusion matrices (holdout)")
                ncols = min(3, len(present))
                rows = int(np.ceil(len(present)/ncols))
                fig, axes = plt.subplots(rows, ncols, figsize=(5*ncols, 4*rows))
                axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
                for i, lbl in enumerate(present):
                    ax = axes[i]
                    plot_confusion(ax, y_te[:, i], y_pred[:, i], title=lbl)
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')
                st.pyplot(fig, clear_figure=True)

                # ROC curves
                st.subheader("ROC curves (holdout)")
                for i, lbl in enumerate(present):
                    y_t = y_te[:, i]
                    prob = y_prob_list[i][:, 1]
                    if len(np.unique(y_t)) < 2:  # need both classes
                        st.info(f"ROC not available for {lbl} (needs both classes).")
                        continue
                    fpr, tpr, _ = roc_curve(y_t, prob)
                    roc_auc = auc(fpr, tpr)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
                    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
                    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(lbl); plt.legend(loc="lower right")
                    st.pyplot(fig, clear_figure=True)
            else:
                # k-fold CV (stratify by composite label if possible)
                st.subheader(f"{k_folds}-fold cross-validation")
                comp = composite_label_for_stratify(Y)
                try:
                    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
                    splitter = skf.split(X, comp)
                except Exception:
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
                    splitter = kf.split(X)

                all_rows = []
                fold_idx = 0
                for tr, te in splitter:
                    fold_idx += 1
                    model_cv = build_stacker()
                    with st.spinner(f"Training fold {fold_idx}/{k_folds} ..."):
                        model_cv.fit(X[tr], Y[tr])
                    y_pred = model_cv.predict(X[te])
                    y_prob_list = model_cv.predict_proba(X[te])
                    fold_metrics = evaluate_all(Y[te], y_prob_list, y_pred, present)
                    fold_metrics.insert(0, "Fold", fold_idx)
                    all_rows.append(fold_metrics)

                metrics_df = pd.concat(all_rows, ignore_index=True)
                st.dataframe(metrics_df, use_container_width=True)
                avg = metrics_df.groupby("Endpoint").agg({
                    "Accuracy":"mean","Precision":"mean","Recall":"mean","F1":"mean","MCC":"mean","ROC_AUC":"mean",
                    "Positives":"sum","Negatives":"sum"
                }).reset_index()
                st.subheader("CV macro-averages")
                st.dataframe(avg, use_container_width=True)

                # Refit final model on all data (common practice)
                model = build_stacker()
                with st.spinner("Refitting final model on all data..."):
                    model.fit(X, Y)

            # Save & download model with versioning + metadata
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_fn = f"model_env_stack_{model_version}_{ts}.joblib"
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            st.download_button("Download model (.joblib)", data=model_bytes.getvalue(),
                               file_name=model_fn, mime="application/octet-stream")

            meta = {
                "version": model_version,
                "timestamp": ts,
                "labels_used": present,
                "test_size": None if use_cv else float(test_size),
                "cv_folds": int(k_folds) if use_cv else None,
                "rdkit_fps": "ECFP4/2048",
                "flags": list(ALERTS_EXTENDED.keys()),
                "name_hints": list(NAME_HINTS.keys()),
                "random_state": RANDOM_STATE,
            }
            # Offer metrics/alerts download + meta
            download_excel(metrics_df, None, alerts_tbl, meta, fn=f"training_results_{model_version}_{ts}.xlsx")

        except Exception as e:
            st.exception(e)

st.header("Predict")
tab1, tab2 = st.tabs(["File upload", "Paste SMILES"])

with tab1:
    pred_file = st.file_uploader("Upload SMILES file (.xlsx/.csv). Labels optional.", type=["xlsx", "csv"], key="pred_up")
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    if pred_file is not None:
        try:
            name = pred_file.name.lower()
            data = pred_file.read()
            bio = io.BytesIO(data)
            if name.endswith(".xlsx"):
                df_pred = pd.read_excel(bio)
            elif name.endswith(".csv"):
                df_pred = pd.read_csv(bio)
            else:
                st.error("Upload .xlsx or .csv"); st.stop()

            df_pred = ensure_smiles_ok(df_pred)
            Xp, alerts_tbl = build_features(df_pred)

            st.subheader("Alerts & Hazard (preview)")
            st.dataframe(alerts_tbl.head(25), use_container_width=True, height=360)

            model = model  # keep reference
            if model is not None:
                try:
                    expected = model.estimators_[0].n_features_in_
                except Exception:
                    expected = None
                if expected is not None and Xp.shape[1] != expected:
                    st.error(f"Feature size mismatch: built {Xp.shape[1]} vs model expects {expected}. Ensure SMARTS/NAME_HINTS match training.")
                    st.stop()

                y_prob_list = model.predict_proba(Xp)
                prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])
                y_bin = (prob_mat >= thr).astype(int)

                preds_df = pd.DataFrame({"SMILES": df_pred[SMILES_COL].values})
                used_labels = CANON_LABELS[:y_bin.shape[1]]
                for j, ep in enumerate(used_labels):
                    preds_df[f"prob_{ep}"] = prob_mat[:, j]
                    preds_df[f"pred_{ep}"] = y_bin[:, j]

                st.subheader("Predictions")
                st.dataframe(preds_df.head(50), use_container_width=True, height=360)

                label_map = map_label_columns(df_pred)
                present = [c for c in used_labels if c in label_map]
                metrics_df = None
                if present:
                    inv = {v: k for k, v in label_map.items()}
                    dfe = df_pred.rename(columns=inv)
                    Y_true = dfe[present].astype(int).values
                    idxs = [used_labels.index(c) for c in present]
                    y_bin_used = y_bin[:, idxs]
                    y_prob_used = prob_mat[:, idxs]

                    rows = []
                    st.subheader("Confusion matrices (uploaded data)")
                    ncols = min(3, len(present))
                    rows_fig = int(np.ceil(len(present)/ncols))
                    fig, axes = plt.subplots(rows_fig, ncols, figsize=(5*ncols, 4*rows_fig))
                    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
                    for i, lbl in enumerate(present):
                        y_t = Y_true[:, i]
                        y_p = y_bin_used[:, i]
                        plot_confusion(axes[i], y_t, y_p, title=lbl)
                    for j in range(i+1, len(axes)):
                        axes[j].axis('off')
                    st.pyplot(fig, clear_figure=True)

                    for i, lbl in enumerate(present):
                        y_t = Y_true[:, i]
                        y_p = y_bin_used[:, i]
                        y_s = y_prob_used[:, i]
                        rows.append({
                            "Endpoint": lbl,
                            "Accuracy": accuracy_score(y_t, y_p),
                            "Precision": precision_score(y_t, y_p, zero_division=0),
                            "Recall": recall_score(y_t, y_p, zero_division=0),
                            "F1": f1_score(y_t, y_p, zero_division=0),
                            "MCC": matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan,
                            "ROC_AUC": roc_auc_score(y_t, y_s) if len(np.unique(y_t)) > 1 else np.nan,
                            "Positives": int(y_t.sum()),
                            "Negatives": int((1 - y_t).sum()),
                        })
                    metrics_df = pd.DataFrame(rows)
                    st.subheader("Metrics (on uploaded file)")
                    st.dataframe(metrics_df, use_container_width=True)

                    st.subheader("ROC curves (uploaded data)")
                    for i, lbl in enumerate(present):
                        y_t = Y_true[:, i]
                        y_s = y_prob_used[:, i]
                        if len(np.unique(y_t)) < 2:
                            st.info(f"ROC not available for {lbl} (needs both classes).")
                            continue
                        fpr, tpr, _ = roc_curve(y_t, y_s)
                        roc_auc = auc(fpr, tpr)
                        fig = plt.figure()
                        plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
                        plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
                        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(lbl); plt.legend(loc="lower right")
                        st.pyplot(fig, clear_figure=True)

                meta = {"inference_threshold": float(thr)}
                download_excel(metrics_df, preds_df, alerts_tbl, meta, fn="predictions_results.xlsx")
            else:
                st.info("No model loaded. Showing alerts/hazard only. Upload or train a model to get probabilities.")
                download_excel(None, df_pred[[SMILES_COL]], alerts_tbl, {"note":"alerts only"}, fn="alerts_only.xlsx")

        except Exception as e:
            st.exception(e)

with tab2:
    st.write("Paste one SMILES per line. Optional: add a Name after a comma, e.g., `CCO, Ethanol`.")
    text = st.text_area("SMILES input", height=160, placeholder="CCN(CC)N=O\nClCCCl\nO=C(NC1=CC=CC=C1)OC")
    thr2 = st.slider("Decision threshold (text input)", 0.0, 1.0, 0.50, 0.01, key="thr2")
    go = st.button("Run alerts / Predict")
    if go and text.strip():
        smiles, names = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            if "," in line:
                s, n = line.split(",", 1)
                smiles.append(s.strip()); names.append(n.strip())
            else:
                smiles.append(line); names.append("")
        df_tmp = pd.DataFrame({SMILES_COL: smiles, "Name": names})
        try:
            df_tmp = ensure_smiles_ok(df_tmp)
            Xp, alerts_tbl = build_features(df_tmp)
            st.subheader("Alerts & Hazard")
            st.dataframe(alerts_tbl, use_container_width=True, height=300)

            if model is not None:
                try:
                    expected = model.estimators_[0].n_features_in_
                except Exception:
                    expected = None
                if expected is not None and Xp.shape[1] != expected:
                    st.error(f"Feature size mismatch: built {Xp.shape[1]} vs model expects {expected}.")
                else:
                    y_prob_list = model.predict_proba(Xp)
                    prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])
                    y_bin = (prob_mat >= thr2).astype(int)
                    preds_df = pd.DataFrame({"SMILES": smiles, "Name": names})
                    used_labels = CANON_LABELS[:y_bin.shape[1]]
                    for j, ep in enumerate(used_labels):
                        preds_df[f"prob_{ep}"] = prob_mat[:, j]
                        preds_df[f"pred_{ep}"] = y_bin[:, j]
                    st.subheader("Predictions")
                    st.dataframe(preds_df, use_container_width=True, height=320)
                    download_excel(None, preds_df, alerts_tbl, {"inference_threshold": float(thr2)}, fn="predictions_from_text.xlsx")
            else:
                st.info("Upload/train a model in the sidebar to get probabilities.")
                download_excel(None, df_tmp[[SMILES_COL, "Name"]], alerts_tbl, {"note":"alerts only"}, fn="alerts_from_text.xlsx")
        except Exception as e:
            st.exception(e)

st.divider()
with st.expander("About / Notes"):
    st.markdown("""
- **Features**: 2048-bit ECFP4 + SMARTS alert flags + name-hint flags + `num_alerts` + `hazard_score`.
- **Model**: Multi-output StackingClassifier with 10 base learners (RF, ET, GB, AdaBoost, Bagging, RBF-SVM, KNN, MLP, GNB, DT) and a Logistic Regression meta-learner.
- **Training modes**: single holdout (with stratification over composite label) or **k-fold CV** (macro-averages reported), with **final refit** on all data in CV mode.
- **Versioning**: every trained model/Excel export includes your **version tag** and **timestamp** in filenames + a **meta** sheet.
- **Confusion matrices**: shown for holdout and for prediction files that include ground-truth labels.
- Keep `ALERTS_EXTENDED` and `NAME_HINTS` identical between training and prediction to avoid feature-size mismatches.
""")
