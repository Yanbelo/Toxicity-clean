%%writefile app.py
# =============================================================================
# Environmental Structural Alerts — Streamlit App
# - Upload .xlsx/.csv with a SMILES column (case-insensitive)
# - (Optional) Include any subset of labels:
#       "Skin sensitization", "Genotoxic", "Non-genotoxic carcinogenicity"
#   (Synonyms auto-detected; see SYNONYMS below)
# - Train 10-learner Stacking model, download .pkl
# - Predict with saved model, export Excel with predictions (+metrics if labels)
#
# Run:
#   pip install streamlit rdkit-pypi scikit-learn pandas numpy openpyxl
#   streamlit run app.py
# =============================================================================

import io, os, re, unicodedata, pickle, json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
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

# =========================
# Config
# =========================
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

# Label synonyms (normalized)
SYNONYMS = {
    "Skin sensitization": [
        "skin sensitization", "skin sensitisation", "skin-sensitization",
        "skin_sensitization", "sensitization skin", "sensitisation skin",
        "skin eye", "skin irritation", "skin allergy",
    ],
    "Genotoxic": [
        "genotoxic", "genotoxicity", "genotoxic carcinogenicity mutagenicity",
        "genotoxic carcinogenicity", "mutagenicity", "geno", "genotoxicty",
    ],
    "Non-genotoxic carcinogenicity": [
        "non genotoxic carcinogenicity", "non-genotoxic carcinogenicity",
        "nongenotoxic carcinogenicity", "non genotoxic", "non-genotoxic",
        "non genotoxicity carcinogenicity", "cancer", "carcinogenicity",
    ],
}

# =========================
# Helpers
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

def read_table_from_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.read()
    bio = io.BytesIO(data)
    if name.endswith(".xlsx"):
        return pd.read_excel(bio)
    elif name.endswith(".csv"):
        return pd.read_csv(bio)
    else:
        raise ValueError("Please upload .xlsx or .csv")

def _coerce_smiles_cell(x):
    import pandas as pd
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
    if SMILES_COL not in df.columns:
        nmap = {normalize(c): c for c in df.columns}
        if "smiles" in nmap:
            df = df.rename(columns={nmap["smiles"]: SMILES_COL})
        else:
            raise ValueError(f"Missing SMILES column. Found columns: {list(df.columns)}")
    df[SMILES_COL] = df[SMILES_COL].map(_coerce_smiles_cell)
    df = df[df[SMILES_COL].str.len() > 0].copy()
    ok = df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df.loc[ok].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid SMILES after cleaning.")
    return df

# =========================
# Features
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

def check_alerts_for_smiles(smiles: str, alerts: Dict[str, List[str]]) -> Dict[str, int]:
    res = {k: 0 for k in alerts.keys()}
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
            name_val = row[col]; break
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
    X = np.hstack([X_fp, flags_full.values, num_alerts, hazard_scores]).astype(float)

    # alerts table for display/export
    alerts_table = pd.DataFrame({
        "SMILES": df[SMILES_COL].values,
        "num_alerts": num_alerts.reshape(-1),
        "hazard_score": hazard_scores.reshape(-1),
    })
    for c in alert_df.columns:
        alerts_table[f"alert_{c}"] = alert_df[c].values
    for c in hints_df.columns:
        alerts_table[f"hint_{c}"] = hints_df[c].values
    return X, alerts_table

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

def metrics_table(y_true: np.ndarray, y_pred: np.ndarray, prob_list: List[np.ndarray], labels_used: List[str]) -> pd.DataFrame:
    rows = []
    for i, lbl in enumerate(labels_used):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ps = prob_list[i][:,1] if prob_list[i].ndim == 2 else prob_list[i]
        rows.append({
            "Endpoint": lbl,
            "Accuracy":  accuracy_score(yt, yp),
            "Precision": precision_score(yt, yp, zero_division=0),
            "Recall":    recall_score(yt, yp, zero_division=0),
            "F1":        f1_score(yt, yp, zero_division=0),
            "MCC":       matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
            "ROC_AUC":   roc_auc_score(yt, ps) if len(np.unique(yt)) > 1 else np.nan,
            "Positives": int(yt.sum()),
            "Negatives": int((1-yt).sum()),
        })
    return pd.DataFrame(rows)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Environmental Structural Alerts (Stacking)", layout="wide")
st.title("Environmental Structural Alerts — Stacking Model")
st.caption("Upload data, train a 10-learner stacking model, and predict with probabilities per endpoint.")

with st.sidebar:
    st.header("Model")
    test_size = st.slider("Test size for training split", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)
    decision_threshold = st.slider("Decision threshold (for predictions)", 0.0, 1.0, 0.50, 0.01)

    st.markdown("---")
    st.subheader("Load/Save Model")
    model_file_up = st.file_uploader("Upload model (.pkl)", type=["pkl"], key="model_up")
    loaded_model = None
    if model_file_up is not None:
        try:
            loaded_model = pickle.loads(model_file_up.read())
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

tabs = st.tabs(["🔧 Train", "🔮 Predict", "⚠️ Alerts-only"])

# ---------------- TRAIN TAB ----------------
with tabs[0]:
    st.subheader("Train a model from a labeled file")
    train_up = st.file_uploader("Upload labeled .xlsx or .csv", type=["xlsx", "csv"], key="train")
    if train_up is not None and st.button("Train model"):
        try:
            df_train = read_table_from_upload(train_up)
            df_train = ensure_smiles_ok(df_train)

            label_map = map_label_columns(df_train)  # canonical -> actual
            if len(label_map) == 0:
                st.error("No recognizable label columns found. Include at least one of the endpoints."); st.stop()
            inv = {v: k for k, v in label_map.items()}
            df_train = df_train.rename(columns=inv)
            present = [c for c in CANON_LABELS if c in df_train.columns]

            X, alerts_tbl = build_features(df_train)
            Y = df_train[present].astype(int).values

            X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE)
            model = build_stacker()
            with st.spinner("Training..."):
                model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            y_prob_list = model.predict_proba(X_te)
            mdf = metrics_table(y_te, y_pred, y_prob_list, present)

            st.success("Training complete.")
            st.dataframe(mdf, use_container_width=True)

            # Download model (.pkl)
            buf = io.BytesIO()
            pickle.dump(model, buf)
            st.download_button(
                "Download model (.pkl)",
                data=buf.getvalue(),
                file_name="model_env_stack.pkl",
                mime="application/octet-stream"
            )

            # Download metrics Excel
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as w:
                mdf.to_excel(w, sheet_name="metrics", index=False)
                pd.DataFrame({"canonical_label": list(label_map.keys()), "actual_column": list(label_map.values())}).to_excel(
                    w, sheet_name="label_mapping", index=False
                )
            st.download_button(
                "Download training metrics (.xlsx)",
                data=excel_buf.getvalue(),
                file_name="results_train.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.exception(e)

# ---------------- PREDICT TAB ----------------
with tabs[1]:
    st.subheader("Predict with a trained model")
    pred_up = st.file_uploader("Upload .xlsx or .csv (labels optional)", type=["xlsx", "csv"], key="pred")
    model_src = st.radio("Model source", ["Use model from sidebar", "Upload model here"], index=0, horizontal=True)
    model_here = None
    if model_src == "Upload model here":
        model_file_here = st.file_uploader("Model (.pkl)", type=["pkl"], key="model_here")
        if model_file_here is not None:
            try:
                model_here = pickle.loads(model_file_here.read())
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    go_pred = st.button("Run prediction")
    if go_pred:
        if pred_up is None:
            st.error("Upload a file for prediction."); st.stop()
        model_to_use = model_here if model_src == "Upload model here" else loaded_model
        if model_to_use is None:
            st.error("No model loaded. Upload a .pkl in the sidebar or in this tab."); st.stop()

        try:
            df_pred = read_table_from_upload(pred_up)
            df_pred = ensure_smiles_ok(df_pred)

            Xp, alerts_tbl = build_features(df_pred)
            y_prob_list = model_to_use.predict_proba(Xp)
            prob_mat = np.column_stack([p[:, 1] for p in y_prob_list])
            bin_preds = (prob_mat >= decision_threshold).astype(int)

            used_labels = CANON_LABELS[:prob_mat.shape[1]]
            preds_df = pd.DataFrame({"SMILES": df_pred[SMILES_COL].values})
            for j, ep in enumerate(used_labels):
                preds_df[f"prob_{ep}"] = prob_mat[:, j]
                preds_df[f"pred_{ep}"] = bin_preds[:, j]

            st.subheader("Predictions (top rows)")
            st.dataframe(preds_df.head(50), use_container_width=True)

            # If labels exist in file, compute metrics
            label_map = map_label_columns(df_pred)
            present = [c for c in used_labels if c in label_map]
            metrics_df = None
            if present:
                inv = {v: k for k, v in label_map.items()}
                dfe = df_pred.rename(columns=inv)
                Y_true = dfe[present].astype(int).values
                idxs = [used_labels.index(c) for c in present]
                y_used = bin_preds[:, idxs]
                prob_used = [y_prob_list[i] for i in idxs]
                metrics_df = metrics_table(Y_true, y_used, prob_used, present)
                st.subheader("Metrics (computed from your file)")
                st.dataframe(metrics_df, use_container_width=True)

            # Download results Excel
            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="openpyxl") as w:
                preds_df.to_excel(w, sheet_name="predictions", index=False)
                alerts_tbl.to_excel(w, sheet_name="alerts", index=False)
                if metrics_df is not None:
                    metrics_df.to_excel(w, sheet_name="metrics", index=False)
            st.download_button(
                "Download predictions (.xlsx)",
                data=out_buf.getvalue(),
                file_name="results_predict.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.exception(e)

# ---------------- ALERTS-ONLY TAB ----------------
with tabs[2]:
    st.subheader("Export SMARTS alerts & hazard score (no labels or model required)")
    alerts_up = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"], key="alerts")
    if st.button("Compute alerts"):
        if alerts_up is None:
            st.error("Upload a file."); st.stop()
        try:
            df_alert = read_table_from_upload(alerts_up)
            df_alert = ensure_smiles_ok(df_alert)
            _, alerts_tbl = build_features(df_alert)

            st.dataframe(alerts_tbl.head(50), use_container_width=True)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                alerts_tbl.to_excel(w, sheet_name="alerts", index=False)
            st.download_button(
                "Download alerts (.xlsx)",
                data=buf.getvalue(),
                file_name="alerts_only.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.exception(e)

st.markdown("---")
with st.expander("Notes / Tips"):
    st.markdown("""
- **Required column**: `SMILES` (any case; synonyms like `smiles` auto-detected).
- **Training labels (optional)**: any subset of
  - `Skin sensitization`
  - `Genotoxic`
  - `Non-genotoxic carcinogenicity`  
  Label synonyms are auto-detected; see `SYNONYMS` in the source.
- Invalid SMILES rows are **silently dropped** (not an error).
- Use the **decision threshold** in the sidebar to change how probabilities map to 0/1 predictions.
- Models are saved/loaded as **.pkl** (via `pickle`).
""")
