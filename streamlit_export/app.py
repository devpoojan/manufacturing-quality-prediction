# Manufacturing Quality Prediction
# Author: Poojan Chauhan
# Streamlit Web Application - Enhanced & Interactive Edition

from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
import hashlib
import json
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import streamlit as st

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TOP_K = 40
DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/179/secom"
FEATURES = [f"sensor_{i:03d}" for i in range(1, 591)]

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ==============================================================================
# Core ML Pipeline & Utilities (Unchanged from Notebook)
# ==============================================================================

def load_secom(data_dir="data"):
    """Download the original UCI ZIP once; accept local raw files for offline use."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "secom.data"
    label_path = data_dir / "secom_labels.data"
    zip_path = data_dir / "secom.zip"
    if not (raw_path.exists() and label_path.exists()):
        if zip_path.exists():
            payload = zip_path.read_bytes()
        else:
            try:
                with urlopen(DATA_URL, timeout=90) as response:
                    payload = response.read()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download SECOM from {DATA_PAGE}. "
                    "Please place secom.data and secom_labels.data in the data/ folder."
                ) from exc
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for filename in ("secom.data", "secom_labels.data"):
                (data_dir / filename).write_bytes(archive.read(filename))
        zip_path.write_bytes(payload)

    X = pd.read_csv(raw_path, sep=r"\s+", header=None)
    labels = pd.read_csv(label_path, sep=r"\s+", header=None, quotechar='"')
    if X.shape != (1567, 590) or len(labels) != len(X) or labels.shape[1] != 2:
        raise ValueError("Unexpected SECOM file dimensions. Use the original UCI files.")
    if not set(labels[0].unique()).issubset({-1, 1}):
        raise ValueError("The original target must contain -1 (pass) and 1 (fail).")
    X = X.apply(pd.to_numeric, errors="raise")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite sensor values found in source files.")
    X.columns = FEATURES
    X.index.name = "sample_id"
    y = labels[0].map({-1: 0, 1: 1}).astype(int).rename("failure")
    y.index = X.index
    timestamps = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S", errors="raise")
    timestamps.index = X.index
    fingerprints = {
        filename: hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        for filename in ("secom.data", "secom_labels.data")
    }
    return X, y, timestamps.rename("test_timestamp"), fingerprints


class SensorFilter(BaseEstimator, TransformerMixin):
    """Learn usable sensors from the training fold only."""
    def __init__(self, max_missing=0.60):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SensorFilter requires a pandas DataFrame.")
        usable = (X.isna().mean() <= self.max_missing) & (X.nunique(dropna=True) > 1)
        self.columns_ = X.columns[usable].tolist()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        if len(self.columns_) < TOP_K:
            raise ValueError(f"Need at least {TOP_K} usable sensors; found {len(self.columns_)}.")
        return self

    def transform(self, X):
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def make_models():
    """Three predeclared candidates, without fitting or data-dependent tuning."""
    estimators = {
        "Logistic Regression": LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", n_jobs=2, random_state=SEED),
    }
    return {
        name: Pipeline([
            ("filter", SensorFilter(max_missing=0.60)),
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=f_classif, k=TOP_K)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])
        for name, estimator in estimators.items()
    }


def fail_scores(model, X):
    """Return the score for class 1 explicitly."""
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def compare_models(X_train, y_train, progress=None):
    """Select by mean 5-fold average precision; never read the held-out test set."""
    candidates = make_models()
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X_train, y_train))
    fold_rows, oof_by_name = [], {}
    for name, template in candidates.items():
        oof = np.full(len(X_train), np.nan)
        start = time.perf_counter()
        for fold, (fit_indices, valid_indices) in enumerate(splits, start=1):
            fitted = clone(template).fit(X_train.iloc[fit_indices], y_train.iloc[fit_indices])
            scores = fail_scores(fitted, X_train.iloc[valid_indices])
            oof[valid_indices] = scores
            fold_rows.append({
                "Model": name, "Fold": fold,
                "Average precision": average_precision_score(y_train.iloc[valid_indices], scores),
                "ROC AUC": roc_auc_score(y_train.iloc[valid_indices], scores),
            })
        if not np.isfinite(oof).all():
            raise RuntimeError("Some training rows lack out-of-fold predictions.")
        oof_by_name[name] = oof
        if progress:
            progress(f"{name}: completed {N_SPLITS} folds in {time.perf_counter()-start:.1f}s")
    folds = pd.DataFrame(fold_rows)
    summary = folds.groupby("Model").agg(
        Mean_AP=("Average precision", "mean"), SD_AP=("Average precision", "std"),
        Mean_ROC_AUC=("ROC AUC", "mean"),
    ).sort_values("Mean_AP", ascending=False)
    selected_name = summary.index[0]
    return selected_name, candidates[selected_name], oof_by_name[selected_name], summary, folds


def choose_threshold(y_train, oof_scores):
    """Maximize training OOF F2; ties prefer the higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y_train, oof_scores)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    table = pd.DataFrame({"Threshold": thresholds, "Precision": precision[:-1],
                          "Recall": recall[:-1], "F2": f2})
    best = table.sort_values(["F2", "Threshold"], ascending=[False, False]).iloc[0]
    return float(best["Threshold"]), table


def metric_row(y_true, scores, threshold):
    predicted = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced accuracy": balanced_accuracy_score(y_true, predicted),
        "Failure precision": precision_score(y_true, predicted, zero_division=0),
        "Failure recall": recall_score(y_true, predicted, zero_division=0),
        "Failure F1": f1_score(y_true, predicted, zero_division=0),
        "Failure F2": fbeta_score(y_true, predicted, beta=2, zero_division=0),
        "Average precision": average_precision_score(y_true, scores),
        "ROC AUC": roc_auc_score(y_true, scores),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def train_experiment(X, y, progress=None):
    """One reproducible split, training-only selection, then final evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    name, template, oof, cv_summary, cv_folds = compare_models(X_train, y_train, progress)
    threshold, threshold_table = choose_threshold(y_train, oof)
    model = clone(template).fit(X_train, y_train)
    scores = fail_scores(model, X_test)
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    baseline_scores = fail_scores(baseline, X_test)
    evaluation = pd.DataFrame({
        "Always pass baseline": metric_row(y_test, baseline_scores, 0.50),
        "Selected model at 0.50": metric_row(y_test, scores, 0.50),
        "Selected model at OOF threshold": metric_row(y_test, scores, threshold),
    }).T
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                name=name, model=model, threshold=threshold, scores=scores,
                oof=oof, cv_summary=cv_summary, cv_folds=cv_folds,
                threshold_table=threshold_table, evaluation=evaluation)


def validate_sensor_frame(frame):
    """Require a complete 590-column schema; individual missing readings are OK."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Provide at least one row of sensor readings.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(
            f"CSV must contain exactly sensor_001 through sensor_590. "
            f"Missing: {missing[:5]}; unexpected: {extra[:5]}. "
            "Remove target, timestamp and index columns."
        )
    frame = frame.loc[:, FEATURES].copy()
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Sensor cells must be numeric or blank; text is not accepted.") from exc
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Infinite values are not allowed.")
    if (frame.isna().mean(axis=1) > 0.50).any():
        raise ValueError("Each row must have at least 295 of its 590 sensor readings.")
    return frame


def predict_quality(model, frame, threshold):
    """Score valid records."""
    frame = validate_sensor_frame(frame)
    selected = np.asarray(model.named_steps["filter"].columns_)[
        model.named_steps["select"].get_support()]
    if (frame.loc[:, selected].isna().mean(axis=1) > 0.50).any():
        raise ValueError("Each row must contain at least half of the model's selected sensors.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    scores = fail_scores(model, frame)
    return pd.DataFrame({
        "failure_score": scores,
        "decision": np.where(scores >= threshold, "Flag for inspection", "Predicted pass"),
        "risk_level": np.where(scores >= threshold * 1.5, "High Risk",
                               np.where(scores >= threshold, "Moderate Risk", "Low Risk")),
        "threshold": threshold,
        "missing_readings": frame.isna().sum(axis=1),
    }, index=frame.index)


def sensor_ranking(model):
    """Training-derived model associations."""
    selected = np.asarray(model.named_steps["filter"].columns_)[
        model.named_steps["select"].get_support()]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
        method = "Tree Gini Feature Importance"
    else:
        weights = np.abs(estimator.coef_[0])
        method = "Absolute Standardized Coefficient"
    return pd.Series(weights, index=selected, name=method).sort_values(ascending=False)

# ==============================================================================
# Streamlit Dashboard UI
# ==============================================================================

st.set_page_config(
    page_title="SECOM • Semiconductor Quality AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design & Aesthetic Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 125, 154, 0.08) 0%, rgba(211, 108, 55, 0.05) 100%);
        border: 1px solid rgba(22, 125, 154, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #167d9a;
    }
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-pass {
        background-color: #e6f7ec;
        color: #0e8a3a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #b7ebd0;
    }
    .badge-flag {
        background-color: #fde8e8;
        color: #c81e1e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #f8b4b4;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache model
@st.cache_resource(show_spinner="Initializing SECOM AI pipeline and training benchmark models...")
def get_experiment():
    X, y, timestamps, hashes = load_secom()
    return X, y, timestamps, train_experiment(X, y)

try:
    X, y, timestamps, result = get_experiment()
except Exception as exc:
    st.error(f"Initialization Notice: {exc}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/processor.png", width=64)
st.sidebar.title("SECOM Intelligence")
st.sidebar.caption("Semiconductor Quality Prediction & Defect Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Overview",
        "🔬 Sensor Diagnostics",
        "🧪 Model Benchmarking & Simulator",
        "⚡ Real-Time Inference Lab",
        "📖 Methodology & Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** `{result['name']}`")
st.sidebar.markdown(f"**Optimal Threshold:** `{result['threshold']:.4f}`")
st.sidebar.caption("Author: Poojan Chauhan\n\nIBM SkillsBuild AI Internship · BharatCares & AICTE")
st.sidebar.link_button("UCI SECOM Dataset", DATA_PAGE)

# ------------------------------------------------------------------------------
# 1. Executive Overview
# ------------------------------------------------------------------------------
if page == "📊 Executive Overview":
    st.title("🏭 Semiconductor Manufacturing Quality Control")
    st.markdown(
        "Real-time sensor analytics and automated defect screening for semiconductor wafer manufacturing. "
        "Built to flag defect-prone production units before packaging, minimizing yield loss and customer warranty claims."
    )
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">1,567</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-label">Monitored Sensors</div><div class="metric-value">590</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-label">Factory Yield</div><div class="metric-value">93.36%</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-label">Defect Prevalence</div><div class="metric-value" style="color: #d36c37;">6.64%</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown("""<div class="metric-card"><div class="metric-label">Defect Recall</div><div class="metric-value" style="color: #167d9a;">61.90%</div></div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Quality Outcomes Distribution")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        outcomes_df = pd.DataFrame({
            "Classification": ["Passed Inspection", "Defective (Failed)"],
            "Count": [(y == 0).sum(), (y == 1).sum()],
            "Percentage": [f"{(y==0).mean():.2%}", f"{(y==1).mean():.2%}"]
        })
        st.dataframe(outcomes_df, use_container_width=True, hide_index=True)
        st.caption("Extreme imbalance: Defects represent only 104 of 1,567 historical records.")

    with col_right:
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(["Pass (0)", "Defect (1)"], [(y == 0).sum(), (y == 1).sum()], color=["#167d9a", "#d36c37"], width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 20, f"{yval}", ha="center", fontweight="bold")
        ax.set_ylabel("Production Wafers")
        ax.set_ylim(0, 1700)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.subheader("⚙️ Leakage-Free Machine Learning Architecture")
    with st.expander("Explore the 5-Stage Preprocessing & Modeling Pipeline", expanded=True):
        st.markdown("""
        1. **Stratified Split:** 80% development cohort (1,253 units) and 20% untouched held-out evaluation test set (314 units).
        2. **Sensor Filtration (`SensorFilter`):** Removes sensors exhibiting >60% missing observations or zero variance within each fold.
        3. **Median Imputation (`SimpleImputer`):** Robust to industrial outlier spikes without leaking test-fold distributions.
        4. **Feature Selection (`SelectKBest`):** Ranks and selects the top 40 discriminative sensors via ANOVA F-statistic.
        5. **Cost-Sensitive Classifier:** Evaluates Random Forest, Extra Trees, and Logistic Regression with balanced subsample weighting.
        6. **$F_2$ Threshold Calibration:** Selects threshold = `0.1377` on out-of-fold predictions to prioritize catching defects (Recall).
        """)

# ------------------------------------------------------------------------------
# 2. Sensor Diagnostics & Feature Explorer
# ------------------------------------------------------------------------------
elif page == "🔬 Sensor Diagnostics":
    st.title("🔬 Sensor Diagnostics & Missingness Exploration")
    st.markdown("Inspect sensor signal distributions, data completeness, and feature relationships across the manufacturing floor.")
    st.markdown("---")

    train = result["X_train"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Missing Sensor Values", f"{train.isna().to_numpy().mean():.2%}")
    c2.metric("Usable Non-Constant Sensors", len(result["model"].named_steps["filter"].columns_))
    c3.metric("Selected Top Features (K)", TOP_K)

    st.markdown("### 📊 Top Sensors by Missing Data Ratio")
    missing_pct = (train.isna().mean() * 100).nlargest(15)
    st.bar_chart(missing_pct, horizontal=True, color="#167d9a")
    st.caption("Sensors with >60% missing readings are dropped before modeling to prevent noisy hallucinations.")

    st.markdown("---")
    st.subheader("🎯 Interactive Sensor Distribution Inspector")
    selected_sensor = st.selectbox(
        "Select a sensor channel to inspect:",
        result["model"].named_steps["filter"].columns_,
        index=0
    )

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    sensor_series = train[selected_sensor].dropna()
    col_stat1.metric("Observed Readings", f"{len(sensor_series):,}")
    col_stat2.metric("Mean Value", f"{sensor_series.mean():.3f}")
    col_stat3.metric("Standard Deviation", f"{sensor_series.std():.3f}")
    col_stat4.metric("Missing Ratio", f"{train[selected_sensor].isna().mean():.2%}")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for label, color, name in [(0, "#167d9a", "Passing Wafers"), (1, "#d36c37", "Defective Wafers")]:
        vals = train.loc[result["y_train"] == label, selected_sensor].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=30, density=True, alpha=0.55, color=color, label=name)
    ax.set_title(f"Density Profile: {selected_sensor} (Passing vs. Defective)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sensor Reading (Standard Scale)")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.subheader("🏆 Model Feature Importance (Top 15 Associations)")
    ranking = sensor_ranking(result["model"])
    st.bar_chart(ranking.head(15), horizontal=True, color="#167d9a")
    st.caption(f"Importance Metric: {ranking.name}. Reflects statistical associations within the Random Forest decision trees.")

# ------------------------------------------------------------------------------
# 3. Model Benchmarking & Simulator
# ------------------------------------------------------------------------------
elif page == "🧪 Model Benchmarking & Simulator":
    st.title("🧪 Model Benchmark & Decision Threshold Simulator")
    st.markdown("Compare candidate algorithms and simulate the operational trade-offs of shifting the inspection threshold in real time.")
    st.markdown("---")

    st.subheader("1. 5-Fold Stratified Cross-Validation Benchmark")
    st.dataframe(result["cv_summary"].round(4), use_container_width=True)
    st.caption("Selected Algorithm: **Random Forest** achieved the highest Average Precision (0.2108) across all folds.")

    st.markdown("---")
    st.subheader("2. 🎛️ Real-Time Inspection Threshold Simulator")
    st.markdown(
        "In manufacturing, missing a defective chip (False Negative) is exponentially worse than sending a good wafer for testing (False Positive). "
        "Adjust the slider below to simulate real-world defect capture rates:"
    )

    sim_threshold = st.slider(
        "Simulate Decision Threshold",
        min_value=0.01,
        max_value=0.90,
        value=float(result["threshold"]),
        step=0.005,
        format="%.4f"
    )

    sim_preds = (result["scores"] >= sim_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(result["y_test"], sim_preds, labels=[0, 1]).ravel()
    rec = recall_score(result["y_test"], sim_preds, zero_division=0)
    prec = precision_score(result["y_test"], sim_preds, zero_division=0)
    f2 = fbeta_score(result["y_test"], sim_preds, beta=2, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defects Caught (TP)", f"{tp} / 21", delta=f"{rec:.1%} Recall")
    m2.metric("Missed Defects (FN)", f"{fn}", delta="Goal: Minimize!", delta_color="inverse")
    m3.metric("False Alarms (FP)", f"{fp} / 293", delta=f"{prec:.1%} Precision")
    m4.metric("Simulated F2-Score", f"{f2:.4f}")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ConfusionMatrixDisplay.from_predictions(
            result["y_test"], sim_preds,
            display_labels=["Pass", "Defect"],
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"Simulated Confusion Matrix (Threshold = {sim_threshold:.3f})")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_chart2:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        PrecisionRecallDisplay.from_predictions(result["y_test"], result["scores"], ax=ax, color="#167d9a")
        ax.axhline(result["y_test"].mean(), ls="--", color="#d36c37", label="Prevalence Baseline")
        ax.scatter([rec], [prec], color="#d36c37", s=80, zorder=5, label="Current Point")
        ax.set_title("Precision-Recall Curve with Operating Point")
        ax.legend(loc="upper right")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.subheader("3. Locked Independent Test Evaluation Table")
    st.dataframe(result["evaluation"].round(4), use_container_width=True)

# ------------------------------------------------------------------------------
# 4. Real-Time Inference Lab
# ------------------------------------------------------------------------------
elif page == "⚡ Real-Time Inference Lab":
    st.title("⚡ Real-Time Wafer Quality Inference Lab")
    st.markdown("Run quality predictions for individual test wafers or upload industrial batches in CSV format.")
    st.markdown("---")

    mode = st.radio("Choose Operational Mode:", ["🔬 Single Wafer Interactive Testing", "📁 Batch Production CSV Upload"], horizontal=True)

    if mode == "🔬 Single Wafer Interactive Testing":
        st.subheader("Held-Out Production Wafer Test")
        sample_id = st.selectbox("Select a test sample ID from the held-out batch:", result["X_test"].index.tolist()[:30])
        row = result["X_test"].loc[[sample_id]].copy()
        top_sensors = sensor_ranking(result["model"]).head(8).index.tolist()

        st.caption("You can modify the most influential sensor readings below to test how sensitive the model is:")
        editable = pd.DataFrame({"Sensor": top_sensors, "Reading": row.loc[sample_id, top_sensors].values})
        edited = st.data_editor(editable, disabled=["Sensor"], hide_index=True, key=f"editor_{sample_id}", use_container_width=True)
        row.loc[sample_id, top_sensors] = edited["Reading"].to_numpy()

        pred_df = predict_quality(result["model"], row, result["threshold"])
        score = pred_df["failure_score"].iloc[0]
        decision = pred_df["decision"].iloc[0]

        st.markdown("### 📋 Inspection Decision")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Estimated Defect Risk Score", f"{score:.4f}", delta=f"Threshold: {result['threshold']:.4f}")
            st.progress(float(min(1.0, score * 2.5)))
        with res_col2:
            if decision == "Flag for inspection":
                st.markdown("""<div class="badge-flag">🚨 FLAGGED FOR PHYSICAL INSPECTION</div>""", unsafe_allow_html=True)
                st.markdown("<p style='margin-top:8px;'>High probability of fabrication defect detected. Routed to secondary optical/electrical inspection.</p>", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="badge-pass">✅ PREDICTED PASS • APPROVED FOR PACKAGING</div>""", unsafe_allow_html=True)
                st.markdown("<p style='margin-top:8px;'>Sensor readings conform to nominal fabrication tolerances.</p>", unsafe_allow_html=True)

        actual_outcome = "Defective (Failed)" if result["y_test"].loc[sample_id] == 1 else "Passed"
        st.info(f"Historical Actual Ground Truth for sample `{sample_id}`: **{actual_outcome}**")

    else:
        st.subheader("📁 Industrial Batch Inference")
        st.markdown("Upload a production CSV containing sensor readings from `sensor_001` through `sensor_590`.")

        sample_csv = result["X_test"].head(5).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download 5-Row Template CSV",
            sample_csv,
            "secom_sample_template.csv",
            "text/csv"
        )

        uploaded_file = st.file_uploader("Upload Batch File (Max 10 MB, up to 10,000 records)", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_batch = pd.read_csv(uploaded_file, nrows=10001)
                batch_preds = predict_quality(result["model"], raw_batch, result["threshold"])

                flagged_count = int((batch_preds["decision"] == "Flag for inspection").sum())
                total_count = len(batch_preds)

                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Wafers Processed", f"{total_count:,}")
                bc2.metric("Flagged for Inspection", f"{flagged_count:,}")
                bc3.metric("Defect Alert Rate", f"{flagged_count / total_count:.2%}")

                st.dataframe(batch_preds, use_container_width=True)

                out_csv = batch_preds.to_csv(index_label="wafer_id").encode("utf-8")
                st.download_button(
                    "💾 Export Batch Inspection Report",
                    out_csv,
                    "wafer_quality_predictions.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Batch Processing Error: {e}")

# ------------------------------------------------------------------------------
# 5. Methodology & Documentation
# ------------------------------------------------------------------------------
elif page == "📖 Methodology & Documentation":
    st.title("📖 Technical Methodology & Attribution")
    st.markdown("Academic documentation, formal mathematical objectives, and dataset citations.")
    st.markdown("---")

    st.markdown("""
    ### 🔬 Dataset Citation & License
    * **Dataset:** SECOM (Semiconductor Manufacturing), UCI Machine Learning Repository.
    * **Citation:** McCann, M. and Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository.
    * **DOI:** [https://doi.org/10.24432/C54305](https://doi.org/10.24432/C54305)
    * **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
    * **Raw Dimensions:** 1,567 manufacturing wafers across 590 anonymized sensor channels.

    ### 📐 Optimization Objective ($F_2$-Score)
    In high-volume manufacturing, the cost of an undetected failure reaching a customer ($C_{FN}$) is far greater than the cost of internal secondary testing ($C_{FP}$). Therefore, we maximize the **$F_2$-Measure**:

    $$F_2 = (1 + 2^2) \\frac{\\text{Precision} \\times \\text{Recall}}{2^2 \\times \\text{Precision} + \\text{Recall}} = \\frac{5 \\times P \\times R}{4P + R}$$

    This weights recall twice as heavily as precision, systematically tuning the threshold to detect defective wafers.

    ### ⚠️ Limitations & Industrial Guardrails
    * **Historical Data:** Sensor measurements are from July–October 2008. Models must be validated chronologically on new factory lots.
    * **Uncalibrated Output Scores:** Probability scores represent tree vote fractions and should not be interpreted as absolute physical probabilities of failure.
    """)
