
# Manufacturing Quality Prediction
# Author: Poojan Chauhan

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
from pathlib import Path

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.20
TOP_K = 40
DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/179/secom"
FEATURES = [f"sensor_{i:03d}" for i in range(1, 591)]
plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def load_secom(data_dir="data"):
    """Download the original UCI ZIP once; accept local raw files for offline use.

    Sensor names are one-based: sensor_001 is original raw column zero.
    No sensor measurements are synthesized or given invented physical units.
    """
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
                    "Could not download SECOM. Download the ZIP from " + DATA_PAGE +
                    " and put secom.data and secom_labels.data in the data folder, "
                    "then rerun this cell. No demonstration data will be substituted."
                ) from exc
        # Read only known members; do not extract arbitrary archive paths.
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
    # The entire pipeline is refitted independently within each CV fold.
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
    """Return the score for class 1 explicitly; do not assume class ordering."""
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
    """Maximize training OOF F2; ties prefer the higher threshold.

    F2 weighs recall more than precision. This is an educational objective,
    not an estimated factory cost function. Test labels are never used here.
    """
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
    """Score valid records. Scores are not calibrated failure probabilities."""
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
        "threshold": threshold,
        "missing_readings": frame.isna().sum(axis=1),
    }, index=frame.index)


def sensor_ranking(model):
    """Training-derived model associations, not physical causes of failure."""
    selected = np.asarray(model.named_steps["filter"].columns_)[
        model.named_steps["select"].get_support()]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
        method = "Tree impurity importance"
    else:
        weights = np.abs(estimator.coef_[0])
        method = "Absolute standardized coefficient"
    return pd.Series(weights, index=selected, name=method).sort_values(ascending=False)

# Streamlit UI appended to the same functions used in the notebook.
import streamlit as st

st.set_page_config(page_title="Manufacturing Quality Prediction", page_icon="🏭", layout="wide")
st.title("Manufacturing Quality Prediction")
st.caption("Poojan Chauhan · SECOM semiconductor analytics · Academic prototype")

@st.cache_resource(show_spinner="Loading SECOM and fitting the models. The first run takes longer.")
def get_experiment():
    X, y, timestamps, hashes = load_secom()
    return X, y, timestamps, train_experiment(X, y)

try:
    X, y, timestamps, result = get_experiment()
except Exception as exc:
    st.error(f"Unable to initialize the project: {exc}")
    st.stop()

page = st.sidebar.radio("Navigate", ["Overview", "Sensor analysis", "Model evaluation", "Predict quality"])
st.sidebar.markdown("**Manufacturing Quality Prediction**")
st.sidebar.caption("IBM SkillsBuild Data Analytics with AI Academic Internship\n\nBharatCares in association with AICTE")
st.sidebar.caption("Pass = 0 · Failure = 1\n\nSensors are anonymous; their physical units are unknown.")
st.sidebar.link_button("Original UCI dataset", DATA_PAGE)

if page == "Overview":
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Production records", "Sensors", "Recorded failures", "Failure rate"],
                                 [f"{len(X):,}", X.shape[1], int(y.sum()), f"{y.mean():.2%}"]):
        col.metric(label, value)
    st.subheader("Detect products that may need inspection")
    st.write("This app explores semiconductor sensor measurements and uses machine learning to flag potential test failures. "
             "Failure detection is assessed using recall, precision, average precision and a confusion matrix.")
    left, right = st.columns(2)
    with left:
        st.subheader("Recorded quality outcomes")
        st.bar_chart(y.map({0: "Pass", 1: "Failure"}).value_counts(), color="#167d9a")
    with right:
        st.subheader("Experiment design")
        st.write("80% development data; 20% held-out test data. Five-fold cross-validation on development data selects "
                 "the model by average precision. Out-of-fold F2 selects the inspection threshold.")
        st.metric("Selected model", result["name"])
        st.write(f"Locked inspection threshold: **{result['threshold']:.4f}**")
    st.info("Educational demonstration on historical data. Predicted pass is not a quality certificate. "
            "Scores are uncalibrated model outputs, not measured probabilities of failure.")

elif page == "Sensor analysis":
    st.subheader("Explore development data")
    st.caption("These charts use only the training partition. The test set is reserved for final evaluation.")
    train = result["X_train"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing sensor cells", f"{train.isna().to_numpy().mean():.2%}")
    c2.metric("Usable sensors before selection", len(result["model"].named_steps["filter"].columns_))
    c3.metric("Selected sensors", TOP_K)
    st.bar_chart((train.isna().mean() * 100).nlargest(20).rename("Missing readings (%)"), horizontal=True)
    sensor = st.selectbox("Sensor distribution", result["model"].named_steps["filter"].columns_)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, color in [(0, "#167d9a"), (1, "#d36c37")]:
        values = train.loc[result["y_train"] == label, sensor].dropna()
        if len(values):
            ax.hist(values, bins=25, density=True, alpha=.55,
                    label="Pass" if label == 0 else "Failure", color=color)
    ax.set(xlabel=f"{sensor} (original scale; units unknown)", ylabel="Density")
    ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.subheader("Training-derived feature associations")
    ranking = sensor_ranking(result["model"])
    st.bar_chart(ranking.head(15), horizontal=True)
    st.caption(f"Method: {ranking.name}. Associations do not prove causation; correlated sensors can distort importance.")

elif page == "Model evaluation":
    st.subheader("Training-only model selection")
    st.dataframe(result["cv_summary"].round(4), width="stretch")
    st.caption("Mean_AP is mean five-fold average precision; SD_AP is fold-to-fold variation. "
               "These are model-selection scores, not independent final performance estimates.")
    st.subheader("Held-out test results")
    st.dataframe(result["evaluation"].round(4), width="stretch")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        predicted = (result["scores"] >= result["threshold"]).astype(int)
        ConfusionMatrixDisplay.from_predictions(result["y_test"], predicted, labels=[0, 1],
                                                display_labels=["Pass", "Failure"], cmap="Blues", ax=ax,
                                                colorbar=False)
        ax.set_title("Selected model at the locked threshold")
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(result["y_test"], result["scores"], ax=ax)
        ax.axhline(result["y_test"].mean(), ls="--", color="#777777", label="Failure prevalence")
        ax.legend(); ax.set_title("Precision and recall trade-off")
        st.pyplot(fig); plt.close(fig)
    st.write("FN means a failed product was missed; FP means a passing product was unnecessarily flagged. "
             "The F2 threshold emphasizes catching failures and can increase false alarms.")
    st.warning("The test split contains few failures. This random-split result does not demonstrate future factory performance. "
               "A chronological or new-production validation set is required before practical use.")

else:
    st.subheader("Predict quality from sensor readings")
    threshold = st.slider("Inspection threshold", 0.0, 1.0, float(result["threshold"]), step=0.001,
                          format="%.3f")
    st.caption("Lower thresholds flag more records. Changing this control does not change the locked evaluation results.")
    mode = st.radio("Input mode", ["Held-out sample demo", "Upload CSV"], horizontal=True)
    if mode == "Held-out sample demo":
        sample_id = st.selectbox("Choose a held-out sample", result["X_test"].index.tolist())
        row = result["X_test"].loc[[sample_id]].copy()
        top_sensors = sensor_ranking(result["model"]).head(10).index.tolist()
        editable = pd.DataFrame({"Sensor": top_sensors, "Reading": row.loc[sample_id, top_sensors].values})
        st.caption("Edit up to ten model-relevant readings. Other sensors retain this sample's recorded values. "
                   "Blank readings use training medians. This is a model sensitivity demo, not a causal intervention.")
        edited = st.data_editor(editable, disabled=["Sensor"], hide_index=True,
                                key=f"sample_{sample_id}", width="stretch")
        original = row.copy()
        row.loc[sample_id, top_sensors] = edited["Reading"].to_numpy()
        try:
            prediction = predict_quality(result["model"], row, threshold)
            st.dataframe(prediction, width="stretch")
            actual = "Failure" if result["y_test"].loc[sample_id] else "Pass"
            st.write(f"Recorded outcome of the original sample: **{actual}**")
            if not row.equals(original):
                st.caption("The recorded outcome belongs to the unedited sample only.")
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.write("Use exactly sensor_001 through sensor_590 as headers; one production record per row. "
                 "Leave missing readings blank. Do not include labels, timestamps or an index column.")
        sample = result["X_test"].head(5).to_csv(index=False).encode("utf-8")
        st.download_button("Download five-row example CSV", sample, "secom_example.csv", "text/csv")
        upload = st.file_uploader("Sensor CSV (maximum 10 MB and 10,000 rows)", type=["csv"])
        if upload is not None:
            try:
                if upload.size > 10 * 1024 * 1024:
                    raise ValueError("File exceeds the 10 MB limit.")
                raw = pd.read_csv(upload, nrows=10001)
                if len(raw) > 10000:
                    raise ValueError("Use no more than 10,000 records per upload.")
                output = predict_quality(result["model"], raw, threshold)
                c1, c2 = st.columns(2)
                c1.metric("Rows scored", len(output))
                c2.metric("Flagged for inspection", int((output["decision"] == "Flag for inspection").sum()))
                st.dataframe(output, width="stretch")
                st.download_button("Download predictions", output.to_csv(index_label="row_id").encode("utf-8"),
                                   "quality_predictions.csv", "text/csv")
            except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))
    st.caption("Use readings from the same SECOM sensor schema and scale. Unrelated factory measurements are not supported.")

