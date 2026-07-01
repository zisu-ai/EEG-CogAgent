from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


class NuisanceResidualizer(BaseEstimator, TransformerMixin):
    """Remove nuisance-associated EEG variance using training-fold data only.

    The input matrix must contain EEG features first and nuisance variables last.
    The transformer returns only residualized EEG features. Because it follows the
    scikit-learn estimator API, fitting occurs independently inside every CV fold.
    """

    def __init__(self, n_eeg_features: int):
        self.n_eeg_features = n_eeg_features

    def fit(self, X, y=None):
        values = self._as_array(X)
        if self.n_eeg_features <= 0 or self.n_eeg_features >= values.shape[1]:
            raise ValueError("n_eeg_features must leave at least one nuisance column.")

        eeg = values[:, : self.n_eeg_features]
        nuisance = values[:, self.n_eeg_features :]
        self.eeg_imputer_ = SimpleImputer(strategy="median")
        self.nuisance_imputer_ = SimpleImputer(strategy="median")
        eeg_filled = self.eeg_imputer_.fit_transform(eeg)
        nuisance_filled = self.nuisance_imputer_.fit_transform(nuisance)
        self.nuisance_scaler_ = StandardScaler()
        nuisance_scaled = self.nuisance_scaler_.fit_transform(nuisance_filled)
        design = np.column_stack([np.ones(len(nuisance_scaled)), nuisance_scaled])
        self.coefficients_ = np.linalg.lstsq(design, eeg_filled, rcond=None)[0]
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = self._as_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} input columns, got {values.shape[1]}."
            )
        eeg = self.eeg_imputer_.transform(values[:, : self.n_eeg_features])
        nuisance = self.nuisance_imputer_.transform(values[:, self.n_eeg_features :])
        nuisance_scaled = self.nuisance_scaler_.transform(nuisance)
        design = np.column_stack([np.ones(len(nuisance_scaled)), nuisance_scaled])
        return eeg - design @ self.coefficients_

    @staticmethod
    def _as_array(X) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Expected a two-dimensional feature matrix.")
        return values


def _model_grid(name: str, random_state: int):
    if name == "logistic_regression":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        return pipe, {"model__C": [0.1, 1.0, 10.0]}
    if name == "svm_rbf":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
        ])
        return pipe, {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]}
    if name == "random_forest":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state)),
        ])
        return pipe, {"model__n_estimators": [300], "model__max_depth": [None, 3, 5]}
    raise ValueError(f"Unknown model: {name}")


def evaluate_models(features: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml_cfg = cfg["ml"]
    label_col = ml_cfg.get("label_column", "label")
    meta_cols = set(ml_cfg.get("exclude_columns", []))
    meta_cols.update({"participant_id", label_col, "label", "Group", "Gender", "Age", "MMSE", "n_epochs"})
    feature_cols = [
        col for col in features.select_dtypes(include=[np.number]).columns
        if col not in meta_cols
    ]
    if not feature_cols:
        raise ValueError("No EEG feature columns remain after excluding metadata columns.")
    X = features[feature_cols]
    labels = features[label_col].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = list(encoder.classes_)

    min_class = labels.value_counts().min()
    outer_splits = min(int(ml_cfg.get("cv_folds", 5)), int(min_class))
    if outer_splits < 2:
        raise ValueError("Need at least two samples per class for cross-validation.")

    random_state = int(ml_cfg.get("random_state", 42))
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    rows = []
    predictions = []

    for model_name in ml_cfg.get("models", ["logistic_regression"]):
        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            estimator, grid = _model_grid(model_name, random_state + fold)
            train_min_class = int(np.bincount(y[train_idx]).min())
            inner_splits = min(max(outer_splits - 1, 2), train_min_class)
            if inner_splits >= 2:
                inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold)
                search = GridSearchCV(
                    estimator,
                    grid,
                    cv=inner,
                    scoring=ml_cfg.get("scoring", "balanced_accuracy"),
                    n_jobs=-1,
                )
                search.fit(X.iloc[train_idx], y[train_idx])
                fitted = search
            else:
                estimator.fit(X.iloc[train_idx], y[train_idx])
                fitted = estimator
            pred = fitted.predict(X.iloc[test_idx])
            prob = fitted.predict_proba(X.iloc[test_idx])

            y_true_all.extend(y[test_idx])
            y_pred_all.extend(pred)
            y_prob_all.extend(prob)
            for idx, pred_label in zip(test_idx, encoder.inverse_transform(pred)):
                predictions.append({
                    "model": model_name,
                    "fold": fold,
                    "participant_id": features.iloc[idx]["participant_id"],
                    "true_label": labels.iloc[idx],
                    "pred_label": pred_label,
                })

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)
        y_prob_arr = np.array(y_prob_all)
        auc = np.nan
        try:
            if len(classes) == 2:
                auc = roc_auc_score(y_true_arr, y_prob_arr[:, 1])
            else:
                auc = roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", labels=np.arange(len(classes)))
        except ValueError:
            pass

        rows.append({
            "model": model_name,
            "n_features": len(feature_cols),
            "cv_folds": outer_splits,
            "accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
            "auc_ovr": auc,
        })

    return pd.DataFrame(rows), pd.DataFrame(predictions)


def evaluate_residualized_models(
    features: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate EEG classifiers after leakage-safe nuisance residualization.

    Age, binary gender, and log retained-epoch count are used only to remove
    nuisance-associated variance from EEG features. They are not passed to the
    classifier as predictive features.
    """
    ml_cfg = cfg["ml"]
    label_col = ml_cfg.get("label_column", "label")
    meta_cols = set(ml_cfg.get("exclude_columns", []))
    meta_cols.update({"participant_id", label_col, "label", "Group", "Gender", "Age", "MMSE", "n_epochs"})
    feature_cols = [
        col for col in features.select_dtypes(include=[np.number]).columns
        if col not in meta_cols
    ]
    if not feature_cols:
        raise ValueError("No EEG feature columns remain after excluding metadata columns.")

    gender = features["Gender"].astype("string").str.upper()
    nuisance = pd.DataFrame({
        "nuisance_age": pd.to_numeric(features["Age"], errors="coerce"),
        "nuisance_gender_m": gender.map({"F": 0.0, "M": 1.0}).astype(float),
        "nuisance_log_epochs": np.log1p(
            pd.to_numeric(features["n_epochs"], errors="coerce").clip(lower=0)
        ),
    }, index=features.index)
    X = pd.concat([features[feature_cols].astype(float), nuisance], axis=1)
    labels = features[label_col].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = list(encoder.classes_)

    min_class = labels.value_counts().min()
    outer_splits = min(int(ml_cfg.get("cv_folds", 5)), int(min_class))
    if outer_splits < 2:
        raise ValueError("Need at least two samples per class for cross-validation.")

    random_state = int(ml_cfg.get("random_state", 42))
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    rows = []
    predictions = []

    for model_name in ml_cfg.get("models", ["logistic_regression"]):
        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            base_estimator, grid = _model_grid(model_name, random_state + fold)
            estimator = Pipeline([
                ("residualize", NuisanceResidualizer(n_eeg_features=len(feature_cols))),
                *base_estimator.steps,
            ])
            train_min_class = int(np.bincount(y[train_idx]).min())
            inner_splits = min(max(outer_splits - 1, 2), train_min_class)
            if inner_splits >= 2:
                inner = StratifiedKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=random_state + fold,
                )
                fitted = GridSearchCV(
                    estimator,
                    grid,
                    cv=inner,
                    scoring=ml_cfg.get("scoring", "balanced_accuracy"),
                    n_jobs=-1,
                )
            else:
                fitted = estimator
            fitted.fit(X.iloc[train_idx], y[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
            prob = fitted.predict_proba(X.iloc[test_idx])

            y_true_all.extend(y[test_idx])
            y_pred_all.extend(pred)
            y_prob_all.extend(prob)
            for row_position, pred_code, probabilities in zip(test_idx, pred, prob):
                row = {
                    "model": model_name,
                    "fold": fold,
                    "participant_id": features.iloc[row_position]["participant_id"],
                    "true_label": labels.iloc[row_position],
                    "pred_label": encoder.inverse_transform([pred_code])[0],
                }
                row.update({
                    f"prob_{label}": float(probabilities[class_index])
                    for class_index, label in enumerate(classes)
                })
                predictions.append(row)

        y_true_arr = np.asarray(y_true_all)
        y_pred_arr = np.asarray(y_pred_all)
        y_prob_arr = np.asarray(y_prob_all)
        try:
            if len(classes) == 2:
                auc = roc_auc_score(y_true_arr, y_prob_arr[:, 1])
            else:
                auc = roc_auc_score(
                    y_true_arr,
                    y_prob_arr,
                    multi_class="ovr",
                    labels=np.arange(len(classes)),
                )
        except ValueError:
            auc = np.nan

        rows.append({
            "model": model_name,
            "n_features": len(feature_cols),
            "n_nuisance_variables": nuisance.shape[1],
            "cv_folds": outer_splits,
            "accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
            "auc_ovr": auc,
        })

    return pd.DataFrame(rows), pd.DataFrame(predictions)
