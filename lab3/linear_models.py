"""Собственные линейные модели на пакетном градиентном спуске."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import (
    BaselineComparisonResult, ClassificationInput, ClassificationMetrics,
    ClassificationResult, RegressionInput, RegressionResult,
)


def _as_2d_features(values):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Признаки должны быть конечным двумерным числовым массивом.")
    return x


def _as_target(values):
    y = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(y).all():
        raise ValueError("Целевая переменная содержит NaN или бесконечность.")
    return y


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучает линейную регрессию пакетным градиентным спуском."""
    x = _as_2d_features(data.train_features)
    y = _as_target(data.train_target)
    x_test = _as_2d_features(data.test_features)
    if len(x) != len(y) or x.shape[1] != x_test.shape[1]:
        raise ValueError("Несогласованные размеры train/test.")
    if len(x) == 0 or data.epochs <= 0 or data.learning_rate <= 0:
        raise ValueError("Нужны непустые данные, положительные epochs и learning_rate.")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    xs = (x - mean) / scale
    x_test_s = (x_test - mean) / scale

    y_mean = float(y.mean())
    y_scale = float(y.std())
    if y_scale == 0:
        y_scale = 1.0
    ys = (y - y_mean) / y_scale

    weights_s = np.zeros(xs.shape[1], dtype=float)
    bias_s = 0.0
    history = []
    n = len(xs)
    for _ in range(int(data.epochs)):
        pred = xs @ weights_s + bias_s
        error = pred - ys
        history.append(float(np.mean((pred * y_scale + y_mean - y) ** 2)))
        grad_w = (2.0 / n) * (xs.T @ error)
        grad_b = 2.0 * float(error.mean())
        weights_s -= data.learning_rate * grad_w
        bias_s -= data.learning_rate * grad_b

    weights = y_scale * weights_s / scale
    bias = y_mean + y_scale * bias_s - float(mean @ weights)
    predictions = x_test @ weights + bias
    return RegressionResult(
        weights=weights.copy(), bias=float(bias),
        loss_history=history, predictions=predictions,
    )


def _sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучает бинарную логистическую регрессию пакетным градиентным спуском."""
    x = _as_2d_features(data.train_features)
    y = _as_target(data.train_target)
    x_test = _as_2d_features(data.test_features)
    if len(x) != len(y) or x.shape[1] != x_test.shape[1]:
        raise ValueError("Несогласованные размеры train/test.")
    classes = np.unique(y)
    if not np.array_equal(classes, np.array([0.0, 1.0])):
        raise ValueError("Для логистической регрессии target должен содержать оба класса 0 и 1.")
    if data.epochs <= 0 or data.learning_rate <= 0:
        raise ValueError("epochs и learning_rate должны быть положительными.")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    xs = (x - mean) / scale
    x_test_s = (x_test - mean) / scale

    w = np.zeros(xs.shape[1], dtype=float)
    b = 0.0
    history = []
    n = len(y)
    eps = 1e-12
    for _ in range(int(data.epochs)):
        probabilities = _sigmoid(xs @ w + b)
        loss = -np.mean(y * np.log(probabilities + eps)
                        + (1.0 - y) * np.log(1.0 - probabilities + eps))
        history.append(float(loss))
        error = probabilities - y
        w -= data.learning_rate * (xs.T @ error) / n
        b -= data.learning_rate * float(error.mean())

    weights = w / scale
    bias = b - float(mean @ weights)
    test_probabilities = _sigmoid(x_test @ weights + bias)
    predictions = (test_probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights.copy(), bias=float(bias), loss_history=history,
        probabilities=test_probabilities, predictions=predictions,
    )


def _metrics(y_true, probabilities, predictions):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=int).reshape(-1)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Для ROC-AUC в тестовой выборке должны присутствовать оба класса.")
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнивает три sklearn-модели на одном и том же train/test split."""
    x_train = _as_2d_features(data.train_features)
    x_test = _as_2d_features(data.test_features)
    y_train = _as_target(data.train_target).astype(int)
    y_test = _as_target(data.test_target).astype(int)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("В train и test должны присутствовать оба класса 0 и 1.")

    own = train_logistic_classifier(data)
    own_metrics = _metrics(y_test, own.probabilities, own.predictions)

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    logreg.fit(x_train, y_train)
    logreg_prob = logreg.predict_proba(x_test)[:, 1]
    logreg_pred = (logreg_prob >= 0.5).astype(int)

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x_train, y_train)
    tree_prob = tree.predict_proba(x_test)[:, 1]
    tree_pred = tree.predict(x_test)

    k = min(5, len(x_train))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(x_train, y_train)
    knn_prob = knn.predict_proba(x_test)[:, 1]
    knn_pred = knn.predict(x_test)

    return BaselineComparisonResult(
        logistic_regression=_metrics(y_test, logreg_prob, logreg_pred),
        decision_tree=_metrics(y_test, tree_prob, tree_pred),
        knn=_metrics(y_test, knn_prob, knn_pred),
    )
