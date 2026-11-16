"""Запуск экспериментов и сохранение фактических метрик в experiment_results.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def metrics_dict(m):
    return {name: float(getattr(m, name)) for name in
            ("roc_auc", "accuracy", "precision", "recall", "f1")}


def main():
    # Регрессия
    students = pd.read_csv(ROOT / "Student_Performance.csv")
    students["Extracurricular Activities"] = (
        students["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )
    if students.isna().any().any():
        raise ValueError("Student_Performance.csv содержит пропуски или неизвестные значения.")
    x = students.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    y = students["Performance Index"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Масштабирование признаков выполняется внутри собственной модели только
    # по статистикам обучающей выборки.
    reg = train_linear_regression(RegressionInput(
        x_train, y_train, x_test, y_test,
        learning_rate=0.05, epochs=800
    ))
    regression_mse = float(np.mean((reg.predictions - y_test) ** 2))

    # Классификация. Файл German Credit использует разделитель ';'.
    german = pd.read_csv(ROOT / "german.csv", sep=";")
    target = german.iloc[:, 0]
    features = german.iloc[:, 1:]
    classes = sorted(pd.unique(target.dropna()).tolist())
    if len(classes) != 2:
        raise ValueError(
            "Невозможно выполнить бинарную классификацию: в доступном german.csv "
            f"обнаружено {len(classes)} класс(а): {classes}. Нужен полный CSV "
            "с обоими классами Creditability."
        )
    # Для исходной разметки 0/1 сохраняем значения; если метки 1/2,
    # преобразуем стандартную разметку German Credit: 1 — хороший, 2 — плохой.
    if set(classes) == {1, 2}:
        target = (target == 1).astype(int)
    else:
        target = target.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(dtype=float), target.to_numpy(dtype=int),
        test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )
    clf_data = ClassificationInput(
        x_train, y_train, x_test, y_test,
        learning_rate=0.1, epochs=1000
    )
    own = train_logistic_classifier(clf_data)
    own_metrics = metrics_dict(__import__("linear_models")._metrics(
        y_test, own.probabilities, own.predictions
    ))
    baselines = compare_classical_models(clf_data)

    result = {
        "linear_regression": {"test_mse": regression_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": metrics_dict(baselines.logistic_regression),
            "decision_tree": metrics_dict(baselines.decision_tree),
            "knn": metrics_dict(baselines.knn),
        },
    }
    (ROOT / "experiment_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
