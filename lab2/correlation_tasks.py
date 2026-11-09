
"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import (
    BrainCorrelationSummary,
    BrainDataInput,
)


def analyze_brain_correlations(
    data: BrainDataInput,
) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt."""

    df = pd.read_csv(
        data.csv_path,
        sep="\t",
        na_values=["NA"],
    )
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]
    men_mri_correlation = {
        feature: float(
            men[feature].corr(men["MRI_Count"], method="pearson")
        )
        for feature in features
    }

    women_mri_correlation = {
        feature: float(
            women[feature].corr(women["MRI_Count"], method="pearson")
        )
        for feature in features
    }
    strongest_mri_feature = max(
        features,
        key=lambda feature: max(
            abs(men_mri_correlation[feature]),
            abs(women_mri_correlation[feature]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )
