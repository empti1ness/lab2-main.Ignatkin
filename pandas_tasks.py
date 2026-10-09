"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    filepath = getattr(data, 'file_path', getattr(data, 'path', str(data)))

    df = pd.read_csv(filepath)

    missing_counts = df.isna().sum().to_dict()

    over_30_count = int((df['Age'] > 30).sum())

    mean_age = df.groupby('Pclass')['Age'].mean().to_dict()

    survival_rate = df.groupby('Pclass')['Survived'].mean().to_dict()

    top_5_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        missing_counts = missing_counts,
        over_30_count = over_30_count,
        mean_age = mean_age,
        survival_rate = survival_rate,
        top_5_fares = top_5_fares
    )