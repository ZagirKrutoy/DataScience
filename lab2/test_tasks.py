
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput


# Проверка Titanic
titanic = analyze_titanic(
    TitanicInput(csv_path="titanic.csv")
)

print("=== TITANIC ===")
print("Количество строк:", titanic.row_count)
print("Пропуски:", titanic.missing_by_column)
print("Старше 30 лет:", titanic.adults_over_30_count)
print("Средний возраст:", titanic.mean_age_by_pclass)
print("Выживаемость:", titanic.survival_rate_by_pclass)
print("Пять максимальных тарифов:", titanic.highest_fares)


# Проверка корреляций
brain = analyze_brain_correlations(
    BrainDataInput(csv_path="brainsize.txt")
)

print("\n=== BRAIN CORRELATIONS ===")
print("Мужчин:", brain.men_count)
print("Женщин:", brain.women_count)
print("Корреляции женщин:", brain.women_mri_correlation)
print("Корреляции мужчин:", brain.men_mri_correlation)
print("Самый коррелирующий признак:", brain.strongest_mri_feature)
